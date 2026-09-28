from __future__ import annotations

import logging
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from phaseforge.models.base import BaseManipulationModel, ModelOutput
from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.clustering import (
    compute_hierarchical_phase_prototypes,
    compute_phase_centroids,
    compute_phase_head_router_weights,
    spherical_kmeans,
)
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.expert import (
    ExpertMLP,
    hash_dropped_indices,
    one_warm_experts_from_action_head,
    partial_reinit_experts_from_action_head,
    warm_start_experts_from_action_head,
)
from phaseforge.models.components.impedance_expert import (
    ImpedanceExpert,
    ResidualImpedanceExpert,
)
from phaseforge.models.components.moe_layer import MoELayer
from phaseforge.models.components.phase_head import PhaseClassificationHead
from phaseforge.models.components.prototype_router import PrototypeRouter
from phaseforge.models.components.router import TopKRouter
from phaseforge.models.components.soft_mapping import (
    build_hierarchical_uniform_mapping,
    build_prototype_softmax_mapping,
    validate_soft_mapping,
)

logger = logging.getLogger(__name__)

#: V2-E evaluation-time routing interventions.
EVAL_ROUTER_MODES: frozenset[str] = frozenset({"learned", "sticky", "uniform", "oracle"})


def _hash_dropped_indices(indices: list[int]) -> str:
    """Backward-compatible alias of the shared audit hash."""
    return hash_dropped_indices(indices)


def _install_router_matrix(router: Any, matrix: Tensor, description: str) -> None:
    """Install bootstrapped directions into either router flavor.

    Prototype routers store them as Voronoi prototypes; legacy TopK routers
    store them as gate hyperplanes with a zeroed bias. Shapes must match
    exactly — a mismatch fails closed instead of silently misrouting.
    """
    proto = getattr(router, "prototypes", None)
    if isinstance(proto, torch.nn.Parameter):
        if tuple(proto.shape) != tuple(matrix.shape):
            raise ValueError(
                "Cannot install bootstrapped prototypes: router expects "
                f"{tuple(proto.shape)}, computed {tuple(matrix.shape)}."
            )
        proto.data.copy_(matrix)
        logger.info("Initialized prototype router with %s", description)
        return
    router.gate_linear.weight.data.copy_(matrix)
    router.gate_linear.bias.data.zero_()
    logger.info("Initialized router weights with %s", description)


class PhaseBootstrappedMoE(BaseManipulationModel):
    # Static annotation for the registered buffer so mypy resolves
    # ``self.soft_mapping`` as a Tensor instead of nn.Module.__getattr__'s
    # ``Tensor | Module`` fallback.
    soft_mapping: Tensor

    """Phase-Bootstrapped Mixture-of-Experts.

    This model operates in two distinct stages:
    Stage 1: Generalist pretraining with auxiliary phase supervision.
             Forward pass uses encoder -> action_head + phase_head.
    Stage 2: Bootstrapped MoE specialization.
             Forward pass uses encoder -> moe_layer.

    The transition between stages is mediated by the `bootstrap_moe()` method,
    which transfers privileged regime structure (or generic clustering / phase head
    weights) into the MoE router and warm-starts or resets experts according to
    configuration.

    Args:
        encoder: The StateEncoder instance.
        action_head: The ActionHead used in Stage 1.
        phase_head: The PhaseClassificationHead used in Stage 1.
        router: The TopKRouter (or PrototypeRouter) for Stage 2.
        expert: A single ExpertMLP (or ImpedanceExpert) template to be cloned
            for Stage 2. Impedance experts output target/gain controller
            parameters through the action adapter instead of raw actions.
        router_init: Optional configuration dictionary for router initialization.
            Keys: 'type' ('centroid' | 'spherical_centroid' | 'spherical_kmeans' |
                  'kmeans' | 'phase_head' | 'soft_mapping' | 'random'), 'seed' (int),
                  'mapping_mode'/'temperature' (soft_mapping only).
        expert_init: Optional configuration dictionary for expert initialization.
            Keys: 'type' ('warmstart' | 'partial_warm' | 'one_warm' | 'random'),
            'jitter_std' (float; warmstart/one_warm), 'drop_rate' (float;
            partial_warm, default 0.5), 'warm_idx' (int; one_warm, default 0),
            'rotate_warm_idx_by_seed' (bool; one_warm, default false),
            'seed' (int; partial_warm, default 42).
        soft_mapping: Optional configuration dictionary for the soft
            phase-to-expert mapping M (V2-B; see
            phaseforge.models.components.soft_mapping). The (P, E) matrix
            is a persistent zero-initialized buffer, built during
            ``bootstrap_moe()`` when ``router_init.type='soft_mapping'``,
            and carried in checkpoints from then on.
        teacher_routing: Optional configuration dictionary for V2-D
            teacher-distilled routing. Only ``enabled`` is read here; the
            trainer-side knobs (lambda0, annealing) live under
            ``train.teacher_routing``.
    """

    def __init__(
        self,
        encoder: StateEncoder,
        action_head: ActionHead,
        phase_head: PhaseClassificationHead,
        router: TopKRouter | PrototypeRouter,
        expert: ExpertMLP | ImpedanceExpert | ResidualImpedanceExpert,
        router_init: dict[str, Any] | None = None,
        expert_init: dict[str, Any] | None = None,
        soft_mapping: dict[str, Any] | None = None,
        teacher_routing: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder

        # Stage 1 components
        self.action_head = action_head
        self.phase_head = phase_head

        # Stage 2 components
        self.moe_layer = MoELayer(router=router, experts=expert)

        # Initialization strategy configurations
        self.router_init_cfg = (
            dict(router_init) if router_init is not None else {"type": "centroid"}
        )
        self.expert_init_cfg = (
            dict(expert_init)
            if expert_init is not None
            else (
                {"type": "random"}
                if self._uses_impedance_experts()
                else {"type": "warmstart", "jitter_std": 0.02}
            )
        )
        self.soft_mapping_cfg = dict(soft_mapping) if soft_mapping is not None else {}
        self.teacher_routing_cfg = dict(teacher_routing) if teacher_routing is not None else {}

        # Audit metadata populated by ``bootstrap_moe`` after a successful
        # Stage 1 -> Stage 2 transition. Initialized to ``None`` so the model
        # state is well-defined before Stage 2 and consumers (cli metadata
        # writer, tests) can rely on attribute existence rather than relying
        # on dynamic attribute creation.
        self._expert_init_info: dict[str, Any] | None = None

        # Persistent soft phase->expert mapping M (P, E). Zero-initialized:
        # a checkpoint that predates M (or a stage-1 checkpoint) loads into
        # this model leaving M zeroed, and the teacher/oracle paths fail
        # closed on the all-zero matrix instead of silently routing with it.
        self.register_buffer(
            "soft_mapping",
            torch.zeros(
                int(self.phase_head.num_phases),
                int(self.moe_layer.router.num_experts),
            ),
        )

        # Internal state to track which stage the model is currently configured for
        self._stage = 1
        self._encoder_frozen = False

        # V2-E: evaluation-time routing intervention ("learned" = the trained
        # router; the others are evaluation-only baselines).
        self._eval_mode = "learned"

        # Storage for the most recent routing information for metrics tracking
        self._last_gate_logits: Tensor | None = None

    @property
    def eval_mode(self) -> str:
        """The active V2-E evaluation-time routing intervention.

        One of ``learned`` / ``sticky`` / ``uniform`` / ``oracle`` (default
        ``learned``). The interventions are evaluation-time only: they
        replace the dispatch selection, never the trained gate weights.
        """
        return self._eval_mode

    @eval_mode.setter
    def eval_mode(self, mode: str) -> None:
        mode = str(mode).lower()
        if mode not in EVAL_ROUTER_MODES:
            raise ValueError(
                f"Unknown eval_mode {mode!r}. Supported: "
                + ", ".join(sorted(EVAL_ROUTER_MODES))
                + "."
            )
        self._eval_mode = mode

    def deployment_contract(self) -> dict[str, Any]:
        """Deployment contract for the switched MoE policy.

        Reports the live routing configuration (expert count, top-k,
        router class). The policy itself stays memoryless: ``get_action``
        consumes one state and returns one action with no cross-step
        state (the sticky-EMA lives only in the legacy ``sticky`` eval
        mode and is cleared per episode by :meth:`reset`).
        """
        router = self.moe_layer.router
        return {
            "memoryless": True,
            "router_type": type(router).__name__,
            "expert_type": (
                "residual_impedance"
                if isinstance(self.moe_layer.experts[0], ResidualImpedanceExpert)
                else ("impedance" if self._uses_impedance_experts() else "direct")
            ),
            "top_k": int(router.top_k),
            "num_experts": int(router.num_experts),
        }

    def reset(self) -> None:
        """Clear per-episode evaluation state (start of a new rollout episode).

        The rollout runner calls ``reset()`` on the model before every
        episode when the callable exists; the sticky-EMA eval mode must
        start each episode from a clean EMA.
        """
        self.moe_layer.router.reset_sticky_ema()

    def _uses_impedance_experts(self) -> bool:
        """True when the Stage 2 experts are impedance-parameterized."""
        return isinstance(self.moe_layer.experts[0], ImpedanceExpert)

    def _task_state(self, state: Tensor) -> Tensor:
        """Extract the task state ``y = ψ(x)`` for impedance experts."""
        from phaseforge.models.components.task_state import extract_task_state

        mean, std = self.get_normalizer_stats()
        return extract_task_state(state, mean=mean, std=std)

    @property
    def teacher_routing_enabled(self) -> bool:
        """Whether the V2-D teacher path is active for this model.

        Single source of truth is ``models.teacher_routing.enabled`` (the
        same config block the trainer reads), mirrored here so the forward
        pass and the trainer cannot disagree.
        """
        return bool(self.teacher_routing_cfg.get("enabled", False))

    @property
    def stage(self) -> int:
        return self._stage

    @stage.setter
    def stage(self, value: int) -> None:
        if value not in (1, 2):
            raise ValueError(f"Stage must be 1 or 2, got {value}")
        self._stage = value
        logger.info(f"PhaseBootstrappedMoE transitioned to Stage {value}.")

    def freeze_encoder(self) -> None:
        """Freeze the encoder for Stage 2 training.

        Disables gradients AND keeps the encoder in eval mode: a frozen
        encoder must not apply dropout, or its latent representation stays
        stochastic during training despite being frozen.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        self._encoder_frozen = True
        self.encoder.eval()
        logger.info("Encoder weights frozen; encoder kept in eval mode (no dropout).")

    def unfreeze_encoder(self) -> None:
        """Unfreeze the encoder for fine-tuning in Stage 2 (PhaseForge-FT)."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self._encoder_frozen = False
        logger.info("Encoder weights unfrozen for Stage 2 fine-tuning.")

    def train(self, mode: bool = True) -> PhaseBootstrappedMoE:
        """Override so a frozen encoder stays deterministic during Stage 2.

        The training loop calls ``model.train()`` every epoch, which would
        otherwise re-enable the encoder's dropout when frozen.
        """
        super().train(mode)
        if mode and self._encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, batch: dict[str, Tensor]) -> ModelOutput:
        """Forward pass depends on the active stage.

        Args:
            batch: Dictionary containing "state", "action", "phase", etc.

        Returns:
            ModelOutput containing predictions, logits, and auxiliary losses.
        """
        state = batch["state"]
        latent = self.encoder(state)

        if self._stage == 1:
            # Stage 1: Generalist action prediction + phase classification
            action_pred = self.action_head(latent)
            phase_logits = self.phase_head(latent)

            return ModelOutput(
                action_pred=action_pred,
                phase_logits=phase_logits,
                # MoE fields are empty in Stage 1
                routing_weights=None,
                expert_indices=None,
                gate_logits=None,
                latent=latent,
            )

        elif self._stage == 2:
            # Stage 2: MoE routing
            override = None if self._eval_mode == "learned" else self._eval_mode
            oracle_phase_logits: Tensor | None = None
            oracle_mapping: Tensor | None = None
            if self._eval_mode == "oracle":
                # The phase-head oracle routes by M^T softmax(phase_head(z));
                # fails closed when M is all-zero (checkpoint predates
                # soft-mapping persistence).
                oracle_phase_logits = self.phase_head(latent)
                oracle_mapping = self.require_soft_mapping()
            task_state: Tensor | None = None
            if self._uses_impedance_experts():
                task_state = self._task_state(state)
            moe_out = self.moe_layer(
                latent,
                trajectory_id=batch.get("trajectory_id"),
                trajectory_position=batch.get("trajectory_position"),
                task_state=task_state,
                router_override=override,
                phase_logits=oracle_phase_logits,
                mapping=oracle_mapping,
            )

            # V2-D teacher path: when teacher routing is enabled the phase
            # head also runs in Stage 2, emitting phase_logits for the
            # trainer's KL target T = M^T softmax(phase_logits). Fails closed
            # when M is all-zero (checkpoint predates soft-mapping).
            if self.teacher_routing_enabled:
                self.require_soft_mapping()
                teacher_phase_logits: Tensor | None = self.phase_head(latent)
            else:
                teacher_phase_logits = None

            # Store gate logits for metric callbacks
            self._last_gate_logits = moe_out.gate_logits.detach()

            return ModelOutput(
                action_pred=moe_out.combined_output,
                phase_logits=teacher_phase_logits,
                routing_weights=moe_out.routing_weights,
                expert_indices=moe_out.expert_indices,
                gate_logits=moe_out.gate_logits,
                aux_losses={
                    "balance": moe_out.balance_loss,
                    "sticky": moe_out.sticky_loss,
                },
                latent=latent,
                info=moe_out.info,
                clean_gate_logits=moe_out.clean_gate_logits,
            )
        else:
            raise RuntimeError(f"Invalid stage {self._stage}")

    @torch.no_grad()
    def describe_step(self, state: Tensor) -> dict[str, Any]:
        """Snapshot one inference step for full rollout tracing.

        Runs the *learned* routing path (never oracle/sticky/uniform) on a
        single normalized state ``(1, S)`` and reports routing internals:
        latent norm, per-expert scores, selected/top-2 experts, margin,
        entropy (soft routers only), impedance ``(T, κ, e, u)`` when the
        experts are impedance-parameterized, and top-1 vs top-2 action
        disagreement. Values the active configuration cannot provide are
        ``None`` (never fabricated). Never raises on shape-valid input;
        callers still guard (tracing must not break rollouts).
        """
        report: dict[str, Any] = {
            "latent_norm": None,
            "dists": None,
            "selected_expert": None,
            "top2_expert": None,
            "router_margin": None,
            "router_entropy": None,
            "task_vars": None,
            "expert_target": None,
            "expert_gains": None,
            "task_error": None,
            "pre_clip_u": None,
            "expert_disagreement": None,
        }
        latent = self.encoder(state)
        report["latent_norm"] = latent.norm(dim=-1)
        if self._stage != 2:
            return report
        task_state: Tensor | None = None
        if self._uses_impedance_experts():
            task_state = self._task_state(state)
            report["task_vars"] = task_state
        moe_out = self.moe_layer(latent, task_state=task_state)
        gate = moe_out.gate_logits
        order = gate.argsort(dim=-1, descending=(not self._router_is_distance_based()))
        report["selected_expert"] = moe_out.expert_indices[:, 0]
        if gate.size(-1) > 1:
            report["top2_expert"] = order[:, 1]
        if self._router_is_distance_based():
            dists = -gate
            report["dists"] = dists
            if gate.size(-1) > 1:
                ranked = dists.gather(1, order)
                report["router_margin"] = ranked[:, 1] - ranked[:, 0]
        else:
            probs = F.softmax(gate, dim=-1)
            ranked = gate.gather(1, order)
            if gate.size(-1) > 1:
                report["router_margin"] = ranked[:, 0] - ranked[:, 1]
            report["router_entropy"] = -(probs * (probs + 1e-12).log()).sum(dim=-1)
        if moe_out.info is not None:
            report["expert_target"] = moe_out.info.get("target")
            report["expert_gains"] = moe_out.info.get("gains")
            report["task_error"] = moe_out.info.get("task_error")
            report["pre_clip_u"] = moe_out.info.get("pre_clip_u")
        first, second = self._top2_expert_actions(latent, task_state, moe_out)
        if first is not None and second is not None:
            report["expert_disagreement"] = (first - second).norm(dim=-1)
        return report

    def _router_is_distance_based(self) -> bool:
        """True when gate logits are negative distances (prototype router)."""
        return isinstance(getattr(self.moe_layer.router, "prototypes", None), torch.nn.Parameter)

    def _expert_action_at(
        self, latent: Tensor, task_state: Tensor | None, index: int
    ) -> Tensor | None:
        """Action of one expert by index, or None when not computable."""
        experts = self.moe_layer.experts
        if index < 0 or index >= len(experts):
            return None
        expert = experts[index]
        if isinstance(expert, ImpedanceExpert):
            if task_state is None:
                return None
            from phaseforge.models.components.action_adapter import impedance_action

            target, gains = expert.params(latent)
            action, _parts = impedance_action(target, gains, task_state, expert.action_scale)
            return action
        try:
            return expert(latent)
        except Exception:
            return None

    def _top2_expert_actions(
        self, latent: Tensor, task_state: Tensor | None, moe_out: Any
    ) -> tuple[Tensor | None, Tensor | None]:
        """Actions of the top-1 and runner-up experts (either may be None)."""
        gate = moe_out.gate_logits
        if gate.size(-1) < 2:
            return None, None
        order = gate.argsort(dim=-1, descending=(not self._router_is_distance_based()))
        first = self._expert_action_at(latent, task_state, int(order[0, 0].item()))
        second = self._expert_action_at(latent, task_state, int(order[0, 1].item()))
        return first, second

    def get_action(self, state: Tensor) -> Tensor:
        """Inference path without auxiliary outputs or gradients."""
        latent = self.encoder(state)

        if self._stage == 1:
            return self.action_head(latent)
        else:
            override = None if self._eval_mode == "learned" else self._eval_mode
            oracle_phase_logits: Tensor | None = None
            oracle_mapping: Tensor | None = None
            if self._eval_mode == "oracle":
                oracle_phase_logits = self.phase_head(latent)
                oracle_mapping = self.require_soft_mapping()
            task_state: Tensor | None = None
            if self._uses_impedance_experts():
                task_state = self._task_state(state)
            moe_out = self.moe_layer(
                latent,
                task_state=task_state,
                router_override=override,
                phase_logits=oracle_phase_logits,
                mapping=oracle_mapping,
            )
            return moe_out.combined_output

    def num_parameters(self) -> int:
        """Count all trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_routing_info(self) -> dict[str, Tensor] | None:
        """Expose raw routing logits for evaluation metrics."""
        if self._stage == 2 and self._last_gate_logits is None:
            logger.warning("get_routing_info called before any forward passes.")
        if self._stage == 1 or self._last_gate_logits is None:
            return None
        return {"gate_logits": self._last_gate_logits}

    def require_soft_mapping(self) -> Tensor:
        """Return the soft phase->expert mapping M, failing closed when absent.

        A zero-initialized buffer means the checkpoint predates M (or was
        loaded from a stage-1 checkpoint); routing with an all-zero matrix
        would silently produce NaN weights. Every consumer of M (teacher
        distillation, oracle evaluation) must go through this accessor so
        the failure is loud instead.
        """
        if not bool(self.soft_mapping.any()):
            raise RuntimeError(
                "The soft phase->expert mapping M is all-zero: this model was "
                "loaded from a checkpoint that predates soft-mapping "
                "persistence (or from a stage-1 checkpoint). Re-train/re-bootstrap "
                "with router_init.type='soft_mapping' before using a consumer "
                "of M (teacher routing, oracle evaluation)."
            )
        return self.soft_mapping

    @torch.no_grad()
    def bootstrap_moe(
        self,
        dataloader: DataLoader,
        device: torch.device | str = "cuda",
        router_init: dict[str, Any] | None = None,
        expert_init: dict[str, Any] | None = None,
        training_seed: int | None = None,
    ) -> None:
        """Bootstrapping the MoE from Stage 1 knowledge.

        Supports the full factorial and unconfounded prototype generation:
        - Router initialization:
            * 'centroid' / 'phase_centroid': unconfounded hierarchical phase prototypes.
            * 'spherical_centroid': phase centroids with spherical pre-normalization.
            * 'spherical_kmeans': unsupervised spherical K-means on all training latents.
            * 'kmeans': Euclidean K-means on all training latents.
            * 'phase_head': normalized rows of linear phase head classifier.
            * 'random': keeps random router initialization.
        - Expert initialization:
            * 'warmstart': ActionHead weights copied with configurable jitter_std.
            * 'partial_warm': ActionHead copied (exact, no jitter) and a
              fraction ``drop_rate`` of intermediate neurons reinitialized
              per expert using the standard Kaiming-uniform draw (Drop-
              Upcycling-style; shared index set across experts; see
              ``partial_reinit_experts_from_action_head``).
            * 'one_warm': expert ``warm_idx`` receives the standard warm
              start; all other experts are reset via Kaiming-uniform draws.
            * 'random': independent Kaiming draws for every expert.

        Args:
            dataloader: Training dataloader to compute centroids/clusters over.
            device: Compute device.
            router_init: Optional override for router initialization config.
            expert_init: Optional override for expert initialization config.
            training_seed: Optional training seed used to deterministically
                rotate ``warm_idx`` for ``one_warm`` when
                ``expert_init.rotate_warm_idx_by_seed`` is true. Ignored
                otherwise.
        """
        r_cfg = router_init or self.router_init_cfg
        e_cfg = expert_init or self.expert_init_cfg

        r_type = str(r_cfg.get("type", "centroid")).lower()
        e_type = str(e_cfg.get("type", "warmstart")).lower()
        jitter_std = float(e_cfg.get("jitter_std", 0.02))  # type: ignore[arg-type]
        drop_rate = float(e_cfg.get("drop_rate", 0.5))  # type: ignore[arg-type]
        init_seed = int(e_cfg.get("seed", 42))  # type: ignore[arg-type]
        warm_idx = int(e_cfg.get("warm_idx", 0))  # type: ignore[arg-type]
        rotate_warm_idx_by_seed = bool(e_cfg.get("rotate_warm_idx_by_seed", False))  # type: ignore[arg-type]
        cluster_seed = int(r_cfg.get("seed", 42))

        logger.info(
            f"Starting MoE bootstrapping process (router_init='{r_type}', "
            f"expert_init='{e_type}', jitter_std={jitter_std})..."
        )
        self.to(device)
        self.eval()

        num_phases = self.phase_head.num_phases
        num_experts = self.moe_layer.router.num_experts
        latent_dim = self.encoder.latent_dim
        non_blocking = torch.device(device).type == "cuda"

        # 1. Collect training latents and phase labels if needed for data-driven router inits
        # and task states for impedance experts.
        needs_latents = r_type in (
            "centroid",
            "phase_centroid",
            "spherical_centroid",
            "spherical_kmeans",
            "kmeans",
            "soft_mapping",
        )
        needs_task_states = self._uses_impedance_experts()
        needs_data = needs_latents or needs_task_states
        latents_list: list[Tensor] = []
        phases_list: list[Tensor] = []
        task_states_list: list[Tensor] = []

        # PhaseForge 2.0: prototype_source selects which label vocabulary the
        # router centroids are built from. "rule" = canonical phase labels
        # (phase_rule or phase), "dynamic" = SLDS regimes (phase_dynamic),
        # "topo" = topological regimes (phase_topo, Phase 2).
        # The 1.0 default is "rule" for backward compatibility.
        # `r_cfg` may be a plain dict (from PhaseBootstrappedMoE.__init__) or a
        # Hydra DictConfig — handle both via duck-typed .get.
        try:
            proto_raw = r_cfg.get("prototype_source", "rule") if hasattr(r_cfg, "get") else "rule"
        except Exception:
            proto_raw = "rule"
        proto_source = str(proto_raw).lower()
        if proto_source not in (
            "rule",
            "dynamic",
            "topo",
            "phase",
            "phase_dynamic",
            "phase_rule",
            "phase_topo",
        ):
            raise ValueError(
                f"Unknown router_init.prototype_source {proto_source!r}. "
                "Expected 'rule' (phase_rule/phase), 'dynamic' (phase_dynamic), "
                "or 'topo' (phase_topo)."
            )
        # Normalize aliases
        if proto_source in ("phase", "phase_rule", "rule"):
            proto_source_norm = "rule"
        elif proto_source in ("phase_topo", "topo"):
            proto_source_norm = "topo"
        else:
            proto_source_norm = "dynamic"

        all_phases: Tensor | None = None
        # Label-mapping provenance recorded in _expert_init_info below.
        # Defaults cover data-free router inits (e.g. "random");
        # data-driven paths override them.
        effective_num_phases: int = num_phases
        topo_label_mapping: list[int] | None = None
        if needs_data:
            logger.info(
                "Scanning training dataloader for bootstrap "
                "(router prototype_source=%r, needs_latents=%s, needs_task_states=%s)…",
                proto_source_norm,
                needs_latents,
                needs_task_states,
            )
            for batch in dataloader:
                state = batch["state"].to(device, non_blocking=non_blocking)
                # Select label source for prototype computation
                if proto_source_norm == "dynamic":
                    if "phase_dynamic" in batch:
                        phase = batch["phase_dynamic"].to(device, non_blocking=non_blocking)
                    elif "phase" in batch and batch["phase"] is not None:
                        # Fallback check: if data was trained on dynamic, primary `phase`
                        # already is dynamic. Detect via metadata? We fail closed
                        # with a hint instead of silently using rule labels.
                        raise RuntimeError(
                            "router_init.prototype_source='dynamic' but batch has no "
                            "'phase_dynamic' key. The dataloader was built without "
                            "dynamics (data.dynamics.enabled=false or stale cache). "
                            "Enable dynamics and re-ingest, or set prototype_source='rule'."
                        )
                    elif needs_latents:
                        raise RuntimeError(
                            "Batch missing phase labels for dynamic prototype source."
                        )
                    else:
                        phase = None
                elif proto_source_norm == "topo":
                    if "phase_topo" in batch:
                        phase = batch["phase_topo"].to(device, non_blocking=non_blocking)
                    elif needs_latents:
                        raise RuntimeError(
                            "router_init.prototype_source='topo' but batch has no "
                            "'phase_topo' key. The dataloader was built without "
                            "topo discovery (data.topo.enabled=false or stale cache). "
                            "Enable topo and re-ingest, or set prototype_source='rule'."
                        )
                    else:
                        phase = None
                else:  # rule
                    if "phase_rule" in batch:
                        phase = batch["phase_rule"].to(device, non_blocking=non_blocking)
                    elif "phase" in batch and batch["phase"] is not None:
                        phase = batch["phase"].to(device, non_blocking=non_blocking)
                    elif needs_latents:
                        raise RuntimeError("Batch missing phase labels for rule prototype source.")
                    else:
                        phase = None

                if state.ndim == 3:
                    state = state.view(-1, state.size(-1))
                    if phase is not None:
                        phase = phase.view(-1)

                if needs_latents:
                    latent = self.encoder(state)
                    latents_list.append(latent)
                    if phase is not None:
                        phases_list.append(phase)

                if needs_task_states:
                    try:
                        t_state = self._task_state(state)
                        task_states_list.append(t_state)
                        if not needs_latents and phase is not None:
                            phases_list.append(phase)
                    except Exception as exc:
                        logger.warning("Could not extract task state in bootstrap_moe: %s", exc)

            if needs_latents:
                if not latents_list:
                    raise ValueError("Bootstrap dataloader is empty. Cannot compute prototypes.")

                all_latents = torch.cat(latents_list, dim=0)
                all_phases = torch.cat(phases_list, dim=0)
                logger.info(
                    f"Collected {all_latents.size(0)} training latent vectors (dim={latent_dim}) "
                    f"for prototype_source={proto_source_norm} "
                    f"(unique phases: {torch.unique(all_phases).tolist()})."
                )

                if proto_source_norm in ("topo", "dynamic"):
                    unique_labels, inverse_indices = torch.unique(all_phases, return_inverse=True)
                    effective_num_phases = len(unique_labels)
                    # Record the documented unique-to-contiguous
                    # remapping so every task/seed consumer can prove it used
                    # the same label mapping. The raw vocabulary (e.g. PELT
                    # regime ids) maps to contiguous prototype indices via
                    # this ordered list: contiguous_id -> raw_label.
                    topo_label_mapping = [int(v) for v in unique_labels.tolist()]
                    all_phases = inverse_indices
                    logger.info(
                        f"Dynamic/topo regimes mapped to {effective_num_phases} "
                        f"contiguous clusters: {unique_labels.tolist()}"
                    )
                else:
                    effective_num_phases = num_phases
                    topo_label_mapping = None
            elif phases_list:
                all_phases = torch.cat(phases_list, dim=0)

        # 2. Router Initialization
        if r_type in ("centroid", "phase_centroid"):
            prototypes = compute_hierarchical_phase_prototypes(
                all_latents,
                all_phases,
                effective_num_phases,
                num_experts,
                seed=cluster_seed,
                spherical=False,
            )
            _install_router_matrix(
                self.moe_layer.router,
                prototypes,
                f"{num_experts} hierarchical phase prototypes.",
            )

        elif r_type == "spherical_centroid":
            prototypes = compute_hierarchical_phase_prototypes(
                all_latents,
                all_phases,
                effective_num_phases,
                num_experts,
                seed=cluster_seed,
                spherical=True,
            )
            _install_router_matrix(
                self.moe_layer.router,
                prototypes,
                f"{num_experts} spherical phase prototypes.",
            )

        elif r_type == "spherical_kmeans":
            centroids, _ = spherical_kmeans(all_latents, k=num_experts, seed=cluster_seed)
            _install_router_matrix(
                self.moe_layer.router,
                centroids,
                f"{num_experts} Spherical K-Means centroids.",
            )

        elif r_type == "kmeans":
            from sklearn.cluster import KMeans

            latents_np = all_latents.cpu().numpy()
            km = KMeans(n_clusters=num_experts, random_state=cluster_seed, n_init=10).fit(
                latents_np
            )
            centroids = torch.from_numpy(km.cluster_centers_).to(
                device=device, dtype=all_latents.dtype
            )
            centroids_normalized = F.normalize(centroids, p=2, dim=-1)
            _install_router_matrix(
                self.moe_layer.router,
                centroids_normalized,
                f"{num_experts} Euclidean KMeans centroids.",
            )

        elif r_type == "phase_head":
            phase_weight = self.phase_head.classifier.weight.data
            weights = compute_phase_head_router_weights(phase_weight, num_experts)
            _install_router_matrix(
                self.moe_layer.router,
                weights,
                "normalized phase-head directions "
                f"({num_experts} rows).",
            )

        elif r_type == "soft_mapping":
            mapping_mode = str(r_cfg.get("mapping_mode", "prototype_softmax")).lower()
            temperature = float(r_cfg.get("temperature", 1.0))
            if mapping_mode == "prototype_softmax":
                prototypes = compute_hierarchical_phase_prototypes(
                    all_latents,
                    all_phases,
                    effective_num_phases,
                    num_experts,
                    seed=cluster_seed,
                    spherical=True,
                )
                phase_centroids = compute_phase_centroids(
                    all_latents, all_phases, effective_num_phases, spherical=True
                )
                mapping = build_prototype_softmax_mapping(
                    phase_centroids, prototypes, temperature=temperature
                )
                _install_router_matrix(
                    self.moe_layer.router,
                    prototypes,
                    f"{num_experts} spherical hierarchical phase prototypes "
                    f"(soft_mapping={mapping_mode}, temperature={temperature}).",
                )
            elif mapping_mode == "hierarchical_uniform":
                mapping = build_hierarchical_uniform_mapping(effective_num_phases, num_experts)
                logger.info(
                    "Soft mapping built data-free as hierarchical uniform rows "
                    "(%d phases, %d experts); router weights keep random "
                    "initialization.",
                    effective_num_phases,
                    num_experts,
                )
            else:
                raise ValueError(
                    f"Unknown soft_mapping mapping_mode '{mapping_mode}'. Supported: "
                    "prototype_softmax, hierarchical_uniform."
                )
            validate_soft_mapping(mapping)
            self.register_buffer(
                "soft_mapping",
                mapping.to(device=self.soft_mapping.device, dtype=self.soft_mapping.dtype),
            )
            logger.info(
                "Persisted soft phase->expert mapping M of shape %s "
                "(rows right-stochastic, validated).",
                tuple(self.soft_mapping.shape),
            )

        elif r_type == "random":
            logger.info("Router initialization: keeping random router weights (standard init).")
        else:
            raise ValueError(
                f"Unknown router_init type '{r_type}'. Supported: "
                "centroid, spherical_centroid, spherical_kmeans, kmeans, "
                "phase_head, soft_mapping, random."
            )

        # 3. Expert Initialization
        expert_init_info: dict[str, Any] = {
            "type": e_type,
            "jitter_std": jitter_std,
        }
        if self._uses_impedance_experts():
            logger.info(
                f"Initializing {num_experts} ImpedanceExperts with task-space centroids..."
            )
            if task_states_list:
                all_task_states = torch.cat(task_states_list, dim=0)
                global_task_mean = all_task_states.mean(dim=0)
                if global_task_mean.shape[-1] >= 8:
                    q = global_task_mean[3:7]
                    q = q / q.norm().clamp(min=1e-12)
                    if q[0] < 0:
                        q = -q
                    global_task_mean[3:7] = q

                for k, expert in enumerate(self.moe_layer.experts):
                    expert_typed = cast(ImpedanceExpert, expert)
                    expert_typed.reset_parameters()
                    mask = (
                        (all_phases == k)
                        if all_phases is not None
                        else torch.tensor([], dtype=torch.bool)
                    )
                    if mask.sum() > 0:
                        k_mean = all_task_states[mask].mean(dim=0)
                        if k_mean.shape[-1] >= 8:
                            q = k_mean[3:7]
                            q = q / q.norm().clamp(min=1e-12)
                            if q[0] < 0:
                                q = -q
                            k_mean[3:7] = q
                        centroid_k = k_mean
                    else:
                        centroid_k = global_task_mean.clone()

                    expert_typed.target_init_bias = centroid_k.detach().cpu()
                    expert_typed.target_head.bias.data.copy_(
                        centroid_k.to(
                            device=expert_typed.target_head.bias.device,
                            dtype=expert_typed.target_head.bias.dtype,
                        )
                    )
                    nn.init.constant_(expert_typed.gain_head.bias, 0.54132485)
                expert_init_info["task_centroids_initialized"] = True
            else:
                for expert in self.moe_layer.experts:
                    cast(ImpedanceExpert, expert).reset_parameters()
                expert_init_info["task_centroids_initialized"] = False
        elif e_type == "warmstart":
            warm_start_experts_from_action_head(
                self.moe_layer.experts, self.action_head, jitter_std=jitter_std
            )
            logger.info(
                f"Warm-started all {num_experts} experts from ActionHead (jitter_std={jitter_std})."
            )
        elif e_type == "partial_warm":
            first_typed = cast(ExpertMLP, self.moe_layer.experts[0]) if num_experts > 0 else None
            hidden_dim = (
                int(first_typed.hidden[0].weight.size(0))  # type: ignore[union-attr,operator]
                if first_typed is not None
                else 0
            )
            dropped_indices = partial_reinit_experts_from_action_head(
                self.moe_layer.experts,
                self.action_head,
                drop_rate=drop_rate,
                seed=init_seed,
            )
            logger.info(
                f"Partial-warm-started all {num_experts} experts from ActionHead "
                f"(drop_rate={drop_rate}, seed={init_seed}, "
                f"dropped={len(dropped_indices)}/{hidden_dim}, jitter_std=0.0)."
            )
            expert_init_info.update(
                {
                    "drop_rate": drop_rate,
                    "init_seed": init_seed,
                    "hidden_dim": hidden_dim,
                    "num_dropped_neurons": len(dropped_indices),
                    "dropped_neuron_indices": dropped_indices,
                    "dropped_indices_sha256": _hash_dropped_indices(dropped_indices),
                }
            )
        elif e_type == "one_warm":
            effective_warm_idx = warm_idx
            if rotate_warm_idx_by_seed:
                if training_seed is None:
                    raise ValueError(
                        "expert_init.rotate_warm_idx_by_seed=true but no "
                        "training_seed was passed to bootstrap_moe"
                    )
                effective_warm_idx = (warm_idx + int(training_seed)) % num_experts
            one_warm_experts_from_action_head(
                self.moe_layer.experts,
                self.action_head,
                jitter_std=jitter_std,
                warm_idx=effective_warm_idx,
            )
            logger.info(
                f"Warm-started expert {effective_warm_idx}/{num_experts} from ActionHead "
                f"(jitter_std={jitter_std}, requested_warm_idx={warm_idx}, "
                f"rotate_warm_idx_by_seed={rotate_warm_idx_by_seed}, "
                f"training_seed={training_seed}); reset the other {num_experts - 1} "
                "experts to independent random draws."
            )
            expert_init_info.update(
                {
                    "warm_idx": int(effective_warm_idx),
                    "requested_warm_idx": int(warm_idx),
                    "rotate_warm_idx_by_seed": rotate_warm_idx_by_seed,
                    "training_seed": int(training_seed) if training_seed is not None else None,
                }
            )
        elif e_type == "random":
            for expert in self.moe_layer.experts:
                expert.reset_parameters()  # type: ignore[union-attr,operator]
            logger.info(
                f"Reset all {num_experts} experts to independent random draws "
                "(scratch distribution)."
            )
        else:
            raise ValueError(
                f"Unknown expert_init type '{e_type}'. Supported: "
                "warmstart, partial_warm, one_warm, random."
            )

        router_cfg = r_cfg
        self._expert_init_info = {
            "expert_init": expert_init_info,
            "router": {
                "num_experts": int(num_experts),
                "top_k": int(self.moe_layer.router.top_k),
                "init_type": r_type,
                "init_seed": int(cluster_seed),
                "prototype_source": str(proto_source_norm),
                "effective_num_phases": int(effective_num_phases),
                "label_mapping": (
                    list(topo_label_mapping) if topo_label_mapping is not None else None
                ),
                "init_mapping_mode": (
                    str(router_cfg.get("mapping_mode")) if "mapping_mode" in router_cfg else None
                ),
                "init_temperature": (
                    float(router_cfg.get("temperature"))  # type: ignore[arg-type]
                    if "temperature" in router_cfg
                    else None
                ),
            },
            "training_seed": int(training_seed) if training_seed is not None else None,
        }

        # Exclude Stage 1 heads from the optimizer in Stage 2
        for param in self.action_head.parameters():
            param.requires_grad = False
        for param in self.phase_head.parameters():
            param.requires_grad = False

        # Transition to Stage 2
        self.stage = 2
        logger.info("MoE Bootstrapping complete. Ready for Stage 2.")

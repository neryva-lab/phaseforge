"""Final plain-representation control with a centroid-bootstrapped router.

Uses a normalized BC provider, ``PrototypeRouter``, ``phase_topo`` labels,
and an explicit centroid/random router-init policy. It shares the final
residual-expert contract with the proposed method; they differ only in the
representation/router factors.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from phaseforge.models.base import BaseManipulationModel, ModelOutput
from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.expert import (
    ExpertMLP,
    hash_dropped_indices,
    partial_reinit_experts_from_action_head,
    warm_start_experts_from_action_head,
)
from phaseforge.models.components.impedance_expert import ResidualImpedanceExpert
from phaseforge.models.components.moe_layer import MoELayer
from phaseforge.models.components.prototype_router import PrototypeRouter
from phaseforge.models.components.router import TopKRouter

logger = logging.getLogger(__name__)

#: Label vocabularies accepted for the centroid bootstrap. Mirrors
#: ``phaseforge.trains.loops.label_contract.ALLOWED_LABEL_FIELDS`` (kept
#: local so model construction never imports the trainer package).
_BOOTSTRAP_LABEL_FIELDS = frozenset({"phase", "phase_rule", "phase_topo", "phase_dynamic"})

#: Router initialization modes registered for this model.
#: ``centroid`` installs own-latent prototypes; ``random`` keeps the
#: construction-time weights (factorial floor). Anything else fails closed.
_ROUTER_INIT_TYPES = frozenset({"centroid", "random"})


class PrecisionResidualPlainEncoderModel(BaseManipulationModel):
    """MoE bootstrapped from a plain (BC) encoder using phase centroids.

    Structure mirrors :class:`WarmStartMoEModel` (no phase head — the BC
    checkpoint has none), but ``bootstrap_moe`` initializes the router's
    gate weights with phase latent centroids exactly like
    :class:`PhaseBootstrappedMoE`. The phase structure is therefore induced
    into the plain encoder's latent space via the router init alone.
    Expert initialization is config-driven (``expert_init``); the
    registered config pins the canonical 50% partial warm-start so the
    cell is an exact factorial control for the H2 representation claim.

    Args:
        encoder: The StateEncoder instance (loaded from the BC Stage 1 ckpt).
        action_head: The ActionHead used in Stage 1.
        router: The router for Stage 2 (TopKRouter for historical cells,
            PrototypeRouter for final-aligned rows).
        expert: A single expert template to be cloned for Stage 2
            (ExpertMLP for historical cells, ResidualImpedanceExpert with
            beta zero for final-aligned rows).
        num_phases: Number of phases used by the phase labeler (drives the
            centroid computation; there is no phase head in this model).
        expert_init: Config-driven expert init (``warmstart`` or
            ``partial_warm``); the registered configs pin the canonical 50%
            partial warm-start.
        router_init: Router init policy (``centroid`` default preserves the
            historical behavior; ``random`` keeps construction weights for
            the factorial floor).
        bootstrap_label_field: Batch label key the centroids are computed
            from (``phase`` default preserves the historical behavior;
            final-aligned rows use ``phase_topo``).
    """

    def __init__(
        self,
        encoder: StateEncoder,
        action_head: ActionHead,
        router: TopKRouter | PrototypeRouter,
        expert: ExpertMLP | ResidualImpedanceExpert,
        num_phases: int,
        expert_init: dict[str, Any] | None = None,
        router_init: dict[str, Any] | None = None,
        bootstrap_label_field: str = "phase",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.action_head = action_head
        self.moe_layer = MoELayer(router=router, experts=expert)
        self.num_phases = num_phases
        self.expert_init_cfg: dict[str, Any] = (
            dict(expert_init)
            if expert_init is not None
            else {"type": "warmstart", "jitter_std": 0.02}
        )
        self.router_init_cfg: dict[str, Any] = (
            dict(router_init) if router_init is not None else {"type": "centroid"}
        )
        if str(self.router_init_cfg.get("type", "centroid")).lower() not in _ROUTER_INIT_TYPES:
            raise ValueError(
                f"Unknown router_init type {self.router_init_cfg.get('type')!r}. Supported: "
                "centroid, random."
            )
        if str(bootstrap_label_field) not in _BOOTSTRAP_LABEL_FIELDS:
            raise ValueError(
                f"Unknown bootstrap_label_field {bootstrap_label_field!r}. Allowed: "
                f"{sorted(_BOOTSTRAP_LABEL_FIELDS)}."
            )
        self.bootstrap_label_field: str = str(bootstrap_label_field)
        self._expert_init_info: dict[str, Any] | None = None
        self._stage = 1
        self._encoder_frozen = False
        self._last_gate_logits: Tensor | None = None

    @property
    def stage(self) -> int:
        return self._stage

    @stage.setter
    def stage(self, value: int) -> None:
        if value not in (1, 2):
            raise ValueError(f"Stage must be 1 or 2, got {value}")
        self._stage = value
        logger.info(f"PrecisionResidualPlainEncoderModel transitioned to Stage {value}.")

    def freeze_encoder(self) -> None:
        """Freeze the encoder for Stage 2 (weights + eval mode, no dropout)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self._encoder_frozen = True
        self.encoder.eval()
        logger.info("Encoder weights frozen; encoder kept in eval mode (no dropout).")

    def train(self, mode: bool = True) -> PrecisionResidualPlainEncoderModel:
        """Override so a frozen encoder stays deterministic during Stage 2."""
        super().train(mode)
        if mode and self._encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, batch: dict[str, Tensor]) -> ModelOutput:
        state = batch["state"]
        latent = self.encoder(state)

        if self._stage == 1:
            action_pred = self.action_head(latent)
            return ModelOutput(
                action_pred=action_pred,
                phase_logits=None,
                routing_weights=None,
                expert_indices=None,
                gate_logits=None,
            )
        elif self._stage == 2:
            moe_out = self.moe_layer(latent)
            self._last_gate_logits = moe_out.gate_logits.detach()
            return ModelOutput(
                action_pred=moe_out.combined_output,
                phase_logits=None,
                routing_weights=moe_out.routing_weights,
                expert_indices=moe_out.expert_indices,
                gate_logits=moe_out.gate_logits,
                aux_losses={"balance": moe_out.balance_loss},
            )
        else:
            raise RuntimeError(f"Invalid stage {self._stage}")

    def get_action(self, state: Tensor) -> Tensor:
        latent = self.encoder(state)
        if self._stage == 1:
            return self.action_head(latent)
        return self.moe_layer(latent).combined_output

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_routing_info(self) -> dict[str, Tensor] | None:
        if self._stage == 2 and self._last_gate_logits is None:
            logger.warning("get_routing_info called before any forward passes.")
        if self._stage == 1 or self._last_gate_logits is None:
            return None
        return {"gate_logits": self._last_gate_logits}

    @torch.no_grad()
    def bootstrap_moe(
        self,
        dataloader: DataLoader,
        device: torch.device | str = "cuda",
        training_seed: int | None = None,
    ) -> None:
        """Centroid-bootstrap a MoE on top of the plain (BC) encoder.

        Identical router algorithm to :class:`PhaseBootstrappedMoE.bootstrap_moe`:
        1. Compute phase centroids in the (plain) encoder's latent space.
        2. Initialize the router gate weights with the normalized centroids.
        3. Initialize the experts per the config-driven ``expert_init``
           (``warmstart`` or R50-matched ``partial_warm``; the registered
           model config pins the canonical 50% partial warm-start so this
           cell is an exact factorial control for the H2 representation
           claim).

        Args:
            dataloader: Training dataloader to compute centroids over.
            device: Compute device.
            training_seed: The run's training seed (passed by the CLI;
                audit metadata and partial-warm seed fallback).
        """
        logger.info("Starting centroid bootstrap over the plain (BC) encoder...")
        self.to(device)
        self.eval()

        num_experts = self.moe_layer.router.num_experts
        latent_dim = self.encoder.latent_dim
        label_field = self.bootstrap_label_field
        r_init_type = str(self.router_init_cfg.get("type", "centroid")).lower()

        if self.num_phases != num_experts:
            logger.warning(
                f"Number of phases ({self.num_phases}) != number of experts "
                f"({num_experts}). Phase-bootstrapping works best when E >= P. "
                "Centroids will be mapped 1:1 for the first P experts, the rest "
                "remain random."
            )

        router_init_info: dict[str, Any] = {
            "init_type": r_init_type,
            "bootstrap_label_field": label_field,
        }

        if r_init_type == "random":
            # Factorial floor: no centroid computation, no dataloader scan.
            # The router keeps its construction-time weights (seeded by the
            # run's global seed at build time); experts are still initialized
            # per expert_init below.
            logger.info(
                "PlainEncoderPhaseBootstrap: leaving router randomly initialized "
                f"(router_init='random', bootstrap_label_field={label_field!r} unused)."
            )
        else:
            # 1. Compute latent centroids over the plain encoder's OWN latents
            # (latent isolation: never reuse the proposed-method latents)
            # using the resolved label field (final rows: phase_topo).
            phase_sums = torch.zeros((self.num_phases, latent_dim), device=device)
            phase_counts = torch.zeros((self.num_phases,), device=device)

            for batch in dataloader:
                state = batch["state"].to(device)
                raw_labels = batch.get(label_field)
                if raw_labels is None:
                    raise RuntimeError(
                        "Phase centroid bootstrap: label field "
                        f"{label_field!r} is missing from the bootstrap batch "
                        f"(available: {sorted(str(k) for k in batch.keys())}). "
                        "Enable the matching discovery source and re-ingest, "
                        "or point bootstrap_label_field at 'phase'."
                    )
                phase = raw_labels.to(device)

                if state.ndim == 3:
                    state = state.view(-1, state.size(-1))
                    phase = phase.view(-1)

                if bool((phase < 0).any()) or bool((phase >= self.num_phases).any()):
                    raise ValueError(
                        "Phase centroid bootstrap: label field "
                        f"{label_field!r} has values outside [0, {self.num_phases - 1}] "
                        f"(min={int(phase.min().item())}, max={int(phase.max().item())})."
                    )

                latent = self.encoder(state)
                phase_expanded = phase.unsqueeze(1).expand_as(latent)
                phase_sums.scatter_add_(0, phase_expanded, latent)
                counts = torch.bincount(phase, minlength=self.num_phases).float()
                phase_counts += counts

            # An absent phase would produce a zero centroid reported as if it had
            # one sample — that silently corrupts the router init, so it is a
            # hard failure instead.
            absent = phase_counts == 0
            if absent.any():
                raise ValueError(
                    "Phase centroid bootstrap: "
                    f"{int(absent.sum().item())} phase(s) have zero samples in the "
                    f"bootstrap dataloader (missing: "
                    f"{[int(i) for i in absent.nonzero().flatten().tolist()]}). "
                    "Refusing to bootstrap with zero centroids; every phase must "
                    "be present in the training data."
                )
            centroids = phase_sums / phase_counts.unsqueeze(1)  # (P, D)

            logger.info(f"Computed latent centroids for {self.num_phases} phases.")

            # 2. Initialize Router with the (normalized) centroids. Prototype
            # routers store them as Voronoi prototypes; legacy TopK routers
            # store them as gate hyperplanes with a zeroed bias. Shapes must
            # match exactly — a mismatch fails closed.
            centroids_normalized = torch.nn.functional.normalize(centroids, p=2, dim=-1)

            proto = getattr(self.moe_layer.router, "prototypes", None)
            if isinstance(proto, torch.nn.Parameter):
                if tuple(proto.shape) != tuple(centroids_normalized.shape):
                    raise ValueError(
                        "Cannot install bootstrapped prototypes: router expects "
                        f"{tuple(proto.shape)}, computed {tuple(centroids_normalized.shape)}."
                    )
                proto.data.copy_(centroids_normalized)
                logger.info(
                    f"Initialized prototype router with {self.num_phases} phase centroids "
                    f"(label field {label_field!r})."
                )
            elif hasattr(self.moe_layer.router, "gate_linear"):
                router_weight = self.moe_layer.router.gate_linear.weight.data
                router_bias = self.moe_layer.router.gate_linear.bias.data

                limit = min(self.num_phases, num_experts)
                router_weight[:limit] = centroids_normalized[:limit]
                router_bias.zero_()

                logger.info(f"Initialized router weights with {limit} phase centroids.")
            else:
                raise ValueError(
                    "Phase centroid bootstrap: the router has neither 'prototypes' "
                    f"nor 'gate_linear' (got {type(self.moe_layer.router).__name__}); "
                    "cannot install centroids."
                )

        # 3. Initialize Experts per the config-driven init type.
        e_cfg = self.expert_init_cfg
        e_type = str(e_cfg.get("type", "warmstart")).lower()
        jitter_std = float(e_cfg.get("jitter_std", 0.02))
        num_experts = len(self.moe_layer.experts)
        expert_init_info: dict[str, Any] = {"type": e_type, "jitter_std": jitter_std}

        if e_type == "warmstart":
            # Exact, strict copy + small symmetry-breaking jitter.
            warm_start_experts_from_action_head(
                self.moe_layer.experts, self.action_head, jitter_std=jitter_std
            )
            logger.info(
                f"Warm-started all {num_experts} experts from ActionHead "
                f"(jitter_std={jitter_std})."
            )
        elif e_type == "partial_warm":
            init_seed = int(e_cfg.get("seed", training_seed if training_seed is not None else 42))
            drop_rate = float(e_cfg.get("drop_rate", 0.5))
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
            expert_init_info.update(
                {
                    "drop_rate": drop_rate,
                    "init_seed": init_seed,
                    "hidden_dim": hidden_dim,
                    "num_dropped_neurons": len(dropped_indices),
                    "dropped_neuron_indices": dropped_indices,
                    "dropped_indices_sha256": hash_dropped_indices(dropped_indices),
                }
            )
            logger.info(
                f"Partial-warm-started all {num_experts} experts from ActionHead "
                f"(drop_rate={drop_rate}, seed={init_seed}, "
                f"dropped={len(dropped_indices)}/{hidden_dim}, jitter_std=0.0)."
            )
        else:
            raise ValueError(
                f"Unknown expert_init type '{e_type}'. Supported: "
                "warmstart, partial_warm."
            )

        self._expert_init_info = {
            "expert_init": expert_init_info,
            "router": {
                "num_experts": int(num_experts),
                "top_k": int(self.moe_layer.router.top_k),
                **router_init_info,
            },
            "training_seed": int(training_seed) if training_seed is not None else None,
        }

        # The Stage 1 action head is unused in Stage 2; exclude it from the
        # optimizer so trainable-parameter counts are correct.
        for param in self.action_head.parameters():
            param.requires_grad = False

        # Automatically transition to Stage 2
        self.stage = 2
        logger.info("Centroid bootstrap complete. Ready for Stage 2.")

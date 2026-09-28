"""Final privileged oracle for offline routing diagnostics.

Final-aligned configurations use ``phase_topo`` labels, beta-zero residual
experts warm-started from the Stage 1 action head, and a fail-closed
refusal of ordinary state-only rollout.
"""

from __future__ import annotations

import logging

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
from phaseforge.models.components.router import TopKRouter

logger = logging.getLogger(__name__)

#: Label vocabularies accepted for oracle routing. Mirrors
#: ``phaseforge.trains.loops.label_contract.ALLOWED_LABEL_FIELDS`` (kept
#: local so model construction never imports the trainer package).
_ORACLE_LABEL_FIELDS = frozenset({"phase", "phase_rule", "phase_topo", "phase_dynamic"})


class PrecisionResidualOracleModel(BaseManipulationModel):
    """MoE trained with Oracle routing (ground truth phases).

    During training, the router is bypassed, and the ground truth phase
    labels are used to perfectly select the corresponding expert.
    Provides an upper bound on performance.

    Args:
        encoder: The StateEncoder instance.
        router: The router (structural parity only; never consulted).
        expert: A single expert template to be cloned (ExpertMLP for
            historical cells, ResidualImpedanceExpert with beta zero for
            final-aligned rows).
        num_phases: Number of routing phases (must be <= num_experts).
        label_field: Batch label key supplying the privileged routing
            labels (``phase`` default preserves the historical behavior;
            final-aligned rows use ``phase_topo``).
        action_head: Optional Stage 1 ActionHead enabling warm-started
            experts via :meth:`bootstrap_moe` (final-aligned rows). When
            absent, experts train from their construction-time random
            initialization (historical behavior).
        expert_init: Config-driven expert init (``warmstart`` /
            ``partial_warm``); only consulted when ``action_head`` is set.
        allow_rollout: Whether the label-free ``get_action`` fallback is
            permitted. Historical default ``True`` preserves the legacy
            behavior exactly; final-aligned rows set ``False`` so ordinary
            state-only rollout is refused and the oracle stays an offline
            diagnostic.
    """

    def __init__(
        self,
        encoder: StateEncoder,
        router: TopKRouter,
        expert: ExpertMLP | ResidualImpedanceExpert,
        num_phases: int,
        label_field: str = "phase",
        action_head: ActionHead | None = None,
        expert_init: dict | None = None,
        allow_rollout: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.action_head = action_head

        # We enforce deterministic routing, Top-1
        router.top_k = 1
        router.noise_std = 0.0

        self.moe_layer = MoELayer(router=router, experts=expert)
        self.num_phases = num_phases
        if num_phases < 1:
            raise ValueError(f"num_phases must be positive, got {num_phases}")
        if num_phases > router.num_experts:
            raise ValueError(
                f"Oracle routing requires at least one expert per phase: "
                f"num_phases={num_phases}, num_experts={router.num_experts}."
            )
        if str(label_field) not in _ORACLE_LABEL_FIELDS:
            raise ValueError(
                f"Unknown label_field {label_field!r}. Allowed: "
                f"{sorted(_ORACLE_LABEL_FIELDS)}."
            )
        self.label_field: str = str(label_field)
        self.expert_init_cfg: dict | None = (
            dict(expert_init) if expert_init is not None else None
        )
        self.allow_rollout: bool = bool(allow_rollout)
        self._expert_init_info: dict | None = None
        self._last_gate_logits: Tensor | None = None
        # Stage 2-only baseline (no Stage 1 exists). Kept as a plain
        # attribute so checkpointing/eval metadata records the right stage.
        self.stage = 2
        self._encoder_frozen = False

    @property
    def requires_stage1_checkpoint(self) -> bool:
        """Whether Stage 2 setup needs a Stage 1 checkpoint loaded first.

        Final-aligned rows (action head and/or expert init configured)
        warm-start from Stage 1 weights and fail closed without them. The
        historical scratch configuration (neither set) trains fully from
        random initialization and must not be forced through a checkpoint
        it never consumed. The training CLI consults this before demanding
        ``train.stage1_ckpt_path``.
        """
        return self.action_head is not None or self.expert_init_cfg is not None

    def freeze_encoder(self) -> None:
        """Freeze the encoder for Stage 2 (weights + eval mode, no dropout)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self._encoder_frozen = True
        self.encoder.eval()
        logger.info("Encoder weights frozen; encoder kept in eval mode (no dropout).")

    def train(self, mode: bool = True) -> PrecisionResidualOracleModel:
        """Override so a frozen encoder stays deterministic during Stage 2."""
        super().train(mode)
        if mode and self._encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, batch: dict[str, Tensor]) -> ModelOutput:
        state = batch["state"]
        phase = batch.get(self.label_field)

        if phase is None:
            raise RuntimeError(
                "PrecisionResidualOracleModel requires ground-truth labels "
                f"(field {self.label_field!r}) in every forward pass and never "
                "falls back to the (untrained) router: routing by it would "
                "silently corrupt the oracle upper bound. The oracle is a "
                "routing-signature reference only and is not deployable "
                "without phase labels "
                f"(available: {sorted(str(k) for k in batch.keys())})."
            )

        # ORACLE ROUTING (Training)
        latent = self.encoder(state)
        B = latent.size(0)

        # Flatten time dim if sequence
        if latent.ndim == 3:
            latent = latent.view(-1, latent.size(-1))
            phase = phase.view(-1)
            B = latent.size(0)

        # Ensure E >= P for oracle mapping
        E = self.moe_layer.router.num_experts
        if phase.numel() and (phase.min() < 0 or phase.max() >= self.num_phases):
            raise ValueError(
                f"Oracle phase labels must be in [0, {self.num_phases - 1}], "
                f"got range [{int(phase.min())}, {int(phase.max())}]."
            )
        expert_indices = phase.unsqueeze(-1)  # (B, 1)

        # Oracle weights are 1.0 (perfect certainty)
        routing_weights = torch.ones((B, 1), device=latent.device)  # (B, 1)

        # Generate dummy logits for metric compatibility
        gate_logits = torch.zeros((B, E), device=latent.device)
        gate_logits.scatter_(1, expert_indices, 100.0)  # Highly peaked
        self._last_gate_logits = gate_logits.detach()

        # Gather output
        out_dim = self.moe_layer.experts[0].output_dim
        combined_output = torch.zeros((B, out_dim), device=latent.device)

        for expert_idx, expert_net in enumerate(self.moe_layer.experts):
            match_mask = (expert_indices == expert_idx).squeeze(-1)
            if not match_mask.any():
                continue

            batch_idx = torch.where(match_mask)[0]
            expert_inputs = latent[batch_idx]
            expert_outputs = expert_net(expert_inputs)

            combined_output.index_copy_(0, batch_idx, expert_outputs)

        return ModelOutput(
            action_pred=combined_output,
            phase_logits=None,
            routing_weights=routing_weights,
            expert_indices=expert_indices,
            gate_logits=gate_logits,
            # No balance loss needed for oracle routing
            aux_losses={"balance": torch.tensor(0.0, device=latent.device)},
        )

    def get_action(self, state: Tensor) -> Tensor:
        """Label-free inference path (rollouts).

        Final-aligned rows refuse this path (``allow_rollout=False``): the
        oracle is an offline diagnostic on held-out demonstration states
        with supplied labels, and a rollout success rate would require a
        separately validated privileged labeler (out of scope). The legacy
        fallback (routing by the never-trained gate) is preserved only when
        explicitly allowed; its scores are NOT a policy-deployable signal.
        """
        if not self.allow_rollout:
            raise RuntimeError(
                "PrecisionResidualOracleModel refuses ordinary state-only rollout: "
                "no privileged labeler supplies routing labels for "
                "policy-generated states. Evaluate through the privileged "
                "offline path with recorded labels "
                f"(field {self.label_field!r})."
            )
        logger.debug(
            "PrecisionResidualOracleModel.get_action falls back to the router's gate, "
            "which was never trained for the oracle (routing is by labels "
            "during training). Rollout scores from this path are NOT a "
            "policy-deployable signal."
        )
        latent = self.encoder(state)
        moe_out = self.moe_layer(latent)
        return moe_out.combined_output

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_routing_info(self) -> dict[str, Tensor] | None:
        if self._last_gate_logits is None:
            return None
        return {"gate_logits": self._last_gate_logits}

    @torch.no_grad()
    def bootstrap_moe(
        self,
        dataloader: DataLoader,
        device: torch.device | str = "cuda",
        training_seed: int | None = None,
    ) -> None:
        """Transition to Stage 2: initialize experts for oracle routing.

        Final-aligned rows load the Stage 1 checkpoint through the CLI
        (encoder, and action head when configured) before this call, then
        warm-start the experts from the action head per ``expert_init``.
        Without an action head the experts keep their construction-time
        random initialization (historical scratch behavior).

        Args:
            dataloader: Training dataloader (kept for CLI signature parity;
                not iterated — oracle routing needs no centroids).
            device: Compute device.
            training_seed: The run's training seed (audit metadata and
                partial-warm seed fallback).
        """
        self.to(device)
        self.eval()

        num_experts = len(self.moe_layer.experts)
        if self.expert_init_cfg is None:
            logger.info(
                "OraclePhaseMoE: no expert_init configured; experts keep "
                "their construction-time random initialization."
            )
            expert_init_info: dict = {"type": "random"}
        else:
            if self.action_head is None:
                raise ValueError(
                    "OraclePhaseMoE: expert_init is configured but the model "
                    "has no action_head to warm-start from. Add an action_head "
                    "or remove expert_init."
                )
            e_type = str(self.expert_init_cfg.get("type", "warmstart")).lower()
            jitter_std = float(self.expert_init_cfg.get("jitter_std", 0.02))
            expert_init_info = {"type": e_type, "jitter_std": jitter_std}
            if e_type == "warmstart":
                warm_start_experts_from_action_head(
                    self.moe_layer.experts, self.action_head, jitter_std=jitter_std
                )
                logger.info(
                    f"OraclePhaseMoE: warm-started all {num_experts} experts "
                    f"from ActionHead (jitter_std={jitter_std})."
                )
            elif e_type == "partial_warm":
                init_seed = int(
                    self.expert_init_cfg.get(
                        "seed", training_seed if training_seed is not None else 42
                    )
                )
                drop_rate = float(self.expert_init_cfg.get("drop_rate", 0.5))
                first = self.moe_layer.experts[0] if num_experts > 0 else None
                hidden_dim = (
                    int(first.hidden[0].weight.size(0)) if first is not None else 0
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
                    f"OraclePhaseMoE: partial-warm-started all {num_experts} experts "
                    f"from ActionHead (drop_rate={drop_rate}, seed={init_seed})."
                )
            else:
                raise ValueError(
                    f"Unknown expert_init type '{e_type}'. Supported: "
                    "warmstart, partial_warm."
                )
            for param in self.action_head.parameters():
                param.requires_grad = False

        self._expert_init_info = {
            "expert_init": expert_init_info,
            "router": {
                "num_experts": int(num_experts),
                "top_k": int(self.moe_layer.router.top_k),
                "init_type": "oracle_bypass",
            },
            "label_field": self.label_field,
            "diagnostic": "privileged_offline",
            "allow_rollout": bool(self.allow_rollout),
            "training_seed": int(training_seed) if training_seed is not None else None,
        }
        logger.info("OraclePhaseMoE bootstrap complete. Ready for Stage 2.")

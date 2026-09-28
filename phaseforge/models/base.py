"""Abstract base class for all PhaseForge manipulation models.

Every model (BC, ScratchMoE, WarmStartMoE, OracleMoE, PhaseBootstrappedMoE)
MUST inherit from BaseManipulationModel and implement every abstract method.
The trainer interacts only through this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ModelOutput:
    """Standardized output returned by every model's forward pass."""

    action_pred: Tensor
    """(B, A) or (B, T, A) — predicted action(s)."""

    phase_logits: Tensor | None = None
    """(B, P) — raw phase classification logits. None for non-phase models."""

    routing_weights: Tensor | None = None
    """(B, K) — top-k normalized gating weights. None for non-MoE models."""

    expert_indices: Tensor | None = None
    """(B, K) — top-k expert indices. None for non-MoE models."""

    gate_logits: Tensor | None = None
    """(B, E) — raw gate logits over all experts (for metric logging)."""

    clean_gate_logits: Tensor | None = None
    """(B, E) — deterministic pre-exploration gate logits.

    Reported by MoE models whose router exposes them (both in-tree
    routers); ``None`` for dense models and legacy baselines. The Stage
    2 margin loss consumes this when present so training-time
    exploration noise never shifts the margin targets."""

    aux_losses: dict[str, Tensor] = field(default_factory=dict)
    """Auxiliary losses: keys may include ``"balance"``, ``"phase"``."""

    latent: Tensor | None = None
    """(B, Dz) — encoder latents (for contrastive losses). None when the
    model does not expose them; trainers must fail closed in that case."""

    info: dict[str, Tensor] | None = None
    """Per-sample diagnostics for impedance experts and tracing
    (keys may include ``"target"``, ``"gains"``, ``"task_error"``,
    ``"pre_clip_u"``, ``"task_state"``, ``"expert_index"``). None for
    direct-action models; consumers must fail closed when a required key
    is absent."""


class BaseManipulationModel(nn.Module, ABC):
    """Shared interface for all PhaseForge model variants.

    Subclasses must implement:
        - :meth:`forward`
        - :meth:`get_action`
        - :meth:`num_parameters`
    """

    @abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> ModelOutput:
        """Training forward pass.

        Args:
            batch: Dict with keys:
                - ``"state"``:   (B, state_dim) or (B, T, state_dim)
                - ``"action"``:  (B, action_dim) or (B, T, action_dim)
                  — ground truth
                - ``"phase"``:   (B,) or (B, T) int — phase labels
                  (may be ignored)
                - ``"task_id"``: (B,) int
        """
        ...

    @abstractmethod
    def get_action(self, state: Tensor) -> Tensor:
        """Inference-only path.

        No phase labels, no auxiliary losses, no gradients required.

        Args:
            state: (B, state_dim) or (1, state_dim)

        Returns:
            action: (B, action_dim). Temporal models may consume one
                timestep at a time and retain recurrent state; sequence
                inputs return (B, T, action_dim).
        """
        ...

    @abstractmethod
    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        ...

    def freeze_encoder(self) -> None:
        """Freeze the encoder sub-module. Default: no-op.

        Override in models that have a distinct encoder attribute.
        """

    def set_normalizer_stats(
        self,
        mean: Tensor | np.ndarray | None,
        std: Tensor | np.ndarray | None,
    ) -> None:
        """Set normalizer statistics for physical task-space reconstruction."""
        if mean is not None:
            if not isinstance(mean, torch.Tensor):
                mean = torch.as_tensor(mean, dtype=torch.float32)
            self.register_buffer("normalizer_mean", mean.detach().clone(), persistent=True)
        elif hasattr(self, "normalizer_mean"):
            delattr(self, "normalizer_mean")

        if std is not None:
            if not isinstance(std, torch.Tensor):
                std = torch.as_tensor(std, dtype=torch.float32)
            self.register_buffer("normalizer_std", std.detach().clone(), persistent=True)
        elif hasattr(self, "normalizer_std"):
            delattr(self, "normalizer_std")

    def get_normalizer_stats(self) -> tuple[Tensor | None, Tensor | None]:
        """Return cached normalizer statistics ``(mean, std)``, if registered."""
        mean = getattr(self, "normalizer_mean", None)
        std = getattr(self, "normalizer_std", None)
        return mean, std

    def get_routing_info(self) -> dict[str, Tensor] | None:
        """Return the most recent routing state for metric logging.

        Returns:
            Dict with ``"gate_logits": Tensor(B, E)`` or None if not an MoE model.
        """
        return None

    def deployment_contract(self) -> dict[str, Any]:
        """Describe the model's deployment contract.

        The base contract carries only what holds for every model: the
        policy is memoryless (one state in, one action out, no recurrent
        hidden state). Subclasses with routing/action specifics (e.g.
        :class:`PhaseBootstrappedMoE`) override and extend this dict.
        The rollout runner logs (never enforces beyond) this contract.
        """
        return {"memoryless": True}

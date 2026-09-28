"""ImpedanceExpert: per-expert local feedback-controller parameters.

Each expert maps the shared latent to a target task state and positive
feedback gains::

    T_k(z_t) = target task state      (Dy = 8: pos 3, quat 4, gripper 1)
    κ_k(z_t) = positive feedback gains (De = 7: pos 3, rot 3, gripper 1)

Positivity (``κ > 0``) is enforced by softplus plus ``[kappa_min,
kappa_max]`` clamping; a non-positive gain would flip the feedback sign
and is rejected downstream as well (see ``impedance_action``).

Unlike :class:`ExpertMLP` (which regresses absolute 7D actions), this
head cannot be warm-started from the Stage 1 ``ActionHead`` — the output
structure differs — so IS-PhaseForge uses random expert init. Stage 1
keeps its direct action head for warm-starting the shared encoder only.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from phaseforge.models.components.action_adapter import impedance_action
from phaseforge.models.components.task_state import ACTION_DIM, TASK_ERROR_DIM, TASK_STATE_DIM


class ImpedanceExpert(nn.Module):
    """One impedance-parameterized expert.

    Args:
        input_dim: Dimension of the latent vector from the encoder.
        hidden_dim: Width of the shared trunk hidden layer.
        task_state_dim: Target task-state width (default 8).
        error_dim: Gain/error width (default 7).
        action_dim: Environment action width (default 7).
        action_scale: Command scale ``s`` in ``a = tanh(u / s)`` (> 0, scalar or per-dim vector).
        kappa_min: Lower gain clamp (must be > 0).
        kappa_max: Upper gain clamp (>= kappa_min).
        target_init_bias: Optional initial bias for the target task state head.
    """

    IS_IMPEDANCE = True

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        task_state_dim: int = TASK_STATE_DIM,
        error_dim: int = TASK_ERROR_DIM,
        action_dim: int = ACTION_DIM,
        action_scale: float | Sequence[float] | Tensor = 1.0,
        kappa_min: float = 0.1,
        kappa_max: float = 5.0,
        target_init_bias: Sequence[float] | Tensor | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if task_state_dim < 1 or error_dim < 1 or action_dim < 1:
            raise ValueError("task/error/action dims must be positive.")
        if isinstance(action_scale, (int, float)):
            if float(action_scale) <= 0.0:
                raise ValueError(f"action_scale must be > 0.0, got {action_scale}.")
            self.action_scale: float | tuple[float, ...] = float(action_scale)
        else:
            try:
                scale_seq = tuple(float(v) for v in action_scale)
            except Exception as exc:
                raise ValueError(
                    f"action_scale must be a float or sequence of floats, got {action_scale}."
                ) from exc
            if any(v <= 0.0 for v in scale_seq):
                raise ValueError(f"action_scale elements must be > 0.0, got {action_scale}.")
            self.action_scale = scale_seq

        kappa_lo, kappa_hi = float(kappa_min), float(kappa_max)
        if kappa_lo <= 0.0 or kappa_hi < kappa_lo:
            raise ValueError(f"Require 0 < kappa_min <= kappa_max, got {kappa_lo, kappa_hi}.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.task_state_dim = int(task_state_dim)
        self.error_dim = int(error_dim)
        self.output_dim = int(action_dim)
        self.kappa_min = kappa_lo
        self.kappa_max = kappa_hi
        self.target_init_bias = (
            torch.as_tensor(target_init_bias, dtype=torch.float32).clone()
            if target_init_bias is not None
            else None
        )

        self.trunk = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.GELU())
        self.target_head = nn.Linear(self.hidden_dim, self.task_state_dim)
        self.gain_head = nn.Linear(self.hidden_dim, self.error_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="linear")
                nn.init.zeros_(module.bias)
        if hasattr(self, "target_head"):
            nn.init.normal_(self.target_head.weight, mean=0.0, std=1e-3)
            if self.target_init_bias is not None:
                self.target_head.bias.data.copy_(self.target_init_bias.to(self.target_head.bias.dtype))
        if hasattr(self, "gain_head"):
            nn.init.normal_(self.gain_head.weight, mean=0.0, std=1e-3)
            # softplus(x) = ln(1 + e^x) = 1.0 => x = ln(e - 1) ≈ 0.54132485
            # gives exact nominal stiffness
            nn.init.constant_(self.gain_head.bias, 0.54132485)

    def reset_parameters(self) -> None:
        """Re-initialize all weights (symmetry-breaking, MoE template clones)."""
        self._init_weights()

    def params(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        """Predict ``(target (..., Dy), gains (..., De))`` with ``κ > 0``."""
        hidden = self.trunk(latent)
        raw_target = self.target_head(hidden)
        if self.task_state_dim >= 8:
            pos = raw_target[..., 0:3]
            raw_quat = raw_target[..., 3:7]
            quat_norm = raw_quat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            quat = raw_quat / quat_norm
            sign = torch.where(quat[..., 0:1] < 0.0, -1.0, 1.0)
            quat = quat * sign
            rest = raw_target[..., 7:]
            target = torch.cat([pos, quat, rest], dim=-1)
        else:
            target = raw_target
        gains = torch.nn.functional.softplus(self.gain_head(hidden))
        gains = gains.clamp(min=self.kappa_min, max=self.kappa_max)
        return target, gains

    def forward(self, latent: Tensor, task_state: Tensor | None = None) -> Tensor:
        """Predict a clipped action from ``(latent, task_state)``.

        ``task_state`` is required (the feedback error needs ``y_t``);
        a missing value fails closed instead of silently reusing zeros.
        """
        if task_state is None:
            raise ValueError(
                "ImpedanceExpert.forward needs task_state=y_t; pass the task "
                "state alongside the latent (MoELayer threads it through)."
            )
        target, gains = self.params(latent)
        action, _info = impedance_action(target, gains, task_state, self.action_scale)
        return action


class ResidualImpedanceExpert(nn.Module):
    """Precision-Residual Expert.

    Combines:
    1. Direct base action head matching ActionHead / ExpertMLP architecture
       (warm-started via partial_reinit_experts_from_action_head).
    2. Residual impedance compliance branch (delta, kappa) initialized with beta=0.0.
    3. Direct gripper channel (un-lagged by impedance).

    When beta == 0.0, output is strictly the warm-started direct action.
    """

    IS_IMPEDANCE = False
    IS_RESIDUAL = True

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        hidden_dim: int = 256,
        output_dim: int = ACTION_DIM,
        action_scale: float | Sequence[float] | Tensor = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 0.04),
        kappa_min: float = 0.1,
        kappa_max: float = 5.0,
        beta: float = 0.0,
    ) -> None:
        super().__init__()
        from phaseforge.models.components.expert import ExpertMLP

        h_dim = hidden_dims[0] if (hidden_dims and len(hidden_dims) > 0) else hidden_dim
        self.base_expert = ExpertMLP(
            input_dim=input_dim, hidden_dims=[h_dim], output_dim=output_dim
        )

        self.input_dim = int(input_dim)
        self.hidden_dim = int(h_dim)
        self.output_dim = int(output_dim)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.beta = float(beta)

        # Multi-arm geometry decomposition:
        # Standard Robosuite operational-space controllers allocate 7 dims per arm
        # (6 pose + 1 gripper).
        if self.output_dim % 7 == 0:
            self.num_arms = self.output_dim // 7
            self.pose_dim = 6 * self.num_arms
        else:
            self.num_arms = 1
            self.pose_dim = max(1, self.output_dim - 1)

        # Action scale expansion for multi-arm tasks (e.g. Transport has 14 dims)
        if isinstance(action_scale, (int, float)):
            self.action_scale = (float(action_scale),) * self.output_dim
        else:
            scales = tuple(float(v) for v in action_scale)
            if len(scales) == 7 and self.num_arms > 1:
                scales = scales * self.num_arms
            self.action_scale = scales

        # Extract pose scale components corresponding to translation + rotation for each arm
        if self.output_dim % 7 == 0:
            pose_scale_list = []
            for arm in range(self.num_arms):
                start = arm * 7
                pose_scale_list.extend(self.action_scale[start : start + 6])
            self.pose_action_scale = tuple(pose_scale_list)
        else:
            self.pose_action_scale = self.action_scale[: self.pose_dim]

        # Residual compliance heads (operational space pose: translation + rotation)
        self.delta_head = nn.Linear(self.hidden_dim, self.pose_dim)
        self.gain_head = nn.Linear(self.hidden_dim, self.pose_dim)
        self._init_residual_weights()

    @property
    def hidden(self):
        return self.base_expert.hidden

    @property
    def output_proj(self):
        return self.base_expert.output_proj

    def _init_residual_weights(self) -> None:
        nn.init.normal_(self.delta_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.normal_(self.gain_head.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.gain_head.bias, 0.54132485)

    def reset_parameters(self) -> None:
        self.base_expert.reset_parameters()
        self._init_residual_weights()

    def forward(self, latent: Tensor, task_state: Tensor | None = None) -> Tensor:
        base_action = self.base_expert(latent)
        if self.beta == 0.0:
            return base_action

        h = self.base_expert.hidden(latent)
        delta = self.delta_head(h)
        kappa = torch.nn.functional.softplus(self.gain_head(h)).clamp(
            min=self.kappa_min, max=self.kappa_max
        )
        scale_pose = torch.as_tensor(
            self.pose_action_scale, dtype=latent.dtype, device=latent.device
        )
        u_imp = torch.clamp((kappa * delta) / scale_pose, -1.0, 1.0)

        if self.output_dim % 7 == 0:
            arm_actions = []
            for arm in range(self.num_arms):
                act_start = arm * 7
                pose_start = arm * 6
                a_pose = torch.clamp(
                    base_action[..., act_start : act_start + 6]
                    + self.beta * u_imp[..., pose_start : pose_start + 6],
                    -1.0,
                    1.0,
                )
                a_grip = base_action[..., act_start + 6 : act_start + 7]
                arm_actions.extend([a_pose, a_grip])
            return torch.cat(arm_actions, dim=-1)
        else:
            a_pose = torch.clamp(
                base_action[..., : self.pose_dim] + self.beta * u_imp,
                -1.0,
                1.0,
            )
            a_rest = base_action[..., self.pose_dim :]
            return torch.cat([a_pose, a_rest], dim=-1)


__all__ = ["ImpedanceExpert", "ResidualImpedanceExpert"]

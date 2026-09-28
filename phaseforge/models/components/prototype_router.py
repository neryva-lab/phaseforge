"""PrototypeRouter: memoryless nearest-prototype (Voronoi) expert router.

Replaces the unconstrained softmax-MLP gate with one prototype per regime::

    d_{tk} = ‖z_t − c_k‖₂        k*_t = argmin_k d_{tk}

Training uses an explicit large-margin loss::

    L_margin = 1/|B| Σ_i Σ_{j ≠ y_i} max(0, m − (d_{ij} − d_{iy_i}))

which pushes decision boundaries into low-density regions and removes the
50/50 boundary splits that cause action chattering. Deployment is hard
top-1; top-2 dispatch exists only for the impedance blending ablation and
is never the primary path.

The router reports ``gate_logits = −dists`` so downstream consumers
(init diagnostics, validation entropy, offline evaluation) keep working
unchanged. Memoryless: no history inputs, no cross-step state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from phaseforge.models.components.router import RouterOutput


class PrototypeRouter(nn.Module):
    """Nearest-prototype router with large-margin training.

    Args:
        latent_dim: Dimension of the input latent vector.
        num_experts: Total number of experts/regimes (E/K).
        top_k: Experts dispatched per input. ``1`` is the primary hard
            routing path; ``2`` exists only for the top-2 ablation.
        margin: Distance margin ``m`` for :meth:`margin_loss_from_logits`.
        balance_coeff: Multiplier for the auxiliary balance loss
            (``λ_bal ≪ 1`` per the spec; prevents dead experts only).
    """

    def __init__(
        self,
        latent_dim: int,
        num_experts: int,
        top_k: int = 1,
        margin: float = 0.5,
        balance_coeff: float = 0.0001,
    ) -> None:
        super().__init__()
        if not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError(f"latent_dim must be a positive int, got {latent_dim!r}")
        if not isinstance(num_experts, int) or num_experts < 1:
            raise ValueError(f"num_experts must be a positive int, got {num_experts!r}")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive int, got {top_k!r}")
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot exceed num_experts ({num_experts}). "
                "The config routes to more experts than exist."
            )
        if margin < 0.0:
            raise ValueError(f"margin must be >= 0.0, got {margin}")
        if balance_coeff < 0.0:
            raise ValueError(f"balance_coeff must be >= 0.0, got {balance_coeff}")

        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.margin = float(margin)
        self.balance_coeff = float(balance_coeff)

        # One prototype per regime; overwritten by bootstrap_moe() with the
        # hierarchical regime centroids. Random init keeps the module usable
        # standalone (e.g. unit tests) before bootstrapping.
        self.prototypes = nn.Parameter(torch.randn(num_experts, latent_dim) * 0.02)

    def forward(
        self,
        latent: Tensor,
        trajectory_id: Tensor | None = None,
        trajectory_position: Tensor | None = None,
    ) -> RouterOutput:
        """Route latents to the nearest prototype(s).

        ``trajectory_id`` / ``trajectory_position`` are accepted (and
        ignored) solely for call-site compatibility with ``TopKRouter``:
        this router is memoryless and must never consume history.
        """
        if latent.ndim != 2 or latent.size(-1) != self.latent_dim:
            raise ValueError(
                "PrototypeRouter expects (B, latent_dim) latents, got "
                f"{tuple(latent.shape)} (latent_dim={self.latent_dim})."
            )
        dists = torch.cdist(latent.float(), self.prototypes.float()).to(latent.dtype)
        gate_logits = -dists
        if self.top_k == 1:
            indices = dists.argmin(dim=-1, keepdim=True)
            weights = torch.ones_like(indices, dtype=latent.dtype)
        else:
            topk_dists, indices = torch.topk(dists, self.top_k, dim=-1, largest=False)
            weights = F.softmax(-topk_dists, dim=-1)
        balance_loss = self._balance_loss(dists) * self.balance_coeff
        return RouterOutput(
            weights=weights,
            indices=indices,
            gate_logits=gate_logits,
            balance_loss=balance_loss,
            sticky_loss=torch.zeros((), device=latent.device, dtype=latent.dtype),
            # No exploration noise exists on this path: the reported logits
            # already are the deterministic pre-exploration signal.
            clean_gate_logits=gate_logits,
        )

    def _balance_loss(self, dists: Tensor) -> Tensor:
        """Switch-style balance loss on the primary (top-1) assignment."""
        top1 = dists.argmin(dim=-1)
        one_hot = F.one_hot(top1, num_classes=self.num_experts).float()
        frac = one_hot.mean(dim=0)
        probs = F.softmax(-dists, dim=-1).mean(dim=0)
        return self.num_experts * torch.sum(frac * probs)

    def margin_loss_from_logits(
        self, gate_logits: Tensor, targets: Tensor, margin: float | None = None
    ) -> Tensor:
        """Large-margin loss from reported gate logits (``−dists``).

        Args:
            gate_logits: ``(B, K)`` logits as reported by :meth:`forward`.
            targets: ``(B,)`` true regime ids.
            margin: Override for ``self.margin`` (default: use it).

        Returns:
            Scalar ``1/|B| Σ_i Σ_{j≠y_i} max(0, m − (d_j − d_y))``.
        """
        dists = -gate_logits.float()
        return self.margin_loss(dists, targets, margin=margin).to(gate_logits.dtype)

    def margin_loss(
        self, dists: Tensor, targets: Tensor, margin: float | None = None
    ) -> Tensor:
        """Large-margin loss from explicit distances."""
        m = self.margin if margin is None else float(margin)
        if m < 0.0:
            raise ValueError(f"margin must be >= 0.0, got {m}.")
        flat_targets = targets.reshape(-1).long()
        if dists.ndim != 2 or dists.size(0) != flat_targets.numel():
            raise ValueError("dists (B, K) and targets (B,) must share the batch size.")
        if dists.size(-1) != self.num_experts:
            raise ValueError(
                f"dists width {dists.size(-1)} != num_experts {self.num_experts}."
            )
        if flat_targets.numel() == 0:
            return torch.zeros((), device=dists.device, dtype=dists.dtype)
        if bool((flat_targets < 0).any()) or bool((flat_targets >= self.num_experts).any()):
            raise ValueError("margin targets are out of range.")
        correct = dists.gather(1, flat_targets.unsqueeze(-1))
        gaps = dists - correct
        margins = torch.clamp(m - gaps, min=0.0)
        eye = torch.zeros_like(margins).scatter_(1, flat_targets.unsqueeze(-1), 1.0)
        margins = margins * (1.0 - eye)
        return margins.sum(dim=-1).mean()

    @torch.no_grad()
    def ema_update(self, latents: Tensor, labels: Tensor, decay: float = 0.99) -> None:
        """Exponential-moving-average prototype update (alternative to gradients).

        ``c_k ← decay·c_k + (1−decay)·mean(z[label==k])`` for regimes present
        in the batch; absent regimes are untouched.
        """
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}.")
        flat_labels = labels.reshape(-1).long()
        flat_latents = latents.reshape(-1, self.latent_dim).to(self.prototypes.dtype)
        for regime in range(self.num_experts):
            mask = flat_labels == regime
            if bool(mask.any()):
                mean = flat_latents[mask].mean(dim=0)
                self.prototypes.data[regime] = decay * self.prototypes.data[regime] + (
                    1.0 - decay
                ) * mean.to(self.prototypes.device)

    def uniform_selection(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        """Equal weight over ALL experts (top-2 ablation baseline)."""
        count = latent.size(0)
        indices = torch.arange(self.num_experts, device=latent.device).unsqueeze(0)
        indices = indices.expand(count, self.num_experts)
        weights = torch.full(
            (count, self.num_experts), 1.0 / self.num_experts, device=latent.device
        )
        return weights, indices

    def sticky_selection(self, gate_logits: Tensor) -> tuple[Tensor, Tensor]:
        """Not supported: sticky routing is a legacy TopKRouter eval mode."""
        raise RuntimeError(
            "sticky_selection is a TopKRouter evaluation-time intervention and is "
            "not defined for PrototypeRouter (memoryless hard routing has no EMA)."
        )

    def oracle_selection(self, phase_logits: Tensor, mapping: Tensor) -> tuple[Tensor, Tensor]:
        """Not supported: oracle routing is a legacy TopKRouter eval mode."""
        raise RuntimeError(
            "oracle_selection is a TopKRouter evaluation-time intervention and is "
            "not defined for PrototypeRouter (no soft mapping M exists here)."
        )

    def reset_sticky_ema(self) -> None:
        """No-op for rollout-runner compatibility (no cross-step state)."""


__all__ = ["PrototypeRouter"]

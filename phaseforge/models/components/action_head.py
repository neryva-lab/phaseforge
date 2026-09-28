"""ActionHead: maps latent → action prediction.

The final output is squashed with ``tanh`` so predicted actions lie in
``(-1, 1)`` — the standard robomimic pattern for the ``[-1, 1]`` action
contract ("actions are assumed to lie in [-1, 1], and most networks will
have a final tanh activation to help ensure this range").
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ActionHead(nn.Module):
    """Maps a latent vector to an action prediction.

    Args:
        input_dim:  Latent dimension from encoder.
        output_dim: Action dimension.
        head_type:  ``"deterministic"`` or ``"gaussian"``.
        hidden_dim: Width of the intermediate hidden layer.

    Outputs:
        ``tanh(mean_head(trunk(x)))`` — bounded to ``(-1, 1)``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_type: str = "deterministic",
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.head_type = head_type
        self.output_dim = output_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )

        if head_type == "deterministic":
            self.mean_head = nn.Linear(hidden_dim, output_dim)
            self.log_std_head = None
        elif head_type == "gaussian":
            self.mean_head = nn.Linear(hidden_dim, output_dim)
            self.log_std_head = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(
                f"Unknown head_type: '{head_type}'. Use 'deterministic' or 'gaussian'."
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, latent: Tensor) -> Tensor:
        """Predict action from latent.

        Args:
            latent: (B, input_dim)

        Returns:
            action_pred: (B, output_dim)
                For gaussian head, returns the *mean* during training.
                Use :meth:`sample` for stochastic output.
        """
        h = self.trunk(latent)
        return torch.tanh(self.mean_head(h))

    def sample(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        """Sample from the Gaussian distribution (gaussian head only).

        Returns:
            (sampled_action, log_prob) — both (B, output_dim).
            The sampled action is tanh-squashed to ``(-1, 1)``, consistent
            with the deterministic head's output range.
        """
        if self.log_std_head is None:
            raise RuntimeError("sample() called on a deterministic ActionHead.")
        h = self.trunk(latent)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-5.0, 2.0)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        action = mean + eps * std
        log_prob = -0.5 * ((action - mean) / std) ** 2 - log_std - 0.9189  # -0.5*log(2π)
        # Squash to (-1, 1) and apply the tanh change-of-variables to the
        # log-probability (same correction as robomimic's TanhWrappedDistribution).
        action = torch.tanh(action)
        log_prob = log_prob - torch.log(1.0 - action**2 + 1e-6)
        return action, log_prob

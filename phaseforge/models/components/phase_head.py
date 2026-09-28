"""PhaseClassificationHead: Auxiliary head for Stage 1 phase supervision."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class PhaseClassificationHead(nn.Module):
    """Auxiliary head to predict phase labels from the shared latent representation.

    This head is intentionally kept as a single linear layer (no hidden layers).
    By forcing the classification to be linear, we guarantee that the phase
    structure is explicitly encoded within the latent space itself, rather than
    being learned by a deep classification head. This is critical for the MoE
    bootstrapping step, where the router is initialized directly from latent space
    centroids.

    Args:
        latent_dim: Dimension of the input latent vector.
        num_phases: Number of distinct phase classes to predict.
    """

    def __init__(self, latent_dim: int, num_phases: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_phases = num_phases

        self.classifier = nn.Linear(latent_dim, num_phases)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights appropriately for a linear classifier."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, latent: Tensor) -> Tensor:
        """Predict phase logits.

        Args:
            latent: Tensor of shape (B, latent_dim) or (B, T, latent_dim).

        Returns:
            phase_logits: Raw classification logits of shape (B, num_phases)
                or (B, T, num_phases). Do not apply softmax here; it is
                handled by the CrossEntropyLoss during training.
        """
        return self.classifier(latent)

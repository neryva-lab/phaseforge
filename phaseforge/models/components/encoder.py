"""StateEncoder: MLP with residual connections."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class StateEncoder(nn.Module):
    """MLP encoder mapping proprioceptive state → latent representation.

    Args:
        input_dim:    Raw state dimension.
        hidden_dims:  List of hidden layer widths.
        latent_dim:   Output latent dimension.
        activation:   Activation function name (``"gelu"``, ``"relu"``, ``"silu"``).
        dropout:      Dropout rate applied after each hidden layer.
        use_residual: Add a residual shortcut from the input to the output.
            When ``input_dim != latent_dim`` a learned projection
            (``res_proj``) bridges the dims; ``res_proj`` is ``Identity``
            when the dims already match.
        normalize_output: L2-normalize the output (``z / ‖z‖``). Required
            for contrastive (SupCon) training and prototype routing whose
            distances assume unit latents. ``False`` preserves the legacy
            unnormalized path bit-for-bit.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_residual: bool = True,
        normalize_output: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        act_cls = _ACTIVATIONS.get(activation, nn.GELU)

        layers: list[nn.Module] = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.hidden = nn.Sequential(*layers)
        self.output_proj = nn.Linear(in_dim, latent_dim)

        # Residual shortcut from the input to the output. A learned
        # projection bridges the dims when they differ (e.g. 151 -> 128).
        self.use_residual = bool(use_residual)
        if self.use_residual and input_dim != latent_dim:
            self.res_proj: nn.Module = nn.Linear(input_dim, latent_dim)
        else:
            self.res_proj = nn.Identity()
        self.normalize_output = bool(normalize_output)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, state: Tensor) -> Tensor:
        """Encode state to latent vector.

        Args:
            state: (B, input_dim)

        Returns:
            latent: (B, latent_dim), L2-normalized when
                ``normalize_output`` is True.
        """
        h = self.hidden(state)
        out = self.output_proj(h)
        if self.use_residual:
            out = out + self.res_proj(state)
        if self.normalize_output:
            out = nn.functional.normalize(out, p=2, dim=-1)
        return out

"""ExpertMLP: A single specialized neural network expert."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

# Reuse the activations dictionary from encoder.py for consistency
_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class ExpertMLP(nn.Module):
    """A single expert network within the Mixture-of-Experts layer.

    All experts share identical architectures but maintain independent weights.
    The architecture is typically a small multi-layer perceptron mapping the
    latent representation to the action space.

    Args:
        input_dim: Dimension of the latent vector from the encoder.
        hidden_dims: List of hidden layer widths.
        output_dim: Dimension of the action prediction.
        activation: Activation function (e.g., "gelu").
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        act_cls = _ACTIVATIONS.get(activation.lower(), nn.GELU)

        layers: list[nn.Module] = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_cls())
            in_dim = h_dim

        self.hidden = nn.Sequential(*layers)
        self.output_proj = nn.Linear(in_dim, output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize expert weights.

        Using Kaiming initialization to preserve variance through the MLP.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def reset_parameters(self) -> None:
        """Re-initialize all weights with an independent fresh draw.

        Used to break symmetry when expert clones are created by
        ``copy.deepcopy`` (bit-identical clones make the combined MoE
        output invariant to the router's weights).
        """
        self._init_weights()

    def forward(self, latent: Tensor) -> Tensor:
        """Map latent vector to action prediction.

        Args:
            latent: Tensor of shape (..., input_dim)

        Returns:
            action_pred: Tensor of shape (..., output_dim), tanh-squashed
                to ``(-1, 1)`` so MoE outputs honour the action contract
                (matches the ActionHead's output range).
        """
        h = self.hidden(latent)
        return torch.tanh(self.output_proj(h))


def hash_dropped_indices(indices: list[int]) -> str:
    """Stable sha256 of the dropped-neuron index set for audit metadata.

    Shared by every partial-warm-start site (``PhaseBootstrappedMoE`` and the
    R50-matched baseline bootstraps) so all runs record a comparable,
    content-addressed fingerprint of the dropped index set. Byte-for-byte
    identical to the original private ``phase_moe._hash_dropped_indices``.
    """
    import hashlib

    h = hashlib.sha256()
    for i in sorted(indices):
        h.update(int(i).to_bytes(8, "little", signed=False))
    return h.hexdigest()


def warm_start_experts_from_action_head(
    experts: nn.ModuleList,
    action_head: nn.Module,
    jitter_std: float = 0.02,
) -> None:
    """Copy the Stage 1 ActionHead into every expert — exactly and strictly.

    The ActionHead is a single-hidden-layer policy (``trunk`` =
    ``Linear -> GELU``, ``mean_head`` = output projection, final ``tanh``
    squash). The ExpertMLP must therefore be configured with a matching
    single-hidden-layer structure (``hidden_dims=[hidden_dim]`` with
    ``hidden_dim`` == ``action_head.hidden_dim``). Any expert parameter
    left unfilled is a hard error: a partial warm start (e.g. a randomly
    initialized second expert hidden layer) is not a functional warm start
    and would invalidate the Stage 1 -> Stage 2 initialization claim.

    A small independent Gaussian jitter is added to every copied parameter
    so the experts are not bit-identical after bootstrapping. Identical
    experts make the combined output invariant to the router's weights
    (top-k softmax weights sum to 1), so the action loss can never shape
    routing; the jitter breaks that symmetry while remaining a true warm
    start.

    Args:
        experts: The MoE layer's ``nn.ModuleList`` of experts.
        action_head: The Stage 1 ``ActionHead`` whose weights are copied.
        jitter_std: Standard deviation of the symmetry-breaking noise.
            ``0.0`` disables the jitter (exact copy).
    """
    if jitter_std < 0.0:
        raise ValueError(f"jitter_std must be >= 0.0, got {jitter_std}")
    action_sd = action_head.state_dict()
    mapping = {
        "trunk.0.weight": "hidden.0.weight",
        "trunk.0.bias": "hidden.0.bias",
        "mean_head.weight": "output_proj.weight",
        "mean_head.bias": "output_proj.bias",
    }

    for expert in experts:
        target_expert = getattr(expert, "base_expert", expert)
        expert_sd = target_expert.state_dict()

        new_dict: dict[str, Tensor] = {}
        for src_key, dst_key in mapping.items():
            if src_key not in action_sd:
                raise ValueError(
                    "Cannot warm-start experts from the ActionHead: it has no "
                    f"'{src_key}' (available keys: {sorted(action_sd)})."
                )
            if dst_key not in expert_sd:
                raise ValueError(
                    "Cannot warm-start experts: the ExpertMLP has no "
                    f"'{dst_key}' (available keys: {sorted(expert_sd)})."
                )
            new_dict[dst_key] = action_sd[src_key].clone()

        missing = sorted(set(expert_sd) - set(new_dict))
        if missing:
            raise ValueError(
                "Cannot warm-start experts: the ExpertMLP has parameters that the "
                f"ActionHead cannot fill (uninitialized keys: {missing}). The "
                "ExpertMLP must match the ActionHead's single-hidden-layer "
                "structure: use expert.hidden_dims=[<action_head.hidden_dim>]."
            )
        target_expert.load_state_dict(new_dict, strict=True)

        if jitter_std > 0.0:
            for param in target_expert.parameters():
                param.data.add_(torch.randn_like(param.data) * jitter_std)


def partial_reinit_experts_from_action_head(
    experts: nn.ModuleList,
    action_head: nn.Module,
    drop_rate: float = 0.5,
    seed: int = 42,
) -> list[int]:
    """Drop-Upcycling-style partial expert reinitialization from the ActionHead.

    Copies the Stage 1 ``ActionHead`` into every expert (exact, no jitter), then
    re-initializes a fraction ``drop_rate`` of each expert's intermediate
    neurons ("Drop-Upcycling", Nakamura et al., ICLR 2025) using the expert's
    normal Kaiming-uniform initialization. Kept neurons retain the copied
    ActionHead weights bit-exactly; reinitialized neurons differ per expert.

    This implements the structure of Drop-Upcycling (shared index set across
    experts, drop along the intermediate dimension) with one deviation from the
    paper: the paper reinitializes dropped weights from N(μ_dropped, σ_dropped)
    (the statistics of the dropped weights themselves), while this function
    uses the expert's standard Kaiming-uniform draw. All other semantics
    match: shared index set per layer applied to every expert, exact copy for
    the kept portion, drop along the intermediate dimension so each dropped
    neuron is consistently reinitialized across the row of the hidden-layer
    weight matrix, the corresponding hidden-layer bias entry, and the
    matching column of the output-projection weight matrix.

    Args:
        experts: The MoE layer's ``nn.ModuleList`` of ``ExpertMLP`` instances.
        action_head: The Stage 1 ``ActionHead`` whose weights are copied.
        drop_rate: Fraction of intermediate neurons to reinitialize per expert.
            ``0.0`` reduces to an exact-copy warm start; ``1.0`` reinitializes
            every neuron (no shared "FFN_common" structure remains). Must be
            in ``[0.0, 1.0]``.
        seed: Seed for the shared index-set draw and the per-(expert, neuron)
            Kaiming draws. Same seed → bit-identical expert weights across
            calls; same convention as ``router_init.seed``. Deterministic for
            the entire ``[0.0, 1.0]`` range, including the ``0.0`` and ``1.0``
            boundaries.

    Returns:
        The sorted list of dropped intermediate neuron indices (the shared
        index set applied to every expert). Useful for audit / metadata
        persistence.

    Raises:
        ValueError: If ``drop_rate`` is outside ``[0.0, 1.0]`` or the expert
            structure does not match the ActionHead (propagated from
            ``warm_start_experts_from_action_head``).
    """
    if not 0.0 <= drop_rate <= 1.0:
        raise ValueError(
            f"drop_rate must be in [0.0, 1.0], got {drop_rate}"
        )

    warm_start_experts_from_action_head(
        experts, action_head, jitter_std=0.0
    )

    if drop_rate == 0.0:
        return []

    num_experts = len(experts)
    if num_experts == 0:
        return []

    first_expert: ExpertMLP = cast(ExpertMLP, getattr(experts[0], "base_expert", experts[0]))
    hidden_weight: Tensor = first_expert.hidden[0].weight  # type: ignore[assignment]
    output_weight: Tensor = first_expert.output_proj.weight  # type: ignore[assignment]
    hidden_dim, input_dim = hidden_weight.shape
    output_dim, _ = output_weight.shape

    k = int(round(drop_rate * hidden_dim))
    k = max(0, min(hidden_dim, k))
    if k == 0:
        return []

    index_gen = torch.Generator(device="cpu").manual_seed(seed)
    shared_indices: Tensor = torch.randperm(hidden_dim, generator=index_gen)[
        :k
    ]
    dropped_indices: list[int] = sorted(int(i) for i in shared_indices.tolist())

    bound_h = math.sqrt(3.0) / math.sqrt(float(input_dim))
    bound_o = math.sqrt(3.0) / math.sqrt(float(hidden_dim))

    for expert_idx, expert in enumerate(experts):
        typed_expert = cast(ExpertMLP, getattr(expert, "base_expert", expert))
        hw: Tensor = typed_expert.hidden[0].weight  # type: ignore[assignment]
        hb: Tensor = typed_expert.hidden[0].bias  # type: ignore[assignment]
        ow: Tensor = typed_expert.output_proj.weight  # type: ignore[assignment]
        device = hw.device

        for j, i in enumerate(dropped_indices):
            gen_h = torch.Generator(device="cpu").manual_seed(
                seed * 100_003 + expert_idx * 1_000 + j
            )
            new_h = torch.empty(
                input_dim, dtype=hw.dtype
            ).uniform_(-bound_h, bound_h, generator=gen_h)
            hw.data[i, :].copy_(new_h.to(device))

            hb.data[i] = 0.0

            gen_o = torch.Generator(device="cpu").manual_seed(
                seed * 100_003 + expert_idx * 1_000 + j + 50_000
            )
            new_o = torch.empty(
                output_dim, dtype=ow.dtype
            ).uniform_(-bound_o, bound_o, generator=gen_o)
            ow.data[:, i].copy_(new_o.to(device))

    return dropped_indices


def one_warm_experts_from_action_head(
    experts: nn.ModuleList,
    action_head: nn.Module,
    jitter_std: float = 0.0,
    warm_idx: int = 0,
) -> None:
    """Initialize exactly one expert as a warm-started ActionHead copy.

    Expert ``warm_idx`` receives the standard warm start (exact copy +
    ``jitter_std`` noise, matching ``warm_start_experts_from_action_head``).
    Every other expert is reset via ``ExpertMLP.reset_parameters`` — an
    independent Kaiming-uniform draw from the standard scratch distribution.

    Diagnostic for the question: does one transferred generalist expert help at
    all, or is complete random diversity better?

    Args:
        experts: The MoE layer's ``nn.ModuleList`` of ``ExpertMLP`` instances.
        action_head: The Stage 1 ``ActionHead`` whose weights are copied into
            the warm expert.
        jitter_std: Symmetry-breaking noise added to the warm expert's copied
            weights. Defaults to ``0.0`` (exact copy) so the diagnostic cleanly
            isolates "one generalist copy" without confounding jitter.
        warm_idx: Index of the expert that receives the warm start. Must be in
            ``[0, len(experts))``.

    Raises:
        ValueError: If ``warm_idx`` is out of range or ``experts`` is empty.
    """
    num_experts = len(experts)
    if num_experts == 0:
        raise ValueError(
            f"warm_idx must be in [0, {num_experts}), got {warm_idx}"
        )
    if not 0 <= warm_idx < num_experts:
        raise ValueError(
            f"warm_idx must be in [0, {num_experts}), got {warm_idx}"
        )

    for expert_idx, expert in enumerate(experts):
        if expert_idx == warm_idx:
            warm_start_experts_from_action_head(
                nn.ModuleList([cast(ExpertMLP, expert)]),
                action_head,
                jitter_std=jitter_std,
            )
        else:
            cast(ExpertMLP, expert).reset_parameters()

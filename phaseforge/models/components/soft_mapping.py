"""Soft phase-to-expert mapping (V2-B): construction and validation.

The mapping ``M`` is a ``(P, E)`` right-stochastic matrix whose rows
``M[p, :]`` distribute phase ``p``'s routing mass across experts. It is
used three ways:

* as the bootstrap target for the Stage-2 router (each expert is
  initialized near its phase's prototype geometry);
* as the teacher ``T = M^T softmax(phase_logits)`` for V2-D
  teacher-distilled routing;
* as the oracle evaluator (V2-E) that routes by the phase head through M.

Two constructions are supported:

* ``prototype_softmax``: ``M[p, e] = softmax(cos(c_p, pi_e) / tau)``
  over experts, where ``c_p`` is phase ``p``'s spherical centroid and
  ``pi_e`` expert ``e``'s hierarchical phase prototype — both unit-norm,
  matching the router's cosine geometry (``normalize_input=true``).
* ``hierarchical_uniform``: data-free uniform rows over the contiguous
  expert blocks that ``compute_hierarchical_phase_prototypes`` assigns
  to each phase when ``E > P`` (e.g. E=8, P=6: phases 0-1 own two
  experts, phases 2-5 own one).

Every construction validates the result (finite, non-negative,
right-stochastic rows); a mapping that fails validation is a hard error,
never a silent fallback.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

_ALLOWED_MODES = ("prototype_softmax", "hierarchical_uniform")


def validate_soft_mapping(mapping: Tensor) -> None:
    """Hard-validate a ``(P, E)`` mapping: finite, non-negative, rows sum to 1.

    Raises:
        ValueError: the mapping violates any of the invariants. The
            check is exact (``atol=1e-5``), so a numerically degenerate
            construction cannot silently reach the router or teacher.
    """
    if not isinstance(mapping, Tensor) or mapping.ndim != 2:
        raise ValueError(
            f"Soft mapping must be a 2D tensor (P, E), got {getattr(mapping, 'shape', None)}"
        )
    if torch.isnan(mapping).any() or torch.isinf(mapping).any():
        raise ValueError("Soft mapping contains non-finite entries.")
    if (mapping < 0).any():
        raise ValueError("Soft mapping contains negative entries.")
    row_sums = mapping.sum(dim=-1)
    expected = torch.ones_like(row_sums)
    if not torch.allclose(row_sums, expected, atol=1e-5):
        bad = torch.nonzero(~torch.isclose(row_sums, expected, atol=1e-5)).flatten().tolist()
        raise ValueError(
            f"Soft mapping rows must sum to 1 (right-stochastic); offending rows: {bad}."
        )


def build_hierarchical_uniform_mapping(num_phases: int, num_experts: int) -> Tensor:
    """Data-free construction mirroring the hierarchical prototype layout.

    ``compute_hierarchical_phase_prototypes`` distributes ``E > P`` experts
    across phases as evenly as possible: phases ``0..rem-1`` own
    ``base_k + 1`` experts, the rest own ``base_k``, in contiguous blocks.
    This builds the matching uniform rows, so each phase's routing mass is
    split equally over exactly the experts whose prototypes live inside
    that phase.

    Args:
        num_phases: Number of phases P.
        num_experts: Number of experts E.

    Returns:
        Tensor of shape ``(P, E)``, right-stochastic.
    """
    if int(num_phases) < 1 or int(num_experts) < 1:
        raise ValueError(
            f"num_phases and num_experts must be positive, got ({num_phases}, {num_experts})"
        )
    num_phases = int(num_phases)
    num_experts = int(num_experts)
    base_k = num_experts // num_phases
    rem = num_experts % num_phases
    mapping = torch.zeros((num_phases, num_experts))
    offset = 0
    for p in range(num_phases):
        k_p = base_k + (1 if p < rem else 0)
        mapping[p, offset : offset + k_p] = 1.0 / k_p
        offset += k_p
    validate_soft_mapping(mapping)
    return mapping


def build_prototype_softmax_mapping(
    phase_centroids: Tensor,
    expert_prototypes: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """``M[p, e] = softmax(cos(c_p, pi_e) / tau)`` over experts.

    ``phase_centroids`` (P, D) and ``expert_prototypes`` (E, D) are
    L2-normalized (as produced by ``compute_phase_centroids`` and
    ``compute_hierarchical_phase_prototypes`` with ``spherical=True``),
    so the similarities live in the same cosine geometry the router
    operates in. A temperature of 1.0 keeps rows soft; smaller values
    sharpen toward one-hot within-phase assignments.

    Args:
        phase_centroids: Unit-norm phase centroids, shape ``(P, D)``.
        expert_prototypes: Unit-norm expert prototypes, shape ``(E, D)``.
        temperature: Positive softmax temperature ``tau``.

    Returns:
        Tensor of shape ``(P, E)``, right-stochastic.
    """
    if phase_centroids.ndim != 2 or expert_prototypes.ndim != 2:
        raise ValueError(
            "phase_centroids and expert_prototypes must be 2D tensors, got "
            f"{tuple(phase_centroids.shape)} and {tuple(expert_prototypes.shape)}"
        )
    if phase_centroids.shape[1] != expert_prototypes.shape[1]:
        raise ValueError(
            f"Dimension mismatch: centroids D={phase_centroids.shape[1]}, "
            f"prototypes D={expert_prototypes.shape[1]}"
        )
    temperature = float(temperature)
    if not temperature > 0 or not torch.isfinite(torch.tensor(temperature)):
        raise ValueError(
            f"Soft-mapping temperature must be a positive finite float, got {temperature}"
        )
    sims = (
        F.normalize(phase_centroids, p=2, dim=-1) @ F.normalize(expert_prototypes, p=2, dim=-1).t()
    )
    mapping = F.softmax(sims / temperature, dim=-1)
    validate_soft_mapping(mapping)
    return mapping


def build_soft_mapping(
    mode: str,
    *,
    num_phases: int | None = None,
    num_experts: int | None = None,
    phase_centroids: Tensor | None = None,
    expert_prototypes: Tensor | None = None,
    temperature: float = 1.0,
) -> Tensor:
    """Dispatch to the requested construction mode.

    ``prototype_softmax`` requires ``phase_centroids`` and
    ``expert_prototypes``; ``hierarchical_uniform`` requires
    ``num_phases`` and ``num_experts``. Any other mode, or a missing
    requirement, raises ``ValueError``.
    """
    mode = str(mode).lower()
    if mode not in _ALLOWED_MODES:
        raise ValueError(
            f"Unknown soft-mapping mode {mode!r}. Supported: {', '.join(_ALLOWED_MODES)}."
        )
    if mode == "hierarchical_uniform":
        if num_phases is None or num_experts is None:
            raise ValueError("hierarchical_uniform requires num_phases and num_experts.")
        return build_hierarchical_uniform_mapping(num_phases, num_experts)
    if phase_centroids is None or expert_prototypes is None:
        raise ValueError("prototype_softmax requires phase_centroids and expert_prototypes.")
    return build_prototype_softmax_mapping(
        phase_centroids, expert_prototypes, temperature=temperature
    )


__all__ = [
    "build_soft_mapping",
    "build_hierarchical_uniform_mapping",
    "build_prototype_softmax_mapping",
    "validate_soft_mapping",
]

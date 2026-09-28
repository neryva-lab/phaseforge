"""Clustering algorithms and prototype extraction for MoE routing initialization.

Provides:
- Deterministic Spherical K-Means (Dhillon & Modha 2001) for directional cosine geometry.
- Standard and Spherical phase centroid computation.
- Hierarchical phase prototype generation handling any expert count E relative to phase count P:
    * E == P: direct phase centroids (c_1, ..., c_P).
    * E > P (e.g. K=12, P=6): intra-phase spherical K-means on each phase's latents to discover
      fine-grained sub-prototypes (c_{p,1}, c_{p,2}, ...).
    * E < P (e.g. K=3, P=6): super-prototypes via spherical K-means clustering over
      the P phase centroids.
- Phase-head linear weight router initialization (normalized discriminative directions).
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def spherical_kmeans(
    x: Tensor,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-5,
    seed: int = 42,
) -> tuple[Tensor, Tensor]:
    """Deterministic Spherical K-Means on PyTorch tensors.

    Maximizes sum_{j=1}^k sum_{i in C_j} cos(x_i, c_j) subject to ||c_j||_2 = 1.

    Args:
        x: Input tensor of shape (N, D).
        k: Number of clusters (1 <= k <= N).
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance for centroid cosine change.
        seed: Random seed for deterministic K-means++ initialization.

    Returns:
        centroids: Tensor of shape (k, D), L2-normalized.
        cluster_assignments: Tensor of shape (N,) with integer cluster IDs.

    Raises:
        ValueError: If inputs are invalid or k > N.
    """
    if x.ndim != 2:
        raise ValueError(f"Input tensor must be 2D (N, D), got shape {tuple(x.shape)}")
    n, d = x.shape
    if k < 1:
        raise ValueError(f"Number of clusters k must be >= 1, got {k}")
    if n < k:
        raise ValueError(f"Sample count N ({n}) is less than cluster count k ({k})")

    # Work on unit-normalized vectors
    x_norm = F.normalize(x, p=2, dim=-1)

    # Deterministic K-means++ initialization on unit sphere
    g = torch.Generator(device="cpu").manual_seed(seed)
    centroids = torch.zeros((k, d), dtype=x.dtype, device=x.device)

    # 1. Pick first centroid uniformly at random
    first_idx = int(torch.randint(0, n, (1,), generator=g).item())
    centroids[0] = x_norm[first_idx]

    # 2. Pick remaining centroids proportional to squared angular distance: (1 - cos(x, c))^2
    for c_idx in range(1, k):
        # Cosine similarity to already chosen centroids: (N, c_idx)
        cos_sims = torch.mm(x_norm, centroids[:c_idx].t())
        max_cos, _ = cos_sims.max(dim=1)  # max similarity = closest centroid
        # Distance = 1 - cos (clamped to >= 0)
        dists = torch.clamp(1.0 - max_cos, min=1e-8)
        probs = (dists ** 2) / (dists ** 2).sum()
        probs_cpu = probs.cpu()
        next_idx = int(torch.multinomial(probs_cpu, 1, generator=g).item())
        centroids[c_idx] = x_norm[next_idx]

    # Lloyd's iterations on the sphere
    cluster_assignments = torch.zeros((n,), dtype=torch.long, device=x.device)
    for it in range(max_iter):
        # E-step: assign each point to the closest centroid (highest cosine similarity)
        # sim: (N, k)
        sim = torch.mm(x_norm, centroids.t())
        new_assignments = sim.argmax(dim=1)

        # M-step: compute new centroids by vector sum and L2 normalization
        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            mask = new_assignments == j
            if mask.any():
                sum_vec = x_norm[mask].sum(dim=0)
                new_centroids[j] = F.normalize(sum_vec, p=2, dim=-1)
            else:
                # Handle empty cluster: reassign to the point furthest from its assigned center
                dists = 1.0 - sim.max(dim=1).values
                furthest_idx = dists.argmax().item()
                new_centroids[j] = x_norm[furthest_idx]

        # Check convergence: maximum centroid shift
        shift = 1.0 - (centroids * new_centroids).sum(dim=1)
        max_shift = shift.max().item()

        centroids = new_centroids
        cluster_assignments = new_assignments

        if max_shift < tol:
            logger.debug(
                f"Spherical K-Means converged in {it + 1} iterations "
                f"(max shift {max_shift:.2e} < {tol})."
            )
            break

    return centroids, cluster_assignments


def compute_phase_centroids(
    latents: Tensor,
    phases: Tensor,
    num_phases: int,
    spherical: bool = False,
) -> Tensor:
    """Compute phase centroids from latent representations.

    Args:
        latents: Tensor of shape (N, D).
        phases: Tensor of shape (N,) with integer phase IDs.
        num_phases: Total expected number of phases P.
        spherical: If True, normalize each latent before averaging (spherical mean).
                   If False, average raw latents then normalize (Euclidean mean).

    Returns:
        Tensor of shape (num_phases, D), L2-normalized.

    Raises:
        ValueError: If any phase in [0, num_phases - 1] is missing from the dataset.
    """
    if latents.ndim != 2:
        raise ValueError(f"latents must be (N, D), got {tuple(latents.shape)}")
    n, d = latents.shape
    phases_flat = phases.view(-1)
    if phases_flat.numel() != n:
        raise ValueError(f"Shape mismatch: {n} latents vs {phases_flat.numel()} phases")

    device = latents.device
    work_latents = F.normalize(latents, p=2, dim=-1) if spherical else latents

    phase_sums = torch.zeros((num_phases, d), dtype=latents.dtype, device=device)
    phase_counts = torch.zeros((num_phases,), dtype=latents.dtype, device=device)

    phase_expanded = phases_flat.unsqueeze(1).expand_as(work_latents)
    phase_sums.scatter_add_(0, phase_expanded, work_latents)
    counts = torch.bincount(phases_flat, minlength=num_phases).float().to(device)
    phase_counts += counts

    absent = phase_counts == 0
    if absent.any():
        missing = [int(i) for i in absent.nonzero().flatten().tolist()]
        raise ValueError(
            f"Phase centroid computation: {len(missing)} phase(s) have zero samples "
            f"in the bootstrap dataset (missing phases: {missing}). Every phase must be present."
        )

    centroids = phase_sums / phase_counts.unsqueeze(1)
    return F.normalize(centroids, p=2, dim=-1)


def compute_hierarchical_phase_prototypes(
    latents: Tensor,
    phases: Tensor,
    num_phases: int,
    num_experts: int,
    seed: int = 42,
    spherical: bool = False,
) -> Tensor:
    """Generate unconfounded expert prototypes for arbitrary expert counts E relative to P.

    Principled Scaling Protocol:
    1. E == P: Exact 1:1 phase centroids.
    2. E > P (e.g. E=12, P=6): Intra-phase spherical K-means.
       Splits each phase p into sub-clusters to discover fine-grained sub-regimes:
       Phase 0 -> {c_{0,1}, c_{0,2}}, ..., Phase 5 -> {c_{5,1}, c_{5,2}}.
       Tests whether the coarse phase scaffold supports finer specialization.
    3. E < P (e.g. E=3, P=6): Super-prototypes via spherical K-means over the P phase centroids.
       {c_0, ..., c_5} -> {c_super_0, c_super_1, c_super_2}.
       Tests whether coarse routing regimes can be formed by aggregating privileged phase geometry.

    Args:
        latents: Tensor of shape (N, D).
        phases: Tensor of shape (N,) with integer phase IDs.
        num_phases: Number of ground-truth phases P.
        num_experts: Target number of experts E.
        seed: Random seed for deterministic clustering.
        spherical: Whether to use spherical centroid averaging for base centroids.

    Returns:
        Tensor of shape (num_experts, D), L2-normalized.
    """
    d = latents.shape[-1]
    device = latents.device

    # Case 1: Exact match E == P
    if num_experts == num_phases:
        logger.info(
            f"Prototype generation: E={num_experts} == P={num_phases} -> 1:1 phase centroids."
        )
        return compute_phase_centroids(latents, phases, num_phases, spherical=spherical)

    # Case 2: Expert scaling E > P (Intra-phase sub-prototypes)
    if num_experts > num_phases:
        logger.info(
            f"Prototype generation: E={num_experts} > P={num_phases} -> "
            f"intra-phase spherical K-means sub-prototypes."
        )
        prototypes = torch.zeros((num_experts, d), dtype=latents.dtype, device=device)

        # Distribute experts across phases as evenly as possible
        base_k = num_experts // num_phases
        rem = num_experts % num_phases
        k_per_phase = [base_k + (1 if p < rem else 0) for p in range(num_phases)]

        expert_offset = 0
        for p in range(num_phases):
            k_p = k_per_phase[p]
            mask = phases.view(-1) == p
            p_latents = latents[mask]
            if len(p_latents) < k_p:
                raise ValueError(
                    f"Phase {p} has only {len(p_latents)} samples, cannot form {k_p} sub-clusters."
                )

            if k_p == 1:
                # Single centroid for this phase
                dummy_phase = torch.zeros(len(p_latents), dtype=torch.long, device=device)
                c_p = compute_phase_centroids(p_latents, dummy_phase, 1, spherical=spherical)
                prototypes[expert_offset] = c_p[0]
            else:
                # Spherical K-means within phase p
                sub_centroids, _ = spherical_kmeans(p_latents, k=k_p, seed=seed + p)
                prototypes[expert_offset : expert_offset + k_p] = sub_centroids

            expert_offset += k_p

        return prototypes

    # Case 3: Expert reduction E < P (Super-prototypes)
    logger.info(
        f"Prototype generation: E={num_experts} < P={num_phases} -> "
        f"clustering {num_phases} phase centroids into {num_experts} super-prototypes."
    )
    base_centroids = compute_phase_centroids(latents, phases, num_phases, spherical=spherical)
    super_centroids, _ = spherical_kmeans(base_centroids, k=num_experts, seed=seed)
    return super_centroids


def compute_phase_head_router_weights(
    phase_head_weight: Tensor,
    num_experts: int,
) -> Tensor:
    """Extract and L2-normalize rows of the linear phase classification head.

    Args:
        phase_head_weight: Tensor of shape (P, D) from PhaseClassificationHead.
        num_experts: Expected number of experts E (must match P).

    Returns:
        Tensor of shape (num_experts, D), L2-normalized.
    """
    p, d = phase_head_weight.shape
    if p != num_experts:
        raise ValueError(
            f"Phase-head router init requires num_phases ({p}) == num_experts ({num_experts})."
        )
    return F.normalize(phase_head_weight, p=2, dim=-1)

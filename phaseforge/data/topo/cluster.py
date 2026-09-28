"""Segment-prototype clustering and evidence-based K selection.

PELT produces variable-length segments; this module converts them into a
finite regime set:

1. :func:`segment_features` — one feature vector per segment (mean/variance
   of the task variables; action statistics only when explicitly enabled
   for diagnostics, never for routing labels by default).
2. :func:`cluster_segments` — cluster segment prototypes across all
   training demonstrations (``kmeans`` / ``spherical_kmeans`` /
   ``agglomerative``).
3. :func:`select_K` — evidence-based regime-count selection::

       K* = argmax_K [Observability + ActionExplanation + Stability
                      - Complexity]

   Each term is a caller-supplied scalar per candidate K (see
   :mod:`phaseforge.data.topo.observability` for Observability); this
   module only performs the argmax, so no import cycle is possible.

CPU-only. ``kmeans``/``agglomerative`` use scikit-learn; ``spherical``
reuses the repo's deterministic spherical K-means on torch tensors.
"""

from __future__ import annotations

import numpy as np

_VALID_METHODS: tuple[str, ...] = ("kmeans", "spherical_kmeans", "agglomerative")


def segment_features(
    segments: list[np.ndarray],
    *,
    include_action_stats: bool = False,
    actions: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Featurize variable-length segments into a fixed-width matrix.

    Args:
        segments: One ``(L_j, Ds)`` task-space array per segment.
        include_action_stats: When True, append per-segment action mean
            and variance (diagnostic only; routing labels must keep this
            False so regimes stay observable from ``x_t`` alone).
        actions: One ``(L_j, Da)`` action array per segment; required
            when ``include_action_stats`` is True.

    Returns:
        Array of shape ``(Nseg, Df)`` with ``Df = 2*Ds`` (+ ``2*Da`` when
        action stats are included).
    """
    if not segments:
        raise ValueError("Cannot featurize an empty segment list.")
    if include_action_stats and actions is None:
        raise ValueError("include_action_stats=True requires per-segment actions.")
    if actions is not None and len(actions) != len(segments):
        raise ValueError("segments and actions must have equal length.")
    rows: list[np.ndarray] = []
    for idx, seg in enumerate(segments):
        seg_arr = np.asarray(seg, dtype=np.float64)
        if seg_arr.ndim != 2 or seg_arr.shape[0] == 0:
            raise ValueError(f"Segment {idx} must be a non-empty (L, Ds) array.")
        if not np.isfinite(seg_arr).all():
            raise ValueError(f"Non-finite values in segment {idx}.")
        parts = [np.mean(seg_arr, axis=0), np.var(seg_arr, axis=0)]
        if include_action_stats:
            assert actions is not None
            act = np.asarray(actions[idx], dtype=np.float64)
            if act.shape[0] != seg_arr.shape[0]:
                raise ValueError(f"Action length mismatch in segment {idx}.")
            parts.extend([np.mean(act, axis=0), np.var(act, axis=0)])
        rows.append(np.concatenate(parts, axis=0))
    return np.stack(rows, axis=0)


def cluster_segments(
    features: np.ndarray,
    num_clusters: int,
    method: str = "kmeans",
    seed: int = 42,
) -> np.ndarray:
    """Cluster segment prototypes into regime ids.

    Args:
        features: Array of shape ``(Nseg, Df)`` from :func:`segment_features`.
        num_clusters: Number of regimes K (``1 <= K <= Nseg``).
        method: ``kmeans`` | ``spherical_kmeans`` | ``agglomerative``.
        seed: Deterministic seed.

    Returns:
        Integer labels of shape ``(Nseg,)`` in ``[0, K)``.
    """
    feats = np.asarray(features, dtype=np.float64)
    if feats.ndim != 2:
        raise ValueError(f"Expected features shape (Nseg, Df), got {feats.shape}.")
    nseg = feats.shape[0]
    k = int(num_clusters)
    if k < 1:
        raise ValueError(f"num_clusters must be >= 1, got {k}.")
    if nseg < k:
        raise ValueError(f"Cannot form {k} clusters from {nseg} segments.")
    if not np.isfinite(feats).all():
        raise ValueError("Non-finite values in segment features.")
    name = str(method).lower()
    if name not in _VALID_METHODS:
        raise ValueError(f"Unknown clustering method {method!r}; expected {_VALID_METHODS}.")
    if name == "kmeans":
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=k, random_state=int(seed), n_init=10)
        return np.asarray(model.fit_predict(feats), dtype=np.int64)
    if name == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(n_clusters=k)
        return np.asarray(model.fit_predict(feats), dtype=np.int64)
    # Spherical K-means on torch tensors (deterministic, CPU).
    import torch

    from phaseforge.models.components.clustering import spherical_kmeans

    with torch.no_grad():
        _, assign = spherical_kmeans(
            torch.from_numpy(feats).float(), k=k, seed=int(seed)
        )
    return np.asarray(assign.cpu().numpy(), dtype=np.int64)


def select_K(scores: dict[int, dict[str, float]]) -> int:
    """Select the regime count by evidence.

    Args:
        scores: Mapping ``K -> {"observability": .., "action_explanation": ..,
            "stability": .., "complexity": ..}`` with finite scalars.

    Returns:
        The ``K`` maximizing ``observability + action_explanation +
        stability - complexity``. Ties resolve to the smaller K.
    """
    if not scores:
        raise ValueError("Cannot select K from an empty score table.")
    best_k: int | None = None
    best_total = -np.inf
    for key in sorted(scores):
        entry = scores[int(key)]
        try:
            total = (
                float(entry["observability"])
                + float(entry["action_explanation"])
                + float(entry["stability"])
                - float(entry["complexity"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Score entry for K={key} is missing terms: {entry!r}.") from exc
        if not np.isfinite(total):
            raise ValueError(f"Non-finite total score for K={key}: {total}.")
        if total > best_total:
            best_total = total
            best_k = int(key)
    assert best_k is not None
    return best_k


def k_sweep_candidates(k_min: int = 3, k_max: int = 10) -> list[int]:
    """Candidate regime counts for the evidence sweep (inclusive range)."""
    lo, hi = int(k_min), int(k_max)
    if lo < 2:
        raise ValueError(f"k_min must be >= 2, got {lo}.")
    if hi < lo:
        raise ValueError(f"k_max ({hi}) must be >= k_min ({lo}).")
    return list(range(lo, hi + 1))


__all__: list[str] = [
    "cluster_segments",
    "k_sweep_candidates",
    "segment_features",
    "select_K",
]

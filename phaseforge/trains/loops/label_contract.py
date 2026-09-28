"""Shared label-field contract for Stage 1 / Stage 2 trainers.

Single source of truth for the baseline program's label policy:

- ``train.phase_label_field`` selects the phase-classification / diagnostic
  label (Stage 1 CE + train/val phase accuracy, Stage 2 routing accuracy /
  phase-weighted losses).
- ``train.supcon.label_field`` selects the Stage 1 SupCon label.
- ``train.margin.label_field`` selects the Stage 2 margin-routing label.

Topology rows resolve all three to ``"phase_topo"``; the Static Rule
comparison resolves all three to ``"phase"``. Plain / factorial controls
have no Stage 1 phase/SupCon loss but resolve Stage 2 prototype + margin
fields to ``"phase_topo"``.

Every failure here is fail-closed with method / task / split / requested
field context. Invalid labels stop before training: no silent remapping
beyond the documented unique-to-contiguous mapping used by router bootstrap
for topo/dynamic vocabularies.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

#: Label vocabularies the data pipeline can produce. ``phase`` is the
#: canonical rule-derived vocabulary; ``phase_rule`` is its explicit alias;
#: ``phase_topo`` is the PELT topology vocabulary (``topo_pelt_k6``);
#: ``phase_dynamic`` is the legacy dynamics vocabulary (backward compat).
ALLOWED_LABEL_FIELDS = frozenset({"phase", "phase_rule", "phase_topo", "phase_dynamic"})

#: Locked regime count under ``topo_pelt_k6``. All final rows retain
#: six phase-head outputs and six experts.
LOCKED_NUM_CLASSES = 6


def describe_run(cfg: Any) -> dict[str, Any]:
    """Extract method / task / seed context for fail-closed error messages."""
    try:
        models = cfg.get("models") if hasattr(cfg, "get") else None
    except Exception:
        models = None
    try:
        project = cfg.get("project") if hasattr(cfg, "get") else None
    except Exception:
        project = None
    try:
        data = cfg.get("data") if hasattr(cfg, "get") else None
    except Exception:
        data = None

    def _get(mapping: Any, key: str, default: Any = None) -> Any:
        try:
            if mapping is None:
                return default
            if hasattr(mapping, "get"):
                value = mapping.get(key, default)
                return default if value is None else value
            return default
        except Exception:
            return default

    model_name = _get(models, "name", None)
    if model_name is None and models is not None:
        try:
            target = _get(models, "_target_", "")
            model_name = str(target).rsplit(".", 1)[-1] if target else "unknown"
        except Exception:
            model_name = "unknown"
    method = _get(project, "method", None) or model_name or "unknown"

    task = "unknown"
    if data is not None:
        try:
            source = data.get("source") if hasattr(data, "get") else None
            if source is not None and hasattr(source, "get"):
                task = str(source.get("task_name", "unknown"))
        except Exception:
            pass

    seed = _get(project, "seed", "unknown")
    return {"method": str(method), "task": str(task), "seed": seed}


def infer_split() -> str:
    """Infer the active data split from the autograd state.

    Validation runs under ``torch.no_grad()`` / ``inference_mode()`` while
    training runs with grad enabled. This keeps per-batch errors accurate
    without threading a split argument through every ``_compute_loss`` call.
    Preflight checks pass the split explicitly.
    """
    try:
        return "train" if torch.is_grad_enabled() else "val"
    except Exception:
        return "train/val"


def peek_loader_batch(loader: Any) -> Any | None:
    """Return a representative batch without touching the live loader.

    This function runs immediately before training and validation. Calling
    ``iter(loader)`` here is unsafe: with persistent workers, the iterator
    owns worker and prefetch state that cannot be restored by resetting RNG
    states or the sampler generator. The next real iterator can therefore
    start from a different prefetched/permuted sequence even though all
    visible RNG states appear unchanged.

    The project loaders are map-style ``DataLoader`` instances. Sample the
    underlying dataset directly and apply the loader's collator, which
    validates the same batch structure without creating an iterator, worker,
    sampler, or prefetch state. Returns ``None`` for an empty or unsupported
    loader. Global RNG states are still restored because a dataset or
    collator may legitimately use process-level randomness while producing
    its sample.
    """
    torch_state = torch.get_rng_state().clone()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    try:
        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            return None
        try:
            dataset_size = len(dataset)
        except (TypeError, AttributeError):
            return None
        if dataset_size <= 0:
            return None

        batch_size = getattr(loader, "batch_size", None)
        collate_fn = getattr(loader, "collate_fn", None)

        # DataLoader(batch_size=None) passes one dataset item directly to its
        # collate function; handle that mode here.
        if batch_size is None:
            sample = dataset[0]
            return collate_fn(sample) if collate_fn is not None else sample

        batch_size = int(batch_size)
        if batch_size <= 0:
            return None
        if bool(getattr(loader, "drop_last", False)) and dataset_size < batch_size:
            return None

        samples = [dataset[index] for index in range(min(batch_size, dataset_size))]
        return collate_fn(samples) if collate_fn is not None else samples
    except (IndexError, KeyError):
        return None
    finally:
        torch.set_rng_state(torch_state)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def resolve_label_field(
    train_cfg: Any,
    key: str,
    *,
    cfg: Any,
    purpose: str,
) -> str:
    """Read and validate a configured label field name (fail-closed).

    Args:
        train_cfg: The ``cfg.train`` subtree.
        key: Top-level field name inside ``cfg.train`` (``"phase_label_field"``).
        cfg: Full run config (for method/task/seed context).
        purpose: Human-readable loss/diagnostic name for the error message.

    Returns:
        The validated field name string.
    """
    try:
        raw = train_cfg.get(key, "phase") if hasattr(train_cfg, "get") else "phase"
    except Exception:
        raw = "phase"
    field = str(raw) if raw is not None else "phase"
    if field not in ALLOWED_LABEL_FIELDS:
        ctx = describe_run(cfg)
        raise ValueError(
            f"{purpose}: unknown label field {field!r} (train.{key}). "
            f"Allowed: {sorted(ALLOWED_LABEL_FIELDS)}. "
            f"(method={ctx['method']}, task={ctx['task']}, seed={ctx['seed']})"
        )
    return field


def resolve_supcon_label_field(train_cfg: Any, *, cfg: Any) -> str:
    """Read and validate ``train.supcon.label_field`` (fail-closed)."""
    try:
        supcon = train_cfg.get("supcon") if hasattr(train_cfg, "get") else None
    except Exception:
        supcon = None
    try:
        if supcon is not None and hasattr(supcon, "get"):
            raw = supcon.get("label_field", "phase")
        else:
            raw = "phase"
    except Exception:
        raw = "phase"
    field = str(raw) if raw is not None else "phase"
    if field not in ALLOWED_LABEL_FIELDS:
        ctx = describe_run(cfg)
        raise ValueError(
            f"Stage 1 SupCon: unknown label field {field!r} (train.supcon.label_field). "
            f"Allowed: {sorted(ALLOWED_LABEL_FIELDS)}. "
            f"(method={ctx['method']}, task={ctx['task']}, seed={ctx['seed']})"
        )
    return field


def resolve_margin_label_field(train_cfg: Any, *, cfg: Any) -> str:
    """Read and validate ``train.margin.label_field`` (fail-closed)."""
    try:
        margin = train_cfg.get("margin") if hasattr(train_cfg, "get") else None
    except Exception:
        margin = None
    try:
        if margin is not None and hasattr(margin, "get"):
            raw = margin.get("label_field", "phase")
        else:
            raw = "phase"
    except Exception:
        raw = "phase"
    field = str(raw) if raw is not None else "phase"
    if field not in ALLOWED_LABEL_FIELDS:
        ctx = describe_run(cfg)
        raise ValueError(
            f"Stage 2 margin: unknown label field {field!r} (train.margin.label_field). "
            f"Allowed: {sorted(ALLOWED_LABEL_FIELDS)}. "
            f"(method={ctx['method']}, task={ctx['task']}, seed={ctx['seed']})"
        )
    return field


def validate_label_values(
    labels: torch.Tensor,
    num_classes: int,
    *,
    cfg: Any,
    field: str,
    purpose: str,
    split: str,
) -> None:
    """Fail closed on non-integer, negative, or out-of-range labels.

    Checks integer dtype, ``0 <= labels < num_classes``. Empty tensors pass
    (masked batches with no valid steps contribute no loss). No silent
    remapping is performed here; the only documented remapping is the
    unique-to-contiguous mapping applied by router bootstrap for
    topo/dynamic vocabularies, which is recorded in the bootstrap metadata.
    """
    if labels.numel() == 0:
        return
    # Integer-valued regime ids only. Float labels are never valid; the
    # prefix check covers every integer width (int8..int64, uint8) and bool
    # (accepted: a bool mask of two regimes still indexes [0, 1]).
    if not str(labels.dtype).startswith(
        ("torch.int", "torch.long", "torch.uint", "torch.bool")
    ):
        ctx = describe_run(cfg)
        raise ValueError(
            f"{purpose}: label field {field!r} must be integer-valued, "
            f"got dtype {labels.dtype}. "
            f"(method={ctx['method']}, task={ctx['task']}, split={split}, "
            f"seed={ctx['seed']}, requested_field={field})"
        )
    try:
        lo = int(labels.min().item())
        hi = int(labels.max().item())
    except Exception as exc:
        ctx = describe_run(cfg)
        raise ValueError(
            f"{purpose}: could not inspect label range for field {field!r}: {exc}. "
            f"(method={ctx['method']}, task={ctx['task']}, split={split}, "
            f"seed={ctx['seed']}, requested_field={field})"
        ) from exc
    if lo < 0 or hi >= int(num_classes):
        ctx = describe_run(cfg)
        raise ValueError(
            f"{purpose}: label field {field!r} has out-of-range values "
            f"[{lo}, {hi}] for {num_classes} classes (expected [0, {int(num_classes) - 1}]). "
            f"(method={ctx['method']}, task={ctx['task']}, split={split}, "
            f"seed={ctx['seed']}, requested_field={field})"
        )


def resolve_label_tensor(
    batch: dict[str, torch.Tensor],
    field: str,
    *,
    cfg: Any,
    purpose: str,
    num_classes: int,
    split: str | None = None,
) -> torch.Tensor:
    """Fetch ``batch[field]`` with fail-closed context and range validation."""
    active_split = split or infer_split()
    labels = batch.get(field)
    if labels is None:
        ctx = describe_run(cfg)
        available = sorted(str(k) for k in batch.keys())
        raise RuntimeError(
            f"{purpose}: label field {field!r} is missing from the batch. "
            f"(method={ctx['method']}, task={ctx['task']}, split={active_split}, "
            f"seed={ctx['seed']}, requested_field={field}, available_keys={available}). "
            "Enable the matching discovery source (data.topo.enabled for "
            "'phase_topo' / data.dynamics.enabled for 'phase_dynamic') and "
            "re-ingest, or point the label field at 'phase'."
        )
    if not isinstance(labels, torch.Tensor):
        ctx = describe_run(cfg)
        raise ValueError(
            f"{purpose}: label field {field!r} must be a Tensor, "
            f"got {type(labels).__name__}. "
            f"(method={ctx['method']}, task={ctx['task']}, split={active_split}, "
            f"seed={ctx['seed']}, requested_field={field})"
        )
    validate_label_values(
        labels, num_classes, cfg=cfg, field=field, purpose=purpose, split=active_split
    )
    return labels


__all__ = [
    "ALLOWED_LABEL_FIELDS",
    "LOCKED_NUM_CLASSES",
    "describe_run",
    "infer_split",
    "peek_loader_batch",
    "resolve_label_field",
    "resolve_supcon_label_field",
    "resolve_margin_label_field",
    "resolve_label_tensor",
    "validate_label_values",
]

"""Final label/data contract gates (ledger TEST-02, TEST-09).

TEST-02 validates representative batches — the real per-task state/action
dimensions from ``phaseforge/config/data/*.yaml`` with all required label
vocabularies — through both trainers' preflight gates. TEST-09 pins the
fail-closed matrix: missing, non-integer, out-of-range, non-contiguous,
and wrong-cardinality labels, plus unknown field names, all stop before
training with method/task/split/field context.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from phaseforge.trains.loops.stage1_loop import Stage1Trainer
from phaseforge.trains.loops.stage2_loop import Stage2Trainer

REPO = Path(__file__).resolve().parents[2]
TASKS = ["lift", "can", "square", "tool_hang", "transport"]


class _DictDataset(Dataset):
    def __init__(self, batches: list[dict]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> dict:
        return self.batches[idx]


class _S1Probe(torch.nn.Module):
    """Minimal Stage 1 model surface: device placement + phase-head width."""

    def __init__(self) -> None:
        super().__init__()
        self.probe = torch.nn.Parameter(torch.zeros(1))
        self.phase_head = SimpleNamespace(num_phases=6)


class _S2Probe(torch.nn.Module):
    """Minimal Stage 2 model surface: device placement + router width."""

    def __init__(self) -> None:
        super().__init__()
        self.probe = torch.nn.Parameter(torch.zeros(1))
        self.moe_layer = SimpleNamespace(router=SimpleNamespace(num_experts=6))


def _cfg(train: dict) -> DictConfig:
    return DictConfig(
        {
            "project": {"device": "cpu", "seed": 42, "method": "probe_method"},
            "train": train,
            "models": {"name": "probe_method"},
            "data": {"source": {"task_name": "ProbeTask"}},
        }
    )


def _s1_cfg(**over: object) -> DictConfig:
    base: dict = {
        "epochs": 1,
        "grad_clip_norm": 0.0,
        "log_every_n_steps": 1,
        "lambda_phase": 1.0,
        "phase_label_field": "phase_topo",
        "supcon": {
            "enabled": True,
            "lambda_sc": 1.0,
            "temperature": 0.07,
            "label_field": "phase_topo",
        },
        "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1.0e-4},
        "scheduler": {"_target_": "torch.optim.lr_scheduler.CosineAnnealingLR", "T_max": 1},
    }
    base.update(over)
    return _cfg(base)


def _s2_cfg(**over: object) -> DictConfig:
    base: dict = {
        "epochs": 1,
        "grad_clip_norm": 0.0,
        "log_every_n_steps": 1,
        "phase_label_field": "phase_topo",
        "margin": {
            "enabled": True,
            "lambda_margin": 1.0,
            "margin": 0.5,
            "label_field": "phase_topo",
        },
        "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1.0e-4},
        "scheduler": {"_target_": "torch.optim.lr_scheduler.CosineAnnealingLR", "T_max": 1},
    }
    base.update(over)
    return _cfg(base)


def _batch(state_dim: int, action_dim: int, n: int = 8, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    return {
        "state": torch.randn(n, state_dim, generator=gen),
        "action": torch.randn(n, action_dim, generator=gen),
        "phase": torch.randint(0, 6, (n,), generator=gen),
        "phase_rule": torch.randint(0, 6, (n,), generator=gen),
        "phase_topo": torch.randint(0, 6, (n,), generator=gen),
    }


def _loaders(batch: dict) -> tuple[DataLoader, DataLoader]:
    train = DataLoader(_DictDataset([batch]), batch_size=None)
    val = DataLoader(_DictDataset([batch]), batch_size=None)
    return train, val


def _check_s1(train: DataLoader, val: DataLoader, **over: object) -> None:
    Stage1Trainer(
        cfg=_s1_cfg(**over), model=_S1Probe(), train_loader=train, val_loader=val
    ).validate_label_contract()


def _check_s2(train: DataLoader, val: DataLoader, **over: object) -> None:
    Stage2Trainer(
        cfg=_s2_cfg(**over), model=_S2Probe(), train_loader=train, val_loader=val
    ).validate_label_contract()


def _assert_context(exc: Exception) -> None:
    text = str(exc)
    for token in ("probe_method", "ProbeTask", "phase_topo", "42"):
        assert token in text, f"context {token!r} missing from: {text}"
    assert "split=" in text or "split" in text


def test_representative_batches_pass_both_preflights() -> None:
    """TEST-02: real task dims + required labels validate for all tasks."""
    for task in TASKS:
        data = OmegaConf.load(str(REPO / "phaseforge" / "config" / "data" / f"{task}.yaml"))
        batch = _batch(int(data.state_dim), int(data.action_dim))
        train, val = _loaders(batch)
        _check_s1(train, val)
        _check_s2(train, val)


def test_missing_label_field_fails_closed_with_context() -> None:
    """TEST-09: absent phase_topo stops before training (both stages)."""
    batch = _batch(19, 7)
    del batch["phase_topo"]
    train, val = _loaders(batch)
    with pytest.raises(RuntimeError) as exc:
        _check_s1(train, val)
    _assert_context(exc.value)
    with pytest.raises(RuntimeError) as exc:
        _check_s2(train, val)
    _assert_context(exc.value)


def test_out_of_range_and_non_contiguous_labels_fail() -> None:
    """TEST-09: values outside [0, 6) stop (wrong-cardinality included)."""
    for bad in (
        torch.tensor([0, 1, 2, 3, 4, 6, 0, 1]),  # 7 distinct values, max 6
        torch.tensor([0, 1, 2, 3, 4, 6, 6, 6]),  # non-contiguous gap at 5
        torch.tensor([-1, 0, 1, 2, 3, 4, 5, 0]),  # negative
    ):
        batch = _batch(19, 7)
        batch["phase_topo"] = bad
        train, val = _loaders(batch)
        with pytest.raises(ValueError):
            _check_s1(train, val)
        with pytest.raises(ValueError):
            _check_s2(train, val)


def test_float_labels_fail() -> None:
    """TEST-09: malformed (non-integer) labels stop."""
    batch = _batch(19, 7)
    batch["phase_topo"] = torch.zeros(8)
    train, val = _loaders(batch)
    with pytest.raises(ValueError, match="integer"):
        _check_s1(train, val)


def test_unknown_field_name_fails() -> None:
    """TEST-09: an unregistered label field name is rejected at config time."""
    batch = _batch(19, 7)
    train, val = _loaders(batch)
    with pytest.raises(ValueError, match="unknown label field"):
        _check_s1(train, val, phase_label_field="phase_foo")
    with pytest.raises(ValueError, match="unknown label field"):
        _check_s2(train, val, margin={"enabled": True, "label_field": "phase_foo"})


def _order_states() -> list[dict]:
    return [
        {
            "state": torch.tensor([float(i)]),
            "action": torch.zeros(1),
            "phase": torch.tensor(i % 6),
            "phase_topo": torch.tensor(i % 6),
        }
        for i in range(64)
    ]


def _shuffled_loader(seed: int) -> tuple[DataLoader, torch.Generator]:
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(_DictDataset(_order_states()), batch_size=8, shuffle=True, generator=gen)
    return loader, gen


def _full_order(loader: DataLoader) -> list[float]:
    return [s for batch in loader for s in batch["state"].flatten().tolist()]


def test_preflight_preserves_sampler_order() -> None:
    """Preflight peeks must not shift the training shuffle sequence."""
    from phaseforge.trains.loops.label_contract import peek_loader_batch

    peeked_loader, peeked_gen = _shuffled_loader(1234)
    before = peeked_gen.get_state().clone()
    assert peek_loader_batch(peeked_loader) is not None
    # Generator state is bit-identical after the peek ...
    assert torch.equal(peeked_gen.get_state(), before)
    # ... so the peeked loader shuffles exactly like an untouched twin.
    fresh_loader, _ = _shuffled_loader(1234)
    assert _full_order(peeked_loader) == _full_order(fresh_loader)


def test_preflight_preserves_persistent_worker_order() -> None:
    """Preflight must not consume a multi-worker loader's prefetched data."""
    from phaseforge.trains.loops.label_contract import peek_loader_batch

    peeked_loader, _ = _shuffled_loader(5678)
    peeked_loader = DataLoader(
        peeked_loader.dataset,
        batch_size=8,
        shuffle=True,
        generator=torch.Generator().manual_seed(5678),
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
    )
    assert peek_loader_batch(peeked_loader) is not None

    fresh_loader = DataLoader(
        _DictDataset(_order_states()),
        batch_size=8,
        shuffle=True,
        generator=torch.Generator().manual_seed(5678),
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
    )
    assert _full_order(peeked_loader) == _full_order(fresh_loader)


def test_stage1_compute_loss_fails_closed_per_batch() -> None:
    """TEST-09: the per-batch path fails too (not only the preflight)."""
    from phaseforge.models.base import ModelOutput

    class _Forward(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.probe = torch.nn.Parameter(torch.zeros(1))
            self.phase_head = SimpleNamespace(num_phases=6)

        def forward(self, batch: dict) -> ModelOutput:
            action_pred = self.probe * torch.zeros_like(batch["action"])
            return ModelOutput(
                action_pred=action_pred,
                phase_logits=torch.zeros(batch["action"].size(0), 6),
                routing_weights=None,
                expert_indices=None,
                gate_logits=None,
                latent=torch.randn(batch["action"].size(0), 4),
            )

    batch = _batch(19, 7)
    del batch["phase_topo"]
    loader = DataLoader(_DictDataset([batch]), batch_size=None)
    trainer = Stage1Trainer(cfg=_s1_cfg(), model=_Forward(), train_loader=loader, val_loader=loader)
    with pytest.raises(RuntimeError) as exc:
        trainer._compute_loss(next(iter(loader)))
    _assert_context(exc.value)

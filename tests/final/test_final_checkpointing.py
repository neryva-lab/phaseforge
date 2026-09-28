"""Final checkpoint-selection gates (ledger TEST-13, TRAIN-05).

Selection is locked to minimum validation action MSE (``val/loss_action``)
on the held-out split: rollout success and eval-bank results can never
select a checkpoint. Pinned three ways: (1) no manifest row tampers with
the monitor, (2) a representative resolved subset carries the locked
values, (3) the callback unit-proves min-selection tracks only the
action loss even when a rollout-style metric disagrees.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize

from phaseforge.runner.protocol import load_protocol
from phaseforge.trains.callbacks.checkpointing import CheckpointCallback

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "experiments" / "final_causal_matrix.json"


def test_no_manifest_row_tampers_with_selection() -> None:
    """Static scan over all 50 rows: monitor/mode/early-stopping untouched."""
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for row in raw["methods"]:
        for override in row["overrides"]:
            assert not override.startswith("train.checkpoint."), (row["name"], override)
            assert not override.startswith("train.early_stopping"), (row["name"], override)
    assert "train.early_stopping.enabled=false" in raw["defaults"]


def test_resolved_selection_values_are_locked() -> None:
    """Representative subset resolves monitor/min + full-length runs."""
    protocol = load_protocol(MANIFEST)
    identities = sorted({m.name for m in protocol.methods})
    sample = [m for m in protocol.methods if m.task == "Lift"]
    sample += [
        m
        for m in protocol.methods
        if m.name == "precision_residual_phaseforge" and m.task != "Lift"
    ]
    seen = set()
    with initialize(version_base="1.3", config_path="../../phaseforge/config"):
        for method in sample:
            key = (method.name, method.task)
            if key in seen:
                continue
            seen.add(key)
            stages = ["train=stage1"] if method.stages == (1,) else ["train=stage2"]
            cfg = compose(
                config_name="main",
                overrides=[f"models={method.model}", f"data={method.data}", stages[0]]
                + [o for o in method.overrides if not o.startswith("eval.")],
            )
            assert cfg.train.checkpoint.monitor == "val/loss_action", key
            assert cfg.train.checkpoint.mode == "min", key
            assert cfg.train.early_stopping.enabled is False, key
    assert len(seen) == len(identities) + 4  # 10 Lift rows + proposed x 4 tasks


class _StubTrainer:
    """Minimal trainer surface for driving the checkpoint callback."""

    def __init__(self, tmp: Path) -> None:
        self.current_epoch = 0
        self.global_step = 0
        self.should_stop = False
        self.model = torch.nn.Linear(2, 2)
        self.model.stage = 1
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.scheduler = None
        self.callbacks: list = []


def test_callback_selects_minimum_action_mse_only() -> None:
    """Min-selection follows val/loss_action even as rollout metrics disagree."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cb = CheckpointCallback(
            output_dir=Path(tmp),
            every_n_epochs=100,
            monitor="val/loss_action",
            mode="min",
            save_top_k=1,
        )
        trainer = _StubTrainer(Path(tmp))
        # Rollout-style metric improves while the action loss worsens: the
        # best checkpoint must still be the action-MSE minimum (epoch 1).
        for epoch, (action, rollout) in enumerate(
            [(0.50, 0.10), (0.60, 0.90), (0.55, 0.95), (0.70, 0.99)], start=1
        ):
            trainer.current_epoch = epoch
            cb.on_epoch_end(trainer, {"loss_action": action, "rollout_success": rollout})
        assert cb.best_score == pytest.approx(0.50)
        assert (Path(tmp) / "checkpoint_best.pt").is_file()
        assert len(cb._topk) == 1 and cb._topk[0][1] == 1


def test_callback_uses_resolved_monitor_values() -> None:
    """The CLI wires the callback from the resolved config (mirrored here)."""
    protocol = load_protocol(MANIFEST)
    method = protocol.method_by_name("precision_residual_plain_encoder", task="Lift")
    assert method is not None
    with initialize(version_base="1.3", config_path="../../phaseforge/config"):
        cfg = compose(
            config_name="main",
            overrides=["models=precision_residual_plain_encoder", "data=lift", "train=stage2"]
            + [o for o in method.overrides if not o.startswith("eval.")],
        )
    # Mirrors phaseforge/cli.py construction exactly.
    cb = CheckpointCallback(
        output_dir=".",
        every_n_epochs=cfg.train.checkpoint.every_n_epochs,
        monitor=cfg.train.checkpoint.monitor,
        mode=cfg.train.checkpoint.mode,
        save_top_k=cfg.train.checkpoint.save_top_k,
    )
    assert (cb.monitor, cb.mode) == ("val/loss_action", "min")

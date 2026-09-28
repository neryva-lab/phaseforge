"""CPU-only tests for the Stage 2 trainer routing diagnostics.

Verifies that every epoch's validation records the balance-vs-specialization
trajectory (``val/phase_expert_nmi``, ``val/topk_balance_score``,
``val/topk_collapse_rate``, ``val/top1_balance_score``,
``val/top1_collapse_rate``, ``val/routing_entropy``) and that validation reuses
the model forward outputs (no double forward pass).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from phaseforge.models.base import ModelOutput
from phaseforge.trains.loops.stage2_loop import Stage2Trainer


class _DictDataset(Dataset):
    def __init__(
        self, num: int = 32, num_phases: int = 3, seed: int = 0, action_dim: int = 2
    ) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.states = torch.randn(num, 4, generator=gen)
        self.actions = torch.randn(num, action_dim, generator=gen)
        self.phases = torch.randint(0, num_phases, (num,), generator=gen)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "phase": self.phases[idx],
        }


class CountingMoEModel(torch.nn.Module):
    """Deterministic fake MoE: routes every sample to its GT phase."""

    def __init__(self, num_experts: int = 3) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.forward_calls = 0
        # A real parameter so the trainer's optimizer has something to own.
        self.probe = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        self.forward_calls += 1
        phase = batch["phase"]
        B = phase.size(0)
        A = batch["action"].size(-1)
        indices = phase.unsqueeze(-1)
        weights = torch.ones(B, 1)
        gate = torch.zeros(B, self.num_experts)
        gate.scatter_(1, indices, 100.0)
        return ModelOutput(
            action_pred=self.probe * torch.zeros(B, A),
            phase_logits=None,
            routing_weights=weights,
            expert_indices=indices,
            gate_logits=gate,
            aux_losses={"balance": torch.tensor(0.0)},
        )

    def freeze_encoder(self) -> None:
        pass


class NaNMoEModel(CountingMoEModel):
    """Fake MoE with a poisoned forward: NaN action predictions.

    The NaN is injected through the parameter-carrying output
    (``probe * zeros * nan``) so the loss has a grad_fn, matching real
    models: backward() runs, produces NaN gradients, and the per-step
    gradient guard aborts. (A *constant* NaN loss would instead crash
    backward() with a bare RuntimeError — the epoch-end loss flag remains
    the backstop for that pathological case.)"""

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        out = super().forward(batch)
        return ModelOutput(
            action_pred=out.action_pred * float("nan"),
            phase_logits=None,
            routing_weights=out.routing_weights,
            expert_indices=out.expert_indices,
            gate_logits=out.gate_logits,
            aux_losses=out.aux_losses,
        )


class SqrtPoisonModel(torch.nn.Module):
    """Finite forward but a NaN gradient (``sqrt`` derivative at zero).

    ``sqrt(0)`` is 0 (finite loss) but its local derivative is infinite;
    multiplying by a zero constant makes the chain-rule product
    ``0 * inf = NaN``, so the gradient guard must catch it while the loss
    guard sees a perfectly finite loss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        B = batch["action"].size(0)
        action_pred = torch.zeros(B, 2) + torch.sqrt(self.a) * 0.0
        return ModelOutput(
            action_pred=action_pred,
            phase_logits=None,
            routing_weights=None,
            expert_indices=None,
            gate_logits=None,
            aux_losses={"balance": torch.tensor(0.0)},
        )

    def freeze_encoder(self) -> None:
        pass


def _make_trainer(
    model: torch.nn.Module,
    val_loader: DataLoader,
    models_cfg: dict | None = None,
    train_cfg: dict | None = None,
) -> Stage2Trainer:
    base_train = {
        "epochs": 1,
        "grad_clip_norm": 0.0,
        "log_every_n_steps": 1,
        "freeze_encoder": False,
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": 1.0e-4,
        },
        "scheduler": {
            "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "T_max": 1,
        },
    }
    if train_cfg:
        base_train.update(train_cfg)
    cfg = DictConfig(
        {
            "project": {"device": "cpu"},
            "train": base_train,
            **({"models": models_cfg} if models_cfg else {}),
        }
    )
    return Stage2Trainer(cfg=cfg, model=model, train_loader=val_loader, val_loader=val_loader)


def test_validate_reports_routing_diagnostics() -> None:
    model = CountingMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(seed=1), batch_size=8))

    val_metrics = trainer._validate()

    # Perfect GT routing => perfect phase-expert alignment (NMI ~ 1.0) and
    # near-uniform usage => high balance score.
    assert "val/phase_expert_nmi" in val_metrics
    assert val_metrics["val/phase_expert_nmi"] > 0.99
    assert "val/topk_balance_score" in val_metrics
    assert val_metrics["val/topk_balance_score"] > 0.9
    assert "val/top1_balance_score" in val_metrics
    assert "val/topk_collapse_rate" in val_metrics
    assert "val/top1_collapse_rate" in val_metrics
    assert "val/routing_entropy" in val_metrics
    # Loss metrics still reported.
    assert "loss_total" in val_metrics
    assert "loss_action" in val_metrics


def test_validate_single_forward_pass() -> None:
    model = CountingMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(num=16, seed=2), batch_size=4))

    trainer._validate()

    # 16 samples / batch 4 = 4 batches; the diagnostics must reuse the same
    # forward outputs instead of running a second pass.
    assert model.forward_calls == 4


def test_compute_loss_accepts_reused_output() -> None:
    model = CountingMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(num=8, seed=3), batch_size=4))

    batch = DataLoader(_DictDataset(num=1, seed=4), batch_size=1)
    first = next(iter(batch))
    out = model(first)

    calls_before = model.forward_calls
    total_loss, metrics = trainer._compute_loss(first, out=out)
    assert model.forward_calls == calls_before
    assert total_loss.requires_grad
    assert set(metrics) == {
        "loss_total",
        "loss_action",
        "loss_balance",
        "loss_sticky",
        "loss_teacher_kl",
        "teacher_lambda",
    }


def test_expert_count_uses_configured_count_not_max_index() -> None:
    # Phases restricted to {0, 1} while the model has 3 experts: expert 2
    # never fires. The collapse rate must count it as dead (1/3) — deriving
    # the expert count from the largest observed index would report 0.5 and
    # hide the dead expert.
    model = CountingMoEModel(num_experts=3)
    dataset = _DictDataset(num=32, seed=6)
    dataset.phases.clamp_(max=1)
    trainer = _make_trainer(model, DataLoader(dataset, batch_size=8))

    val_metrics = trainer._validate()

    assert val_metrics["val/topk_collapse_rate"] == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_fit_with_epoch_progressbar_completes() -> None:
    """The epoch-level tqdm progressbar path runs a full fit() cleanly.

    The progressbar wraps the epoch iterator, formats the postfix from
    validation metrics, and must be closed even on the normal path.
    """
    model = CountingMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(num=16, seed=7), batch_size=8))
    trainer.train_cfg.epoch_progressbar = True

    trainer.fit()

    assert trainer.current_epoch == 1
    assert model.forward_calls > 0


def test_validate_losses_are_sample_weighted() -> None:
    # 7 samples in batches of 4 + 3: the short final batch must not weigh
    # as much as a full one. The reported loss is the mean over ALL samples.
    model = CountingMoEModel(num_experts=3)
    dataset = _DictDataset(num=7, seed=5)
    trainer = _make_trainer(model, DataLoader(dataset, batch_size=4))

    val_metrics = trainer._validate()

    # action_pred == 0 => per-sample MSE == action^2.
    expected = float((dataset.actions**2).mean())
    assert val_metrics["loss_action"] == pytest.approx(expected, rel=1e-6)
    assert val_metrics["loss_total"] == pytest.approx(expected, rel=1e-6)


def test_non_finite_loss_fails_fast() -> None:
    # With clipping disabled the per-step fused gradient guard is the first
    # abort surface for a NaN loss (NaN forward -> NaN backward -> guard fires
    # before optimizer.step()); the epoch-end on-device loss flag is the
    # backstop for non-finite losses that never reach a gradient (e.g.
    # detached constants). Either way the run aborts with a non-finite error
    # before any epoch-end artifact is written.
    model = NaNMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(seed=8), batch_size=8))

    with pytest.raises(FloatingPointError, match="Non-finite"):
        trainer.fit()


def test_non_finite_gradient_fails_fast() -> None:
    # The loss is finite (sqrt(0) = 0) but the gradient w.r.t. `a` is NaN
    # (0 * inf through the chain rule): only the gradient guard can catch it.
    model = SqrtPoisonModel()
    trainer = _make_trainer(model, DataLoader(_DictDataset(seed=9), batch_size=8))

    with pytest.raises(FloatingPointError, match="Non-finite gradient"):
        trainer.fit()


class StickyMoEModel(CountingMoEModel):
    """CountingMoEModel that also emits a fixed raw stickiness loss."""

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        out = super().forward(batch)
        return ModelOutput(
            action_pred=out.action_pred,
            phase_logits=None,
            routing_weights=out.routing_weights,
            expert_indices=out.expert_indices,
            gate_logits=out.gate_logits,
            aux_losses={"balance": torch.tensor(0.0), "sticky": torch.tensor(2.0)},
        )


def test_sticky_loss_scaled_by_train_coeff() -> None:
    model = StickyMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(num=4, seed=10), batch_size=4))
    trainer.train_cfg.sticky_coeff = 0.5

    batch = DataLoader(_DictDataset(num=1, seed=11), batch_size=1)
    first = next(iter(batch))
    out = model(first)
    _, metrics = trainer._compute_loss(first, out=out)

    assert metrics["loss_sticky"] == pytest.approx(1.0)  # 0.5 * 2.0
    # default coeff 0.0 keeps the total identical to the plain loss.
    trainer2 = _make_trainer(model, DataLoader(_DictDataset(num=4, seed=12), batch_size=4))
    total_default, metrics_default = trainer2._compute_loss(first, out=out)
    assert metrics_default["loss_sticky"].item() == pytest.approx(0.0)
    assert total_default.item() == pytest.approx(
        metrics_default["loss_action"].item(), rel=1e-6
    )


class TrajDictDataset(Dataset):
    """Fixed samples with explicit trajectory keys for switch-rate tests.

    traj 0: phases [0, 0]  -> adjacent pair with NO switch.
    traj 1: phases [1, 2]  -> adjacent pair WITH a switch.
    """

    def __init__(self) -> None:
        self.samples = [
            {"state": torch.randn(4), "action": torch.randn(2), "phase": torch.tensor(0),
             "trajectory_id": torch.tensor(0), "trajectory_position": torch.tensor(0)},
            {"state": torch.randn(4), "action": torch.randn(2), "phase": torch.tensor(0),
             "trajectory_id": torch.tensor(0), "trajectory_position": torch.tensor(1)},
            {"state": torch.randn(4), "action": torch.randn(2), "phase": torch.tensor(1),
             "trajectory_id": torch.tensor(1), "trajectory_position": torch.tensor(0)},
            {"state": torch.randn(4), "action": torch.randn(2), "phase": torch.tensor(2),
             "trajectory_id": torch.tensor(1), "trajectory_position": torch.tensor(1)},
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def test_validate_reports_switch_rate() -> None:
    # CountingMoEModel routes to the GT phase, so the switch rate mirrors the
    # phase sequence: traj0 (0->0) keeps the expert, traj1 (1->2) flips it.
    model = CountingMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(TrajDictDataset(), batch_size=4))

    val_metrics = trainer._validate()

    assert val_metrics["val/routing_switch_rate"] == pytest.approx(0.5)
    assert val_metrics["val/routing_switch_rate"] != 0.0


class TeacherKLMoEModel(CountingMoEModel):
    """CountingMoEModel that also emits phase_logits and exposes M."""

    def __init__(self, num_experts: int = 3) -> None:
        super().__init__(num_experts)
        self.mapping = torch.eye(num_experts)

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        out = super().forward(batch)
        return ModelOutput(
            action_pred=out.action_pred,
            phase_logits=torch.randn(out.action_pred.size(0), self.num_experts),
            routing_weights=out.routing_weights,
            expert_indices=out.expert_indices,
            gate_logits=out.gate_logits,
            aux_losses=out.aux_losses,
        )

    def require_soft_mapping(self) -> torch.Tensor:
        return self.mapping


class PhaseLogitsNoMappingModel(CountingMoEModel):
    """Emits phase_logits but has no require_soft_mapping accessor."""

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        out = super().forward(batch)
        return ModelOutput(
            action_pred=out.action_pred,
            phase_logits=torch.randn(out.action_pred.size(0), self.num_experts),
            routing_weights=out.routing_weights,
            expert_indices=out.expert_indices,
            gate_logits=out.gate_logits,
            aux_losses=out.aux_losses,
        )


def test_teacher_kl_loss_matches_manual() -> None:
    torch.manual_seed(3)
    model = TeacherKLMoEModel(num_experts=3)
    trainer = _make_trainer(
        model,
        DataLoader(_DictDataset(num=4, seed=14), batch_size=4),
        models_cfg={"teacher_routing": {"enabled": True}},
    )
    trainer.train_cfg.teacher_routing = {"lambda0": 0.5}

    batch = DataLoader(_DictDataset(num=2, seed=15), batch_size=2)
    first = next(iter(batch))
    out = model(first)
    total, metrics = trainer._compute_loss(first, out=out)

    phase_probs = F.softmax(out.phase_logits, dim=-1)
    target = torch.einsum("pe,bp->be", model.mapping, phase_probs)
    expected_kl = F.kl_div(
        F.log_softmax(out.gate_logits, dim=-1), target, reduction="batchmean"
    )
    assert metrics["teacher_lambda"] == pytest.approx(0.5)
    assert metrics["loss_teacher_kl"] == pytest.approx(0.5 * expected_kl, rel=1e-6)
    assert total.item() == pytest.approx(
        metrics["loss_action"].item() + metrics["loss_teacher_kl"].item(), rel=1e-6
    )


def test_teacher_kl_schedule_anneals() -> None:
    model = TeacherKLMoEModel(num_experts=3)
    trainer = _make_trainer(
        model,
        DataLoader(_DictDataset(num=4, seed=16), batch_size=4),
        models_cfg={"teacher_routing": {"enabled": True}},
    )
    trainer.train_cfg.teacher_routing = {"lambda0": 1.0}
    trainer.train_cfg.epochs = 10
    # First half: flat at lambda0.
    trainer.current_epoch = 0
    assert trainer._teacher_kl_coeff() == pytest.approx(1.0)
    trainer.current_epoch = 4
    assert trainer._teacher_kl_coeff() == pytest.approx(1.0)
    trainer.current_epoch = 5
    assert trainer._teacher_kl_coeff() == pytest.approx(1.0)
    # Second half: linear anneal to 0.
    trainer.current_epoch = 6
    assert trainer._teacher_kl_coeff() == pytest.approx(0.8)
    trainer.current_epoch = 8
    assert trainer._teacher_kl_coeff() == pytest.approx(0.4)
    trainer.current_epoch = 10
    assert trainer._teacher_kl_coeff() == pytest.approx(0.0)


def test_teacher_kl_inactive_without_models_block() -> None:
    # phase_logits present but no models.teacher_routing block -> no teacher KL,
    # no crash.
    model = TeacherKLMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(num=2, seed=17), batch_size=2))
    batch = DataLoader(_DictDataset(num=1, seed=18), batch_size=1)
    first = next(iter(batch))
    out = model(first)
    total, metrics = trainer._compute_loss(first, out=out)
    assert metrics["loss_teacher_kl"].item() == pytest.approx(0.0)
    assert metrics["teacher_lambda"] == pytest.approx(0.0)
    assert total.item() == pytest.approx(metrics["loss_action"].item(), rel=1e-6)


def test_teacher_kl_fails_closed_without_mapping() -> None:
    # enabled=true + phase_logits but no require_soft_mapping -> RuntimeError.
    model = PhaseLogitsNoMappingModel(num_experts=3)
    trainer = _make_trainer(
        model,
        DataLoader(_DictDataset(num=2, seed=19), batch_size=2),
        models_cfg={"teacher_routing": {"enabled": True}},
    )
    batch = DataLoader(_DictDataset(num=1, seed=20), batch_size=1)
    first = next(iter(batch))
    out = model(first)
    with pytest.raises(RuntimeError, match="require_soft_mapping"):
        trainer._compute_loss(first, out=out)


class _EpochEndRecorder(torch.nn.Module):
    """No-op probe model + callback pair recording epoch-end invocations."""


class RecordingCallback:
    def __init__(self) -> None:
        self.epoch_end_epochs: list[int] = []

    def on_epoch_end(self, trainer, val_metrics) -> None:
        self.epoch_end_epochs.append(int(trainer.current_epoch))


def test_non_finite_loss_aborts_before_epoch_end_artifacts() -> None:
    """The loss-finiteness guard fires at the epoch boundary — after the
    epoch trained, but BEFORE validation persistence / the checkpoint
    callback, so a corrupted epoch produces no artifacts."""
    model = NaNMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(seed=21), batch_size=8))
    recorder = RecordingCallback()
    trainer.add_callback(recorder)

    with pytest.raises(FloatingPointError, match="Non-finite"):
        trainer.fit()

    assert recorder.epoch_end_epochs == [], "on_epoch_end ran despite a non-finite loss epoch"


def test_epoch_loss_finite_backstop_check() -> None:
    """The epoch-end backstop check itself: a red flag aborts before any
    artifact, a green flag passes, and no flag (empty epoch) is a no-op."""
    model = CountingMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(num=8, seed=23), batch_size=8))

    trainer._epoch_loss_finite = torch.tensor(False)
    with pytest.raises(FloatingPointError, match="Non-finite training loss"):
        trainer._check_epoch_loss_finite()

    trainer._epoch_loss_finite = torch.tensor(True)
    trainer._check_epoch_loss_finite()  # green: no raise

    trainer._epoch_loss_finite = None
    trainer._check_epoch_loss_finite()  # empty: no raise


def test_finite_epoch_sets_green_loss_flag() -> None:
    """A clean epoch leaves the on-device finite flag True (no abort)."""
    model = CountingMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(num=16, seed=22), batch_size=8))

    trainer.fit()

    assert trainer._epoch_loss_finite is not None
    assert bool(trainer._epoch_loss_finite.item())


def test_trainer_zero_scalar_cache_is_stable_constant() -> None:
    """The stage-2 aux-loss zero default is a cached, unmutated 0.0."""
    from phaseforge.trains.loops.stage2_loop import _zero_scalar

    zero = _zero_scalar(torch.device("cpu"))
    assert zero.item() == 0.0
    assert zero.dtype == torch.float32
    assert _zero_scalar(torch.device("cpu")) is zero


def test_validate_aggregation_exact_python_float_math() -> None:
    """The on-device float64 accumulation must reproduce the former
    per-batch Python-float (float64) arithmetic EXACTLY — same operation
    order, same rounding — including a short final batch (sample-weighted
    means). Exact equality, not approx: the guard exists to prove the
    aggregation refactor is numerically invisible."""
    model = CountingMoEModel(num_experts=3)
    dataset = _DictDataset(num=7, seed=24)
    loader = DataLoader(dataset, batch_size=4)
    trainer = _make_trainer(model, loader)

    agg: dict[str, float] = {}
    total = 0
    for batch in loader:
        out = model(batch)
        _, metrics = trainer._compute_loss(batch, out=out)
        n = batch["action"].shape[0]
        for k, v in metrics.items():
            f = float(v.detach().cpu().item()) if isinstance(v, torch.Tensor) else float(v)
            agg[k] = agg.get(k, 0.0) + f * n
        total += n
    expected = {k: v / total for k, v in agg.items()}

    val_metrics = trainer._validate()
    for k, v in expected.items():
        assert k in val_metrics, f"metric {k} missing from validation output"
        assert val_metrics[k] == v, f"{k}: {val_metrics[k]!r} != {v!r}"


def test_dim_weights_scales_action_loss() -> None:
    """Verifies that train.dim_weights correctly scales the per-dimension action MSE."""
    model = CountingMoEModel(num_experts=3)
    dataset = _DictDataset(num=8, seed=50, action_dim=2)
    loader = DataLoader(dataset, batch_size=8)
    weights = [1.0, 4.0]
    trainer = _make_trainer(model, loader, train_cfg={"dim_weights": weights})

    batch = next(iter(loader))
    out = model(batch)
    _, metrics = trainer._compute_loss(batch, out=out)

    sq_diff = (out.action_pred - batch["action"]) ** 2
    expected_loss = (sq_diff * torch.tensor(weights, dtype=torch.float32)).mean()
    assert torch.isclose(metrics["loss_action"], expected_loss, atol=1e-6)


def test_release_loss_active_on_gripper_opening() -> None:
    """Verifies that train.release_loss penalizes lateral velocity only during place with gripper
    opening."""
    model = CountingMoEModel(num_experts=5)
    dataset = _DictDataset(num=4, seed=60, action_dim=7)
    # Set phase=4 (place phase), gripper < 0 (releasing in Robosuite convention)
    dataset.phases = torch.tensor([4, 4, 1, 4])
    dataset.actions[:, 6] = torch.tensor([-1.0, -0.5, -1.0, 1.0])
    loader = DataLoader(dataset, batch_size=4)

    rel_cfg = {
        "release_loss": {
            "enabled": True,
            "lambda_rel": 0.5,
            "gripper_threshold": 0.0,
            "place_phase": 4,
        }
    }
    trainer = _make_trainer(model, loader, train_cfg=rel_cfg)
    batch = next(iter(loader))
    out = model(batch)
    # Set non-zero predicted actions on lateral components
    out.action_pred = torch.tensor([
        [0.2, 0.3, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.1, 0.2, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0],  # phase 1, should be ignored
        [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0],   # gripper >= 0, should be ignored
    ])

    _, metrics = trainer._compute_loss(batch, out=out)
    assert "loss_release" in metrics
    # Samples 0 and 1:
    # Sample 0: 0.2^2 + 0.3^2 = 0.04 + 0.09 = 0.13
    # Sample 1: 0.1^2 + 0.2^2 = 0.01 + 0.04 = 0.05
    # Mean: (0.13 + 0.05) / 2 = 0.09
    # Scaled by lambda_rel (0.5) = 0.045
    expected_rel = 0.5 * 0.09
    assert torch.isclose(metrics["loss_release"], torch.tensor(expected_rel), atol=1e-5)


def test_log_dimension_mse_subspace_metrics() -> None:
    """Verifies that train.log_dimension_mse logs fine-grained subspace action MSEs."""
    model = CountingMoEModel(num_experts=5)
    dataset = _DictDataset(num=4, seed=70, action_dim=7)
    dataset.phases = torch.tensor([4, 4, 2, 3])
    loader = DataLoader(dataset, batch_size=4)
    trainer = _make_trainer(model, loader, train_cfg={"log_dimension_mse": True})

    batch = next(iter(loader))
    out = model(batch)
    _, metrics = trainer._compute_loss(batch, out=out)

    assert "loss_action_pos" in metrics
    assert "loss_action_rot" in metrics
    assert "loss_action_grip" in metrics
    assert "loss_action_place" in metrics


def test_dim_weights_auto_tile_dual_arm() -> None:
    """Verifies that 7-dim dim_weights auto-tiles to 14 dims for dual-arm Transport without
    error."""
    model = CountingMoEModel(num_experts=3)
    dataset = _DictDataset(num=4, seed=80, action_dim=14)
    loader = DataLoader(dataset, batch_size=4)

    # 7-element dimension weights (e.g. z-dim and gripper 2x)
    dim_weights_7d = [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0]
    trainer = _make_trainer(model, loader, train_cfg={"dim_weights": dim_weights_7d})

    batch = next(iter(loader))
    out = model(batch)
    out.action_pred = torch.zeros_like(batch["action"])

    # Compute expected loss with tiled weights
    tiled_weights = torch.tensor(dim_weights_7d * 2, dtype=torch.float32)
    expected_loss = ((batch["action"] ** 2) * tiled_weights).mean()

    total_loss, metrics = trainer._compute_loss(batch, out=out)
    assert "loss_action" in metrics
    assert torch.isclose(metrics["loss_action"], expected_loss, atol=1e-5)


def test_dual_arm_release_loss() -> None:
    """Verifies that release_loss evaluates both arm 0 (dim 6) and arm 1 (dim 13) independently for
    dual-arm."""
    model = CountingMoEModel(num_experts=5)
    dataset = _DictDataset(num=2, seed=90, action_dim=14)
    dataset.phases = torch.tensor([4, 4])
    # Sample 0: arm 0 releasing (-1.0), arm 1 holding (1.0)
    # Sample 1: arm 0 holding (1.0), arm 1 releasing (-1.0)
    dataset.actions[0, 6] = -1.0
    dataset.actions[0, 13] = 1.0
    dataset.actions[1, 6] = 1.0
    dataset.actions[1, 13] = -1.0
    loader = DataLoader(dataset, batch_size=2)

    rel_cfg = {
        "release_loss": {
            "enabled": True,
            "lambda_rel": 1.0,
            "gripper_threshold": 0.0,
            "place_phase": 4,
        }
    }
    trainer = _make_trainer(model, loader, train_cfg=rel_cfg)
    batch = next(iter(loader))
    out = model(batch)
    # Set lateral commands: arm 0 lateral is 0:2, arm 1 lateral is 7:9
    pred = torch.zeros(2, 14)
    pred[0, 0:2] = torch.tensor([0.3, 0.4])  # arm 0 lateral (norm^2 = 0.09 + 0.16 = 0.25)
    pred[1, 7:9] = torch.tensor([0.6, 0.8])  # arm 1 lateral (norm^2 = 0.36 + 0.64 = 1.00)
    out.action_pred = pred

    _, metrics = trainer._compute_loss(batch, out=out)
    assert "loss_release" in metrics
    # arm 0 loss = 0.25 (sample 0 only), arm 1 loss = 1.00 (sample 1 only)
    # sum = 1.25 * lambda_rel (1.0) = 1.25
    expected_rel = 1.25
    assert torch.isclose(metrics["loss_release"], torch.tensor(expected_rel), atol=1e-5)



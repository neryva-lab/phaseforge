"""Unit tests for PhaseForge 2.0 dynamic discovery package."""

import numpy as np
import pytest
import torch
from phaseforge.data.dynamics.artifacts import (
    load_discovery_artifact,
    save_discovery_artifact,
)
from phaseforge.data.dynamics.diagnostics import evaluate_discovery_quality
from phaseforge.data.dynamics.features import (
    extract_dataset_transitions,
    extract_trajectory_transitions,
)
from phaseforge.data.dynamics.switching_linear import (
    SingleDynamicsModel,
    StickySLDS,
)


def _generate_synthetic_trajectories(
    num_trajs: int = 10,
    traj_len: int = 50,
    state_dim: int = 6,
    action_dim: int = 2,
    num_regimes: int = 3,
    seed: int = 42,
) -> list[dict[str, torch.Tensor]]:
    """Generate synthetic piecewise-linear switching trajectories."""
    rng = np.random.default_rng(seed)
    trajectories = []

    A_true = [
        rng.normal(0, 0.2, (state_dim, state_dim)) + np.eye(state_dim) * 0.9
        for _ in range(num_regimes)
    ]
    B_true = [rng.normal(0, 0.5, (state_dim, action_dim)) for _ in range(num_regimes)]

    for traj_idx in range(num_trajs):
        s = np.zeros((traj_len, state_dim), dtype=np.float32)
        a = rng.normal(0, 1.0, (traj_len, action_dim)).astype(np.float32)
        p = np.zeros(traj_len, dtype=np.int64)

        s[0] = rng.normal(0, 0.5, state_dim)
        regime = traj_idx % num_regimes

        for t in range(traj_len - 1):
            # Switch regime with small probability
            if rng.uniform() < 0.1:
                regime = (regime + 1) % num_regimes
            p[t] = regime
            s[t + 1] = (
                s[t] @ A_true[regime].T + a[t] @ B_true[regime].T + rng.normal(0, 0.05, state_dim)
            )
        p[-1] = regime

        trajectories.append(
            {
                "state": torch.from_numpy(s),
                "action": torch.from_numpy(a),
                "phase": torch.from_numpy(p),
                "task_id": 0,
            }
        )

    return trajectories


def test_features_construction():
    trajs = _generate_synthetic_trajectories(num_trajs=3, traj_len=20, state_dim=4, action_dim=2)
    tb_single = extract_trajectory_transitions(trajs[0], traj_idx=0)
    assert tb_single.num_samples == 19
    assert tb_single.state_dim == 4
    assert tb_single.action_dim == 2
    assert tb_single.feature_dim == 4 + 2 + 4

    tb_all = extract_dataset_transitions(trajs)
    assert tb_all.num_samples == 3 * 19
    assert tb_all.trajectory_indices.shape == (57,)
    assert tb_all.timesteps.shape == (57,)
    assert tb_all.phase_semantic is not None


def test_sticky_slds_fit_and_decode():
    train_trajs = _generate_synthetic_trajectories(
        num_trajs=8, traj_len=40, state_dim=4, action_dim=2, num_regimes=3, seed=42
    )
    val_trajs = _generate_synthetic_trajectories(
        num_trajs=3, traj_len=40, state_dim=4, action_dim=2, num_regimes=3, seed=99
    )

    slds = StickySLDS(num_regimes=3, sticky_kappa=30.0, max_em_iter=15, min_duration=2, seed=42)
    slds.fit(train_trajs)

    assert slds.params is not None
    assert slds.params.A.shape == (3, 4, 4)
    assert slds.params.B.shape == (3, 4, 2)
    assert slds.params.transition_matrix.shape == (3, 3)

    # Check Viterbi decoding
    labels = slds.decode_trajectory(train_trajs[0])
    assert len(labels) == 40
    assert np.all((labels >= 0) & (labels < 3))

    # Check out-of-sample scoring
    val_ll = slds.score_trajectory(val_trajs[0])
    assert np.isfinite(val_ll)

    # Test single dynamics baseline
    single = SingleDynamicsModel().fit(train_trajs)
    single_ll = single.score_trajectory(val_trajs[0])
    assert np.isfinite(single_ll)


def test_diagnostics_and_quality_checks():
    train_trajs = _generate_synthetic_trajectories(
        num_trajs=10, traj_len=50, state_dim=4, action_dim=2, num_regimes=3, seed=42
    )
    val_trajs = _generate_synthetic_trajectories(
        num_trajs=4, traj_len=50, state_dim=4, action_dim=2, num_regimes=3, seed=99
    )

    slds = StickySLDS(num_regimes=3, sticky_kappa=20.0, max_em_iter=10, seed=42)
    slds.fit(train_trajs)

    report = evaluate_discovery_quality(slds, train_trajs, val_trajs, min_occupancy_threshold=0.01)
    assert report.total_train_trajs == 10
    assert report.total_train_steps == 500
    assert len(report.occupancy) == 3
    assert np.isfinite(report.held_out_nll_slds)
    assert np.isfinite(report.held_out_nll_single_dynamics)


def test_quality_gate_thresholds_are_enforced(monkeypatch):
    train_trajs = _generate_synthetic_trajectories(
        num_trajs=6, traj_len=30, state_dim=4, action_dim=2, num_regimes=3, seed=42
    )
    val_trajs = _generate_synthetic_trajectories(
        num_trajs=2, traj_len=30, state_dim=4, action_dim=2, num_regimes=3, seed=99
    )
    slds = StickySLDS(num_regimes=3, max_em_iter=3, seed=42).fit(train_trajs)
    monkeypatch.setattr(
        slds,
        "decode_trajectory",
        lambda trajectory: np.zeros(len(trajectory["state"]), dtype=np.int64),
    )

    report = evaluate_discovery_quality(
        slds,
        train_trajs,
        val_trajs,
        min_occupancy_threshold=0.0,
        max_single_regime_fraction=0.5,
        max_switch_rate=1.0,
        min_nll_improvement=-np.inf,
    )
    assert report.passed_all is False
    assert any("Over-persistence" in reason for reason in report.failure_reasons)


def test_artifact_serialization_roundtrip(tmp_path):
    train_trajs = _generate_synthetic_trajectories(
        num_trajs=5, traj_len=30, state_dim=4, action_dim=2, num_regimes=3, seed=42
    )
    val_trajs = _generate_synthetic_trajectories(
        num_trajs=2, traj_len=30, state_dim=4, action_dim=2, num_regimes=3, seed=99
    )

    slds = StickySLDS(num_regimes=3, sticky_kappa=20.0, max_em_iter=5, seed=42).fit(train_trajs)
    report = evaluate_discovery_quality(slds, train_trajs, val_trajs)

    train_labels = [slds.decode_trajectory(t) for t in train_trajs]
    val_labels = [slds.decode_trajectory(t) for t in val_trajs]

    art_dir = save_discovery_artifact(
        output_dir=tmp_path / "discovery_k3",
        slds=slds,
        report=report,
        task_name="test_task",
        data_config_hash="abc123hash",
        train_labels=train_labels,
        val_labels=val_labels,
    )

    loaded_slds, loaded_labels, meta = load_discovery_artifact(art_dir)
    assert meta["num_regimes"] == 3
    assert meta["task_name"] == "test_task"
    assert loaded_slds.params is not None
    assert np.allclose(loaded_slds.params.transition_matrix, slds.params.transition_matrix)
    assert len(loaded_labels["train"]) == 5
    assert len(loaded_labels["val"]) == 2


def test_artifact_checksum_failure_is_rejected(tmp_path):
    train_trajs = _generate_synthetic_trajectories(
        num_trajs=5, traj_len=30, state_dim=4, action_dim=2, num_regimes=3, seed=42
    )
    val_trajs = _generate_synthetic_trajectories(
        num_trajs=2, traj_len=30, state_dim=4, action_dim=2, num_regimes=3, seed=99
    )
    slds = StickySLDS(num_regimes=3, max_em_iter=2, seed=42).fit(train_trajs)
    report = evaluate_discovery_quality(slds, train_trajs, val_trajs, min_occupancy_threshold=0.0)
    artifact_dir = save_discovery_artifact(
        output_dir=tmp_path / "discovery_k3",
        slds=slds,
        report=report,
        task_name="test_task",
        data_config_hash="abc123hash",
        train_labels=[slds.decode_trajectory(t) for t in train_trajs],
        val_labels=[slds.decode_trajectory(t) for t in val_trajs],
    )

    with (artifact_dir / "model_parameters.pt").open("ab") as file:
        file.write(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_discovery_artifact(artifact_dir)

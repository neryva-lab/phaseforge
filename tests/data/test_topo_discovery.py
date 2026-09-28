"""CPU-only tests for Phase 2 topological discovery."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from torch.utils.data import DataLoader

from phaseforge.data.topo.artifacts import (
    TOPO_ARTIFACT_VERSION,
    load_topo_artifact,
    save_topo_artifact,
)
from phaseforge.data.topo.cluster import (
    cluster_segments,
    k_sweep_candidates,
    segment_features,
    select_K,
)
from phaseforge.data.topo.observability import audit_regimes
from phaseforge.data.topo.pelt import run_pelt
from phaseforge.data.topo.task_vars import TASK_VAR_ORDER, concat_task_matrix, extract_task_vars

CAN_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
CAN_DIMS = [3, 4, 2, 14]


def _can_state(traj_len: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1.0, (traj_len, 23)).astype(np.float64)


# ------------------------------------------------------------------
# task_vars
# ------------------------------------------------------------------


def test_task_vars_can_layout_shapes() -> None:
    out = extract_task_vars(_can_state(), CAN_KEYS, CAN_DIMS)
    assert out["eef_pos"].shape == (60, 3)
    assert out["eef_quat"].shape == (60, 4)
    assert out["gripper"].shape == (60, 2)
    assert out["object"].shape == (60, 14)
    assert out["rel_ee_obj"].shape == (60, 3)
    assert out["gripper_aperture"].shape == (60, 1)
    matrix = concat_task_matrix(out)
    assert matrix.shape == (60, 3 + 4 + 2 + 14 + 3 + 1)


def test_task_vars_relative_vector_is_documented_proxy() -> None:
    state = _can_state(traj_len=4, seed=1)
    out = extract_task_vars(state, CAN_KEYS, CAN_DIMS)
    np.testing.assert_allclose(out["rel_ee_obj"], state[:, 0:3] - state[:, 9:12])
    np.testing.assert_allclose(
        out["gripper_aperture"][:, 0], np.max(np.abs(state[:, 7:9]), axis=1)
    )
    assert set(TASK_VAR_ORDER) <= set(out)


def test_task_vars_rejects_bad_layout() -> None:
    state = _can_state()
    with pytest.raises(ValueError):
        extract_task_vars(state, ["robot0_eef_pos"], [3])
    with pytest.raises(ValueError):
        extract_task_vars(state, CAN_KEYS, [3, 4, 2, 5])


# ------------------------------------------------------------------
# PELT
# ------------------------------------------------------------------


def _two_regime_signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    first = rng.normal(0, 0.2, (40, 4))
    second = rng.normal(5.0, 0.2, (40, 4))
    return np.concatenate([first, second], axis=0)


def test_pelt_finds_known_changepoint() -> None:
    bounds = run_pelt(_two_regime_signal(), penalty_beta=10.0, min_segment_len=5)
    assert bounds[0] == 0 and bounds[-1] == 80
    assert any(abs(int(b) - 40) <= 2 for b in bounds[1:-1])


def test_pelt_enforces_min_segment_length() -> None:
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 1.0, (50, 2))
    bounds = run_pelt(signal, penalty_beta=1000.0, min_segment_len=8)
    lengths = np.diff(bounds)
    assert bounds.tolist() == [0, 50]
    assert (lengths >= 8).all()


def test_pelt_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        run_pelt(_two_regime_signal(), penalty_beta=10.0, cost="rbf")
    with pytest.raises(ValueError):
        run_pelt(np.zeros((0, 2)), penalty_beta=10.0)
    with pytest.raises(ValueError):
        run_pelt(np.zeros((10, 2)) + np.nan, penalty_beta=10.0)


# ------------------------------------------------------------------
# clustering + K selection
# ------------------------------------------------------------------


def test_segment_features_shapes() -> None:
    segments = [_two_regime_signal()[:40], _two_regime_signal()[40:]]
    feats = segment_features(segments)
    assert feats.shape == (2, 8)
    actions = [np.zeros((40, 2)), np.ones((40, 2))]
    feats_act = segment_features(segments, include_action_stats=True, actions=actions)
    assert feats_act.shape == (2, 12)
    with pytest.raises(ValueError):
        segment_features(segments, include_action_stats=True, actions=None)


def test_cluster_segments_methods() -> None:
    rng = np.random.default_rng(0)
    feats = np.concatenate(
        [rng.normal(0, 0.1, (10, 6)), rng.normal(5.0, 0.1, (10, 6))], axis=0
    )
    for method in ("kmeans", "agglomerative", "spherical_kmeans"):
        labels = cluster_segments(feats, num_clusters=2, method=method, seed=0)
        assert labels.shape == (20,)
        assert set(np.unique(labels)) == {0, 1}
    with pytest.raises(ValueError):
        cluster_segments(feats, num_clusters=2, method="dbscan")
    with pytest.raises(ValueError):
        cluster_segments(feats, num_clusters=25, method="kmeans")


def test_select_K_argmax_and_tie_break() -> None:
    scores = {
        3: {"observability": 0.5, "action_explanation": 0.2, "stability": 0.1, "complexity": 0.1},
        4: {"observability": 0.9, "action_explanation": 0.2, "stability": 0.1, "complexity": 0.1},
    }
    assert select_K(scores) == 4
    tied = {
        3: {"observability": 0.5, "action_explanation": 0.0, "stability": 0.0, "complexity": 0.0},
        4: {"observability": 0.5, "action_explanation": 0.0, "stability": 0.0, "complexity": 0.0},
    }
    assert select_K(tied) == 3
    assert k_sweep_candidates(3, 5) == [3, 4, 5]
    with pytest.raises(ValueError):
        select_K({})


# ------------------------------------------------------------------
# observability audit
# ------------------------------------------------------------------


def _separable_states(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    states = np.concatenate(
        [rng.normal(0, 0.3, (n, 4)), rng.normal(4.0, 0.3, (n, 4))], axis=0
    )
    labels = np.array([0] * n + [1] * n, dtype=np.int64)
    traj_ids = np.array([i // 40 for i in range(2 * n)], dtype=np.int64)
    return states, labels, traj_ids


def test_audit_passes_separable_regimes() -> None:
    states, labels, traj_ids = _separable_states()
    report = audit_regimes(states, labels, traj_ids, 2)
    assert report.passed
    assert report.macro_f1 > 0.9
    assert report.merge_candidates == []
    payload = report.to_dict()
    assert payload["num_regimes"] == 2


def test_audit_fails_aliased_regimes_with_merge_hint() -> None:
    rng = np.random.default_rng(0)
    states = rng.normal(0, 0.3, (120, 4))
    labels = np.array([0] * 60 + [1] * 60, dtype=np.int64)
    traj_ids = np.array([i // 40 for i in range(120)], dtype=np.int64)
    report = audit_regimes(states, labels, traj_ids, 2)
    assert not report.passed
    assert report.merge_candidates == [[0, 1]]
    assert any("Aliased" in reason for reason in report.failure_reasons)


def test_audit_flags_dead_regime() -> None:
    states, labels, traj_ids = _separable_states()
    report = audit_regimes(states, labels, traj_ids, 3)
    assert not report.passed
    assert any("Dead regime" in reason for reason in report.failure_reasons)


# ------------------------------------------------------------------
# artifacts
# ------------------------------------------------------------------


def test_topo_artifact_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(0)
    train_labels = [rng.integers(0, 3, size=50), rng.integers(0, 3, size=40)]
    val_labels = [rng.integers(0, 3, size=30)]
    train_bounds = [np.array([0, 25, 50]), np.array([0, 40])]
    val_bounds = [np.array([0, 30])]
    out = save_topo_artifact(
        output_dir=tmp_path / "topo_artifact",
        method="pelt",
        task_name="can",
        data_config_hash="abc123",
        num_regimes=3,
        hyper_params={"penalty_beta": 10.0},
        train_labels=train_labels,
        val_labels=val_labels,
        train_boundaries=train_bounds,
        val_boundaries=val_bounds,
        report={"passed": True},
    )
    labels, bounds, metadata = load_topo_artifact(out)
    assert metadata["version"] == TOPO_ARTIFACT_VERSION
    assert metadata["num_regimes"] == 3
    assert len(labels["train"]) == 2 and len(bounds["val"]) == 1


def test_topo_artifact_checksum_rejection(tmp_path) -> None:
    rng = np.random.default_rng(1)
    out = save_topo_artifact(
        output_dir=tmp_path / "topo_artifact",
        method="pelt",
        task_name="can",
        data_config_hash="abc123",
        num_regimes=2,
        hyper_params={},
        train_labels=[rng.integers(0, 2, size=20)],
        val_labels=[rng.integers(0, 2, size=10)],
        train_boundaries=[np.array([0, 20])],
        val_boundaries=[np.array([0, 10])],
        report={},
    )
    (out / "topo_labels.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError):
        load_topo_artifact(out)


# ------------------------------------------------------------------
# bootstrap + config wiring
# ------------------------------------------------------------------


def _tiny_topo_loader(num_samples: int = 128):
    rng = np.random.default_rng(0)
    states = torch.from_numpy(rng.normal(0, 1.0, (num_samples, 10))).float()
    labels = torch.from_numpy(rng.integers(0, 3, size=num_samples)).long()
    rows = [{"state": states[i], "phase_topo": labels[i]} for i in range(num_samples)]
    return DataLoader(rows, batch_size=32)


def _tiny_phase_moe(num_experts: int = 3):
    from phaseforge.models.components.action_head import ActionHead
    from phaseforge.models.components.encoder import StateEncoder
    from phaseforge.models.components.expert import ExpertMLP
    from phaseforge.models.components.phase_head import PhaseClassificationHead
    from phaseforge.models.components.router import TopKRouter
    from phaseforge.models.phase_moe import PhaseBootstrappedMoE

    encoder = StateEncoder(input_dim=10, hidden_dims=[16], latent_dim=8)
    head = ActionHead(input_dim=8, output_dim=7, hidden_dim=16)
    phase_head = PhaseClassificationHead(latent_dim=8, num_phases=3)
    router = TopKRouter(latent_dim=8, num_experts=num_experts, top_k=1, normalize_input=True)
    expert = ExpertMLP(input_dim=8, hidden_dims=[16], output_dim=7)
    return PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=head,
        phase_head=phase_head,
        router=router,
        expert=expert,
        router_init={"type": "centroid", "prototype_source": "topo", "seed": 0},
        expert_init={"type": "random"},
    )


def test_bootstrap_topo_prototypes_finite_unit_norm() -> None:
    model = _tiny_phase_moe()
    model.bootstrap_moe(dataloader=_tiny_topo_loader(), device="cpu")
    weights = model.moe_layer.router.gate_linear.weight.data
    assert weights.shape == (3, 8)
    assert torch.isfinite(weights).all()
    norms = weights.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_bootstrap_topo_missing_labels_fails_closed() -> None:
    model = _tiny_phase_moe()
    rows = [{"state": torch.randn(10), "phase": torch.tensor(0)} for _ in range(32)]
    loader = DataLoader(rows, batch_size=32)
    with pytest.raises(RuntimeError, match="phase_topo"):
        model.bootstrap_moe(dataloader=loader, device="cpu")


def test_topo_config_group_is_composable() -> None:
    with initialize(version_base="1.3", config_path="../../phaseforge/config"):
        cfg = compose(config_name="main", overrides=["topo@_global_=topo_pelt_k6"])
    assert cfg.data.topo.enabled is True
    assert cfg.data.topo.num_regimes == 6
    assert cfg.data.topo.train_label_field == "phase_topo"


def test_default_main_has_no_topo_block() -> None:
    with initialize(version_base="1.3", config_path="../../phaseforge/config"):
        cfg = compose(config_name="main", overrides=[])
    assert cfg.data.get("topo") is None

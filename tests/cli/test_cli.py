"""CPU-only tests for the CLI module's strict checkpoint-loading diagnostics.

``phaseforge.cli`` must stay importable without ``wandb`` installed (the
import is lazy), and checkpoint loads must hard-fail on missing/unexpected
weights instead of silently continuing on random weights — a mismatched
eval model config otherwise runs on random weights and produces meaningless
rollout results with no error. The only allowed skip is the Stage 1 -> Stage 2
bootstrap (the ``moe_layer`` prefix), which is logged at INFO.
"""

from __future__ import annotations

import logging

import pytest
import torch

import phaseforge.cli as cli


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 2)


def test_cli_importable_without_wandb() -> None:
    """The module imports even though wandb is not installed (lazy import)."""
    # This test env intentionally lacks wandb; the import above already proves
    # the module-level import works. Guard against a future regression by
    # asserting the module is not None and wandb is not imported at top level.
    assert cli is not None
    assert "wandb" not in getattr(cli, "__dict__", {})


def _cfg_with_log_level(level: str):
    from omegaconf import DictConfig

    return DictConfig({"project": {"log_level": level}})


def test_apply_log_level_sets_root_level_and_restores() -> None:
    """The helper must respect the configured level (case-insensitive)."""
    previous = logging.getLogger().level
    try:
        cli._apply_log_level(_cfg_with_log_level("info"))
        assert logging.getLogger().level == logging.INFO
        cli._apply_log_level(_cfg_with_log_level("DEBUG"))
        assert logging.getLogger().level == logging.DEBUG
        cli._apply_log_level(_cfg_with_log_level("warning"))
        assert logging.getLogger().level == logging.WARNING
    finally:
        logging.getLogger().setLevel(previous)


def test_apply_log_level_defaults_to_warning() -> None:
    """Missing config key falls back to the cloud-friendly WARNING level."""
    from omegaconf import DictConfig

    previous = logging.getLogger().level
    try:
        cli._apply_log_level(DictConfig({"project": {}}))
        assert logging.getLogger().level == logging.WARNING
    finally:
        logging.getLogger().setLevel(previous)


def test_apply_log_level_invalid_falls_back_to_warning(caplog) -> None:
    """Unknown levels must not crash the run; WARNING is the safe default."""
    previous = logging.getLogger().level
    try:
        cli._apply_log_level(_cfg_with_log_level("banana"))
        assert logging.getLogger().level == logging.WARNING
        assert "falling back to WARNING" in caplog.text
    finally:
        logging.getLogger().setLevel(previous)


def test_apply_log_level_filters_info_records(caplog) -> None:
    """At WARNING, INFO records are suppressed while WARNING records pass."""
    previous = logging.getLogger().level
    try:
        cli._apply_log_level(_cfg_with_log_level("WARNING"))
        run_logger = logging.getLogger("phaseforge.cli")
        run_logger.info("verbose info that must be filtered")
        run_logger.warning("warning that must appear")
        assert "verbose info that must be filtered" not in caplog.text
        assert "warning that must appear" in caplog.text
    finally:
        logging.getLogger().setLevel(previous)


def test_load_state_dict_checked_passes_on_perfect_match() -> None:
    model = _DummyModel()
    cli._load_state_dict_checked(model, model.state_dict(), "Evaluation checkpoint load")


def test_load_state_dict_checked_raises_on_missing_keys() -> None:
    model = _DummyModel()
    state_dict = model.state_dict()
    del state_dict["head.weight"]
    with pytest.raises(RuntimeError, match="1 missing"):
        cli._load_state_dict_checked(model, state_dict, "Evaluation checkpoint load")


def test_load_state_dict_checked_raises_on_unexpected_keys() -> None:
    model = _DummyModel()
    state_dict = dict(model.state_dict())
    state_dict["moe_layer.router.gate_linear.weight"] = torch.zeros(4, 4)
    with pytest.raises(RuntimeError, match="unexpected"):
        cli._load_state_dict_checked(model, state_dict, "Evaluation checkpoint load")


def test_load_state_dict_checked_allows_expected_prefix_at_info(caplog) -> None:
    """The Stage 1 -> Stage 2 bootstrap legitimately skips the MoE block."""
    model = _DummyModel()
    state_dict = dict(model.state_dict())
    state_dict["moe_layer.router.gate_linear.weight"] = torch.zeros(4, 4)
    state_dict["moe_layer.experts.0.hidden.0.weight"] = torch.zeros(4, 4)

    with caplog.at_level(logging.INFO, logger="phaseforge.cli"):
        cli._load_state_dict_checked(
            model,
            state_dict,
            "Stage 1 -> Stage 2 bootstrap load",
            expected_unexpected_prefixes=("moe_layer",),
        )

    assert "expected, skipped" in caplog.text
    assert "2 key(s) differ" in caplog.text


def test_load_state_dict_checked_allows_expected_missing_prefix_at_info(caplog) -> None:
    """A BC checkpoint legitimately lacks the target MoE block."""
    model = _DummyModel()
    model.moe_layer = torch.nn.Linear(4, 4)
    state_dict = {
        key: value for key, value in model.state_dict().items() if not key.startswith("moe_layer")
    }

    with caplog.at_level(logging.INFO, logger="phaseforge.cli"):
        cli._load_state_dict_checked(
            model,
            state_dict,
            "Stage 1 -> Stage 2 bootstrap load",
            expected_unexpected_prefixes=("moe_layer",),
        )

    assert "2 key(s) differ" in caplog.text


def test_load_state_dict_checked_raises_when_unknown_keys_remain() -> None:
    """Unexpected keys OUTSIDE the expected prefixes must still fail."""
    model = _DummyModel()
    state_dict = dict(model.state_dict())
    state_dict["moe_layer.router.gate_linear.weight"] = torch.zeros(4, 4)
    state_dict["encoder.extra_bias"] = torch.zeros(4)

    with pytest.raises(RuntimeError, match="encoder.extra_bias"):
        cli._load_state_dict_checked(
            model,
            state_dict,
            "Stage 1 -> Stage 2 bootstrap load",
            expected_unexpected_prefixes=("moe_layer",),
        )


def _build_model_for_test(name: str) -> torch.nn.Module:
    from omegaconf import DictConfig, OmegaConf

    from phaseforge.utils.registry import build_model

    path = f"phaseforge/config/models/{name}.yaml"
    # Model configs interpolate ${data.state_dim}/${data.action_dim} and the
    # canonical phaseforge ties expert_init.seed to ${project.seed}, so the
    # data and project blocks must be present for the config to resolve —
    # same as the CLI's full composition.
    data_cfg = OmegaConf.load("phaseforge/config/data/common.yaml")
    return build_model(
        DictConfig(
            {"models": OmegaConf.load(path), "data": data_cfg, "project": {"seed": 42}}
        )
    )


def test_stage1_bootstrap_load_matrix_all_cells() -> None:
    """Every real Stage 1 -> Stage 2 source/target pair must load cleanly.

    Cells whose Stage 1 comes from ``bc`` (no phase head) or from the
    phaseforge cell must be loadable through the cli's checked bootstrap
    load, and the load must still reject a missing head the target NEEDS.
    """
    # The normalized BC provider (encoder + action head only) never has phase_head.
    bc = _build_model_for_test("final_aligned_bc")
    bc_sd = bc.state_dict()

    # The proposed final cell's Stage 1 checkpoint includes the phase head.
    pf = _build_model_for_test("precision_residual_phaseforge")
    pf_sd = pf.state_dict()
    assert any(k.startswith("phase_head") for k in pf_sd)

    # The plain representation control consumes the BC provider and has no phase head.
    for name in ("precision_residual_plain_encoder", "precision_residual_factorial_floor"):
        model = _build_model_for_test(name)
        assert not any(k.startswith("phase_head") for k in model.state_dict())
        cli._load_state_dict_checked(
            model,
            bc_sd,
            f"{name} <- final_aligned_bc bootstrap load",
            expected_unexpected_prefixes=("moe_layer",),
        )

    # The proposed and teacher diagnostic share the final phase-supervised bundle.
    for name in ("precision_residual_phaseforge", "precision_residual_teacher_forced"):
        model = _build_model_for_test(name)
        assert any(k.startswith("phase_head") for k in model.state_dict())
        cli._load_state_dict_checked(
            model,
            pf_sd,
            f"{name} <- precision_residual_phaseforge bootstrap load",
            expected_unexpected_prefixes=("moe_layer", "soft_mapping"),
        )

    # The random-router and scratch controls keep the final phase head, while
    # changing only their registered Stage-2 initialization factor.
    model = _build_model_for_test("precision_residual_phase_random_router")
    assert any(k.startswith("phase_head") for k in model.state_dict())
    cli._load_state_dict_checked(
        model,
        pf_sd,
        "precision_residual_phase_random_router <- precision_residual_phaseforge bootstrap load",
        expected_unexpected_prefixes=("moe_layer", "soft_mapping"),
    )


def test_unused_stage1_head_prefixes_are_target_specific() -> None:
    """The cli derives droppable Stage 1 heads from the TARGET architecture.

    A cell without a phase head may drop the checkpoint's phase_head keys;
    a cell that routes by the phase head may not.
    """
    assert cli._unused_stage1_head_prefixes(
        _build_model_for_test("precision_residual_plain_encoder")
    ) == ("phase_head",)
    assert cli._unused_stage1_head_prefixes(
        _build_model_for_test("precision_residual_factorial_floor")
    ) == ("phase_head",)
    assert cli._unused_stage1_head_prefixes(
        _build_model_for_test("precision_residual_phase_random_router")
    ) == ()
    assert cli._unused_stage1_head_prefixes(
        _build_model_for_test("precision_residual_teacher_forced")
    ) == ()


def test_phase_head_required_when_target_uses_it() -> None:
    """A checkpoint missing the phase head must still fail for cells that
    route by it (the prefix is only allowed for heads the target lacks)."""
    teacher_forced = _build_model_for_test("precision_residual_teacher_forced")
    bc_sd = dict(_build_model_for_test("final_aligned_bc").state_dict())
    assert not any(k.startswith("phase_head") for k in bc_sd)

    with pytest.raises(RuntimeError, match="phase_head"):
        cli._load_state_dict_checked(
            teacher_forced,
            bc_sd,
            "precision_residual_teacher_forced <- final_aligned_bc bootstrap load",
            expected_unexpected_prefixes=("moe_layer",),
        )


def test_load_state_dict_checked_normalizer_stats() -> None:
    """Checkpoints containing normalizer buffers load cleanly into models with and without
    normalizer support."""
    # Model without set_normalizer_stats legitimately drops normalizer_ keys when allowed
    dummy = _DummyModel()
    sd = dict(dummy.state_dict())
    sd["normalizer_mean"] = torch.zeros(4)
    sd["normalizer_std"] = torch.ones(4)
    cli._load_state_dict_checked(
        dummy,
        sd,
        "Evaluation checkpoint load",
        expected_unexpected_prefixes=("soft_mapping", "normalizer_"),
    )

    # Model with set_normalizer_stats has the buffers registered and matches exactly
    pf = _build_model_for_test("precision_residual_phaseforge")
    dim = pf.encoder.hidden[0].in_features
    mean = torch.randn(dim)
    std = torch.ones(dim)
    pf.set_normalizer_stats(mean, std)
    sd_pf = pf.state_dict()
    assert "normalizer_mean" in sd_pf
    assert "normalizer_std" in sd_pf

    fresh_pf = _build_model_for_test("precision_residual_phaseforge")
    fresh_pf.set_normalizer_stats(sd_pf["normalizer_mean"], sd_pf["normalizer_std"])
    cli._load_state_dict_checked(
        fresh_pf,
        sd_pf,
        "Evaluation checkpoint load",
        expected_unexpected_prefixes=("soft_mapping", "normalizer_"),
    )
    m_ret, s_ret = fresh_pf.get_normalizer_stats()
    assert torch.allclose(m_ret, mean)
    assert torch.allclose(s_ret, std)


def test_build_eval_model_restores_normalizer_stats(tmp_path) -> None:
    """build_eval_model automatically restores normalizer stats from checkpoint."""
    from omegaconf import DictConfig, OmegaConf

    data_cfg = OmegaConf.load("phaseforge/config/data/common.yaml")
    model_cfg = OmegaConf.load(
        "phaseforge/config/models/precision_residual_phaseforge.yaml"
    )
    cfg = DictConfig(
        {"models": model_cfg, "data": data_cfg, "project": {"seed": 42}, "train": {}}
    )
    model = cli.build_model(cfg)
    dim = model.encoder.hidden[0].in_features
    mean = torch.randn(dim)
    std = torch.ones(dim) * 2.0
    model.set_normalizer_stats(mean, std)

    ckpt_path = tmp_path / "checkpoint_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stage": 2,
        },
        ckpt_path,
    )

    cfg.train.stage1_ckpt_path = str(ckpt_path)
    eval_model = cli.build_eval_model(cfg)
    assert getattr(eval_model, "stage", None) == 2
    m_ret, s_ret = eval_model.get_normalizer_stats()
    assert m_ret is not None and torch.allclose(m_ret, mean)
    assert s_ret is not None and torch.allclose(s_ret, std)


def test_build_eval_model_fails_closed_on_corrupt_unexpected_key(tmp_path) -> None:
    """build_eval_model still hard-fails on truly unexpected keys."""
    from omegaconf import DictConfig, OmegaConf

    data_cfg = OmegaConf.load("phaseforge/config/data/common.yaml")
    model_cfg = OmegaConf.load(
        "phaseforge/config/models/precision_residual_phaseforge.yaml"
    )
    cfg = DictConfig(
        {"models": model_cfg, "data": data_cfg, "project": {"seed": 42}, "train": {}}
    )
    model = cli.build_model(cfg)
    sd = dict(model.state_dict())
    sd["corrupt_head_extra_param"] = torch.zeros(4)

    ckpt_path = tmp_path / "checkpoint_best.pt"
    torch.save({"model_state_dict": sd, "stage": 2}, ckpt_path)

    cfg.train.stage1_ckpt_path = str(ckpt_path)
    with pytest.raises(RuntimeError, match="corrupt_head_extra_param"):
        cli.build_eval_model(cfg)

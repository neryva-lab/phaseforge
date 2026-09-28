"""Tests for the pinned env metadata and the environment parity gate."""

from __future__ import annotations

import pytest

from phaseforge.evaluations.envs.env_metadata import (
    DEV_FALLBACK_ENV_ARGS,
    env_args_to_metadata,
    env_metadata_from_cache,
    verify_environment_parity,
)
from phaseforge.evaluations.envs.errors import EnvParityError

GOOD_ARGS = {
    "env_name": "Lift",
    "env_version": "1.5.1",
    "type": "robosuite",
    "env_kwargs": {"robots": "Panda", "horizon": 500, "reward_shaping": True},
}


class TestParse:
    def test_valid_args(self) -> None:
        meta = env_args_to_metadata(GOOD_ARGS)
        assert meta.env_name == "Lift"
        assert meta.horizon == 500

    def test_missing_keys(self) -> None:
        with pytest.raises(EnvParityError, match="env_name"):
            env_args_to_metadata({"env_kwargs": {}})

    def test_bad_kwargs_type(self) -> None:
        with pytest.raises(EnvParityError, match="env_kwargs"):
            env_args_to_metadata({**GOOD_ARGS, "env_kwargs": 5})

    def test_default_horizon(self) -> None:
        meta = env_args_to_metadata({**GOOD_ARGS, "env_kwargs": {}})
        assert meta.horizon == 500

    def test_canonical_json_stable_and_rendering_insensitive(self) -> None:
        with_renderer = env_args_to_metadata(
            {
                **GOOD_ARGS,
                "env_kwargs": {
                    "has_renderer": True,
                    "use_camera_obs": True,
                    "robots": "Panda",
                },
            }
        )
        headless = env_args_to_metadata({**GOOD_ARGS, "env_kwargs": {"robots": "Panda"}})
        assert with_renderer.canonical_json() == headless.canonical_json()


class TestFromCache:
    def test_recovers_metadata(self, tmp_path) -> None:
        traj_dir = tmp_path / "trajectories"
        traj_dir.mkdir()
        import torch

        torch.save(
            {"env_metadata": GOOD_ARGS, "state": __import__("numpy").zeros((5, 19))},
            traj_dir / "000000.pt",
        )
        meta = env_metadata_from_cache(tmp_path)
        assert meta.env_name == "Lift"
        assert meta.env_version == "1.5.1"

    def test_missing_env_metadata(self, tmp_path) -> None:
        traj_dir = tmp_path / "trajectories"
        traj_dir.mkdir()
        import torch

        torch.save({"state": __import__("numpy").zeros((5, 19))}, traj_dir / "000000.pt")
        with pytest.raises(EnvParityError, match="env_metadata"):
            env_metadata_from_cache(tmp_path)

    def test_empty_cache(self, tmp_path) -> None:
        with pytest.raises(EnvParityError, match="no trajectories"):
            env_metadata_from_cache(tmp_path)


class TestParityGate:
    def test_passes_on_matching_versions(self, monkeypatch) -> None:
        from phaseforge.evaluations.envs import env_metadata as mod

        monkeypatch.setattr(
            mod, "installed_versions", lambda: {"robosuite": "1.5.1", "mujoco": "3.2.7"}
        )
        meta = env_args_to_metadata(GOOD_ARGS)
        installed = verify_environment_parity(
            meta,
            expected_env_name="Lift",
            robosuite_requirement="==1.5.1",
            mujoco_requirement=">=3.2.3",
        )
        assert installed["robosuite"] == "1.5.1"

    def test_version_mismatch_fails_closed(self, monkeypatch) -> None:
        from phaseforge.evaluations.envs import env_metadata as mod

        monkeypatch.setattr(
            mod, "installed_versions", lambda: {"robosuite": "1.4.0", "mujoco": "3.2.7"}
        )
        meta = env_args_to_metadata(GOOD_ARGS)
        with pytest.raises(EnvParityError, match="collected with robosuite 1.5.1"):
            verify_environment_parity(
                meta,
                expected_env_name="Lift",
                robosuite_requirement="==1.5.1",
                mujoco_requirement=">=3.2.3",
            )

    def test_version_mismatch_message_is_actionable(self, monkeypatch) -> None:
        """The ToolHang-vs-1.5.1 case must include the install command."""
        from phaseforge.evaluations.envs import env_metadata as mod

        # Installed 1.5.1, dataset records 1.5.0 (ToolHang).
        monkeypatch.setattr(
            mod, "installed_versions", lambda: {"robosuite": "1.5.1", "mujoco": "3.2.7"}
        )
        toolhang_args = {**GOOD_ARGS, "env_name": "ToolHang", "env_version": "1.5.0"}
        meta = env_args_to_metadata(toolhang_args)
        with pytest.raises(EnvParityError) as exc_info:
            verify_environment_parity(
                meta,
                expected_env_name="ToolHang",
                robosuite_requirement="==1.5.1",
                mujoco_requirement=">=3.2.3",
            )
        msg = str(exc_info.value)
        # Must mention the conflicting versions and the install command.
        assert "1.5.1" in msg and "1.5.0" in msg
        assert "robosuite==1.5.0" in msg
        assert "authoritative" in msg
        # Must surface the per-task pin location.
        assert "data/<task>.yaml" in msg or "per-task" in msg

    def test_missing_robosuite_fails(self, monkeypatch) -> None:
        from phaseforge.evaluations.envs import env_metadata as mod

        monkeypatch.setattr(mod, "installed_versions", lambda: {"robosuite": "", "mujoco": ""})
        meta = env_args_to_metadata(GOOD_ARGS)
        with pytest.raises(EnvParityError, match="not installed"):
            verify_environment_parity(
                meta,
                expected_env_name="Lift",
                robosuite_requirement="==1.5.1",
                mujoco_requirement=">=3.2.3",
            )

    def test_mujoco_requirement_enforced(self, monkeypatch) -> None:
        from phaseforge.evaluations.envs import env_metadata as mod

        monkeypatch.setattr(
            mod, "installed_versions", lambda: {"robosuite": "1.5.1", "mujoco": "3.1.0"}
        )
        meta = env_args_to_metadata(GOOD_ARGS)
        with pytest.raises(EnvParityError, match="mujoco 3.1.0"):
            verify_environment_parity(
                meta,
                expected_env_name="Lift",
                robosuite_requirement="==1.5.1",
                mujoco_requirement=">=3.2.3",
            )

    def test_env_name_mismatch(self, monkeypatch) -> None:
        from phaseforge.evaluations.envs import env_metadata as mod

        monkeypatch.setattr(
            mod, "installed_versions", lambda: {"robosuite": "1.5.1", "mujoco": "3.2.7"}
        )
        meta = env_args_to_metadata(GOOD_ARGS)
        with pytest.raises(EnvParityError, match="environment name mismatch"):
            verify_environment_parity(
                meta,
                expected_env_name="Stack",
                robosuite_requirement="==1.5.1",
                mujoco_requirement=">=3.2.3",
            )

    def test_aggregates_all_problems(self, monkeypatch) -> None:
        from phaseforge.evaluations.envs import env_metadata as mod

        monkeypatch.setattr(mod, "installed_versions", lambda: {"robosuite": "", "mujoco": ""})
        meta = env_args_to_metadata(GOOD_ARGS)
        with pytest.raises(EnvParityError) as exc_info:
            verify_environment_parity(
                meta,
                expected_env_name="Stack",
                robosuite_requirement="==1.5.1",
                mujoco_requirement=">=3.2.3",
            )
        message = str(exc_info.value)
        assert "robosuite" in message and "mujoco" in message and "env" in message


def test_dev_fallback_roundtrip() -> None:
    meta = env_args_to_metadata(DEV_FALLBACK_ENV_ARGS)
    assert meta.env_name == "Lift"
    assert meta.env_version == "1.5.1"
    assert meta.horizon == 500


class TestDevFallbackPerTask:
    """The dev fallback must resolve each of the five benchmark tasks."""

    @pytest.mark.parametrize(
        "protocol_name, env_name, horizon",
        [
            ("Lift", "Lift", 500),
            ("Can", "PickPlaceCan", 500),
            ("Square", "NutAssemblySquare", 500),
            ("ToolHang", "ToolHang", 500),
            ("Transport", "TwoArmTransport", 700),
        ],
    )
    def test_dev_fallback_for_each_task(
        self, protocol_name: str, env_name: str, horizon: int
    ) -> None:
        from phaseforge.evaluations.envs.env_metadata import dev_fallback_metadata

        meta = dev_fallback_metadata(protocol_name)
        assert meta.env_name == env_name
        assert meta.env_version == "1.5.1"
        assert meta.horizon == horizon

    def test_dev_fallback_unknown_protocol_raises(self) -> None:
        from phaseforge.evaluations.envs.env_metadata import dev_fallback_metadata

        with pytest.raises(KeyError, match="Unknown protocol task"):
            dev_fallback_metadata("Hopper")

    def test_dev_fallback_module_default_is_lift(self) -> None:
        """The module-level ``DEV_FALLBACK_ENV_ARGS`` is kept as the Lift
        default for backward compatibility with single-task callers."""
        assert DEV_FALLBACK_ENV_ARGS["env_name"] == "Lift"

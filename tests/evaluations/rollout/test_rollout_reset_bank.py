"""Tests for the frozen reset bank (generation, integrity, tamper detection)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from phaseforge.evaluations.envs.errors import EnvParityError
from phaseforge.evaluations.rollout.reset_bank import (
    ResetBank,
    ResetCase,
    compute_bank_id,
    generate_reset_bank,
)
from tests.rollout_helpers import FakeAdapter, FakeLiftSim, make_meta


def _adapter_factory(**kwargs):
    def factory():
        return FakeAdapter(FakeLiftSim(np.random.default_rng(7)), **kwargs)

    return factory


def _tiny_bank(tmp_path) -> ResetBank:
    cases = [
        ResetCase(index=i, states=np.random.default_rng(i).uniform(-1, 1, 12).astype(np.float32))
        for i in range(3)
    ]
    bank = ResetBank(
        task="Lift",
        bank_id="deadbeef",
        seed=1,
        num_cases=3,
        env_canonical="{}",
        robosuite_version="1.5.1",
        git_commit="",
        generated_at="2026-01-01T00:00:00Z",
        cases=cases,
    )
    bank.save(tmp_path / "bank")
    return bank


class TestGenerate:
    def test_generates_distinct_cases(self) -> None:
        meta = make_meta()
        bank = generate_reset_bank(
            _adapter_factory(),
            meta,
            task="Lift",
            seed=2026,
            num_cases=5,
            robosuite_version="1.5.1",
        )
        assert bank.num_cases == 5
        assert len(bank.cases) == 5
        states = np.stack([c.states for c in bank.cases])
        assert np.isfinite(states).all()
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                assert np.linalg.norm(states[i][1:] - states[j][1:]) > 1e-3

    def test_degenerate_sampler_fails(self) -> None:
        class ConstantSim:
            def __init__(self):
                self.state = np.zeros(12, dtype=np.float32)

            @property
            def sim(self):
                return self

            def get_state(self):
                return self.state

            def reset(self):
                pass

        def constant_factory():
            return FakeAdapter(ConstantSim())  # type: ignore[arg-type]

        with pytest.raises(Exception, match="distinct|degenerate"):
            generate_reset_bank(
                constant_factory,
                make_meta(),
                task="Lift",
                num_cases=3,
                max_attempts_per_case=3,
                robosuite_version="1.5.1",
            )

    def test_seed_controls_legacy_global_sampler_and_restores_caller_state(self) -> None:
        class GlobalRngSim:
            def __init__(self):
                self.state = np.zeros(12, dtype=np.float32)

            @property
            def sim(self):
                return self

            def reset(self):
                self.state = np.random.random(12).astype(np.float32)

            def get_state(self):
                return self.state

        def factory():
            return type("Adapter", (), {"env": GlobalRngSim()})()

        np.random.seed(1234)
        random.seed(1234)
        expected_numpy = np.random.random()
        expected_python = random.random()

        np.random.seed(1234)
        random.seed(1234)
        first = generate_reset_bank(
            factory,
            make_meta(),
            task="Lift",
            seed=2026,
            num_cases=3,
            robosuite_version="1.5.1",
        )
        actual_numpy = np.random.random()
        actual_python = random.random()

        second = generate_reset_bank(
            factory,
            make_meta(),
            task="Lift",
            seed=2026,
            num_cases=3,
            robosuite_version="1.5.1",
        )
        assert actual_numpy == expected_numpy
        assert actual_python == expected_python
        assert np.array_equal(
            np.stack([case.states for case in first.cases]),
            np.stack([case.states for case in second.cases]),
        )


class TestRoundtrip:
    def test_save_load_roundtrip(self, tmp_path) -> None:
        bank = _tiny_bank(tmp_path)
        loaded = ResetBank.load(tmp_path / "bank")
        assert loaded.bank_id == bank.bank_id
        assert loaded.num_cases == 3
        for original, restored in zip(bank.cases, loaded.cases):
            assert np.array_equal(original.states, restored.states)
            assert original.xml == restored.xml

    def test_missing_manifest(self, tmp_path) -> None:
        with pytest.raises(EnvParityError, match="manifest"):
            ResetBank.load(tmp_path / "nope")

    def test_tampered_state_detected(self, tmp_path) -> None:
        bank_dir = tmp_path / "bank"
        _tiny_bank(tmp_path)
        (bank_dir / "cases" / "0000.npy").write_bytes(b"TAMPERED")
        with pytest.raises(EnvParityError, match="SHA-256"):
            ResetBank.load(bank_dir)

    def test_missing_case_file_detected(self, tmp_path) -> None:
        bank_dir = tmp_path / "bank"
        _tiny_bank(tmp_path)
        (bank_dir / "cases" / "0002.npy").unlink()
        with pytest.raises(EnvParityError, match="missing case file"):
            ResetBank.load(bank_dir)

    def test_truncated_cases_detected(self, tmp_path) -> None:
        import json

        bank_dir = tmp_path / "bank"
        _tiny_bank(tmp_path)
        manifest_path = bank_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["num_cases"] = 7
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(EnvParityError, match="declares 7 cases"):
            ResetBank.load(bank_dir)


class TestIdentity:
    def test_bank_id_deterministic_and_content_sensitive(self) -> None:
        meta = make_meta()
        a = compute_bank_id(meta, task="Lift", seed=2026, num_cases=50, robosuite_version="1.5.1")
        b = compute_bank_id(meta, task="Lift", seed=2026, num_cases=50, robosuite_version="1.5.1")
        c = compute_bank_id(meta, task="Lift", seed=2027, num_cases=50, robosuite_version="1.5.1")
        assert a == b
        assert a != c

    def test_case_index_range(self, tmp_path) -> None:
        bank = _tiny_bank(tmp_path)
        with pytest.raises(IndexError):
            bank.case(5)

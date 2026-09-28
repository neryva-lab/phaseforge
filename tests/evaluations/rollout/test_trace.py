"""CPU-only tests for full rollout tracing."""

from __future__ import annotations

import pytest
import torch

from phaseforge.evaluations.rollout.trace import (
    TRACE_FIELDS,
    TraceSchemaError,
    TraceWriter,
    read_trace_rows,
    to_jsonable,
    validate_trace_record,
)


def _row(**overrides):
    row = {
        "episode_id": 0,
        "case_id": 0,
        "timestep": 0,
        "termination_reason": "running",
        "raw_obs_summary": {"state_norm": 1.0, "eef_pos": [0.0, 0.0, 1.0]},
        "normalized_state_norm": 1.0,
        "task_vars": [0.0] * 8,
        "latent_norm": 1.0,
        "dists": [0.1, 0.9],
        "selected_expert": 0,
        "top2_expert": 1,
        "router_margin": 0.8,
        "router_entropy": None,
        "expert_target": [0.0] * 8,
        "expert_gains": [1.0] * 7,
        "task_error": [0.0] * 7,
        "pre_clip_command": [0.0] * 7,
        "final_action": [0.0] * 7,
        "nearest_train_dist": 2.5,
        "expert_disagreement": 0.01,
        "lip_diagnostic": None,
        "done": False,
    }
    row.update(overrides)
    return row


def test_trace_fields_cover_spec_order() -> None:
    assert len(TRACE_FIELDS) == 22
    assert TRACE_FIELDS[0] == "episode_id"
    assert TRACE_FIELDS[-1] == "done"
    assert "termination_reason" in TRACE_FIELDS


def test_validate_accepts_full_and_minimal_rows() -> None:
    validate_trace_record(_row())
    sparse = _row(
        task_vars=None,
        dists=None,
        router_entropy=None,
        expert_target=None,
        expert_gains=None,
        task_error=None,
        pre_clip_command=None,
        nearest_train_dist=None,
        expert_disagreement=None,
        lip_diagnostic=None,
        top2_expert=None,
    )
    validate_trace_record(sparse)


def test_validate_rejects_bad_rows() -> None:
    row = _row()
    del row["timestep"]
    with pytest.raises(TraceSchemaError, match="missing"):
        validate_trace_record(row)
    with pytest.raises(TraceSchemaError, match="unknown"):
        validate_trace_record({**_row(), "extra": 1})
    with pytest.raises(TraceSchemaError):
        validate_trace_record(_row(episode_id="zero"))
    with pytest.raises(TraceSchemaError):
        validate_trace_record(_row(done="yes"))
    with pytest.raises(TraceSchemaError):
        validate_trace_record(_row(selected_expert=1.5))


def test_writer_roundtrip_and_empty_noop(tmp_path) -> None:
    writer = TraceWriter(tmp_path)
    assert writer.append_episode_rows([]) == tmp_path / "trace.jsonl"
    assert read_trace_rows(tmp_path) == []
    writer.append_episode_rows([_row(timestep=0), _row(timestep=1, done=True)])
    rows = read_trace_rows(tmp_path)
    assert [r["timestep"] for r in rows] == [0, 1]
    assert rows[1]["done"] is True
    with pytest.raises(TraceSchemaError):
        writer.append_episode_rows([_row(timestep=2), {"bogus": True}])
    # The failed append wrote nothing (validation precedes the write).
    assert [r["timestep"] for r in read_trace_rows(tmp_path)] == [0, 1]


def test_to_jsonable_conversions() -> None:
    import numpy as np

    assert to_jsonable(None) is None
    assert to_jsonable(torch.tensor([1, 2])) == [1, 2]
    assert to_jsonable(torch.tensor(3.0)) == pytest.approx(3.0)
    assert to_jsonable(np.float64(0.5)) == pytest.approx(0.5)
    assert to_jsonable({"a": torch.tensor(1)}) == {"a": 1}

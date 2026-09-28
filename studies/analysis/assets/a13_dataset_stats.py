"""A13 — dataset & phase-label statistics per task (from data_provenance.json)."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import io as cio
from studies.analysis.common import registry
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.tables import Table, save_table


def _provenance_for(dataset: AnalysisDataset, task: str) -> dict:
    for (t, name, seed, stage), run in dataset.train_runs.items():
        if t == task:
            path = run.path / "metadata" / "data_provenance.json"
            if path.is_file():
                data = cio.read_json(path)
                if isinstance(data, dict):
                    return data
    return {}


def generate(dataset: AnalysisDataset) -> list[Path]:
    rows = []
    for task in registry.tasks():
        prov = _provenance_for(dataset, task)
        p = prov.get("provenance", {}) if prov else {}
        if not isinstance(p, dict):
            p = {}
        # data_provenance.json nests schema under state_schema and the
        # strategy under normalization (flat keys do not exist).
        schema = p.get("state_schema", {})
        if not isinstance(schema, dict):
            schema = {}
        norm = p.get("normalization", {})
        if not isinstance(norm, dict):
            norm = {}
        sampling = p.get("sampling", {})
        if not isinstance(sampling, dict):
            sampling = {}
        state_keys = schema.get("keys", [])
        if not isinstance(state_keys, list):
            state_keys = []
        state_dim = schema.get(
            "state_dim", sum(int(k.get("dim", 0)) for k in state_keys if isinstance(k, dict))
        )
        strategy = norm.get("strategy", "--")
        normalization = (
            f"{strategy} (train split)"
            if norm.get("train_split_only", False) and strategy != "--"
            else str(strategy)
        )
        seq_len = sampling.get("sequence_length", "--")
        seq_cell = (
            f"{seq_len} (single-step window)"
            if isinstance(seq_len, int)
            else str(seq_len)
        )
        rows.append(
            [
                task,
                str(state_dim),
                str(schema.get("action_dim", "--")),
                normalization,
                seq_cell,
                str(schema.get("schema_version", "--")),
                str(len(state_keys)),
            ]
        )
    table = Table(
        headers=[
            "Task",
            "State dim",
            "Action dim",
            "Normalization",
            "Seq. len",
            "Schema",
            "State keys",
        ],
        rows=rows,
        caption="Per-task structured-state schema and normalization (data_provenance.json).",
        notes=(
            "Phase-label distribution statistics are produced by the ingestion "
            "cache; the frozen phase definitions (6 regimes) are identical "
            "across tasks.",
        ),
    )
    return save_table(table, "tables/A13_dataset_stats")

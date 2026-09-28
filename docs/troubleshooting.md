# Troubleshooting

This document lists common failure symptoms, their causes, and their fixes.

## `phaseforge-gates` exits with code 2

**Cause.** The gates could not run at all, for example because the robosuite simulator dependency is missing.

**Fix.** Install the simulator dependency, or run offline evaluation instead of rollout evaluation. See [installation.md](installation.md) for the environment and [evaluation.md](evaluation.md) for the evaluation modes.

## `phaseforge-gates` exits with code 1

**Cause.** One or more required gates failed.

**Fix.** Inspect the per-gate results in the report at `outputs/_gates/<timestamp>/gates_report.json` to identify which gate failed and why.

## Training re-ingests instead of reusing the cache

**Cause.** The cache key (`config_hash`) covers the canonical serialized data config plus provenance context: the git commit and the raw dataset file names and sizes. Any change to phase thresholds, state keys, split ratios, code revision, or raw data invalidates the cache. With `enforce_strict_cache: true`, cache reuse is fail-closed by design.

**Fix.** This is expected behavior after a config or data change. If reuse was expected, confirm that the data config, code revision, and raw files are byte-identical to the run that populated the cache. See [data.md](data.md) for the cache layout and invalidation rules.

## `DataPipelineStateMachine` raises on missing raw data

**Cause.** The `VALIDATE_SOURCE` state fails closed when the raw dataset directory is absent. `data.source.auto_download` defaults to `false`, so training never downloads data on its own.

**Fix.** Provision the data explicitly before training:

```bash
uv run python scripts/utils/download_datasets.py
```

Enable `auto_download` only for explicit provisioning workflows. See [data.md](data.md).

## `--help` crashes with `badly formed help string` on Python 3.14

**Cause.** A known incompatibility between hydra-core 1.3.5 and Python 3.14's strict argparse validation: hydra passes a non-string help object that argparse rejects.

**Fix.** Use the compatibility shim, which runs the identical train/eval code path:

```bash
uv run python scripts/protocol/local_run.py train models=precision_residual_phaseforge train=stage1 ...
uv run python scripts/protocol/local_run.py eval  models=precision_residual_phaseforge eval=rollout ...
```

See [training.md](training.md).

## Import or config errors referencing `legacy/` paths

**Cause.** The `legacy/` tree is an archived, explicitly disowned snapshot. The live protocol is the robomimic ingestion and evaluation stack under `phaseforge/data/ingestion/` and `phaseforge/evaluations/`.

**Fix.** Do not resurrect legacy modules. Point imports and configs at the live packages documented in [data.md](data.md) and [training.md](training.md).

## `generate --check` reports missing cells

**Cause.** The sweep coverage is incomplete: some (method, task, seed) cells expected by the manifests have no completed run.

**Fix.** Complete the missing runs, then regenerate. For dev iteration only, `generate` accepts `--allow-partial` to proceed despite incomplete coverage; results produced this way are not final. See [analysis.md](analysis.md).

## Duplicate completed runs abort generation

**Cause.** The analysis pipeline refuses ambiguous inputs: two completed runs for the same cell would make the figures non-reproducible.

**Fix.** Resolve the duplication in the outputs (keep the intended run, remove or relocate the other) before regenerating. See [analysis.md](analysis.md).

## See also

- [installation.md](installation.md) — environment setup
- [quickstart.md](quickstart.md) — minimal end-to-end run
- [data.md](data.md) — datasets, ingestion, and cache
- [training.md](training.md) — training stages and the sweep runner
- [evaluation.md](evaluation.md) — offline metrics, gates, and rollout reporting

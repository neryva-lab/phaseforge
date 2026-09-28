# Reproducing the Paper Results

This document is the ordered end-to-end checklist for reproducing the paper's results from a clean checkout.

## 1. Install

```bash
uv sync
uv run phaseforge-train --help
```

`uv sync` installs the project and its dependencies; the second command confirms the console entry points are available. See [installation.md](installation.md) for environment details.

## 2. Provision the data

```bash
uv run python scripts/utils/download_datasets.py
```

This downloads the raw robomimic artifacts for all five tasks. Expected result: `data/raw/robomimic/<task>/low_dim_v15.hdf5` for each of lift, can, square, tool_hang, and transport, each SHA-256-verified against the mirror's checksum metadata. See [data.md](data.md) for the ingestion pipeline and cache layout.

## 3. Preflight the experiment matrix

```bash
uv run python scripts/protocol/preflight_configs.py
```

Expected result: exit code 0, meaning every (method, task, stage, seed) cell of the frozen manifest composed and validated. Any failure is reported per cell with the exact failing override; resolve it before spending GPU time. See [experiments.md](experiments.md).

## 4. Run the sweep

```bash
phaseforge-sweep
```

The default manifest is `experiments/final_causal_matrix.json`. Expected result: one run directory per completed cell under `outputs/`, each containing `resolved_config.yaml`, `run_meta.json`, and `checkpoints/checkpoint_best.pt`. See [training.md](training.md).

## 5. Run the rollout gates

```bash
phaseforge-gates eval=rollout
```

Expected result: exit code 0 (every required gate passed) and a report at `outputs/_gates/<timestamp>/gates_report.json`. See [evaluation.md](evaluation.md).

## 6. Aggregate the rollout results

```bash
phaseforge-rollout-report
```

Expected result: paper-table CSVs under `outputs/_results`. The command is idempotent; re-running re-aggregates every completed rollout eval run.

## 7. Generate the figures and tables

```bash
python -m studies.analysis.scripts.generate --check
python -m studies.analysis.scripts.generate
python -m studies.analysis.scripts.verify
```

Expected result: `--check` reports clean coverage; generation writes all 23 assets under `studies/analysis/outputs/figures/{main,appendix}` and `studies/analysis/outputs/tables/`; `verify` confirms the outputs and the manifest. Note: `verify`'s registry-shape check currently reports a divergence because `PLANNED_IDS` omits A16 (see [analysis.md](analysis.md)); this is a known bookkeeping discrepancy, not an asset problem. See [analysis.md](analysis.md).

## Traceability

Every figure and table maps through `generation_manifest.json` to the exact run data that produced it: the manifest records each asset alongside the sha256 of its inputs and outputs. A reproduced figure is therefore auditable back to the specific sweep cells, configs, and dataset cache it was built from.

## See also

- [quickstart.md](quickstart.md) — a shorter single-task run for orientation
- [troubleshooting.md](troubleshooting.md) — failure symptoms, causes, and fixes

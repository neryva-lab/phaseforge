# Data

This document describes the datasets used by PhaseForge, their provenance, and the ingestion pipeline that converts raw HDF5 artifacts into training-ready data loaders.

## Datasets

All experiments use robomimic Proficient-Human (`ph`) low-dimensional HDF5 artifacts, mirrored on Hugging Face at `amandlek/robomimic`. Five tasks are used: `lift`, `can`, `square`, `tool_hang`, and `transport`. The remote path for each task is `v1.5/<task>/ph/low_dim_v15.hdf5`.

## Acquisition

Raw datasets are downloaded with a human-operated script. Training never downloads data.

```bash
uv run python scripts/utils/download_datasets.py
```

```bash
uv run python scripts/utils/download_datasets.py --tasks lift
```

The first command downloads all five tasks; the second downloads a single task. The default data directory is `data/raw/robomimic`; each task lands at `<data-dir>/<task>/low_dim_v15.hdf5`.

## Observation and action schema

The observation schema is privileged structured low-dimensional state. It is neither RGB imagery nor bare robot proprioception. Actions are the 7-dimensional robosuite environment action vector, normalized to [-1, 1] (see `phaseforge/config/data/common.yaml`).

Ingestion is performed by `phaseforge.data.robomimic.ingester.RobomimicHDF5Ingester`. Phase labels come from `phaseforge.data.robomimic.phase_labeler.RuleBasedPhaseLabeler`, which assigns one of 6 phases.

## Data configuration

Per-task configs live in `phaseforge/config/data/`: `lift.yaml`, `can.yaml`, `square.yaml`, `tool_hang.yaml`, and `transport.yaml`. Negative-control variants with robot-only observations are provided as `robot_only.yaml` and `robot_only_<task>.yaml` for each task. Select a config with the Hydra override `data=<name>`.

## Ingestion pipeline

`DataPipelineStateMachine(cfg).run()` is the single entry point for all data loading. It returns `{"train": DataLoader, "val": DataLoader}`; a split with zero samples returns `None` instead of a loader.

The state machine has seven states:

- `CHECK_PERSISTENT_CACHE` — reuse an existing processed cache when the config hash matches.
- `VALIDATE_SOURCE` — verify that the raw dataset directory exists and is complete. A missing raw directory raises here unless auto-download is enabled.
- `PROVISION_SOURCE` — download missing raw data from the mirror. Reached only when `data.source.auto_download=true`.
- `INGEST_AND_STRIP` — parse the HDF5 artifacts and extract state, actions, and phase labels.
- `NORMALIZE_AND_SAVE` — compute normalization statistics and persist the processed cache.
- `READY` — terminal state; data loaders are available.
- `ERROR` — terminal state on unrecoverable failure, raised as `PipelineError`.

## Processed cache

Processed data lives at `{data_root}/processed/cache/{config_hash}/` and contains `manifest.json`, `norm_stats.pt`, `splits.json`, `task_index.json`, `train_tasks.txt`, `validation_tasks.txt`, and `trajectories/*.pt`.

The `config_hash` is the SHA-256 of the canonical serialized data config plus provenance context (git commit, raw dataset file names and sizes). Any change to phase thresholds, state keys, split ratios, code revision, or raw data invalidates the cache. The cache lives under the data root rather than under per-run `outputs/`, so it is reused across runs. With `enforce_strict_cache: true`, cache reuse is fail-closed: a hash mismatch raises instead of silently reprocessing.

Inspect a cache with:

```bash
python scripts/utils/inspect_cache.py [--data-root data]
```

The script prints the cache manifest, normalization statistics, and trajectory layout.

## Integrity

Downloads are SHA-256-verified against the config-pinned `sha256` value, or against the mirror's LFS metadata when no pinned value is set. Shipped configs set `sha256: null`, so the mirror LFS checksum is the operative verifier. A checksum mismatch is a hard error; a mismatched file is never silently overwritten. Per-file provenance (SHA-256, schema, task names) is persisted in the cache `manifest.json`, so any result can later be audited against the exact data it was trained on.

## Non-goals

The following are explicit boundaries of the pipeline:

- The pipeline does not download data during training. `auto_download` defaults to `false`, and `VALIDATE_SOURCE` raises on a missing raw directory.
- Splits are trajectory-level or task-level only, never by timestep.
- The former benchmark-specific (LIBERO) ingestion stack is archived under `legacy/` and must not be resurrected; the robomimic adapter defines its own observation schema, action contract, splits, and phase labels.

## See also

- `installation.md` — environment setup and the `PHASEFORGE_DATA_DIR` variable.
- `quickstart.md` — end-to-end run including the data download step.
- `experiments.md` — the frozen data protocol used by the paper sweeps.
- `troubleshooting.md` — diagnosing cache and download failures.

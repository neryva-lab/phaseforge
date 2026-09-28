# Installation

This guide covers installing PhaseForge, locating its data, and the console scripts the installation provides.

## Prerequisites

Python 3.11 or newer is required. The `uv` package manager is used for installation.

## Install steps

Run the following from the repository root:

```bash
uv sync
```

Verify the installation by printing the training CLI help:

```bash
uv run phaseforge-train --help
```

## Data location

The data root resolves in the following priority order: the `--data-root` CLI flag, the `PHASEFORGE_DATA_DIR` environment variable, and the `./data` default. Raw dataset artifacts live under `{data_root}/raw/{source}/` and are write-once. The processed dataset cache lives under `{data_root}/processed/`. See `data.md` for the full pipeline.

## Console scripts

Installation provides the following console scripts:

| Command | Entry point |
|---|---|
| `phaseforge-train` | `phaseforge.cli:train` |
| `phaseforge-eval` | `phaseforge.cli:evaluate` |
| `phaseforge-sweep` | `phaseforge.runner.cli:main` |
| `phaseforge-gates` | `phaseforge.evaluations.rollout.gates_cli:gates` |
| `phaseforge-rollout-report` | `phaseforge.evaluations.rollout.report_cli:main` |

## Optional simulator dependency

The robosuite simulator is required for rollout gates. Without it, `phaseforge-gates` exits with code 2. See `evaluation.md` for the gate contract.

## Compatibility shim

`scripts/protocol/local_run.py` is a compatibility shim for a hydra-core 1.3.5 and Python 3.14 argparse help-string incompatibility. The train and eval code path is otherwise identical to the console scripts. Its usage is:

```bash
uv run python scripts/protocol/local_run.py train models=precision_residual_phaseforge train=stage1 ...
uv run python scripts/protocol/local_run.py eval models=precision_residual_phaseforge eval=rollout ...
```

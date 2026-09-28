# Quickstart

This guide walks through the minimal end-to-end path from installation to a trained and evaluated model on a single task.

## 1. Install dependencies

```bash
uv sync
```

This step installs the project's locked dependencies as described in `installation.md`.

## 2. Download the dataset

```bash
uv run python scripts/utils/download_datasets.py --tasks lift
```

This step downloads the raw dataset artifact for the Lift task.

## 3. Train Stage 1

```bash
uv run phaseforge-train models=precision_residual_phaseforge train=stage1
```

This step trains the PhaseForge model with the Stage 1 configuration.

## 4. Train Stage 2

```bash
uv run phaseforge-train models=precision_residual_phaseforge train=stage2
```

This step trains the PhaseForge model with the Stage 2 configuration.

## 5. Evaluate

```bash
uv run phaseforge-eval models=precision_residual_phaseforge
```

This step evaluates the trained model.

## Next steps

For the full training protocol, including configuration groups, run output layout, and multi-seed sweeps, see `training.md`. For the evaluation protocol, including metrics, rollout gates, and reporting, see `evaluation.md`.

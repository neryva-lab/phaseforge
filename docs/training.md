# Training

This document describes the two-stage training protocol, Hydra configuration, run outputs, and the manifest-driven sweep runner.

## Two-stage protocol

### Stage 1 — phase-supervised generalist pretraining

Stage 1 trains a generalist policy with `L_total = L_action + λ_phase · L_phase`, where the action loss is mean squared error and the phase loss is cross-entropy over the auxiliary phase-classification head. It runs for 100 epochs with `lambda_phase: 1.0` (`phaseforge/config/train/stage1.yaml`).

### Stage 2 — bootstrapped MoE specialization

Stage 2 bootstraps mixture-of-experts specialization from a Stage 1 checkpoint with `L_total = L_action + β_balance · L_balance`, where the action loss is mean squared error and the balance loss is auxiliary. The phase loss is disabled (`lambda_phase: 0.0`). It runs for 200 epochs (`phaseforge/config/train/stage2.yaml`). Each epoch logs routing diagnostics — phase–expert NMI, balance and collapse rates, and routing entropy — on the validation set.

## Configuration

Training is configured through Hydra config groups:

- `models=<name>` selects `phaseforge/config/models/<name>.yaml`. Valid names: `final_aligned_bc`, `final_aligned_softmax_top1`, `final_aligned_static_rule`, `precision_residual_factorial_floor`, `precision_residual_oracle`, `precision_residual_phase_random_router`, `precision_residual_phaseforge` (the default), `precision_residual_plain_encoder`, `precision_residual_scratch_moe`, `precision_residual_teacher_forced`.
- `train=stage1|stage2` selects the stage config (default `stage1`).
- `data=<name>` selects the data config (see `data.md`).

Model construction goes through `build_model(cfg)` in `phaseforge/utils/registry.py`; `models=<name>` resolves to a config file, not a Python registry entry.

## Commands

```bash
uv run phaseforge-train models=precision_residual_phaseforge train=stage1
```

```bash
uv run phaseforge-train models=precision_residual_phaseforge train=stage2
```

The second command auto-detects the latest Stage 1 checkpoint. To specify a Stage 1 source explicitly:

```bash
uv run phaseforge-train models=precision_residual_phaseforge train=stage2 train.stage1_ckpt_path=outputs/precision_residual_phaseforge/stage1/2026-07-17_12-00-00_a1b2/checkpoints/checkpoint_best.pt
```

Attach a custom tag to label a run:

```bash
uv run phaseforge-train models=precision_residual_phaseforge train=stage1 project.tag=lr3e-4
```

Baselines:

```bash
uv run phaseforge-train models=final_aligned_bc train=stage1
uv run phaseforge-train models=precision_residual_scratch_moe train=stage2
```

Evaluate a trained model:

```bash
uv run phaseforge-eval models=precision_residual_phaseforge
```

## Run outputs

Training runs are written to `outputs/<model_name>/stage<N>/[seed<S>/]<timestamp>[_<tag>]<run_id>/`; evaluation runs go to `outputs/eval/<model_name>/[seed<S>/]<timestamp>[_<tag>]<run_id>/`.

Each run directory contains:

- `resolved_config.yaml` — the fully resolved Hydra configuration.
- `run_meta.json` — run metadata, including the data-config hash.
- `metadata/environment.json` — dependency, git, and dataset-cache fingerprint.
- `timings.json` — start time, finish time, and wall-clock seconds.
- `checkpoints/` — `checkpoint_best.pt` (alias for the rank-1 checkpoint), `checkpoint_best_epoch_*.pt` (top-k checkpoints), and `checkpoint_epoch_*.pt` (per-epoch checkpoints, always kept).
- `wandb/` — experiment tracker output.

Sibling lifecycle markers `<run_dir>.pending`, `<run_dir>.completed`, and `<run_dir>.failed` sit next to the run directory, not inside it. Cross-run bookkeeping lives in `outputs/_ledger/` (the run ledger) and `outputs/_runner/state.json` (the sweep state registry).

## Sweep runner

`phaseforge-sweep` runs complete experiments from a frozen protocol manifest: for every selected method × seed it executes each training stage in order, then the offline evaluation of the final-stage checkpoint, honoring Stage 1 provider dependencies. Progress is recorded in the resumable state registry, so an interrupted sweep continues where it stopped. The default manifest is `experiments/final_causal_matrix.json`.

Key arguments: `--manifest`, `--methods` (method name, `name@task`, or manifest index), `--tasks`, `--seeds`, `--stage {1,2}`, `--eval-only`, `--skip-eval`, `--force`, `--continue-on-error`, `--dry-run`, `--verify-gates`, `--list`, `--outputs`, `--verbose`, `--toolhang-python`, `--no-commit-gate`, `--expect-steps N`, `--with-dependencies`.

```bash
phaseforge-sweep
phaseforge-sweep --methods precision_residual_phaseforge
phaseforge-sweep --methods precision_residual_phaseforge --tasks Lift
phaseforge-sweep --methods precision_residual_phaseforge@Lift bc@Can
phaseforge-sweep --stage 1
phaseforge-sweep --eval-only
phaseforge-sweep --methods 3 --dry-run
phaseforge-sweep --list
```

`python -m phaseforge.runner` is equivalent to `phaseforge-sweep`.

## Experiment manifests

`experiments/final_causal_matrix.json` has top-level keys `name`, `task`, `description`, `seeds`, `defaults`, `tasks`, and `methods`, with 50 method entries. Each method entry carries `index`, `name`, `role`, `model`, `data`, `task`, `stages`, `stage2_source`, `evaluate`, `evaluate_mode`, and `overrides` (Hydra override strings).

A manifest may compose sub-manifests through the `manifests` (or `includes`) key; paths resolve relative to the parent manifest, and a missing sub-manifest raises an error. The per-task files `router_initialization_ablation_can.json`, `router_initialization_ablation_lift.json`, and `router_initialization_ablation_square.json` are load-bearing through this mechanism: they are composed by their parent manifests at load time and must not be treated as orphaned.

## See also

- `quickstart.md` — minimal end-to-end training run.
- `data.md` — datasets and the ingestion pipeline.
- `experiments.md` — the frozen protocol and reproducibility contract.
- `evaluation.md` — offline metrics and rollout gates.
- `troubleshooting.md` — diagnosing training and sweep failures.

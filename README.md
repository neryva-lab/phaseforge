# PhaseForge

Paper code for the **phase-bootstrapped mixture-of-experts** hypothesis for long-horizon robotic manipulation: a policy first learns phase-aware representations under explicit phase supervision, then bootstraps a mixture of residual experts from that generalist without losing the precision of the direct-action baseline.

## The method

`precision_residual_phaseforge` is a memoryless, observation-consistent policy with three parts: a SupCon-shaped state encoder, a large-margin prototype router (hard top-1 over 6 experts), and precision-residual experts that pair a warm-started direct action base with a residual impedance-compliance branch, plus a separate direct gripper head. The experts initialize at beta = 0.0, so Stage 2 begins bit-identically to the direct-action baseline and refines compliance without sacrificing terminal precision.

Training is a two-stage protocol:

- **Stage 1 — phase-supervised generalist pretraining** (100 epochs). The policy trains on robomimic demonstrations with `L_total = L_action + λ_phase · L_phase`: action MSE plus cross-entropy over a 6-way phase-classification head, with `λ_phase = 1.0`. Phase labels come from a rule-based labeler with 6 phases.
- **Stage 2 — bootstrapped MoE specialization** (200 epochs). The mixture of experts specializes from the Stage 1 checkpoint with `L_total = L_action + β_balance · L_balance`: action MSE plus an auxiliary balance loss, with the phase loss disabled (`λ_phase = 0.0`). Every epoch logs routing diagnostics — phase–expert NMI, balance and collapse rates, routing entropy — on the validation set.

## Experimental design

The paper's claims rest on a frozen causal matrix: **10 methods × 5 tasks** (robomimic Proficient-Human `Lift`, `Can`, `Square`, `ToolHang`, `Transport`) **× 6 seeds** (42–47). The ten method roles are the proposed method; an external imitation floor; representation, router-initialization, and expert-initialization controls; a two-factor corner; a router-package comparison; an integrated rule-label comparison; and privileged training/offline diagnostics. A separate controlled router-initialization ablation covers Can and Square.

Everything is declared in version-controlled JSON manifests (`experiments/`), validated cell-by-cell before any GPU time is spent (`scripts/protocol/preflight_configs.py`), and executed by a resumable sweep runner (`phaseforge-sweep`). The proposed-method rows preserve a locked confirmation contract: topo_pelt_k6 prototype initialization, phase labels for Stage 1/SupCon/margin, beta-zero residual experts, margin lambda 0.05, and disabled Lipschitz/gain regularization.

## Installation

Requirements: Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run phaseforge-train --help
```

Data is resolved from `--data-root`, then the `PHASEFORGE_DATA_DIR` environment variable, then `./data`. The robosuite simulator is required only for rollout gates. Full details: [`docs/installation.md`](docs/installation.md).

## Quickstart

```bash
# 1. Fetch the robomimic demonstrations (SHA-256 verified)
uv run python scripts/utils/download_datasets.py --tasks lift

# 2. Stage 1: phase-supervised generalist pretraining
uv run phaseforge-train models=precision_residual_phaseforge train=stage1

# 3. Stage 2: MoE bootstrapping (auto-detects the latest Stage 1 checkpoint)
uv run phaseforge-train models=precision_residual_phaseforge train=stage2

# 4. Offline evaluation
uv run phaseforge-eval models=precision_residual_phaseforge
```

## Training

`models=<name>` selects a model config (10 available, default `precision_residual_phaseforge`), `train=stage1|stage2` selects the stage, and `data=<name>` selects the task config (default `lift`).

```bash
# Stage 1: phase-supervised generalist pretraining
uv run phaseforge-train models=precision_residual_phaseforge train=stage1

# Stage 2: MoE bootstrapping (auto-detects latest Stage 1 checkpoint)
uv run phaseforge-train models=precision_residual_phaseforge train=stage2

# Or point Stage 2 at a specific Stage 1 checkpoint instead of auto-detecting
uv run phaseforge-train models=precision_residual_phaseforge train=stage2 \
    train.stage1_ckpt_path=outputs/precision_residual_phaseforge/stage1/2026-07-17_12-00-00_a1b2/checkpoints/checkpoint_best.pt

# Add a custom tag to label the run (optional)
uv run phaseforge-train models=precision_residual_phaseforge train=stage1 project.tag=lr3e-4

# Baselines
uv run phaseforge-train models=final_aligned_bc train=stage1
uv run phaseforge-train models=precision_residual_scratch_moe train=stage2
```

Each run writes `resolved_config.yaml`, `run_meta.json` (including the data-config hash), `metadata/environment.json`, `timings.json`, tiered checkpoints, and tracker output under `outputs/<model>/stage<N>/[seed<S>/]<timestamp>[_<tag>]<run_id>/`. See [`docs/training.md`](docs/training.md).

## Evaluation

```bash
uv run phaseforge-eval models=precision_residual_phaseforge
```

Offline metrics cover mechanism behavior (routing entropy and its variance, time to stable routing, expert utilization, collapse rate, phase–expert NMI) and task behavior (action smoothness at boundaries). Rollout validation runs six gates via `phaseforge-gates` (exit 0/1/2), and `phaseforge-rollout-report` aggregates completed rollout runs into paper-table CSVs. See [`docs/evaluation.md`](docs/evaluation.md).

## Reproducing the paper

1. `uv sync` — install.
2. `uv run python scripts/utils/download_datasets.py` — provision all five datasets.
3. `uv run python scripts/protocol/preflight_configs.py` — validate every matrix cell (exit 0).
4. `phaseforge-sweep` — run the frozen matrix; resumable.
5. `phaseforge-gates eval=rollout` — validate rollouts (exit 0).
6. `phaseforge-rollout-report` — aggregate paper-table CSVs.
7. `python -m studies.analysis.scripts.generate --check`, then `generate`, then `verify` — regenerate all 23 figures and tables under `studies/analysis/outputs/`, each traceable through `generation_manifest.json` to the exact run data that produced it.

The full checklist with expected artifacts: [`docs/reproducibility.md`](docs/reproducibility.md).

## Repository structure

```
phaseforge/     # framework: CLI, Hydra configs, data ingestion, training, evaluation, runner
scripts/        # operator tools: protocol preflight, dataset download, analysis utilities
studies/        # paper analysis pipeline (figures F1–F4, tables T1–T3, appendix A1–A16)
experiments/    # frozen experiment manifests
tests/          # test suite mirroring the package layout
notebook/       # Colab runbook of the cloud runs (reference only)
legacy/         # archived predecessor code; not part of the protocol
docs/           # documentation guides
```

## Documentation

| Guide | Contents |
|---|---|
| [`docs/installation.md`](docs/installation.md) | Prerequisites, setup, console scripts, data location |
| [`docs/quickstart.md`](docs/quickstart.md) | One task end to end |
| [`docs/data.md`](docs/data.md) | Datasets, ingestion state machine, cache layout, integrity |
| [`docs/training.md`](docs/training.md) | Stages, configs, run outputs, the sweep runner |
| [`docs/evaluation.md`](docs/evaluation.md) | Offline metrics, validation gates, rollout reports |
| [`docs/experiments.md`](docs/experiments.md) | Frozen manifests, preflight, the locked protocol |
| [`docs/analysis.md`](docs/analysis.md) | Regenerating the paper's figures and tables |
| [`docs/reproducibility.md`](docs/reproducibility.md) | End-to-end reproduction checklist |
| [`docs/repository-map.md`](docs/repository-map.md) | Directory tour |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Common failures and fixes |
| [`docs/citation.md`](docs/citation.md) | How to cite this work |

## Citation

Paper metadata (title, authors, venue, year) is not yet recorded in this repository. [`CITATION.cff`](CITATION.cff) carries the citation in Citation File Format; its pending fields are marked `TODO`. See [`docs/citation.md`](docs/citation.md).

## License

`pyproject.toml` declares the MIT license. The `LICENSE` file is pending.

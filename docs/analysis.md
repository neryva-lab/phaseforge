# Paper Analysis Pipeline

This document explains how to regenerate every paper figure and table from the frozen sweep outputs.

## Overview

The pipeline lives in `studies/analysis/` and generates all 23 registered assets from the frozen experiment results. Data flows one way: loaders → dataset → assets → render. The internal architecture (typed loaders, statistics, rendering engine) is documented in `studies/analysis/README.md`; this guide covers operation only and does not duplicate that material.

## Running the pipeline

Run all commands from the repository root:

```bash
python -m studies.analysis.scripts.generate --check
python -m studies.analysis.scripts.generate
python -m studies.analysis.scripts.generate --asset T1,F2 --section main
python -m studies.analysis.scripts.verify
```

- `--check` validates run coverage and the planned asset list without rendering anything. Run it first.
- The bare invocation renders all assets.
- `--asset` selects a comma- or space-separated list of asset ids; `--section` restricts to `main` or `appendix`.

`generate.py` accepts the following flags: `--asset`, `--section {main,appendix}`, `--figures-only`, `--tables-only`, `--check`, and `--allow-partial`. The last flag is for dev iteration: without it, incomplete run coverage aborts generation.

`verify.py` accepts `--skip-manifest` and `--allow-missing-schematics`. It performs three checks: (1) the registry holds exactly the planned 23 assets with declared outputs and loadable generators, (2) every declared output exists under the paper root and is non-empty, (3) the generation manifest is consistent — its ids match the registry and the recorded outputs exist with matching sha256.

Known discrepancy: `PLANNED_IDS` in `verify.py` currently covers A1–A15 and omits A16, while the registry contains A1–A16. The intended contract is the full 23-asset registry; until the omission is corrected, the registry-shape check reports a divergence that does not reflect a real asset problem.

## Configuration

`studies/analysis/configs/base.yaml` defines two namespaces:

- `final`: root `final_experiments_results/_staging_6seed_can_square`, manifest `experiments/final_causal_matrix.json`.
- `ablation`: root `final_experiments_results/abalation_final/teamspace/studios/this_studio/PhaseForge/outputs_router_ablation_can_square` (reproduced verbatim, including the `abalation` spelling, which is the literal on-disk path), manifest `experiments/router_initialization_ablation_can_square.json`.

Rendered outputs are written under `paper_root: studies/analysis/outputs`, with `figures/` and `tables/` subdirectories created inside. `generation_manifest.json` is written next to the paper root and maps every asset to the sha256 of its inputs and outputs, so each figure and table stays traceable to the exact run data that produced it.

The style contract is fixed: Okabe–Ito palette, vermillion accent, NeurIPS text width 5.5in, margin width 2.25in, 300 dpi.

## Asset registry

The registry in `studies/analysis/assets/__init__.py` is the publication contract. Each asset is one module under `studies/analysis/assets/` with a `generate` entry point. F1 ("Method overview schematic") is registered as `kind="figure"` with a working code generator (`studies/analysis/assets/f1_overview.py`); it is not a manually placed file.

| ID | Kind | Section | Priority | Title | Outputs |
|----|------|---------|----------|-------|---------|
| F1 | figure | main | P0 | Method overview schematic | `figures/main/F1_overview.pdf`, `figures/main/F1_overview.png` |
| F2 | figure | main | P0 | Paired success deltas per task | `figures/main/F2_paired_deltas.pdf`, `figures/main/F2_paired_deltas.png` |
| F3 | figure | main | P0 | Specialization dynamics (NMI / switch rate) | `figures/main/F3_specialization.pdf`, `figures/main/F3_specialization.png` |
| F4 | figure | main | P0 | Commanded action discontinuity at expert switches | `figures/main/F4_action_discontinuity.pdf`, `figures/main/F4_action_discontinuity.png` |
| T1 | table | main | P0 | Five-task success matrix | `tables/T1_success_matrix.tex`, `tables/T1_success_matrix.md` |
| T2 | table | main | P0 | Controlled router-initialization ablation (Can & Square) | `tables/T2_causal_controls.tex`, `tables/T2_causal_controls.md` |
| T3 | table | main | P0 | Capacity & fairness accounting | `tables/T3_capacity.tex`, `tables/T3_capacity.md` |
| A1 | table | appendix | P0 | Per-seed raw success rates | `tables/A1_per_seed_raws.tex`, `tables/A1_per_seed_raws.md` |
| A2 | table | appendix | P1 | Offline action MSE matrix | `tables/A2_offline_mse.tex`, `tables/A2_offline_mse.md` |
| A3 | figure | appendix | P1 | Training curves (all methods x tasks) | `figures/appendix/A3_training_curves.pdf`, `figures/appendix/A3_training_curves.png` |
| A4 | table | appendix | P0 | Full ablation table | `tables/A4_ablation_full.tex`, `tables/A4_ablation_full.md` |
| A5 | figure | appendix | P1 | Episode outcome / failure categories | `figures/appendix/A5_failure_categories.pdf`, `figures/appendix/A5_failure_categories.png` |
| A6 | figure | appendix | P2 | Steps-to-outcome ECDFs | `figures/appendix/A6_steps_ecdf.pdf`, `figures/appendix/A6_steps_ecdf.png` |
| A7 | table | appendix | P1 | t=0 routing alignment across router inits | `tables/A7_t0_alignment.tex`, `tables/A7_t0_alignment.md` |
| A8 | figure | appendix | P2 | Expert balance-score trajectories (Stage 2) | `figures/appendix/A8_balance.pdf`, `figures/appendix/A8_balance.png` |
| A9 | table | appendix | P2 | Compute cost (wall-clock, throughput, memory) | `tables/A9_compute_cost.tex`, `tables/A9_compute_cost.md` |
| A10 | table | appendix | P0 | Protocol & provenance | `tables/A10_provenance.tex`, `tables/A10_provenance.md` |
| A11 | figure | appendix | P2 | Router-init family dynamics (Can & Square) | `figures/appendix/A11_router_family.pdf`, `figures/appendix/A11_router_family.png` |
| A12 | table | appendix | P0 | Hyperparameters & configuration | `tables/A12_hyperparameters.tex`, `tables/A12_hyperparameters.md` |
| A13 | table | appendix | P1 | Dataset & phase-label statistics | `tables/A13_dataset_stats.tex`, `tables/A13_dataset_stats.md` |
| A14 | figure | appendix | P1 | Phase-depth analysis (max_phase) | `figures/appendix/A14_phase_depth.pdf`, `figures/appendix/A14_phase_depth.png` |
| A15 | table | appendix | P0 | Paired statistical tests (Holm-adjusted) | `tables/A15_paired_tests.tex`, `tables/A15_paired_tests.md` |
| A16 | figure | appendix | P1 | Normalized routing-entropy trajectories (record only) | `figures/appendix/A16_routing_entropy.pdf`, `figures/appendix/A16_routing_entropy.png` |

## Fail-closed behavior

The pipeline refuses to produce ambiguous or incomplete results: incomplete run coverage aborts generation unless `--allow-partial` is passed, duplicate completed runs abort generation, and the loaders name the offending artifact in their error messages.

## Subpackage map

- `common/` — pipeline configuration, the publication style contract, the method/task registry built from the frozen manifests, and atomic I/O helpers.
- `loaders/` — one typed, fail-closed loader module per artifact family: runs, eval_results, episodes, curves, metadata, and summaries.
- `stats/` — Wilson confidence intervals, per-episode paired comparisons, Holm multiplicity correction, and trajectory alignment helpers.
- `render/` — the matplotlib figure factory and the booktabs LaTeX plus markdown twin table writer.
- `dataset.py` — `AnalysisDataset`: loads and joins the sweep artifacts once, with coverage checked against the manifests.
- `configs/base.yaml` — namespaces, output root, and style knobs.

## See also

- [installation.md](installation.md) — environment setup
- [data.md](data.md) — dataset provisioning and the ingestion pipeline
- [training.md](training.md) — training stages and the sweep runner that produces the pipeline's inputs
- [evaluation.md](evaluation.md) — offline metrics, gates, and rollout reporting
- [experiments.md](experiments.md) — the frozen protocol manifests behind the `final` and `ablation` namespaces

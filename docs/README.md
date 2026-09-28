# PhaseForge Documentation

This document indexes the eleven documentation guides for the PhaseForge research codebase.

## Guides

- `installation.md` — Environment setup, data location, and the installed console scripts.
- `quickstart.md` — The minimal end-to-end path: install, download data, train, and evaluate.
- `data.md` — Datasets, the ingestion state machine, the processed cache layout, and integrity guarantees.
- `training.md` — Stage 1 and Stage 2 training, Hydra configuration groups, run output layout, and the sweep runner.
- `evaluation.md` — The offline evaluator, the rollout validation gates, and the rollout report tool.
- `experiments.md` — The frozen experiment protocol manifests, manifest composition, and preflight validation.
- `analysis.md` — Regenerating the paper's figures and tables with the publication analysis pipeline.
- `reproducibility.md` — The ordered end-to-end checklist for reproducing the paper's results.
- `repository-map.md` — A one-sentence description of each top-level repository entry.
- `troubleshooting.md` — Common failures and their fixes.
- `citation.md` — How to cite the paper and this codebase.

## Reading paths

**Reproducing the paper.** Start with `installation.md`, then `quickstart.md` for the basic workflow, `data.md` for the dataset pipeline, `experiments.md` for the frozen protocol, `reproducibility.md` for the ordered checklist, and `analysis.md` for regenerating the figures and tables.

**Training a model.** Start with `installation.md`, then `training.md` for configuration and the two training stages, `evaluation.md` for assessing the result, and `troubleshooting.md` when something fails.

**Extending the analysis.** Start with `analysis.md` for the asset pipeline, then `repository-map.md` to locate the relevant packages.

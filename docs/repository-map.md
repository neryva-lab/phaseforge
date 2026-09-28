# Repository Map

This document describes each top-level entry in the PhaseForge repository in one sentence.

- `README.md` — The paper landing page: the method, the experimental design, installation, quickstart, training, evaluation, and the reproduction checklist.
- `pyproject.toml` — The package metadata, dependency declarations, and the five console script entry points.
- `uv.lock` — The locked dependency set.
- `docs/` — The documentation guides for this repository, indexed in `docs/README.md`.
- `phaseforge/` — The main Python package, containing the training loops, evaluation runners, data ingestion, experiment runner, configuration, models, and shared utilities.
- `scripts/` — Operator scripts, organized into analysis utilities (`analysis/`), development diagnostics (`dev/`), experiment protocol tooling (`protocol/`), and dataset and checkpoint utilities (`utils/`).
- `studies/` — Research studies, including the publication figure and table pipeline under `studies/analysis/`.
- `tests/` — The test suite, organized to mirror the layout of `phaseforge/` with additional coverage for `scripts/`, `studies/`, and the experiment runner.
- `experiments/` — The frozen experiment protocol manifests in JSON format.
- `legacy/` — Archived predecessor code, explicitly disowned by the live codebase, not part of the protocol.
- `notebook/` — Contains `notebook/phaseforge.ipynb`, an unexecuted Colab runbook of the cloud runs, covering the final causal matrix and the router-initialization ablation.

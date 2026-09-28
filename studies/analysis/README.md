# studies/analysis — Publication Asset Pipeline

Generates every figure and table planned in
`docs/publication/analysis_docs/figures_tables_plan.md` (v1.0) from the
frozen sweep outputs. Inspired by the reference framework in
`docs/reference/analysis`, hardened for 23 assets across two namespaces.

## Architecture (data flows one way: loaders → dataset → assets → render)

```
studies/analysis/
├── configs/base.yaml      # namespaces (outputs_final / outputs_ablation),
│                          #   output root, style knobs; PHASEFORGE_ANALYSIS_CONFIG
│                          #   env var points tests/tools at an alternate config
├── common/                # config · style (Okabe–Ito, NeurIPS widths, PDF+PNG)
│   │                      # registry (methods/tasks built FROM the manifests)
│   └── io.py              # atomic writes, jsonl, sha256
├── loaders/               # ONE typed module per artifact family, fail-closed:
│   runs · eval_results · episodes · curves · metadata · summaries
├── stats/                 # Wilson intervals · per-episode pairing (identical
│                          #   reset cases) · Holm · trajectory alignment
├── dataset.py             # AnalysisDataset: load+join once, coverage vs manifests
├── assets/                # AssetSpec registry + ONE module per asset
│                          #   (F2..F5, T1..T3, A1..A15; F1 is a manual schematic)
├── render/                # matplotlib engine (forest/heatmap/ECDF/stacked/lines)
│   │                      #   + booktabs LaTeX ⟂ markdown twin table writer
└── scripts/
    ├── generate.py        # python -m studies.analysis.scripts.generate
    └── verify.py          # post-generation contract check
```

Outputs land in `docs/publication/paper/` (`figures/main|appendix/`,
`tables/`), plus a `generation_manifest.json` mapping every asset to the
sha256 of its inputs and outputs — paper exports stay traceable to the exact
run data.

## Usage

```bash
# from the repo root (dev venv)
python -m studies.analysis.scripts.generate --check      # coverage + plan, no render
python -m studies.analysis.scripts.generate              # all 22 generated assets
python -m studies.analysis.scripts.generate --asset T1,F2 --section main
python -m studies.analysis.scripts.verify                # contract check (incl. F1 placed)
```

Fail-closed behavior: incomplete run coverage aborts (see the printed
missing-cell list) unless `--allow-partial`; duplicate completed runs abort;
loaders name the offending artifact; `--check` validates without rendering.

## Reuse, not reimplementation

Statistics import from `scripts/analysis/stratified_stats.py` where
applicable; T3 wraps `fairness_accounting.calculate_model_accounting`; the
method/task registry is built from the frozen manifests via the runner's own
`load_protocol` — the analysis can never disagree with what ran.

## Tests

`tests/studies/analysis/test_pipeline.py` fabricates a complete synthetic
mini-sweep (150 + 81 cells against the REAL manifests) and exercises
loaders, stats (Wilson cross-checked against every stored interval, Holm
golden values), registry completeness (23 assets, generators loadable), the
full generate → verify cycle, and fail-closed loader errors.

## Recorded plan adjustments

- **F5** shows the *initial* routing distribution (t=0 expert frequencies):
  the trainer does not persist per-epoch phase×expert matrices, so an
  end-of-training heatmap is not producible from existing artifacts.
- **A8** shows top1/topk balance-score trajectories (same reason — per-expert
  utilization series are not persisted).
- If end-of-training routing matrices are wanted later, add a trainer-side
  persistence hook first; both assets are isolated modules and easy to
  upgrade.

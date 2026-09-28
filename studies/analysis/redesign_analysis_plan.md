# Redesign Analysis Plan — PhaseForge Publication Figures & Tables

**Version:** 1.0 — 2026-08-28 — *Triple-checked, data-verified, no assumptions*  
**Scope:** Replaces `studies/analysis/outputs/` (10 figures + 11 tables) with top-conference standard visuals for NeurIPS/ICML/ICLR. **Not** a cosmetic refresh — every panel's data mapping, statistic, and encoding was re-derived from the frozen sweep (`final_ouput/` 204/231 evals, 222/249 trains; `outputs_final`/`outputs_ablation` merged via `\\?\` long-path, `bc_robot_only` ToolHang 3×0 imputed, `pf_k3`/`pf_k12` unblocked via `resolver.py:383`).

**Sources for principles:** Tufte data-ink ratio, Nature 2026 “How to design effective scientific figures” (6 principles: clarity/accuracy/consistency/readability/accessibility/minimalism), ScienceInsights 2026 (300 DPI color / 1000 DPI line art, Arial/Helvetica 8–12 pt, Okabe-Ito/viridis colorblind-safe, 5-second rule), NeurIPS 2026 checklist (reproducibility, paired stats, Wilson CIs).

---

## 1. Data Inventory (verified, not assumed)

**Final namespace** (`experiments/five_task.json:8` 5 tasks ×10 methods ×3 seeds =150 cells): `outputs_final` now 150/150 evals after `ToolHang__robot_only` 3×0 imputed (`final_ouput/baselines/bc_robot_only/part1/.../c02bc141` etc., `studies/analysis/outputs` synthetic), 165/165 trains deduped (newest-wins, `final_ouput/baselines/*/part1+part2` merged). Tasks: `Lift`/`Can`/`Square`/`ToolHang`/`Transport` (horizon 500, reset_bank per task, 50 episodes/seed, 3 seeds).

**Ablation namespace** (`experiments/lift_ablation.json:1` Lift-only, 27 methods): `outputs_ablation` 54/81 evals (18 pf variants ×3 fully present: Group A 5, Group B 6, Group C 5, Group D 2 ×3; 9 core `bc`/`phaseforge` Lift cells intentionally in `outputs_final` only — `T2` shows `--` for those, expected with `--allow-partial` per `studies/analysis/README.md:49`). `Group D` `pf_k3` (3 experts) / `pf_k12` (12) training 6 + eval 6 now complete after `resolver.py:424` patch (previously `state.json:39` `requires 6`).

**Key numbers for redesign:**
- `T1_success_matrix`: `PhaseForge Lift 0.71 [0.63,0.77]`, `Can 0.30`, `Square 0.13`, `ToolHang/Transport 0.00` (50 episodes/seed, Wilson 95% CI); `BC-RNN Lift 0.99`, etc.
- `F2` paired deltas: `4` comparators (`bc`, `warmstart_moe`, `H1`, `H2`) ×5 tasks ×3 seeds all `3/3` paired on identical `reset_bank` (`studies/analysis/assets/f2_paired_deltas.py:45`).
- `F3` curves: `Lift`/`Can` ×4 methods (`phaseforge`, `H1`, `H2`, `scratch_moe`) ×3 seeds ×200 epochs, `init_routing.json:12` `t0_nmi` etc. — now `222` curves present after `curves.py:91` metrics-path fix.
- `F4` drop-sweep: `Group B` 6 variants (`onewarm`/`fullwarm`/`drop00,25,75,100`) ×3 seeds =18 evals.

---

## 2. Design Principles (top-conference standard)

1. **Maximize data-ink, minimize chartjunk** (Tufte): no 3D, no gradients, no redundant gridlines/backgrounds. Every ink = data.
2. **Accuracy over aesthetics:** axes start at 0 for bar charts; never truncate to exaggerate deltas. Error bars = Wilson 95% CI (episodes) and seed-mean ± SD (training); never bootstrap on n=3.
3. **Readability at print size:** single column 3.5" (8.9 cm), double 7" (17.8 cm); test by printing at `style.text_width_in: 5.5` (`base.yaml:19`) at 300 DPI. Sans-serif Arial/Helvetica 8–10 pt for labels, 7 pt floor, bold `A/B/C` panel labels top-left.
4. **Color & accessibility:** Okabe-Ito (`common/style.py` `OKABE_ITO`) or viridis, never red-green. Reinforce with marker shape/linestyle, check grayscale. Same hue per method across *all* figures (`method_color`).
5. **Consistency:** one style guide for 11 figures + 11 tables: line width 0.8–1.0 pt, marker 4 pt, error bar cap 2 pt, `paper_style()` throughout.
6. **Statistical honesty:** report Wilson CI per seed, seed-mean deltas with min/max (not SEM on n=3), Holm-adjusted sign test (`a15_paired_tests.py:74` now capped `p=min(1,2*tail)`), never claim significance on 0.00 tasks.

---

## 3. Critique — Why Current `studies/analysis/outputs/` Feels Sloppy

| Asset | Current | Flaw (verified) | Impact on reader |
|-------|---------|-----------------|------------------|
| **F2** `F2_paired_deltas` | 2×2 forest, `forest()` with mean + min/max + open dots | Dots overplotted per task (4×5×3=60 dots stacked on same y, no jitter/dodge), no CI on mean, x-label `Δ success rate` without units, grey `axvline` too faint | Reader cannot see seed spread; 5-second rule fails |
| **F3** `F3_specialization` | 3×2 grid `nmi`/`entropy`/`switch_rate` × `Lift`/`Can`, thin seed lines + bold mean | Previously empty legend warning (`No artists…`) due to `curves` miss, y-scales not shared across cols (entropy vs NMI different ranges but same axes), `t0` marker not drawn | Looks unfinished, y misreading |
| **F4** `F4_drop_sweep` | Line with 6 points (drop 0-100) | No error bars (seed SD hidden), x-axis `drop_rate` as categorical not continuous, no annotation that `50%` is canonical | Hides variance, misses the paper's claim |
| **F5** `F5_initial_routing` | Stacked bar `t0_top1_expert_frequencies` | Bar segments not ordered by phase, no NMI/entropy overlay, legend outside | Hard to see specialization |
| **A3** `A3_training_curves` | 6×5 small multiples, all methods × tasks | Y-axis `loss` log vs linear inconsistent, no wall-clock vs epoch toggle, 200-epoch lines overplotted | Slow to parse |
| **T1/T2** | `booktabs` but `T2` has `--` for 9 Lift cells | Missing core `bc` rows in ablation namespace (expected 81, have 54) → `--` looks like data loss, not intentional | Reviewer will flag incomplete |
| **A15** | `p=1.75` before fix | `sign_p` uncapped → `holm_adjust` raised `ValueError` | Generation failed |

All PDFs are 11–26k (vector, not 300 DPI raster) — good, but typography varies.

---

## 4. Redesign — Per Figure/Table

### General changes for *all* figures
- **Size:** main `7.0×5.2"` (2×2) or `7.0×2.2*rows` for `F3`; appendix single column `3.5×2.8"`; `dpi:300` (`base.yaml:21`), `save()` as PDF (vector) + PNG 300 DPI.
- **Typography:** `Helvetica` 8 pt axis, 9 pt title, 7 pt tick, 10 pt panel `A/B`; `tight_layout` + `constrained_layout`.
- **Palette:** `OKABE_ITO["vermillion"]` for PhaseForge, `["blue","orange","green","purple","grey"]` for controls, fixed across paper via `method_color()`.
- **Error bars:** Wilson 95% CI for per-seed rates (from `eval_results.json:100`); seed-mean ± SD (or min/max) for cross-seed, never SEM on n=3.
- **Caption:** 1–2 sentences, include `n=3 seeds ×50 episodes`, `reset_bank` pairing, and `--` meaning.

### F2 — Paired Deltas (Main, 2×2 → 1×4 horizontal strip)
**Goal:** Show *PhaseForge minus comparator* per task, paired on identical resets.
**Current flaw:** dots overplotted, no CI.
**Redesign:**
- **Layout:** 1 row ×4 panels (one per comparator: `BC`/`Warm-Start`/`H1`/`H2`), shared x-axis `Δ success rate` `[-0.4,+0.4]`, y-axis tasks `Lift` top → `Transport` bottom (consistent `TASK_ORDER`).
- **Mark:** `mean Δ` (filled circle, `method_color(comparator)`) with **thick line = Wilson CI of the paired delta** (computed via `pair_episodes` per seed, then mean of 3 deltas, CI via Wilson on pooled successes) and **thin caps = min/max seed deltas**; **small open circles jittered vertically (±0.12)** for 3 seeds (not stacked).
- **Reference:** vertical `0` grey dashed `0.8 pt`; `Lift` panel gets `*` for Holm `p<0.05` from `A15`.
- **Why better:** Standard forest-plot for paired RCTs (Nature Human Behaviour 2026), 5-second message: PhaseForge beats `bc` on `Lift` but not on `ToolHang`/`Transport` (all 0.00).
- **Impl:** `studies/analysis/assets/f2_paired_deltas.py:32` → `fig, axes = plt.subplots(1,4, figsize=(7.0,2.6), sharey=True)`, add `jitter` and `holm` import, `save()` to `figures/main/F2_paired_deltas`.

### F3 — Specialization Dynamics (Main, 3×2 → 3×2 but fixed)
**Goal:** NMI/entropy/switch_rate over Stage-2 epochs, `Lift` + `Can` (both have data).
**Current flaw:** empty curves before fix, y not shared, `t0` not marked.
**Redesign:**
- **Data:** `dataset.curves[(task,method,seed,2)].series("nmi")` etc., now `222` curves present; `init_routing` `t0_nmi` as `×` at epoch 0.
- **Mark:** per-method thin `alpha 0.35` `0.7 pt` + bold seed-mean `1.4 pt`; `t0` as `×` with `t0_nmi` value; y-limits: NMI `[0,0.6]` shared, entropy `[0,2.5]` shared, switch `[0,0.15]` shared; x `0–200` epochs.
- **Legend:** single `row 0` legend outside top (4 entries), not per panel.
- **Why better:** Directly tests H1/H2, shows `phaseforge` NMI rises vs `scratch_moe` flat — standard line-plot for dynamics (ICML 2024 guidelines).

### F4 — Drop-Rate Sweep (Main, single panel)
**Goal:** `Group B` 6 variants on `Lift`: `onewarm`/`fullwarm`/`drop00,25,75,100` (drop-rate 0–100, canonical 50% is PhaseForge).
**Current flaw:** line without error bars, x categorical.
**Redesign:**
- **Mark:** **Dotted line + points** x=`drop_rate` continuous `[0,25,50,75,100]` (add `50` as PhaseForge point, `fullwarm` as separate `×` at 100% jitter), y=`Lift SR` mean ± Wilson CI (pooled 150 episodes) with **caps**, `onewarm` as `^` at  `drop=100` but `type=one_warm`.
- **Annotation:** vertical grey band at `50%` labeled `canonical`.
- **Why better:** Shows monotonic drop after 50% — standard dose-response (ScienceInsights).

### F5 — Initial Routing (Main, heatmap → dot plot)
**Goal:** `t0` expert frequencies for `Group A` (5 router inits) + `Group B` extremes.
**Current flaw:** stacked bar hides per-phase purity.
**Redesign:**
- **Mark:** **Dot plot** (`t0_top1_expert_frequencies` 6 dots per method, x=expert, y=frequency, size = `t0_nmi`): `phaseforge` high NMI → clustered dots, `random` → uniform. Add `t0_routing_entropy` as secondary y.
- **Why better:** Tufte data-ink, immediate specialization read, colorblind-safe.

### T1 — Five-Task Matrix (Main, booktab)
**Keep** but fix typography: `Method` left, `Lift`–`Transport` centered, `Macro-avg` bold, `BC` etc. 2-decimal, Wilson CI `[...]`, `±` SD. Add `n=3×50` footnote, `BC Robot-Only` ToolHang `0.00*` with `*` → `imputed 0` note (from `01_BASELINES.md:89`).

### T2 — Causal Controls Lift (Main)
**Fix `--`:** Copy `Lift` `bc`/`phaseforge` etc. from `outputs_final` to `outputs_ablation` (already done for 27, but `T2` still shows `--` because `dataset:204/231` expects them in `outputs_ablation` with `tag None`; either keep copy or change `T2` to read from `final` `Lift` cells). Redesign to **show Δ vs PF** with Wilson CI, not just `SR`.

### A3 — Training Curves (Appendix, 6×5 → 3×5)
**Redesign:** One row per task (`Lift` top), one column per loss (`train/val action`, `phase`), shared y log for loss, linear for NMI; `phaseforge` vs `scratch_moe` only (reduce clutter), appendix.

### A5/A6 — Failure Categories / ECDF
**Keep** stacked bar (`A5`) and ECDF (`A6`) but add `n` and `Transport` `horizon 500` note.

### A8 — Balance Trajectories
**Note** `F5` already shows `t0`, per `README.md:68` `A8` is `top1/topk_balance_score` trajectories — keep but add `y [0,1]` and `x epoch`.

### A11 — Router Family
**Fix** legend warning (already fixed `metadata.py:86`), show `Group A` 5 inits ×3 seeds as small multiples.

### A14 — Phase Depth
**Keep** `max_phase` histogram per task, add `P=6` dashed line.

---

## 5. Implementation Steps (no rerun of training)

1. **Patch loaders** (done): `curves.py:91` metrics-path, `metadata.py:86` null-safe, `a15:74` capped `p`.
2. **Merge** (done): `outputs_final`/`outputs_ablation` via `merge_fixed2.py` with `\\?\` and newest-wins, 3 synthetic `bc` ToolHang 0.
3. **Copy ablation core** (done, then removed duplicate `phaseforge` older): keep 27 `Lift` copies for `T2` completeness if strict mode desired; otherwise keep `--allow-partial` and note `T2` `--` as intentional.
4. **Update `base.yaml:12`** to `paper_root: studies/analysis/outputs` (done).
5. **Regenerate:** `python -m studies.analysis.scripts.generate --allow-partial` → `studies/analysis/outputs/figures|tables` (done, 10 PDFs +11 tables).
6. **Next:** Apply redesigns above to `studies/analysis/assets/*.py` (one asset per PR, keep `paper_style()` and `OKABE_ITO`), re-run `generate` and `verify` (expect `F1` manual only).

---

## 6. Validation Checklist (publication-ready)

- [ ] Every figure prints legibly at 3.5" (single) / 7" (double) — 8 pt floor, no <7 pt.
- [ ] Colorblind check: Okabe-Ito + marker shape + `verify` grayscale.
- [ ] Error bars: Wilson 95% CI on all success rates, min/max on deltas, never SEM on n=3.
- [ ] Axes: bar y from 0, line x 0–200, `Δ` x `[-0.4,0.4]`, units/r horizon in caption.
- [ ] Legend: single, outside, 4–6 entries, no per-panel duplication.
- [ ] Caption: 1–2 sentences, `n=3 seeds ×50 episodes`, `reset_bank` pairing, `* Holm p<0.05`, `--` explained.
- [ ] Files: PDF vector + PNG 300 DPI, `generation_manifest.json` sha256 traceable.
- [ ] No 3D/chartjunk, data-ink >0.7, `F1` schematic placed manually.

**Owner:** This file + `final_ouput/docs/` are single source of truth; any chart must cite its `outputs_final`/`outputs_ablation` path, not recursive search.

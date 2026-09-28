# Evaluation

This document describes offline metric evaluation, the rollout validation gates, and the rollout report aggregator.

## Offline evaluator

`phaseforge/evaluations/runners/offline_evaluator.py` computes metrics over a validation or test dataset. Static metrics — routing entropy, expert utilization, phase–expert NMI, and action MSE — pool all samples. Temporal metrics — `routing_entropy_variance`, `time_to_stable_routing`, and `boundary_action_smoothness` — are computed per trajectory and averaged; they are never computed across trajectory boundaries.

Default metric toggles live in `phaseforge/config/eval/metrics.yaml` (`mode: "offline"`):

- Mechanism metrics, all enabled: `routing_entropy`, `routing_entropy_variance`, `time_to_stable_routing`, `expert_utilization`, `collapse_rate`, `phase_expert_nmi`.
- Task metrics: `boundary_action_smoothness` is enabled; `action_l2_threshold_rate`, `completion_rate`, and `phase_boundary_error` are disabled by default.

## Rollout validation gates

`phaseforge-gates` runs the rollout validation gates:

```bash
phaseforge-gates eval=rollout
phaseforge-gates eval=rollout project.seed=42
```

Exit codes: 0 means every required gate passed (skipped gates are warnings only); 1 means one or more required gates failed; 2 means the gates could not run at all (for example, robosuite is missing). The report is written to `{outputs}/_gates/{timestamp}/gates_report.json`.

The six gates are:

- `gate_env_schema` — observation keys and dimensions, action spec, and state restore.
- `gate_demo_replay` — demo replay against the raw HDF5.
- `gate_action_contract` — normalized action range and simulator acceptance.
- `gate_native_predicate` — success-predicate availability and type.
- `gate_random_noop_sanity` — no-op and random actions must neither succeed nor crash.
- `gate_checkpoint_smoke` — a few episodes with the configured checkpoint policy.

## Rollout report

`phaseforge-rollout-report` aggregates completed rollout evaluation runs into paper-table CSVs:

```bash
phaseforge-rollout-report [outputs_base] [out_dir]
```

`outputs_base` defaults to `./outputs`; `out_dir` defaults to `{outputs_base}/_results`. The command is idempotent: re-running it re-aggregates every completed rollout evaluation run.

## See also

- `experiments.md` — the frozen evaluation protocol.
- `analysis.md` — generating the paper's figures and tables from evaluation outputs.
- `troubleshooting.md` — diagnosing gate failures and missing metrics.

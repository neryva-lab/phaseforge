"""AnalysisDataset: load + join every namespace artifact once, validate coverage.

The dataset is the single object handed to asset generators. Assembly is
fail-closed in two tiers:

* hard errors — unreadable/malformed artifacts, duplicate completed runs;
* coverage report — cells expected from the manifests but absent on disk,
  returned per tier so the CLI can refuse (strict) or allow partial work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from studies.analysis.common import registry
from studies.analysis.common.config import namespace_root
from studies.analysis.loaders.curves import TrainingCurve, load_curve
from studies.analysis.loaders.episodes import EpisodeRecord, load_episodes
from studies.analysis.loaders.eval_results import EvalResult, load_eval_result
from studies.analysis.loaders.metadata import (
    EnvironmentRecord,
    InitExpert,
    InitRouting,
    TimingsRecord,
    load_environment,
    load_init_expert,
    load_init_routing,
    load_timings,
)
from studies.analysis.loaders.runs import EvalRun, TrainRun, assert_no_duplicates, scan_namespace
from studies.analysis.loaders.summaries import (
    PairedWilcoxon,
    StratifiedStats,
    load_paired_wilcoxon,
    load_stratified,
)

logger = logging.getLogger(__name__)

# EvalKey: (task, method, seed); CurveKey adds stage; namespace implied by dict.
EvalKey = tuple[str | None, str, int]
CurveKey = tuple[str | None, str, int, int]


@dataclass
class CoverageReport:
    missing_evals: list[EvalKey] = field(default_factory=list)
    missing_stage_runs: list[CurveKey] = field(default_factory=list)
    present_evals: int = 0
    present_stage_runs: int = 0

    @property
    def ok(self) -> bool:
        return not self.missing_evals and not self.missing_stage_runs

    def summary(self) -> str:
        return (
            f"evals {self.present_evals} present / {self.present_evals + len(self.missing_evals)} "
            f"expected; train runs {self.present_stage_runs} present / "
            f"{self.present_stage_runs + len(self.missing_stage_runs)} expected"
        )


@dataclass
class AnalysisDataset:
    train_runs: dict[CurveKey, TrainRun] = field(default_factory=dict)
    eval_runs: dict[EvalKey, EvalRun] = field(default_factory=dict)
    evals: dict[EvalKey, EvalResult] = field(default_factory=dict)
    episodes: dict[EvalKey, list[EpisodeRecord]] = field(default_factory=dict)
    curves: dict[CurveKey, TrainingCurve] = field(default_factory=dict)
    init_routing: dict[CurveKey, InitRouting] = field(default_factory=dict)
    init_expert: dict[CurveKey, InitExpert] = field(default_factory=dict)
    environments: dict[str, EnvironmentRecord] = field(default_factory=dict)
    timings: dict[CurveKey, TimingsRecord] = field(default_factory=dict)
    summaries: dict[str, StratifiedStats | None] = field(default_factory=dict)
    paired_wilcoxon: dict[str, PairedWilcoxon | None] = field(default_factory=dict)

    # -- typed views -----------------------------------------------------
    def matrix_eval(self, task: str, method: str, seed: int) -> EvalResult:
        key = (task, method, seed)
        if key not in self.evals:
            raise KeyError(f"Missing matrix eval cell {key}")
        return self.evals[key]

    def matrix_seeds(self, task: str, method: str) -> list[int]:
        return sorted(seed for (t, m, seed) in self.evals if t == task and m == method)

    def ablation_eval(self, method: str, seed: int) -> EvalResult:
        key = (None, method, seed)
        if key not in self.evals:
            raise KeyError(f"Missing ablation eval cell {key}")
        return self.evals[key]

    def curve(self, task: str | None, method: str, seed: int, stage: int) -> TrainingCurve:
        key = (task, method, seed, stage)
        if key not in self.curves:
            raise KeyError(f"Missing curve {key}")
        return self.curves[key]

    def coverage(self) -> CoverageReport:
        report = CoverageReport()
        for task, method in registry.expected_cells("final"):
            for seed in registry.seeds("final"):
                if (task, method, seed) in self.evals:
                    report.present_evals += 1
                else:
                    report.missing_evals.append((task, method, seed))
        for task, method in registry.expected_cells("ablation"):
            for seed in registry.seeds("ablation"):
                if (task, method, seed) in self.evals or (None, method, seed) in self.evals:
                    report.present_evals += 1
                else:
                    report.missing_evals.append((task, method, seed))
        report.missing_stage_runs = sorted(
            (key for key in self._expected_stage_runs() if key not in self.train_runs),
            key=lambda k: (k[0] is not None, k[0] or "", k[1], k[2], k[3]),
        )
        report.present_stage_runs = len(self._expected_stage_runs()) - len(
            report.missing_stage_runs
        )
        return report

    def _expected_stage_runs(self) -> list[CurveKey]:
        keys: list[CurveKey] = []
        for namespace in ("final", "ablation"):
            for method in registry.methods(namespace):
                task = method.task
                for seed in registry.seeds(namespace):
                    for stage in method.stages:
                        keys.append((task, method.name, seed, stage))
        return keys


def build_dataset(strict: bool = True, load_curves: bool = True) -> AnalysisDataset:
    """Scan + load both namespaces. ``strict`` raises on incomplete coverage."""
    dataset = AnalysisDataset()
    for namespace in ("final", "ablation"):
        root = namespace_root(namespace)
        if not root.is_dir():
            logger.warning("namespace %s root %s missing — skipped", namespace, root)
            continue
        train_runs, eval_runs = scan_namespace(namespace, root)
        assert_no_duplicates(train_runs, eval_runs)
        for run in train_runs:
            dataset.train_runs[(run.task, run.method, run.seed, run.stage)] = run
        for run in eval_runs:
            dataset.eval_runs[(run.task, run.method, run.seed)] = run
        dataset.summaries[namespace] = load_stratified(root)
        dataset.paired_wilcoxon[namespace] = load_paired_wilcoxon(root)

    for key, run in dataset.eval_runs.items():
        dataset.evals[key] = load_eval_result(run.path)
        episodes_path = run.path / "episodes.jsonl"
        if episodes_path.is_file():
            dataset.episodes[key] = load_episodes(run.path)

    if load_curves:
        for key, run in dataset.train_runs.items():
            # training_curves may sit at the run root or under metrics/ (final_output layout)
            try:
                c = load_curve(run.path)
                dataset.curves[key] = c
            except ValueError:
                continue

    for key, run in dataset.train_runs.items():
        dataset.init_routing[key] = load_init_routing(run.path)
        dataset.init_expert[key] = load_init_expert(run.path)
        dataset.timings[key] = load_timings(run.path)
        env = load_environment(run.path)
        if env is not None and env.git_sha:
            dataset.environments.setdefault(env.git_sha, env)

    report = dataset.coverage()
    if strict and not report.ok:
        missing_eval = report.missing_evals[:10]
        raise ValueError(
            f"Incomplete coverage: {report.summary()}; first missing evals: {missing_eval}; "
            "first missing stage runs: "
            f"{report.missing_stage_runs[:10]}. Run with strict=False for partial work."
        )
    logger.info("dataset: %s", report.summary())
    return dataset

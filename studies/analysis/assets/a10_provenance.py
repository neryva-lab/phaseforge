"""A10 — protocol & provenance (seeds, banks, hashes, pinned versions, linkage)."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.tables import Table, save_table


def generate(dataset: AnalysisDataset) -> list[Path]:
    rows: list[list[str]] = []
    first = next(iter(dataset.evals.values()), None)
    seeds_final = ",".join(str(s) for s in registry.seeds("final"))
    seeds_abl = ",".join(str(s) for s in registry.seeds("ablation"))
    rows.append(["Seeds (matrix / ablation)", f"{seeds_final} / {seeds_abl}"])
    # Oracle evals are offline-only (no rollouts, no router): they carry no
    # reset bank or router mode, so they are excluded from these sets rather
    # than rendered as "?" placeholders.
    closed_loop = {
        key: ev
        for key, ev in dataset.evals.items()
        if "oracle" not in key[1]
    }
    banks = sorted({ev.reset_bank or "?" for ev in closed_loop.values()})
    rows.append(["Reset banks", ", ".join(banks)])
    reset_seeds = sorted(
        {ev.reset_seed for ev in dataset.evals.values() if ev.reset_seed is not None}
    )
    rows.append(["Reset seeds", ", ".join(str(s) for s in reset_seeds) or "--"])
    router_modes = sorted({ev.router_mode or "?" for ev in closed_loop.values()})
    rows.append(["Eval router modes", ", ".join(router_modes)])
    commits = sorted({run.git_commit or "?" for run in dataset.train_runs.values()})
    rows.append(["Training commits", ", ".join(commits)])
    drops = sorted(
        {
            f"{key[1]}@seed{key[2]}: {init.dropped_indices_sha256 or '—'}"
            for key, init in dataset.init_expert.items()
            if init is not None and init.dropped_indices_sha256
        }
    )
    rows.append(
        [
            "Dropped-neuron hashes",
            f"{len(drops)} recorded" + ("; e.g. " + drops[0] if drops else ""),
        ]
    )
    env = next(iter(dataset.environments.values()), None)
    if env is not None:
        pinned = ", ".join(
            f"{k}=={v}"
            for k, v in sorted(env.packages.items())
            if k in ("torch", "robosuite", "mujoco", "numpy", "python")
        )
        rows.append(["Pinned stack", pinned or env.platform or "--"])
        rows.append(["Platform", env.platform or "--"])
    ckpt_linked = sum(1 for ev in dataset.evals.values() if ev.checkpoint_sha256)
    rows.append(["Eval→checkpoint sha links", f"{ckpt_linked}/{len(dataset.evals)}"])
    if first is not None and first.horizon:
        rows.append(["Rollout horizon", str(first.horizon)])

    table = Table(
        headers=["Item", "Value"],
        rows=rows,
        caption="Protocol and provenance record for the reported sweep.",
        notes=(
            "Reproducibility statement, evaluation-protocol note, sweep-level "
            "_runner/plan.json and state.json, gate reports, and the full "
            "per-run manifest checksums accompany the artifacts; see the "
            "attribution record for the protocol revision history.",
        ),
    )
    return save_table(table, "tables/A10_provenance")

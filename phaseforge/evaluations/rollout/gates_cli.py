"""``phaseforge-gates`` — run the rollout validation gates.

Usage::

    phaseforge-gates eval=rollout                 # default experiment config
    phaseforge-gates eval=rollout project.seed=42

Exit codes:
* 0 — every required gate passed (skipped gates are warnings only)
* 1 — one or more required gates FAILED
* 2 — the gates could not run at all (e.g. robosuite missing)

The report is written to ``{outputs}/_gates/{timestamp}/gates_report.json``
under the project output base.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from phaseforge.utils.config import output_base_dir

logger = logging.getLogger(__name__)


def _result_marker(result) -> str:
    """Render the console marker for one gate result.

    Diagnostic gates use the ``[DIAG-...]`` prefix so their non-blocking
    nature is visible in the CLI output. SKIPPED gates are never marked
    diagnostic (SKIPPED is already a non-blocking warning).
    """
    if result.diagnostic and result.status != "SKIPPED":
        return {"PASS": "[DIAG-PASS]", "FAIL": "[DIAG-FAIL]"}[result.status]
    return {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIPPED": "[SKIP]"}[result.status]


def _compute_exit_code(results) -> int:
    """Return 0 if every required gate passed, else 1.

    Diagnostic FAILs do not block (they are signals to investigate, not
    stop conditions); SKIPPED gates are also non-blocking warnings.
    """
    for result in results:
        if result.status == "FAIL" and not result.diagnostic:
            return 1
    return 0


def _write_report(cfg: DictConfig, results, exit_code: int, base: Path) -> Path:
    timestamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = base / "_gates" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": timestamp,
        "exit_code": exit_code,
        "gates": [
            {
                "gate": r.gate,
                "status": r.status,
                "detail": r.detail,
                "metrics": r.metrics,
                "diagnostic": r.diagnostic,
            }
            for r in results
        ],
    }
    path = out_dir / "gates_report.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# Use an explicit package URI rather than a filesystem-relative path.  The
# entry point is installed in the cloud environment, where Hydra's relative
# path resolution can otherwise reduce this to the unrelated top-level
# ``config`` module.  ``phaseforge.config`` is included in the distribution
# and is therefore stable in both editable and installed deployments.
@hydra.main(version_base="1.3", config_path="pkg://phaseforge.config", config_name="main")
def gates(cfg: DictConfig) -> None:
    """Run all rollout validation gates and report PASS/FAIL/SKIPPED."""
    from phaseforge.evaluations.rollout.gates import GateFailure, run_all_gates

    try:
        results = run_all_gates(cfg)
    except GateFailure as exc:
        logger.error("Gates could not run: %s", exc)
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001 — setup failure = cannot run
        logger.exception("Gates could not run: %s", exc)
        sys.exit(2)

    exit_code = _compute_exit_code(results)
    for result in results:
        print(f"{_result_marker(result)} {result.gate}: {result.detail}")

    base = output_base_dir(cfg)
    report = _write_report(cfg, results, exit_code, base)
    print(f"Gate report written to {report}")
    if exit_code:
        print(
            "Required gates FAILED — do not run rollouts until every required "
            "gate passes. DIAGNOSTIC FAILs and SKIPPED gates are warnings; "
            "they must be investigated but do not block the learned-policy "
            "sweep."
        )
    sys.exit(exit_code)


if __name__ == "__main__":
    gates()

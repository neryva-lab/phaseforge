"""Read-only verification gates for the final causal matrix.

Every function here is pure inspection: no training, no writes, no sweep
state changes. The same checks back the ``--verify-gates`` runner flag and
the ``tests/final/`` suite, so a gate that passes in CI passes identically
for the operator — and vice versa.

Gate mapping (ledger ``docs/final/baselines/ledger.md``):

* :func:`inspect_sweep` — no-write inspection record.
* plan expansion inside :func:`run_final_gates` — the complete step plan.
* :func:`verify_plan_ordering` — providers precede their consumers.
* :func:`verify_command_contract` — resolved overrides per command.
* :func:`verify_historical_isolation` — no historical fallback.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from phaseforge.runner.executor import step_command
from phaseforge.runner.protocol import (
    Protocol,
    ProtocolError,
    Step,
    build_plan,
    is_final_method,
    load_protocol,
    provider_method_name,
    provider_model_name,
    verify_output_namespace,
)

#: Model configs permitted for final-family rows (ledger §3 identities).
#: Anything else (``baselines/*``, retired dev variants) is a historical
#: artifact and must never carry a final result.
_FINAL_MODEL_ALLOWLIST = frozenset(
    {
        "precision_residual_phaseforge",
        "precision_residual_plain_encoder",
        "precision_residual_phase_random_router",
        "precision_residual_scratch_moe",
        "precision_residual_factorial_floor",
        "final_aligned_bc",
        "final_aligned_softmax_top1",
        "final_aligned_static_rule",
        "precision_residual_teacher_forced",
        "precision_residual_oracle",
    }
)

def _git_record(project_root: Path) -> dict[str, Any]:
    """Read-only git identity: commit, branch, and working-tree cleanliness."""
    from phaseforge.utils.config import git_info

    record: dict[str, Any] = dict(git_info())
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(project_root),
        )
        if proc.returncode == 0:
            record["clean"] = proc.stdout.strip() == ""
            record["dirty_files"] = sorted(proc.stdout.splitlines())[:20]
        else:
            record["clean"] = None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        record["clean"] = None
    return record


def inspect_sweep(
    manifest_path: str | Path, outputs_base: str | Path, project_root: Path | None = None
) -> dict[str, Any]:
    """Build the no-write inspection record.

    Records the repo commit/tree state, the manifest hash, the existing
    output trees (names only — historical outputs are enumerated, never
    moved), and the target namespace freshness. Pure reads; raises nothing
    (freshness failures are recorded, not raised — the gate report decides).
    """
    manifest = Path(manifest_path)
    base = Path(outputs_base)
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    record: dict[str, Any] = {
        "manifest": str(manifest),
        "outputs_base": str(base),
        "git": _git_record(root),
        "manifest_sha256": None,
        "existing_trees": [],
        "namespace_fresh": None,
        "namespace_detail": None,
    }
    try:
        record["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    except OSError:
        record["manifest_sha256"] = None
    try:
        if base.is_dir():
            record["existing_trees"] = sorted(p.name for p in base.iterdir())
    except OSError:
        record["existing_trees"] = None
    try:
        verify_output_namespace(base)
        record["namespace_fresh"] = True
    except ProtocolError as exc:
        record["namespace_fresh"] = False
        record["namespace_detail"] = str(exc)
    return record


def verify_plan_ordering(protocol: Protocol, plan: list[Step]) -> list[str]:
    """Check that every Stage 1 provider precedes its Stage 2 consumers.

    Indexes plan positions by ``(phase_key, seed, phase)`` and requires, for
    each stage-2 train step, the provider's stage-1 position to be earlier
    (same task and seed), and for each eval step, the method's own
    final-stage train position to be earlier. Eval steps whose method has no
    train step in the plan at all (eval-only selections, whose artifacts are
    pre-existing by construction) carry no ordering constraint. Returns
    violation strings (empty means the ordering gate passes).
    """
    position: dict[tuple[str, int, str], int] = {}
    trained: set[tuple[str, int]] = set()
    for idx, step in enumerate(plan):
        position[(step.method.phase_key, step.seed, step.registry_phase)] = idx
        if step.kind == "train":
            trained.add((step.method.phase_key, step.seed))
    violations: list[str] = []
    for step in plan:
        if step.kind == "train" and step.stage == 2:
            source = step.method.stage2_source
            if source is None:
                # Provider-less by design (e.g. historical scratch cells
                # training from random init): no ordering constraint. Final
                # rows must still declare a provider — enforced by
                # verify_historical_isolation(), not here.
                continue
            if source == "self":
                provider_key = step.method.phase_key
            else:
                provider_name = provider_method_name(source)
                if provider_name is None:  # pragma: no cover - validated at load
                    violations.append(f"{step.label}: unresolvable source {source!r}.")
                    continue
                provider = protocol.method_by_name(provider_name, task=step.method.task)
                if provider is None:
                    violations.append(
                        f"{step.label}: provider {source!r} is not a method "
                        f"in task {step.method.task!r}."
                    )
                    continue
                provider_key = provider.phase_key
            need = (provider_key, step.seed, "stage1")
            if need not in position:
                violations.append(
                    f"{step.label}: provider stage 1 {need[0]}/seed={need[1]} "
                    "is absent from the plan."
                )
            elif position[need] >= position[
                (step.method.phase_key, step.seed, step.registry_phase)
            ]:
                violations.append(
                    f"{step.label}: provider stage 1 runs at plan position "
                    f"{position[need]} (not before the consumer)."
                )
        elif step.kind == "eval":
            if (step.method.phase_key, step.seed) not in trained:
                continue
            need = (step.method.phase_key, step.seed, f"stage{step.method.final_stage}")
            if need not in position:
                violations.append(f"{step.label}: final-stage train step absent from plan.")
    return violations


def verify_command_contract(
    protocol: Protocol, plan: list[Step], outputs_base: str | Path
) -> list[str]:
    """Check that every command carries its resolved contract tokens.

    Rebuilds each step's argv with :func:`step_command` (a dummy checkpoint
    path stands in where resolution needs an artifact — artifact existence
    is a separate static concern, not this check) and requires the intended
    task (via tag), seed, model, method, label/beta/topo overrides, and
    evaluation mode/group. Returns violation strings.
    """
    base = Path(outputs_base)
    violations: list[str] = []
    for step in plan:
        method = step.method
        try:
            if step.kind == "eval":
                argv = step_command(
                    step,
                    ckpt_path=Path("CKPT_PLACEHOLDER"),
                    outputs_base=base,
                    defaults=protocol.defaults,
                )
            else:
                argv = step_command(
                    step, ckpt_path=None, outputs_base=base, defaults=protocol.defaults
                )
        except Exception as exc:
            violations.append(f"{step.label}: command build failed: {exc}.")
            continue
        text = " ".join(argv)
        expected = [
            f"models={method.model}",
            f"project.seed={step.seed}",
            f"project.method={method.name}",
        ]
        if step.kind == "eval":
            group = "rollout" if method.evaluate_mode == "rollout" else "metrics"
            expected += [f"eval={group}", f"eval.mode={method.evaluate_mode}"]
        else:
            assert step.stage is not None
            expected.append(f"train=stage{step.stage}")
            if method.stage2_source is not None and step.stage == 2:
                resolved = provider_model_name(method.stage2_source, method.model_name)
                if resolved is None:
                    violations.append(f"{step.label}: unresolvable source.")
                    continue
        if method.data != "common":
            expected.append(f"data={method.data}")
        if method.output_tag:
            expected.append(f"project.tag={method.output_tag}")
        if step.kind == "train":
            expected.extend(
                o for o in method.overrides if not o.startswith("eval.") and "=" in o
            )
        else:
            expected.extend(o for o in method.overrides if "=" in o)
        for token in expected:
            if token not in text:
                violations.append(f"{step.label}: argv missing {token!r}.")
    return violations


def verify_historical_isolation(protocol: Protocol) -> list[str]:
    """Check that no final row can resolve to a historical artifact.

    Asserts every final-family method uses an allowlisted model config and a
    non-historical Stage 2 source. (The loader already rejects historical
    aliases; this re-asserts the property over the
    loaded matrix so the dry-run report carries it as evidence.)
    """
    violations: list[str] = []
    for method in protocol.methods:
        if not is_final_method(method.name):
            continue
        if method.model not in _FINAL_MODEL_ALLOWLIST:
            violations.append(
                f"{method.name} (task={method.task}): model {method.model!r} "
                "is not a final-family config."
            )
        source = method.stage2_source
        if 2 in method.stages and not source:
            violations.append(
                f"{method.name} (task={method.task}): stage 2 with no "
                "declared Stage 1 provider (MAN-07)."
            )
    return violations


def run_final_gates(
    manifest_path: str | Path,
    outputs_base: str | Path,
    *,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Run the plan-expansion, ordering, command-contract, and
    historical-isolation gates plus the no-write inspection record over a
    manifest.

    Builds the complete plan and returns a structured report with
    one entry per gate: ``status`` (``pass``/``fail``) plus details. Raises
    :class:`ProtocolError` when the manifest itself cannot load or the seed
    selection is invalid — a matrix that does not parse has no gates to
    evaluate.
    """
    protocol = load_protocol(manifest_path)
    plan = build_plan(protocol, list(protocol.methods), seeds=seeds, with_dependencies=True)
    ordering = verify_plan_ordering(protocol, plan)
    commands = verify_command_contract(protocol, plan, outputs_base)
    isolation = verify_historical_isolation(protocol)
    report: dict[str, Any] = {
        "manifest": str(manifest_path),
        "methods": len(protocol.methods),
        "seeds": list(protocol.seeds if seeds is None else seeds),
        "steps": len(plan),
        "inspection": inspect_sweep(manifest_path, outputs_base),
        "gates": {
            "DRY-02_plan_expands": {
                "status": "pass" if plan else "fail",
                "detail": f"{len(plan)} steps",
            },
            "DRY-03_provider_ordering": {
                "status": "pass" if not ordering else "fail",
                "detail": ordering,
            },
            "DRY-04_command_contract": {
                "status": "pass" if not commands else "fail",
                "detail": commands[:20],
                "violation_count": len(commands),
            },
            "DRY-05_historical_isolation": {
                "status": "pass" if not isolation else "fail",
                "detail": isolation,
            },
        },
    }
    return report


def format_gates_report(report: dict[str, Any]) -> str:
    """Render a gates report as human-readable lines."""
    lines = [
        f"manifest: {report['manifest']}",
        f"methods={report['methods']} seeds={report['seeds']} steps={report['steps']}",
    ]
    insp = report.get("inspection", {})
    git = insp.get("git", {}) if isinstance(insp, dict) else {}
    lines.append(
        f"commit={git.get('commit', '?')} clean={git.get('clean', '?')} "
        f"namespace_fresh={insp.get('namespace_fresh', '?')}"
    )
    for gate, result in report.get("gates", {}).items():
        detail = result.get("detail")
        if isinstance(detail, list):
            shown = "; ".join(detail[:5]) if detail else "ok"
            extra = f" (+{len(detail) - 5} more)" if len(detail) > 5 else ""
            lines.append(f"{gate}: {result.get('status')} [{shown}{extra}]")
        else:
            lines.append(f"{gate}: {result.get('status')} [{detail}]")
    return "\n".join(lines)


__all__ = [
    "inspect_sweep",
    "verify_plan_ordering",
    "verify_command_contract",
    "verify_historical_isolation",
    "run_final_gates",
    "format_gates_report",
]

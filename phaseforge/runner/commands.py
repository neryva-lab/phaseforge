"""Pure argv builders for the CLI commands a plan step executes.

Kept side-effect free and unit-tested: the runner's correctness hinges on
these exact overrides (model, stage, seed, outputs base, data variant tag,
early-stopping lock, and the evaluated checkpoint path).
"""

from __future__ import annotations

from pathlib import Path

from phaseforge.runner.protocol import Method, Step

# Evaluate-mode name -> Hydra eval-config-group selector. The eval group
# defines the schema the CLI's evaluator reads (``rollout.yaml`` vs
# ``metrics.yaml``). A key-level ``eval.mode=...`` override alone leaves
# the default (``metrics``) group in place, and the rollout path then
# crashes on missing ``bank``/``env``/``episodes`` sections — the bug
# this map prevents. Updated atomically with the valid modes in
# ``protocol._VALID_EVAL_MODES``.
_EVAL_GROUP_BY_MODE: dict[str, str] = {
    "rollout": "rollout",
    "offline": "metrics",
}


def _effective_tag(method: Method) -> str | None:
    """Compose the effective output tag.

    The five-task protocol trains the same method name on multiple tasks;
    without a task-prefixed tag, the output trees for, e.g.,
    ``phaseforge/Lift`` and ``phaseforge/Can`` would collide on the same
    directory. The task tag is appended to any explicit ``method.tag``
    the manifest carries, so the negative control's ``robot_only`` tag
    survives (for example ``Lift__robot_only``). The separator is
    filesystem-safe because the tag is embedded in one run-directory name.
    """
    return method.output_tag


def train_command(
    step: Step,
    *,
    outputs_base: Path,
    defaults: tuple[str, ...],
    ckpt_path: Path | None = None,
) -> list[str]:
    """Build ``phaseforge-train`` argv for a training step.

    A stage-2 step that bootstraps from a provider passes that exact Stage 1
    checkpoint as ``train.stage1_ckpt_path``, so the subprocess never falls
    back to its own loose auto-detect (which could select a tagged sibling
    variant that shares the provider's output tree).
    """
    method = step.method
    assert step.stage is not None
    cmd = [
        "phaseforge-train",
        f"models={method.model}",
        f"train=stage{step.stage}",
        f"project.seed={step.seed}",
        f"project.output_dir={outputs_base}",
    ]
    if method.data != "common":
        cmd.append(f"data={method.data}")
    effective_tag = _effective_tag(method)
    if effective_tag:
        cmd.append(f"project.tag={effective_tag}")
    if method.name:
        cmd.append(f"project.method={method.name}")
    if ckpt_path is not None:
        cmd.append(f"train.stage1_ckpt_path={ckpt_path}")
    cmd.extend(defaults)
    if method.overrides:
        cmd.extend([o for o in method.overrides if not o.startswith("eval.")])
    return cmd


def eval_command(
    step: Step,
    *,
    ckpt_path: Path,
    outputs_base: Path,
    defaults: tuple[str, ...],
) -> list[str]:
    """Build ``phaseforge-eval`` argv for an evaluation step.

    The historical override name is ``train.stage1_ckpt_path`` but it is the
    checkpoint actually evaluated (the method's final-stage artifact). The
    evaluation mode comes from the method's ``evaluate_mode`` (rollout by
    default), so the runner never evaluates the state-only protocol with
    the offline metric.

    Two overrides carry the mode:

    - ``eval=<group>`` selects the Hydra config group whose schema the
      evaluator reads. ``rollout.yaml`` defines ``bank``/``env``/
      ``episodes``/``gates``; ``metrics.yaml`` does not. Switching only
      ``eval.mode`` leaves the default ``metrics`` group in place and the
      rollout path raises ``ConfigAttributeError`` on the first missing
      key.
    - ``eval.mode=<mode>`` is an explicit assertion kept for redundancy
      and documentation. It is consistent with the group by construction
      (the map's keys equal the modes) and survives if the group's own
      ``mode`` field is ever renamed.
    """
    method = step.method
    eval_group = _EVAL_GROUP_BY_MODE[method.evaluate_mode]
    cmd = [
        "phaseforge-eval",
        f"models={method.model}",
        f"project.seed={step.seed}",
        f"project.output_dir={outputs_base}",
        f"train.stage1_ckpt_path={ckpt_path}",
        f"eval={eval_group}",
        f"eval.mode={method.evaluate_mode}",
    ]
    if method.data != "common":
        cmd.append(f"data={method.data}")
    effective_tag = _effective_tag(method)
    if effective_tag:
        cmd.append(f"project.tag={effective_tag}")
    if method.name:
        cmd.append(f"project.method={method.name}")
    cmd.extend(defaults)
    if method.overrides:
        cmd.extend(method.overrides)
    return cmd

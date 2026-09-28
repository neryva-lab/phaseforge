"""Subprocess execution for runner steps.

Runs each command with inherited stdio (so progress bars and per-epoch logs
stream to the console) and raises on a non-zero exit. The console entry
points are resolved via ``shutil.which`` so the sweep works regardless of
the active environment, with a loud error if they are not installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from packaging.version import InvalidVersion, Version

from phaseforge.runner.commands import eval_command, train_command
from phaseforge.runner.protocol import Step


class CommandError(RuntimeError):
    """Raised when a runner step's command cannot run or fails."""


def resolve_toolhang_python(project_root: Path, configured: str | None = None) -> Path:
    """Resolve the interpreter reserved for Tool Hang evaluation.

    Tool Hang was collected with robosuite 1.5.0 while the other four tasks
    use 1.5.1.  The sweep therefore selects a separate interpreter for every
    Tool Hang subprocess.  An explicit CLI value or environment variable is
    preferred; otherwise the conventional ``.venv-toolhang`` location is
    detected on POSIX and Windows.
    """
    requested = configured or os.environ.get("PHASEFORGE_TOOLHANG_PYTHON")
    if requested:
        candidate = Path(requested).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        candidate = candidate.absolute()
        if not candidate.is_file():
            raise CommandError(
                f"Configured Tool Hang Python does not exist: {candidate}. "
                "Create the dedicated robosuite 1.5.0 environment or pass "
                "--toolhang-python with its interpreter path."
            )
        return candidate

    candidates = (
        project_root / ".venv-toolhang" / "bin" / "python",
        project_root / ".venv-toolhang" / "Scripts" / "python.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.absolute()
    raise CommandError(
        "Tool Hang steps are present, but the dedicated robosuite 1.5.0 "
        "environment was not found. Create it with `uv venv --python 3.11 "
        ".venv-toolhang`, install the project without the rollout extra, "
        "then install robosuite==1.5.0 and mujoco==3.2.7; or pass "
        "--toolhang-python PATH."
    )


def preflight_toolhang_python(python_executable: Path) -> None:
    """Fail before a sweep if the Tool Hang interpreter has the wrong pins."""
    probe = (
        "import importlib.metadata as m; "
        "print(m.version('robosuite')); "
        "print(m.version('mujoco'))"
    )
    result = subprocess.run(
        [str(python_executable), "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise CommandError(
            f"Tool Hang interpreter preflight failed for {python_executable}: {detail}"
        )
    versions = result.stdout.splitlines()
    if len(versions) < 2 or versions[0].strip() != "1.5.0":
        actual = versions[0].strip() if versions else "unavailable"
        raise CommandError(
            f"Tool Hang interpreter {python_executable} has robosuite {actual}; "
            "the dataset requires robosuite==1.5.0."
        )
    try:
        mujoco_version = Version(versions[1].strip())
    except InvalidVersion as exc:
        raise CommandError(
            f"Tool Hang interpreter {python_executable} reported an invalid "
            f"MuJoCo version: {versions[1].strip()!r}."
        ) from exc
    if mujoco_version < Version("3.2.7"):
        raise CommandError(
            f"Tool Hang interpreter {python_executable} has MuJoCo "
            f"{mujoco_version}; the rollout protocol requires >=3.2.7."
        )


def _resolve_script(name: str, python_executable: Path | None = None) -> str:
    if python_executable is not None:
        script_dir = python_executable.parent
        candidates = (script_dir / name, script_dir / f"{name}.exe", script_dir / f"{name}.cmd")
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        raise CommandError(
            f"Cannot find {name!r} beside the selected interpreter "
            f"{python_executable}. Install PhaseForge into that environment "
            "with `uv pip install -e '.[dev]'`."
        )
    found = shutil.which(name)
    if found is None:
        raise CommandError(
            f"Cannot find the {name!r} entry point on PATH. Run "
            "'uv sync' (or reinstall the package) so the console scripts "
            "are available, then re-run the sweep."
        )
    return found


def step_command(
    step: Step, *, ckpt_path: Path | None, outputs_base: Path, defaults: tuple[str, ...]
) -> list[str]:
    """Return the argv for a step (train or eval)."""
    if step.kind == "eval":
        if ckpt_path is None:
            raise CommandError(f"Eval step {step.label} has no checkpoint to evaluate.")
        return eval_command(step, ckpt_path=ckpt_path, outputs_base=outputs_base, defaults=defaults)
    return train_command(step, outputs_base=outputs_base, defaults=defaults, ckpt_path=ckpt_path)


def run_step(
    step: Step,
    *,
    ckpt_path: Path | None,
    outputs_base: Path,
    defaults: tuple[str, ...],
    cwd: Path,
    log_level: str = "WARNING",
    toolhang_python: Path | None = None,
) -> None:
    """Execute one step's command, raising :class:`CommandError` on failure.

    ``ckpt_path`` is required for eval steps. ``log_level`` is forwarded as
    ``project.log_level`` so sweep output stays readable unless ``--verbose``
    is requested.
    """
    if step.kind == "eval" and ckpt_path is None:
        raise CommandError(f"Eval step {step.label} has no checkpoint to evaluate.")
    cmd = step_command(step, ckpt_path=ckpt_path, outputs_base=outputs_base, defaults=defaults)
    selected_python = toolhang_python if step.method.task == "ToolHang" else None
    executable = _resolve_script(cmd[0], python_executable=selected_python)
    argv = [executable, f"project.log_level={log_level}"] + cmd[1:]

    print(f"\n[runner] $ {' '.join(argv)}", flush=True)
    result = subprocess.run(argv, cwd=str(cwd))
    if result.returncode != 0:
        raise CommandError(
            f"Step {step.label} failed with exit code {result.returncode}.\n"
            f"Command: {' '.join(argv)}"
        )

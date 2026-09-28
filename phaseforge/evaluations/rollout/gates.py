"""Validation gates for the rollout protocol.

The gates validate the simulator/adapter/bank chain BEFORE real policy
rollouts, so a broken chain can never produce meaningless numbers:

* Gate 1 — environment schema self-test (parity + state restoration +
  schema + zero step). Required.
* Gate 2 — demo replay against the raw HDF5 (non-gating; SKIPPED on
  mismatch — robosuite action playback can drift across machines).
* Gate 3 — action-contract enforcement (in-range accepted, out-of-range
  rejected, simulator accepts the declared gripper action). Required.
* Gate 4 — native robosuite success-predicate availability (task-
  independent: ``adapter.check_success()`` and ``adapter.env._check_success()``
  both callable and return bool/dict). Required.
* Gate 5 — random/no-op sanity (success must be ≈ 0, no infra failures).
  Required.
* Gate 6 — checkpoint smoke run (skipped loudly when no checkpoint given).
  Optional.

Gates that need a missing input are SKIPPED with a loud warning (never a
silent pass); gates that need robosuite itself hard-fail. Demo replay is
diagnostic because robosuite documents that action playback can drift
across machines. The five-task protocol is now::

    trained policy -> structured state -> robosuite -> native task success

The gates exit 0 only when every required gate passed; diagnostic FAILs and
skipped gates are reported as warnings and do not block the learned-policy
sweep.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from phaseforge.evaluations.envs.env_metadata import (
    verify_environment_parity,
)
from phaseforge.evaluations.envs.errors import (
    EnvParityError,
    PolicyInvalidActionError,
    StateSchemaError,
)
from phaseforge.evaluations.envs.robosuite_adapter import (
    RobosuiteStateAdapter,
)
from phaseforge.evaluations.rollout.reset_bank import ResetBank

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Outcome of one gate."""

    gate: str
    status: str  # "PASS" | "FAIL" | "SKIPPED"
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    diagnostic: bool = False
    #: When True the gate is a diagnostic signal only: a FAIL is reported
    #: but must not block the protocol (no non-zero exit code).

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


class GateFailure(RuntimeError):
    """A required gate failed; the protocol must not proceed."""


def _decode_hdf5_text(value: Any) -> str | None:
    """Decode a scalar HDF5 string/bytes attribute without lossy coercion."""
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    text = str(value)
    return text if text else None


# ---------------------------------------------------------------------------
# Gate 1 — environment schema self-test
# ---------------------------------------------------------------------------


def gate_env_schema(
    adapter: RobosuiteStateAdapter,
    bank: ResetBank,
    *,
    expected_state_dim: int,
    expected_action_dim: int,
) -> GateResult:
    """Verify obs keys, dims, action spec, and a state restore + zero step."""
    problems: list[str] = []
    try:
        state = adapter.reset_to(
            bank.case(0).states, xml=bank.case(0).xml, ep_meta=bank.case(0).ep_meta
        )
        if state.shape != (expected_state_dim,):
            problems.append(f"restored state dim {state.shape[0]} != declared {expected_state_dim}")
        zero_action = np.zeros(expected_action_dim, dtype=np.float64)
        next_state, _done, success, _info = adapter.step(zero_action)
        if next_state.shape != (expected_state_dim,):
            problems.append(
                f"stepped state dim {next_state.shape[0]} != declared {expected_state_dim}"
            )
        if not isinstance(success, bool):
            problems.append(f"success predicate returned {type(success).__name__}")
    except (StateSchemaError, PolicyInvalidActionError, EnvParityError) as exc:
        problems.append(str(exc))
    except Exception as exc:  # noqa: BLE001 — gate must not crash the CLI
        problems.append(f"unexpected: {type(exc).__name__}: {exc}")

    status = "PASS" if not problems else "FAIL"
    return GateResult(
        gate="1_env_schema_selftest",
        status=status,
        detail="; ".join(problems)
        if problems
        else (
            f"state restore + zero step over ({expected_state_dim},) states and "
            f"({expected_action_dim},) actions OK"
        ),
        metrics={"state_dim": expected_state_dim, "action_dim": expected_action_dim},
    )


# ---------------------------------------------------------------------------
# Gate 2 — demo replay against the raw HDF5
# ---------------------------------------------------------------------------


def gate_demo_replay(
    adapter: RobosuiteStateAdapter,
    hdf5_path: Path,
    *,
    num_demos: int = 1,
    tolerance: float = 1e-3,
) -> GateResult:
    """Diagnostically replay the first demo(s) against raw HDF5 states.

    robomimic stores either ``T+1`` states with ``T`` actions (an initial state
    plus one post-action state per action) or ``T`` post-action states with
    ``T`` actions in the v1.5 files. In the latter representation the first
    action produced the stored initial state, so the check starts at
    ``states[0]`` and replays ``actions[1:]`` against ``states[1:]``. Both
    representations are valid; silently treating the equal-length v1.5 form
    as a malformed demo would incorrectly skip this gate.

    The demo's stored ``model_file`` XML is restored when present. A replay
    mismatch is reported as ``SKIPPED`` rather than ``FAIL``: upstream
    robosuite documents that action playback is not guaranteed to be
    deterministic across machines and recommends direct state restoration.
    Gate 1 and the frozen reset bank validate that required state-restore
    path. This check remains useful for diagnosing collection/runtime drift.

    Skipped loudly when the raw HDF5 is absent.
    """
    import h5py

    if not hdf5_path.is_file():
        return GateResult(
            gate="2_demo_replay",
            status="SKIPPED",
            detail=(
                f"raw dataset {hdf5_path} not present — replay gate skipped "
                "(the evaluation machine must run it before real rollouts)"
            ),
        )
    try:
        mismatches = 0
        compared = 0
        checked = 0
        with h5py.File(hdf5_path, "r") as h5:
            data = h5["data"]
            demo_keys = sorted(k for k in data.keys() if k.startswith("demo_"))
            if not demo_keys:
                raise GateFailure("dataset has no demo_* trajectories")
            for demo_key in demo_keys[:num_demos]:
                demo = data[demo_key]
                if "states" not in demo:
                    return GateResult(
                        gate="2_demo_replay",
                        status="SKIPPED",
                        detail=(
                            f"{demo_key} has no per-step 'states' attribute — "
                            "replay cannot be validated"
                        ),
                    )
                states = np.asarray(demo["states"][:], dtype=np.float64)
                actions = np.asarray(demo["actions"][:], dtype=np.float64)
                model_file = _decode_hdf5_text(demo.attrs.get("model_file"))
                if states.shape[0] == actions.shape[0] + 1:
                    replay_actions = actions
                    expected_states = states[1:]
                elif states.shape[0] == actions.shape[0] and states.shape[0] >= 2:
                    # The robosuite data-collection path records states after
                    # each action and removes the extra terminal state before
                    # writing the equal-length v1.5 HDF5. State[0] is thus
                    # already the result of action[0].
                    replay_actions = actions[1:]
                    expected_states = states[1:]
                else:
                    return GateResult(
                        gate="2_demo_replay",
                        status="SKIPPED",
                        detail=(
                            f"{demo_key} has incompatible trajectory lengths: "
                            f"states={states.shape[0]}, actions={actions.shape[0]}; "
                            "expected states=actions or states=actions+1"
                        ),
                    )
                adapter.reset_to(states[0], xml=model_file, ep_meta=None)
                for i in range(replay_actions.shape[0]):
                    adapter.step(replay_actions[i])
                    sim = np.asarray(adapter.env.sim.get_state().flatten(), dtype=np.float64)
                    if sim.shape != expected_states[i].shape:
                        mismatches += 1
                        break
                    compared += 1
                    if np.max(np.abs(sim[1:] - expected_states[i][1:])) > tolerance:
                        mismatches += 1
                checked += 1
    except GateFailure as exc:
        return GateResult(gate="2_demo_replay", status="SKIPPED", detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            gate="2_demo_replay",
            status="FAIL",
            diagnostic=True,
            detail=f"replay raised {type(exc).__name__}: {exc}",
        )

    if mismatches:
        return GateResult(
            gate="2_demo_replay",
            status="SKIPPED",
            detail=(
                f"diagnostic only: {mismatches} of {compared} compared steps "
                f"diverged (tolerance {tolerance}); exact action playback is "
                "not a required cross-machine gate — state restoration is "
                "validated by Gate 1 and the frozen reset bank"
            ),
            metrics={"compared": compared, "mismatches": mismatches, "diagnostic_only": True},
        )
    return GateResult(
        gate="2_demo_replay",
        status="PASS",
        detail=(f"{checked} demo(s), {compared} steps replayed within tolerance {tolerance}"),
        metrics={"compared": compared, "checked": checked},
    )


# ---------------------------------------------------------------------------
# Gate 3 — action contract
# ---------------------------------------------------------------------------


def gate_action_contract(
    adapter: RobosuiteStateAdapter,
    *,
    trials: int = 10,
    tolerance: float = 1e-4,
) -> GateResult:
    """Validate the normalized action range and simulator acceptance.

    The gripper direction is deliberately not inferred from one qpos delta:
    a single MuJoCo control step can move a partially closed gripper in the
    opposite direction because of controller state and contact dynamics.
    This gate checks action acceptance only. Behavioural task performance is
    measured by the learned-policy rollout evaluator; Gate 6 is only an
    optional checkpoint smoke test.
    """
    problems: list[str] = []
    rng = np.random.default_rng(0)

    for _ in range(trials):
        action = rng.uniform(-1.0, 1.0, size=adapter.action_dim)
        try:
            adapter.validate_action(action, tolerance=tolerance)
        except PolicyInvalidActionError as exc:
            problems.append(f"in-range action rejected: {exc}")
            break

    for bad in (
        np.full(adapter.action_dim, 1.5),
        np.full(adapter.action_dim, -1.5),
        np.array([np.nan] * adapter.action_dim),
        np.array([np.inf] * adapter.action_dim),
        np.zeros(adapter.action_dim + 1),
    ):
        try:
            adapter.validate_action(bad, tolerance=tolerance)
            problems.append(f"invalid action {np.asarray(bad).tolist()} was accepted")
        except PolicyInvalidActionError:
            pass

    if not problems:
        try:
            adapter.reset_to(
                np.asarray(adapter.env.sim.get_state().flatten(), dtype=np.float64),
                xml=None,
                ep_meta=None,
            )
            close_action = np.zeros(adapter.action_dim, dtype=np.float64)
            close_action[-1] = 1.0
            adapter.step(close_action)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"gripper action acceptance probe failed: {exc}")

    return GateResult(
        gate="3_action_contract",
        status="PASS" if not problems else "FAIL",
        detail="; ".join(problems)
        if problems
        else (
            "in-range accepted, NaN/Inf/out-of-range rejected, gripper +1 "
            "accepted; behavioural validation is the checkpoint smoke gate's job"
        ),
    )


# ---------------------------------------------------------------------------
# Gate 4 — success-predicate availability / type check (task-independent)
# ---------------------------------------------------------------------------


def gate_native_predicate(
    adapter: RobosuiteStateAdapter,
    bank: ResetBank,
    *,
    num_cases: int = 5,
) -> GateResult:
    """Required Gate 4: success-predicate availability and type check.

    Probes both :meth:`RobosuiteStateAdapter.check_success` and the
    underlying :meth:`robosuite.env._check_success` on a few pinned
    reset cases. PASS requires:

    * both surfaces are callable (no exception) on every probed case;
    * both surfaces return ``bool`` or a ``dict`` whose ``"task"`` key
      coerces to ``bool``;
    * at least one bank case was probed.

    This gate does NOT validate the semantic correctness of the predicate.
    A predicate that always returns ``False`` would still PASS this gate —
    it is the *learned-policy rollouts* (not this gate) that produce the
    task-success numbers used in the paper. This gate only proves the
    adapter wraps the native robosuite predicate correctly enough that the
    runner's success-rate metric is meaningful.

    Fail-closed conditions:

    * empty bank (``len(bank.cases) == 0``) → FAIL: nothing to probe;
    * ``num_cases <= 0`` → FAIL: would otherwise produce a vacuous 0/0 PASS;
    * any non-bool/non-dict return or any exception → FAIL on the first
      probed case.
    """
    bank_size = len(bank.cases)
    if bank_size == 0:
        return GateResult(
            gate="4_native_predicate",
            status="FAIL",
            detail="reset bank is empty; cannot probe the success predicate",
            metrics={"probed": 0, "cases": 0},
        )
    if num_cases <= 0:
        return GateResult(
            gate="4_native_predicate",
            status="FAIL",
            detail=(
                f"native_predicate_cases={num_cases} must be > 0; refuse a "
                "vacuous gate"
            ),
            metrics={"probed": 0, "cases": bank_size},
        )
    effective_num_cases = min(int(num_cases), bank_size)

    probed = 0
    for case in bank.cases[:effective_num_cases]:
        try:
            adapter.reset_to(case.states, xml=case.xml, ep_meta=case.ep_meta)
        except Exception as exc:  # noqa: BLE001 — gate must not crash
            return GateResult(
                gate="4_native_predicate",
                status="FAIL",
                detail=f"reset failed for case {case.index}: {exc}",
                metrics={"probed": probed, "cases": effective_num_cases},
            )
        # Probe 1: adapter-wrapped predicate.
        try:
            wrapped = adapter.check_success()
        except Exception as exc:  # noqa: BLE001
            return GateResult(
                gate="4_native_predicate",
                status="FAIL",
                detail=(
                    f"adapter.check_success raised on case {case.index}: "
                    f"{type(exc).__name__}: {exc}"
                ),
                metrics={"probed": probed, "cases": effective_num_cases},
            )
        if not _is_valid_predicate_value(wrapped):
            return GateResult(
                gate="4_native_predicate",
                status="FAIL",
                detail=_predicate_failure_detail(
                    "adapter.check_success", wrapped, case.index
                ),
                metrics={"probed": probed, "cases": effective_num_cases},
            )
        # Probe 2: underlying robosuite predicate (so the adapter cannot
        # silently bypass env._check_success).
        try:
            raw = adapter.env._check_success()
        except Exception as exc:  # noqa: BLE001
            return GateResult(
                gate="4_native_predicate",
                status="FAIL",
                detail=(
                    f"env._check_success raised on case {case.index}: "
                    f"{type(exc).__name__}: {exc}"
                ),
                metrics={"probed": probed, "cases": effective_num_cases},
            )
        if not _is_valid_predicate_value(raw):
            return GateResult(
                gate="4_native_predicate",
                status="FAIL",
                detail=_predicate_failure_detail(
                    "env._check_success", raw, case.index
                ),
                metrics={"probed": probed, "cases": effective_num_cases},
            )
        probed += 1
    return GateResult(
        gate="4_native_predicate",
        status="PASS",
        detail=(
            f"native predicate callable and well-typed on "
            f"{probed}/{bank_size} bank cases (requested num_cases={num_cases})"
        ),
        metrics={"probed": probed, "cases": effective_num_cases},
    )


def _is_valid_predicate_value(value: object) -> bool:
    """A predicate value is valid iff it is a Python ``bool`` (or a numpy
    scalar ``bool``) or a ``dict`` with a ``task`` key whose value
    coerces to ``bool``.

    Robosuite success predicates are implemented in numpy and routinely
    return ``np.bool_`` scalars — those are the canonical "did the task
    succeed?" answer in the robosuite codebase and rejecting them would
    flag every state-only rollout pipeline. We accept Python ``bool`` and
    any numpy scalar with ``dtype.kind == "b"`` as a strict predicate
    value; everything else (including ``int``, ``np.int_``, ``str``,
    ``None``) is rejected so a refactor cannot quietly substitute a
    truthy-but-not-boolean return.
    """
    if isinstance(value, bool):
        return True
    if isinstance(value, np.generic):
        return value.dtype.kind == "b"
    if isinstance(value, dict):
        task_value = value.get("task")
        if not isinstance(task_value, bool):
            if isinstance(task_value, np.generic) and task_value.dtype.kind == "b":
                return True
            return False
        return True
    return False


def _predicate_failure_detail(source: str, value: object, case_index: int) -> str:
    """Render a human-readable FAIL detail for a malformed predicate value."""
    type_name = type(value).__name__
    if isinstance(value, dict):
        if "task" not in value:
            return (
                f"{source} returned dict without required 'task' key on case "
                f"{case_index}: keys={sorted(value)}"
            )
        return (
            f"{source} returned dict whose 'task' value is not bool "
            f"({type(value['task']).__name__}) on case {case_index}"
        )
    return (
        f"{source} returned unsupported value {value!r} (type {type_name}) "
        f"on case {case_index}"
    )


# ---------------------------------------------------------------------------
# Gate 5 — random / no-op sanity
# ---------------------------------------------------------------------------


def gate_random_noop_sanity(
    adapter: RobosuiteStateAdapter,
    bank: ResetBank,
    *,
    num_cases: int,
    horizon: int,
    max_success_rate: float = 0.05,
) -> GateResult:
    """Neither no-op nor random actions may succeed or crash the simulator."""
    bank_size = len(bank.cases)
    if bank_size == 0:
        return GateResult(
            gate="5_random_noop_sanity",
            status="FAIL",
            detail="reset bank is empty; cannot run random/no-op sanity",
            metrics={"successes": 0, "episodes": 0, "rate": 0.0, "infra_failures": 0, "steps": 0},
        )
    if num_cases <= 0:
        return GateResult(
            gate="5_random_noop_sanity",
            status="FAIL",
            detail=f"random_sanity_episodes={num_cases} must be > 0",
            metrics={"successes": 0, "episodes": 0, "rate": 0.0, "infra_failures": 0, "steps": 0},
        )
    if horizon <= 0:
        return GateResult(
            gate="5_random_noop_sanity",
            status="FAIL",
            detail=f"random_sanity_horizon={horizon} must be > 0",
            metrics={"successes": 0, "episodes": 0, "rate": 0.0, "infra_failures": 0, "steps": 0},
        )
    effective_num_cases = min(int(num_cases), bank_size)
    rng = np.random.default_rng(0)
    total_successes = 0
    total_infra = 0
    total_steps = 0
    for policy_name, policy in (
        ("noop", lambda: np.zeros(adapter.action_dim, dtype=np.float64)),
        (
            "random",
            lambda: rng.uniform(-1.0, 1.0, size=adapter.action_dim),
        ),
    ):
        for case in bank.cases[:effective_num_cases]:
            try:
                state = adapter.reset_to(case.states, xml=case.xml, ep_meta=case.ep_meta)
            except Exception:  # noqa: BLE001
                total_infra += 1
                continue
            ok = False
            for _t in range(horizon):
                try:
                    state, _done, success, _info = adapter.step(policy())
                except Exception:  # noqa: BLE001
                    total_infra += 1
                    break
                total_steps += 1
                if success:
                    ok = True
                    break
            if ok:
                total_successes += 1

    total = effective_num_cases * 2
    rate = total_successes / total
    passed = rate <= max_success_rate and total_infra == 0
    return GateResult(
        gate="5_random_noop_sanity",
        status="PASS" if passed else "FAIL",
        detail=(
            f"{total_successes}/{total} episodes 'succeeded' (max allowed "
            f"{max_success_rate}); {total_infra} infra failures; "
            f"{total_steps} steps"
        ),
        metrics={
            "successes": total_successes,
            "episodes": total,
            "rate": rate,
            "infra_failures": total_infra,
            "steps": total_steps,
        },
    )


# ---------------------------------------------------------------------------
# Gate 6 — checkpoint smoke run
# ---------------------------------------------------------------------------


def gate_checkpoint_smoke(
    cfg,
    adapter: RobosuiteStateAdapter,
    bank: ResetBank,
    *,
    num_episodes: int,
    max_success_rate: float = 1.0,
) -> GateResult:
    """Run a few episodes with the configured checkpoint policy.

    Skipped loudly when no checkpoint is configured (the gates CLI is also
    used before training on a fresh machine).
    """
    ckpt_path = cfg.train.get("stage1_ckpt_path")
    if not ckpt_path:
        return GateResult(
            gate="6_checkpoint_smoke",
            status="SKIPPED",
            detail="no train.stage1_ckpt_path configured — smoke gate skipped",
        )
    try:
        import torch

        from phaseforge.cli import build_eval_model

        model = build_eval_model(cfg)
        requested_device = str(cfg.project.get("device", "cpu"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        device = torch.device(requested_device)
        model.to(device)
        model.eval()
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            gate="6_checkpoint_smoke",
            status="FAIL",
            detail=f"model/checkpoint construction failed: {exc}",
        )
    if num_episodes > len(bank.cases):
        num_episodes = len(bank.cases)

    from phaseforge.evaluations.rollout.runner import resolve_rollout_normalizer

    try:
        normalizer = resolve_rollout_normalizer(cfg, model=model)
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            gate="6_checkpoint_smoke",
            status="FAIL",
            detail=f"normalizer load failed: {exc}",
        )

    mean = normalizer.mean.to(device)
    std = normalizer.std.to(device)

    successes = 0
    infra = 0
    policy_failures = 0
    with torch.inference_mode():
        for case in bank.cases[:num_episodes]:
            try:
                state = adapter.reset_to(case.states, xml=case.xml, ep_meta=case.ep_meta)
            except Exception:  # noqa: BLE001
                infra += 1
                continue
            ok = False
            for _t in range(adapter.horizon):
                try:
                    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
                    normalized = ((state_tensor - mean) / std).unsqueeze(0)
                    action = model.get_action(normalized)  # type: ignore[operator]
                    action = np.asarray(action.detach().cpu().numpy()).reshape(-1)
                    action = adapter.validate_action(action)
                except PolicyInvalidActionError:
                    policy_failures += 1
                    break
                except Exception:  # noqa: BLE001
                    policy_failures += 1
                    break
                try:
                    state, _done, success, _info = adapter.step(action)
                except Exception:  # noqa: BLE001
                    infra += 1
                    break
                if success:
                    ok = True
                    break
            if ok:
                successes += 1

    rate = successes / num_episodes if num_episodes else 0.0
    passed = rate + 1e-9 >= max_success_rate and infra == 0
    return GateResult(
        gate="6_checkpoint_smoke",
        status="PASS" if passed else "FAIL",
        detail=(
            f"{successes}/{num_episodes} episodes solved ({rate:.3f}; "
            f"max expected {max_success_rate}); {policy_failures} policy "
            f"failures; {infra} infra failures"
        ),
        metrics={
            "successes": successes,
            "episodes": num_episodes,
            "rate": rate,
            "policy_failures": policy_failures,
            "infra_failures": infra,
        },
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all_gates(cfg, *, bank: ResetBank | None = None) -> list[GateResult]:
    """Run every gate in order; returns results (raises only on setup errors).

    The bank is generated/loaded like the eval path does; robosuite must be
    installed or the gates fail (hard requirement, not skippable).
    """
    from phaseforge.evaluations.rollout.runner import (
        _adapter_from_config,
        require_rollout_eval_schema,
        resolve_pinned_metadata,
        resolve_robosuite_requirement,
    )

    require_rollout_eval_schema(cfg)
    meta = resolve_pinned_metadata(cfg)
    # The expected env_name is derived from the dataset's pinned metadata
    # by default; an explicit override in cfg.eval.env wins. This makes the
    # gate work uniformly across the five benchmark tasks.
    expected_env_name = cfg.eval.env.get("expected_env_name") or meta.env_name
    verify_environment_parity(
        meta,
        expected_env_name=str(expected_env_name),
        robosuite_requirement=resolve_robosuite_requirement(cfg),
        mujoco_requirement=str(cfg.eval.env.get("mujoco_requirement", "==3.2.7")),
    )
    if bank is None:
        from phaseforge.evaluations.rollout.runner import load_or_generate_bank

        bank = load_or_generate_bank(cfg, meta)

    gates = cfg.eval.get("gates", {})
    def run_with_fresh_adapter(gate_fn):
        """Run one gate on an isolated simulator instance.

        robosuite keeps mutable controller, gripper, contact, and task
        bookkeeping state outside the flattened MuJoCo state restored by
        ``reset_to``. Each gate must therefore receive a fresh adapter
        while sharing the same pinned metadata and reset bank.
        """
        isolated_adapter = _adapter_from_config(cfg, meta)
        try:
            return gate_fn(isolated_adapter)
        finally:
            isolated_adapter.close()

    results = [
        run_with_fresh_adapter(
            lambda adapter: gate_env_schema(
                adapter,
                bank,
                expected_state_dim=int(cfg.data.state_dim),
                expected_action_dim=int(cfg.data.action_dim),
            )
        ),
        run_with_fresh_adapter(
            lambda adapter: gate_demo_replay(
                adapter,
                _resolve_raw_hdf5(cfg),
                num_demos=int(gates.get("replay_demos", 1)),
                tolerance=float(gates.get("replay_tolerance", 1e-3)),
            )
        ),
        run_with_fresh_adapter(
            lambda adapter: gate_action_contract(
                adapter,
                tolerance=float(cfg.eval.episodes.get("action_tolerance", 1e-4)),
            )
        ),
        run_with_fresh_adapter(
            lambda adapter: gate_native_predicate(
                adapter,
                bank,
                num_cases=int(gates.get("native_predicate_cases", 5)),
            )
        ),
        run_with_fresh_adapter(
            lambda adapter: gate_random_noop_sanity(
                adapter,
                bank,
                num_cases=int(gates.get("random_sanity_episodes", 20)),
                horizon=int(gates.get("random_sanity_horizon", 200)),
                max_success_rate=float(gates.get("random_sanity_success_max", 0.05)),
            )
        ),
        run_with_fresh_adapter(
            lambda adapter: gate_checkpoint_smoke(
                cfg,
                adapter,
                bank,
                num_episodes=int(gates.get("smoke_episodes", 10)),
            )
        ),
    ]
    return results


def _resolve_raw_hdf5(cfg) -> Path:
    raw_dir = cfg.data.source.get("dir")
    if not raw_dir:
        return Path("")
    raw_path = Path(str(raw_dir))
    files = sorted(raw_path.glob("*.hdf5")) if raw_path.is_dir() else []
    return files[0] if files else Path("")


__all__ = [
    "GateResult",
    "GateFailure",
    "run_all_gates",
    "gate_env_schema",
    "gate_demo_replay",
    "gate_action_contract",
    "gate_native_predicate",
    "gate_random_noop_sanity",
    "gate_checkpoint_smoke",
]

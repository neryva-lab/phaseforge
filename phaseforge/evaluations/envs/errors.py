"""Error taxonomy for the rollout environment layer.

Every failure mode is classified explicitly so the rollout runner can
apply the locked strict metric:

* :class:`EnvParityError` — simulator/dataset version or schema mismatch.
  Fail closed before any rollout.
* :class:`StateSchemaError` — the simulator produced observations that do
  not match the declared state schema.
* :class:`PolicyInvalidActionError` — the policy emitted a NaN, non-finite,
  or out-of-range action. Under the strict metric this is a *policy
  failure* (valid episode, success=False), never a simulator error.
* :class:`InfrastructureError` — anything the simulator or environment
  raises while the policy input is valid. These are *infrastructure
  failures* (valid episode=False, excluded from the success denominator)
  and invalidate the run until rerun.
"""

from __future__ import annotations


class EnvParityError(RuntimeError):
    """Simulator/dataset version, environment, or schema mismatch.

    Raised when the installed robosuite/MuJoCo versions, environment name,
    or environment metadata do not match the pinned dataset contract.
    The caller must stop before any rollout.
    """


class StateSchemaError(ValueError):
    """The simulator observation does not match the declared state schema.

    Raised when a declared state key is missing or has the wrong dimension
    (assert the expected state dimension before the first action).
    """


class PolicyInvalidActionError(ValueError):
    """The policy produced a NaN, non-finite, or out-of-range action.

    Under the strict metric this is a policy failure: the episode is
    recorded as valid with ``success=False`` and
    ``failure_category="policy_invalid_action"`` — it is never silently
    removed and never relabeled as a simulator error.
    """


class InfrastructureError(RuntimeError):
    """A simulator/environment failure while the policy input was valid.

    Recorded as an invalid episode (``valid_episode=False``), excluded from
    the success denominator, and it invalidates the run until rerun.
    """


__all__ = [
    "EnvParityError",
    "StateSchemaError",
    "PolicyInvalidActionError",
    "InfrastructureError",
]

"""Observation-consistent topological regime discovery.

Discovers manipulation regimes from task-space variables that are directly
observable in the instantaneous state ``x_t`` — never from privileged
transition features ``phi_t = [x_t, a_t, Δx_t]`` (see
:mod:`phaseforge.data.dynamics.features`). The sticky SLDS stays available
as an offline diagnostic only.
"""

from __future__ import annotations

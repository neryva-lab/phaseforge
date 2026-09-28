"""Experiment runner: orchestrate complete, resumable protocol runs.

See :func:`phaseforge.runner.cli.main` for the command line interface
(``phaseforge-sweep`` or ``python -m phaseforge.runner``).
"""

from phaseforge.runner.cli import main

__all__ = ["main"]

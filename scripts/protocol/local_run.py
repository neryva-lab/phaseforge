"""Local launcher for phaseforge-train / phaseforge-eval.

Compatibility shim: hydra-core 1.3.5 passes a non-string ``LazyCompletionHelp``
object as an argparse help text, which Python 3.14's strict argparse
validation rejects (``badly formed help string``). Coerce non-string help
text before the hydra CLI runs; the training/eval code path is otherwise
identical to the console scripts.

Usage::

    uv run python scripts/protocol/local_run.py train models=precision_residual_phaseforge train=stage1 ...
    uv run python scripts/protocol/local_run.py eval  models=precision_residual_phaseforge eval=rollout ...
"""

import argparse
import sys


def _patch_argparse() -> None:
    orig_expand = argparse.HelpFormatter._expand_help

    def _safe_expand(self, action):
        if not isinstance(action.help, str):
            action.help = str(action.help)
        return orig_expand(self, action)

    argparse.HelpFormatter._expand_help = _safe_expand


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "eval"):
        print(__doc__)
        return 2
    kind = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    _patch_argparse()

    if kind == "train":
        from phaseforge.cli import train

        train()
    else:
        from phaseforge.cli import evaluate

        evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

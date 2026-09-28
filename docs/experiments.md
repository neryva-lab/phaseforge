# Experiment Protocol

This document describes the frozen experiment protocol: the manifests that define the method matrix, the locked confirmation contract, and the preflight validation that runs before any GPU time is spent.

## Frozen manifests

The complete experiment matrix is declared in version-controlled JSON manifests under `experiments/`. These manifests are frozen: the analysis pipeline builds its method and task registry from them via the runner's own protocol loader, so the analysis can never disagree with what ran.

- `experiments/final_causal_matrix.json` — the final same-generation baseline and ablation matrix for the proposed method `precision_residual_phaseforge`. It declares 50 method entries (10 distinct methods × 5 tasks: Lift, Can, Square, ToolHang, Transport) × seeds [42, 43, 44, 45, 46, 47], with `defaults: ["train.early_stopping.enabled=false"]`.
- `experiments/router_initialization_ablation.json` — the controlled router-initialization ablation, together with its `can_square` composition manifest and the per-task `can`, `lift`, and `square` manifests.

The per-task ablation manifests are composed at load time through the `manifests` key (`phaseforge/runner/protocol.py`): the parent manifest lists sub-manifest paths, which are resolved relative to the parent and loaded into a single protocol. These files are load-bearing composition units, not drafts, and must not be deleted or edited independently of the manifests that reference them.

See [training.md](training.md) for how `phaseforge-sweep` executes these manifests, and [data.md](data.md) for provisioning the datasets they require.

## Locked confirmation contract

The proposed-method rows of `final_causal_matrix.json` preserve a locked confirmation contract, quoted from the manifest description:

- topo_pelt_k6 prototype init
- phase labels for Stage 1/SupCon/margin
- beta-zero residual experts
- margin lambda 0.05
- disabled Lipschitz/gain regularization

Any run that deviates from this contract is outside the frozen protocol and is not part of the reported results.

## Preflight validation

`scripts/protocol/preflight_configs.py` validates the full experiment matrix before GPU time is spent. It composes every (method, task, stage, seed) cell from the frozen manifest through Hydra's compose API and checks each resolved config for:

- config-group existence,
- `num_phases` consistency between the data phase labeler and the model phase head,
- the checkpoint monitor rule `val/loss_action` for both stages,
- stage-2 `freeze_encoder` with a resolved Stage 1 source,
- scheduler `T_max` covering the full epoch budget,
- `eval` group consistency.

Usage:

```bash
uv run python scripts/protocol/preflight_configs.py
uv run python scripts/protocol/preflight_configs.py --methods bc
uv run python scripts/protocol/preflight_configs.py --tasks Lift
uv run python scripts/protocol/preflight_configs.py --manifest experiments/final_causal_matrix.json
```

The exit code is 0 when every cell passes. Failures are collected and reported per cell with the exact failing override, so a broken configuration is identified before any training starts.

## See also

- [installation.md](installation.md) — environment setup
- [quickstart.md](quickstart.md) — minimal end-to-end run
- [training.md](training.md) — training stages and the sweep runner
- [evaluation.md](evaluation.md) — offline metrics, gates, and rollout reporting

"""Script to compute and generate fairness, parameter, and compute accounting tables.

Evaluates parameter counts, active capacity per forward pass, Stage-2 trainable parameters,
FLOP estimates, Stage 1 / Stage 2 epoch budgets, and shared Stage-1 sources across all methods.

Deployed policy parameter counts exclude detached Stage 1 heads (ActionHead / PhaseHead).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch.nn as nn
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelFairnessRecord:
    method_name: str
    model_class: str
    deployed_params: int
    excluded_heads_params: int
    stage2_trainable_params: int
    active_params_per_sample: int
    active_param_ratio_vs_bc: float
    forward_flops_approx: int
    stage1_epochs: int
    stage2_epochs: int
    shared_stage1_source: str
    batch_size: int
    total_optimizer_steps: int
    total_examples_seen: int


def _count_parameters(module: nn.Module, requires_grad_only: bool = False) -> int:
    if requires_grad_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def _estimate_mlp_flops(in_dim: int, hidden_dims: list[int], out_dim: int) -> int:
    """Estimate FLOPs for an MLP (Linear layers: 2 * in * out + out for bias/activation)."""
    flops = 0
    curr_in = in_dim
    for h in hidden_dims:
        flops += 2 * curr_in * h + h
        curr_in = h
    flops += 2 * curr_in * out_dim + out_dim
    return flops


def calculate_model_accounting(
    cfg: DictConfig, method_name: str, stage1_source: str
) -> ModelFairnessRecord:
    """Analyze parameter counts and compute metrics for a configured model."""
    from phaseforge.utils.registry import build_model

    model: nn.Module = build_model(cfg)

    state_dim = int(cfg.data.get("state_dim", 19))
    action_dim = int(cfg.data.get("action_dim", 7))
    latent_dim = int(cfg.models.get("encoder", {}).get("latent_dim", 128))
    enc_hidden = list(cfg.models.get("encoder", {}).get("hidden_dims", [256, 256, 256]))

    # Encoder FLOPs
    encoder_flops = _estimate_mlp_flops(state_dim, enc_hidden, latent_dim)

    # Active parameters & FLOPs
    if hasattr(model, "moe_layer"):
        moe = getattr(model, "moe_layer")
        top_k = int(moe.router.top_k)
        num_experts = int(moe.router.num_experts)
        single_expert_params = _count_parameters(moe.experts[0])
        router_params = _count_parameters(moe.router)
        encoder_params = _count_parameters(model.encoder)

        # Deployed parameters (encoder + router + experts, excluding detached Stage 1 heads)
        deployed_params = encoder_params + router_params + num_experts * single_expert_params

        # Calculate excluded heads
        excluded_heads = 0
        if hasattr(model, "action_head"):
            excluded_heads += _count_parameters(model.action_head)
        if hasattr(model, "phase_head"):
            excluded_heads += _count_parameters(model.phase_head)

        # Stage 2 trainable parameters
        models_cfg = cfg.get("models", {})
        freeze_encoder = (
            bool(models_cfg.get("freeze_encoder"))
            if "freeze_encoder" in models_cfg
            else bool(cfg.train.get("freeze_encoder", True))
        )
        if freeze_encoder:
            stage2_trainable = num_experts * single_expert_params + router_params
        else:
            stage2_trainable = deployed_params

        # Active params per sample: encoder + router + top_k * single expert
        active_params = encoder_params + router_params + top_k * single_expert_params

        # FLOPs: encoder + router projection + top_k experts + top_k softmax combine
        router_flops = 2 * latent_dim * num_experts + num_experts
        expert_hidden = list(cfg.models.get("expert", {}).get("hidden_dims", [256]))
        single_expert_flops = _estimate_mlp_flops(latent_dim, expert_hidden, action_dim)
        forward_flops = (
            encoder_flops
            + router_flops
            + top_k * single_expert_flops
            + top_k * action_dim
        )

    elif hasattr(model, "rnn"):
        # Temporal BC (RNN): encoder + recurrence over the declared history
        # window + action head, all run per history step to emit one action.
        deployed_params = _count_parameters(model)
        excluded_heads = 0
        encoder_params = _count_parameters(model.encoder)
        stage2_trainable = deployed_params
        active_params = deployed_params
        seq_len = int(cfg.data.get("sequence_length", 1))
        rnn_hidden = int(model.hidden_dim)
        rnn_layers = int(model.num_layers)
        rnn_flops = 0
        curr_in = latent_dim
        for _ in range(rnn_layers):
            # 4 gates, each an (in -> h) + (h -> h) matmul: 4 * 2*h*(in + h)
            rnn_flops += 4 * 2 * rnn_hidden * (curr_in + rnn_hidden)
            curr_in = rnn_hidden
        rnn_flops *= seq_len
        act_hidden = [int(cfg.models.action_head.get("hidden_dim", 256))]
        action_head_flops = _estimate_mlp_flops(rnn_hidden, act_hidden, action_dim)
        forward_flops = seq_len * (encoder_flops + action_head_flops) + rnn_flops

    elif hasattr(model, "action_head"):
        # Dense BC model
        deployed_params = _count_parameters(model)
        excluded_heads = 0
        encoder_params = _count_parameters(model.encoder)
        stage2_trainable = deployed_params
        active_params = deployed_params
        act_hidden = [int(cfg.models.action_head.get("hidden_dim", 256))]
        action_head_flops = _estimate_mlp_flops(latent_dim, act_hidden, action_dim)
        forward_flops = encoder_flops + action_head_flops
    else:
        deployed_params = _count_parameters(model)
        excluded_heads = 0
        stage2_trainable = deployed_params
        active_params = deployed_params
        forward_flops = encoder_flops

    # Baseline BC active params for ratio
    bc_params = 206983
    active_ratio = active_params / bc_params

    # Epoch allocation
    batch_size = int(cfg.data.get("batch_size", 256))
    stage1_epochs = 100 if method_name in (
        "precision_residual_phaseforge", "bc", "final_aligned_static_rule",
    ) else 0
    stage2_epochs = 200 if method_name != "bc" else 0
    total_epochs = stage1_epochs + stage2_epochs

    # Approx dataset size 10,000 steps
    n_samples = 10000
    steps_per_epoch = n_samples // batch_size
    total_steps = total_epochs * steps_per_epoch
    total_examples = total_steps * batch_size

    return ModelFairnessRecord(
        method_name=method_name,
        model_class=type(model).__name__,
        deployed_params=deployed_params,
        excluded_heads_params=excluded_heads,
        stage2_trainable_params=stage2_trainable,
        active_params_per_sample=active_params,
        active_param_ratio_vs_bc=active_ratio,
        forward_flops_approx=forward_flops,
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        shared_stage1_source=stage1_source,
        batch_size=batch_size,
        total_optimizer_steps=total_steps,
        total_examples_seen=total_examples,
    )


def format_markdown_table(records: list[ModelFairnessRecord]) -> str:
    lines = [
        "| Method | Deployed Params | Excluded Heads* | Stage-2 Trainable "
        "| Active Params / Sample | Active Ratio vs BC | Forward FLOPs | Epochs (S1/S2) "
        "| Shared Stage-1 Source |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for r in records:
        ex_str = f"{r.excluded_heads_params:,}" if r.excluded_heads_params > 0 else "-"
        lines.append(
            f"| `{r.method_name}` | {r.deployed_params:,} | {ex_str} | "
            f"{r.stage2_trainable_params:,} | {r.active_params_per_sample:,} | "
            f"{r.active_param_ratio_vs_bc:.3f}× | {r.forward_flops_approx:,} | "
            f"{r.stage1_epochs} / {r.stage2_epochs} | `{r.shared_stage1_source}` |"
        )
    lines.extend([
        "",
        "\\* *Note: Excluded Heads refers to detached Stage 1 ActionHead/PhaseHead "
        "parameters that are frozen and not in the Stage 2 computation graph or deployed policy.*",
        "\\* *Stage 1 pretraining runs are counted once per provider method "
        "(`phaseforge`, `bc`) and reused across consumers via dependency injection.*",
    ])
    return "\n".join(lines)


def format_latex_table(records: list[ModelFairnessRecord]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Deployed} & \textbf{Excluded} & \textbf{S2 Trainable} "
        r"& \textbf{Active Params} & \textbf{Active / BC} & \textbf{FLOPs} & \textbf{S1 Source} \\",
        r"\midrule",
    ]
    for r in records:
        method_clean = r.method_name.replace("_", r"\_")
        ex_str = f"{r.excluded_heads_params:,}" if r.excluded_heads_params > 0 else "--"
        src_clean = r.shared_stage1_source.replace("_", r"\_")
        lines.append(
            f"{method_clean} & {r.deployed_params:,} & {ex_str} & "
            f"{r.stage2_trainable_params:,} & {r.active_params_per_sample:,} & "
            f"{r.active_param_ratio_vs_bc:.2f}$\\times$ & "
            f"{r.forward_flops_approx:,} & {src_clean} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Parameter count, active inference capacity, and computational fairness. "
        r"Deployed policy parameters match PhaseForge across all 6-expert MoE models.}",
        r"\label{tab:fairness_accounting}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main() -> int:
    from hydra import compose, initialize_config_dir

    config_dir = str(PROJECT_ROOT / "phaseforge" / "config")
    methods = [
        ("bc", "final_aligned_bc", "self"),
        ("precision_residual_phaseforge", "precision_residual_phaseforge", "self"),
        ("precision_residual_plain_encoder", "precision_residual_plain_encoder", "final_aligned_bc"),
        ("precision_residual_phase_random_router", "precision_residual_phase_random_router", "precision_residual_phaseforge"),
        ("precision_residual_scratch_moe", "precision_residual_scratch_moe", "precision_residual_phaseforge"),
        ("precision_residual_factorial_floor", "precision_residual_factorial_floor", "final_aligned_bc"),
        ("final_aligned_softmax_top1", "final_aligned_softmax_top1", "precision_residual_phaseforge"),
        ("final_aligned_static_rule", "final_aligned_static_rule", "self"),
        ("precision_residual_teacher_forced", "precision_residual_teacher_forced", "precision_residual_phaseforge"),
        ("precision_residual_oracle", "precision_residual_oracle", "precision_residual_phaseforge"),
    ]

    records: list[ModelFairnessRecord] = []
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        for method_name, model_override, s1_source in methods:
            data_override = "common"
            cfg = compose(
                config_name="main",
                overrides=[f"models={model_override}", f"data={data_override}", "train=stage2"],
            )
            rec = calculate_model_accounting(cfg, method_name, s1_source)
            records.append(rec)

    print("\n" + "=" * 90)
    print("PHASEFORGE RESEARCH FAIRNESS & COMPUTE ACCOUNTING TABLE")
    print("=" * 90 + "\n")
    md_table = format_markdown_table(records)
    print(md_table)
    print("\n" + "-" * 90)
    print("LATEX SOURCE:")
    print("-" * 90)
    print(format_latex_table(records))
    print("\n" + "=" * 90)

    # Save to markdown file in docs/
    out_path = PROJECT_ROOT / "docs" / "plan" / "reports" / "fairness_accounting.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"# PhaseForge Fairness & Compute Accounting\n\n"
        f"**Scope:** parameter, active-capacity, and compute accounting for the research matrix "
        f"(`../specs/research_definition.md` §5–§6 cells), generated by "
        f"`scripts/analysis/fairness_accounting.py`. Deployed policy "
        f"counts exclude the detached Stage-1 heads.\n\n{md_table}\n",
        encoding="utf-8",
    )
    print(f"Persisted fairness accounting to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

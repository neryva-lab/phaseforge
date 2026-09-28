"""Diagnostic metrics for t=0 initial MoE routing state after bootstrapping.

Evaluates the initial inductive routing prior before any Stage 2 gradient updates:
- NMI(phase, top1_routing)
- Mean routing entropy
- Top-1 and Top-k balance metrics
- Routing collapse rate (fraction of dead experts)
- Router vs PhaseHead prediction agreement (when phase head present)
- PhaseHead classification accuracy
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger("phaseforge.init_diagnostics")


@torch.no_grad()
def compute_init_routing_diagnostics(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device | str = "cuda",
) -> dict[str, Any]:
    """Run one evaluation pass over validation data to measure t=0 routing properties.

    Args:
        model: Model after bootstrap_moe has executed.
        val_loader: Validation DataLoader.
        device: Compute device.

    Returns:
        Dictionary of t=0 routing and phase metrics.
    """
    model.eval()
    has_moe = hasattr(model, "moe_layer") and hasattr(model.moe_layer, "router")
    has_phase_head = hasattr(model, "phase_head")

    if not has_moe:
        logger.info("Init diagnostics: Model has no standard MoE router. Skipping router metrics.")
        return {}

    moe = model.moe_layer
    num_experts = len(moe.experts)
    top_k = moe.router.top_k

    all_phases = []
    all_top1_experts = []
    all_topk_experts = []
    all_routing_entropies = []
    all_phase_head_preds = []

    non_blocking = torch.device(device).type == "cuda"

    for batch in val_loader:
        state = batch["state"].to(device, non_blocking=non_blocking)
        phase = batch["phase"].to(device, non_blocking=non_blocking)

        if state.ndim == 3:
            state = state.view(-1, state.size(-1))
            phase = phase.view(-1)

        latent = model.encoder(state)

        # 1. Routing logits via the model's actual dispatch path (supports
        #    final teacher diagnostic, whose router is structural parity only and never
        #    trained). Falls back to the raw router if the model exposes none.
        model_out = model({"state": state})
        if model_out.gate_logits is not None:
            gate_logits = model_out.gate_logits
        else:
            gate_logits = moe.router(latent).gate_logits
        probs = F.softmax(gate_logits, dim=-1)

        top1 = probs.argmax(dim=-1)
        topk = probs.topk(top_k, dim=-1).indices

        # Routing entropy: - sum p log p
        eps = 1e-12
        entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)

        all_phases.append(phase.cpu())
        all_top1_experts.append(top1.cpu())
        all_topk_experts.append(topk.cpu())
        all_routing_entropies.append(entropy.cpu())

        # 2. Phase head forward if present
        if has_phase_head:
            phase_logits = model.phase_head(latent)
            p_pred = phase_logits.argmax(dim=-1)
            all_phase_head_preds.append(p_pred.cpu())

    if not all_phases:
        return {}

    phases_cat = torch.cat(all_phases, dim=0).numpy()
    top1_cat = torch.cat(all_top1_experts, dim=0).numpy()
    topk_cat = torch.cat(all_topk_experts, dim=0).numpy()
    entropies_cat = torch.cat(all_routing_entropies, dim=0).numpy()

    n_samples = len(phases_cat)

    # 1. NMI(phase, top1_routing)
    from sklearn.metrics import normalized_mutual_info_score

    nmi = float(normalized_mutual_info_score(phases_cat, top1_cat))

    # 2. Mean routing entropy
    mean_entropy = float(entropies_cat.mean())
    max_entropy = math.log(num_experts)
    normalized_entropy = float(mean_entropy / max(max_entropy, 1e-8))

    # 3. Top-1 expert distribution & balance
    top1_counts = np.bincount(top1_cat, minlength=num_experts)
    top1_freqs = top1_counts / n_samples
    top1_cv = float(top1_freqs.std() / (top1_freqs.mean() + 1e-8))

    # 4. Top-k expert distribution & balance
    topk_flat = topk_cat.flatten()
    topk_counts = np.bincount(topk_flat, minlength=num_experts)
    topk_freqs = topk_counts / (n_samples * top_k)
    topk_cv = float(topk_freqs.std() / (topk_freqs.mean() + 1e-8))

    # 5. Dead / collapsed expert rate (< 1% tokens assigned)
    collapse_threshold = 0.01 / num_experts
    dead_experts = int((top1_freqs < collapse_threshold).sum())
    collapse_rate = float(dead_experts / num_experts)

    # 6. Phase head metrics & agreement if available
    phase_head_acc = None
    router_phase_agreement = None
    if has_phase_head and all_phase_head_preds:
        phase_preds_cat = torch.cat(all_phase_head_preds, dim=0).numpy()
        phase_head_acc = float((phase_preds_cat == phases_cat).mean())
        router_phase_agreement = float((top1_cat == phase_preds_cat).mean())

    diagnostics = {
        "t0_nmi_phase_top1": nmi,
        "t0_mean_routing_entropy": mean_entropy,
        "t0_normalized_routing_entropy": normalized_entropy,
        "t0_top1_coefficient_of_variation": top1_cv,
        "t0_topk_coefficient_of_variation": topk_cv,
        "t0_top1_expert_frequencies": top1_freqs.tolist(),
        "t0_collapse_rate": collapse_rate,
        "t0_dead_expert_count": dead_experts,
        "t0_phase_head_accuracy": phase_head_acc,
        "t0_router_phase_head_agreement": router_phase_agreement,
    }

    logger.info(
        f"t=0 Routing Diagnostics: NMI={nmi:.4f}, Entropy={mean_entropy:.3f} "
        f"({normalized_entropy:.1%}), Top-1 CV={top1_cv:.3f}, "
        f"Dead Experts={dead_experts}/{num_experts}"
        + (f", PhaseHead Acc={phase_head_acc:.2%}" if phase_head_acc is not None else "")
    )

    return diagnostics

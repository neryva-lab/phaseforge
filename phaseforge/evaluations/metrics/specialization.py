"""Direct expert specialization metrics.

Computes:
1. M_{z, e} behavioral matrix: MSE(pi_e(x_z), a_z) for every phase z and expert e.
   Runs each expert independently across all validation latents to measure how well
   expert e performs on phase regime z.
2. Best expert per phase: e*(z) = argmin_e M_{z, e} and theoretical minimum error M_{z, e*(z)}
   vs actual routed MSE M_{z, routed}.
3. Expert pairwise output divergence: D(e_i, e_j) = E[||pi_i(x) - pi_j(x)||_2] over shared inputs.
4. Routing contingency matrix: C_{p, e} = P(expert e | phase p).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader

from phaseforge.models.base import BaseManipulationModel

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_specialization_matrix(
    model: BaseManipulationModel,
    dataloader: DataLoader,
    device: torch.device | str = "cuda",
    num_phases: int | None = None,
) -> dict[str, Any]:
    """Compute the comprehensive expert specialization matrix and diagnostics.

    Args:
        model: Trained BaseManipulationModel (must have encoder and moe_layer).
        dataloader: Evaluation / validation DataLoader.
        device: Target compute device.
        num_phases: Total number of phases P (if None, inferred from phase_head or max label).

    Returns:
        Dictionary containing:
        - "expert_phase_mse_matrix": list of lists (P x E) of floats representing M_{z, e}.
        - "best_expert_per_phase": dict mapping phase index -> best expert index.
        - "theoretical_best_phase_mse": dict mapping phase index -> MSE of best expert.
        - "routed_phase_mse": dict mapping phase index -> MSE of routed policy on that phase.
        - "theoretical_mean_mse": mean over phases of theoretical best MSE.
        - "expert_pairwise_divergence": list of lists (E x E) of mean L2 distances
          between expert outputs.
        - "mean_expert_divergence": scalar mean pairwise divergence across all pairs i < j.
        - "phase_expert_contingency": list of lists (P x E) of normalized assignment probabilities.
    """
    model.eval()
    model.to(device)

    if not hasattr(model, "moe_layer") or not hasattr(model, "encoder"):
        logger.warning("Model does not have moe_layer or encoder; skipping specialization matrix.")
        return {}

    moe_layer = getattr(model, "moe_layer")
    num_experts = len(moe_layer.experts)

    # Inferred phase count
    if num_phases is None:
        if hasattr(model, "phase_head") and hasattr(model.phase_head, "num_phases"):
            num_phases = model.phase_head.num_phases
        elif hasattr(model, "num_phases"):
            num_phases = getattr(model, "num_phases")
        else:
            num_phases = 6

    # Accumulators for each (phase z, expert e)
    # phase_expert_sq_err: (P, E), phase_routed_sq_err: (P,), phase_action_counts: (P,)
    phase_expert_sq_err = torch.zeros((num_phases, num_experts), dtype=torch.float64, device=device)
    phase_routed_sq_err = torch.zeros((num_phases,), dtype=torch.float64, device=device)
    phase_action_counts = torch.zeros((num_phases,), dtype=torch.float64, device=device)

    # Accumulator for pairwise expert output differences: (E, E)
    pairwise_l2_sum = torch.zeros((num_experts, num_experts), dtype=torch.float64, device=device)
    total_samples = 0

    # Accumulator for contingency matrix: (P, E)
    contingency_counts = torch.zeros((num_phases, num_experts), dtype=torch.float64, device=device)

    non_blocking = torch.device(device).type == "cuda"

    for batch in dataloader:
        state = batch["state"].to(device, non_blocking=non_blocking)
        action_target = batch["action"].to(device, non_blocking=non_blocking)
        phase = batch["phase"].to(device, non_blocking=non_blocking)

        if state.ndim == 3:
            state = state.view(-1, state.size(-1))
            action_target = action_target.view(-1, action_target.size(-1))
            phase = phase.view(-1)

        b = state.size(0)
        total_samples += b

        # 1. Forward through frozen/trained encoder
        latent = model.encoder(state)

        # 2. Forward through all individual experts independently: expert_preds is (E, B, A)
        expert_preds = torch.stack([expert(latent) for expert in moe_layer.experts], dim=0)

        # 3. Forward through full model (supporting PhaseMoE, TeacherForced, etc.)
        #    forward() takes a batch dict (reads batch["state"]) and returns
        #    ModelOutput with the action prediction in ``action_pred``.
        model_out = model(batch)
        routed_pred = model_out.action_pred
        if model_out.gate_logits is not None:
            top1_expert = model_out.gate_logits.argmax(dim=-1)
        elif model_out.routing_weights is not None:
            top1_expert = model_out.routing_weights.argmax(dim=-1)
        else:
            top1_expert = phase % num_experts

        # 4. Update pairwise expert output divergence
        # expert_preds: (E, B, A)
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                diff = expert_preds[i] - expert_preds[j]
                l2_dist = torch.norm(diff, p=2, dim=-1).sum().to(torch.float64)
                pairwise_l2_sum[i, j] += l2_dist
                pairwise_l2_sum[j, i] += l2_dist

        # 5. Update per-phase error accumulators and contingency
        for p in range(num_phases):
            mask = phase == p
            if not mask.any():
                continue

            n_p = mask.sum().item()
            target_p = action_target[mask]  # (N_p, A)
            routed_p = routed_pred[mask]    # (N_p, A)
            top1_p = top1_expert[mask]      # (N_p,)

            act_dim = target_p.size(-1)
            phase_action_counts[p] += n_p * act_dim

            phase_routed_sq_err[p] += ((routed_p - target_p) ** 2).sum().to(torch.float64)

            bincount = torch.bincount(top1_p, minlength=num_experts).to(torch.float64)
            contingency_counts[p] += bincount

            for e in range(num_experts):
                pred_e_p = expert_preds[e][mask]  # (N_p, A)
                sq_err = ((pred_e_p - target_p) ** 2).sum().to(torch.float64)
                phase_expert_sq_err[p, e] += sq_err

    # Compute final M_{z, e} matrix
    # Avoid division by zero for unobserved phases
    safe_counts = torch.clamp(phase_action_counts, min=1.0).unsqueeze(1)
    mse_matrix_tensor = phase_expert_sq_err / safe_counts
    mse_matrix = mse_matrix_tensor.cpu().tolist()

    # Routed MSE per phase
    safe_counts_1d = torch.clamp(phase_action_counts, min=1.0)
    routed_mse_tensor = phase_routed_sq_err / safe_counts_1d
    routed_mse_dict = {f"phase_{p}": float(routed_mse_tensor[p].item()) for p in range(num_phases)}

    # Best expert per phase
    best_expert_dict = {}
    best_expert_mse_dict = {}
    for p in range(num_phases):
        if phase_action_counts[p].item() > 0:
            best_e = int(mse_matrix_tensor[p].argmin().item())
            best_mse = float(mse_matrix_tensor[p, best_e].item())
            best_expert_dict[f"phase_{p}"] = best_e
            best_expert_mse_dict[f"phase_{p}"] = best_mse
        else:
            best_expert_dict[f"phase_{p}"] = -1
            best_expert_mse_dict[f"phase_{p}"] = float("nan")

    # Pairwise divergence matrix
    safe_samples = max(total_samples, 1)
    pairwise_div_tensor = pairwise_l2_sum / safe_samples
    pairwise_div_matrix = pairwise_div_tensor.cpu().tolist()

    if num_experts > 1:
        num_pairs = num_experts * (num_experts - 1) / 2
        upper_sum = torch.triu(pairwise_div_tensor, diagonal=1).sum().item()
        mean_divergence = float(upper_sum / num_pairs)
    else:
        mean_divergence = 0.0

    # Normalized contingency matrix (each row sums to 1.0)
    row_sums = torch.clamp(contingency_counts.sum(dim=1, keepdim=True), min=1.0)
    norm_contingency = (contingency_counts / row_sums).cpu().tolist()

    valid_phases = [p for p in range(num_phases) if phase_action_counts[p].item() > 0]
    theo_mean = (
        float(sum(best_expert_mse_dict[f"phase_{p}"] for p in valid_phases) / len(valid_phases))
        if valid_phases
        else float("nan")
    )
    routed_mean = (
        float(sum(routed_mse_dict[f"phase_{p}"] for p in valid_phases) / len(valid_phases))
        if valid_phases
        else float("nan")
    )

    logger.info(
        f"Specialization Summary: Mean Expert Divergence={mean_divergence:.4f}, "
        f"Theoretical Best MSE={theo_mean:.6f}, Routed MSE={routed_mean:.6f}"
    )

    return {
        "expert_phase_mse_matrix": mse_matrix,
        "best_expert_per_phase": best_expert_dict,
        "theoretical_best_phase_mse": best_expert_mse_dict,
        "routed_phase_mse": routed_mse_dict,
        "theoretical_mean_mse": theo_mean,
        "routed_mean_mse": routed_mean,
        "expert_pairwise_divergence": pairwise_div_matrix,
        "mean_expert_divergence": mean_divergence,
        "phase_expert_contingency": norm_contingency,
    }

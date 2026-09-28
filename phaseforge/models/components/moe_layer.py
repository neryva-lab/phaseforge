"""MoELayer: Encapsulates routing logic and multiple experts."""

from __future__ import annotations

import copy
from typing import NamedTuple, cast

import torch
import torch.nn as nn
from torch import Tensor

from phaseforge.models.components.expert import ExpertMLP
from phaseforge.models.components.impedance_expert import (
    ImpedanceExpert,
    ResidualImpedanceExpert,
)
from phaseforge.models.components.prototype_router import PrototypeRouter
from phaseforge.models.components.router import RouterOutput, TopKRouter


class MoEOutput(NamedTuple):
    """Output from the MoE Layer."""

    combined_output: Tensor  # (B, output_dim) Final action prediction
    routing_weights: Tensor  # (B, K)
    expert_indices: Tensor  # (B, K)
    balance_loss: Tensor  # scalar
    gate_logits: Tensor  # (B, E)
    sticky_loss: Tensor  # scalar, raw history-stickiness loss (V2-C)
    info: dict[str, Tensor] | None = None  # impedance diagnostics
    # (B, E) deterministic pre-exploration logits.
    clean_gate_logits: Tensor | None = None


def _clean_logits(router_out: RouterOutput) -> Tensor:
    """Return the deterministic pre-exploration logits with fallback.

    Routers reporting ``clean_gate_logits`` use it; any foreign
    router-like returning a legacy 5-field output falls back to
    ``gate_logits`` so dispatch never breaks.
    """
    clean = getattr(router_out, "clean_gate_logits", None)
    return clean if clean is not None else router_out.gate_logits


class MoELayer(nn.Module):
    """Mixture-of-Experts Layer orchestrating the router and experts.

    This layer takes the latent representation, uses the router to determine
    expert assignments and weights, dynamically dispatches inputs to the assigned
    experts, and combines their outputs via a weighted sum. The router is any
    module following the ``TopKRouter`` call protocol — either ``TopKRouter``
    (soft top-k) or ``PrototypeRouter`` (hard top-1 Voronoi); both report the
    standard :class:`RouterOutput`, so dispatch needs no router-specific branch.

    Args:
        router: Instantiated TopKRouter or PrototypeRouter.
        experts: A list (or nn.ModuleList) of instantiated ExpertMLP networks.
            If a single ExpertMLP is provided, it will be cloned `num_experts` times.
    """

    def __init__(
        self,
        router: TopKRouter | PrototypeRouter,
        experts: ExpertMLP
        | ImpedanceExpert
        | ResidualImpedanceExpert
        | nn.ModuleList
        | list[ExpertMLP]
        | list[ImpedanceExpert]
        | list[ResidualImpedanceExpert],
    ) -> None:
        super().__init__()
        self.router = router

        # Handle expert instantiation
        if isinstance(experts, nn.ModuleList):
            self.experts: nn.ModuleList[ExpertMLP] = experts  # type: ignore[type-arg]
        elif isinstance(experts, list):
            self.experts = nn.ModuleList(experts)
        elif isinstance(experts, (ExpertMLP, ImpedanceExpert, ResidualImpedanceExpert)):
            # Clone the single expert template E times.
            self.experts = nn.ModuleList(
                [copy.deepcopy(experts) for _ in range(router.num_experts)]
            )
            # Re-initialize every clone independently: bit-identical clones
            # make the combined output invariant to the router's weights
            # (top-k weights sum to 1), so the router would receive no
            # action-based specialization signal at initialization.
            for expert in self.experts:
                if not isinstance(expert, (ExpertMLP, ImpedanceExpert, ResidualImpedanceExpert)):
                    raise TypeError(
                        "All experts must be ExpertMLP, ImpedanceExpert, "
                        "or ResidualImpedanceExpert instances, "
                        f"got {type(expert).__name__}"
                    )
                expert.reset_parameters()
        else:
            raise TypeError(
                "experts must be an ExpertMLP, ImpedanceExpert, ResidualImpedanceExpert, "
                "list[ExpertMLP], list[ImpedanceExpert], "
                "list[ResidualImpedanceExpert], or nn.ModuleList"
            )

        if len(self.experts) != router.num_experts:
            raise ValueError(
                f"Number of experts ({len(self.experts)}) does not match "
                f"router.num_experts ({router.num_experts})"
            )
        expert_types = {type(expert) for expert in self.experts}
        if len(expert_types) != 1:
            raise ValueError(
                "MoE experts must all share the same class (ExpertMLP, ImpedanceExpert, "
                f"or ResidualImpedanceExpert); mixing is not supported. "
                f"Got: {[t.__name__ for t in expert_types]}"
            )

    def forward(
        self,
        latent: Tensor,
        trajectory_id: Tensor | None = None,
        trajectory_position: Tensor | None = None,
        task_state: Tensor | None = None,
        *,
        router_override: str | None = None,
        phase_logits: Tensor | None = None,
        mapping: Tensor | None = None,
    ) -> MoEOutput:
        """Route latents to experts and combine their outputs.

        Args:
            latent: Tensor of shape (B, latent_dim).
            trajectory_id: Optional (B,) long tensor passed through to the
                router (V2-C history routing).
            trajectory_position: Optional (B,) long tensor passed through to
                the router (V2-C history routing).
            task_state: Optional (B, Dy) task states ``y_t``. Required with
                impedance experts, forbidden with direct experts (fail-closed
                on flavor mismatch).
            router_override: V2-E evaluation-time routing intervention, one
                of ``"sticky"`` / ``"uniform"`` / ``"oracle"`` (or ``None``
                for the learned router). The learned gate logits are still
                computed and reported; only the dispatch selection is
                replaced. Evaluation-time only — training always routes
                through the learned router.
            phase_logits: Optional (B, P) phase-head logits; required by
                ``router_override="oracle"``.
            mapping: Optional right-stochastic (P, E) soft phase->expert
                mapping M; required by ``router_override="oracle"`` (the
                caller must validate it — see
                ``PhaseBootstrappedMoE.require_soft_mapping``).

        Returns:
            MoEOutput containing the combined predictions and routing metadata.
        """
        if latent.ndim != 2:
            raise ValueError(
                f"MoELayer expects 2D latents of shape (B, latent_dim), got "
                f"{tuple(latent.shape)}. Sequence-aware training "
                "(sequence_length > 1) is not implemented — the data pipeline "
                "rejects it; keep data.sequence_length=1."
            )
        B, D = latent.shape

        # 1. Route inputs
        router_out: RouterOutput = self.router(
            latent,
            trajectory_id=trajectory_id,
            trajectory_position=trajectory_position,
        )
        weights = router_out.weights  # (B, K)
        indices = router_out.indices  # (B, K)

        # V2-E: replace the dispatch selection for evaluation-time
        # interventions; the reported gate logits stay the learned ones.
        if router_override == "uniform":
            weights, indices = self.router.uniform_selection(latent)
        elif router_override == "sticky":
            weights, indices = self.router.sticky_selection(router_out.gate_logits)
        elif router_override == "oracle":
            if phase_logits is None or mapping is None:
                raise ValueError(
                    "router_override='oracle' requires phase_logits and the "
                    "soft phase->expert mapping M"
                )
            weights, indices = self.router.oracle_selection(phase_logits, mapping)
        elif router_override is not None:
            raise ValueError(
                f"Unknown router_override {router_override!r}. Supported: "
                "sticky, uniform, oracle."
            )

        use_impedance = isinstance(self.experts[0], ImpedanceExpert)
        if use_impedance and task_state is None:
            raise RuntimeError(
                "Impedance experts need task_state=y_t; pass the task state "
                "alongside the latent (PhaseBootstrappedMoE threads it through)."
            )
        if not use_impedance and task_state is not None:
            raise RuntimeError(
                "Direct (ExpertMLP) experts received a task_state; impedance "
                "experts are required to consume it."
            )
        if use_impedance:
            assert task_state is not None
            return self._forward_impedance(latent, weights, indices, task_state, router_out)

        # Fast-path for single-sample top-1 evaluation/rollout (B=1, K=1)
        if B == 1 and indices.size(-1) == 1:
            expert_idx = int(indices[0, 0].item())
            expert_out = self.experts[expert_idx](latent)
            combined_output = expert_out * weights[..., :1]
            return MoEOutput(
                combined_output=combined_output,
                routing_weights=weights,
                expert_indices=indices,
                balance_loss=router_out.balance_loss,
                gate_logits=router_out.gate_logits,
                sticky_loss=router_out.sticky_loss,
                info=None,
                clean_gate_logits=_clean_logits(router_out),
            )

        out_dim = cast(ExpertMLP, self.experts[0]).output_dim
        # Final combined output tensor: (B, out_dim)
        combined_output = torch.zeros((B, out_dim), dtype=latent.dtype, device=latent.device)

        # 2. Dispatch to experts and combine
        # Implementation note: For small K (e.g., 2) and E (e.g., 6-8), iterating over
        # experts is often faster than complex scatter/gather batched operations due to
        # kernel launch overheads. We use the loop-over-experts approach.
        #
        # There is deliberately NO "expert received zero tokens" early-skip:
        # converting a CUDA tensor to bool (`match_mask.any()`) is a
        # host-device synchronization, so the skip cost 6 syncs per forward
        # on GPU. An expert invoked on an empty (0, D) slice is an exact
        # no-op (empty forward, empty gather, empty index_add_), so always
        # dispatching is bit-identical and sync-free.

        for expert_idx, expert_net in enumerate(self.experts):
            # Find all locations where this expert was selected
            # match_mask: (B, K) boolean tensor
            match_mask = indices == expert_idx

            # Find the batch indices that selected this expert
            # batch_idx: 1D tensor of batch indices
            # k_idx: 1D tensor of the k-th choice (0 to K-1)
            batch_idx, k_idx = torch.where(match_mask)

            # Gather the latents for this expert
            # expert_inputs: (N, D) where N is the number of items routed to this expert
            expert_inputs = latent[batch_idx]

            # Forward pass through the expert
            # expert_outputs: (N, out_dim)
            expert_outputs = expert_net(expert_inputs)

            # Gather the corresponding weights
            # expert_weights: (N, 1)
            expert_weights = weights[batch_idx, k_idx].unsqueeze(-1)

            # Accumulate into the combined output
            # combined_output[batch_idx] += expert_outputs * expert_weights
            # Note: We use scatter_add_ to safely handle cases where the same expert
            # might somehow be selected multiple times for the same batch item
            # (though topk should prevent this, it's safer).
            weighted_outputs = expert_outputs * expert_weights
            combined_output.index_add_(0, batch_idx, weighted_outputs)

        return MoEOutput(
            combined_output=combined_output,
            routing_weights=weights,
            expert_indices=indices,
            balance_loss=router_out.balance_loss,
            gate_logits=router_out.gate_logits,
            sticky_loss=router_out.sticky_loss,
            info=None,
            clean_gate_logits=_clean_logits(router_out),
        )

    def _forward_impedance(
        self,
        latent: Tensor,
        weights: Tensor,
        indices: Tensor,
        task_state: Tensor,
        router_out: RouterOutput,
    ) -> MoEOutput:
        """Dispatch impedance experts and combine controller parameters.

        Selected experts run ``params(z)`` to produce ``(T, κ)``; top-1
        selection maps directly through the action adapter, while wider
        selections (top-2 ablation, uniform override) blend parameters via
        :func:`blend_impedance` first. The reported gate logits and
        auxiliary losses stay the learned router's own.
        """
        from phaseforge.models.components.action_adapter import (
            blend_impedance,
            impedance_action,
        )

        batch_size = latent.size(0)
        select = weights.size(-1)
        first = cast(ImpedanceExpert, self.experts[0])
        task_dim, error_dim = first.task_state_dim, first.error_dim
        scale = getattr(first, "action_scale", 1.0)

        # Fast-path for single-sample top-1 evaluation/rollout (B=1, select=1)
        if batch_size == 1 and select == 1:
            expert_idx = int(indices[0, 0].item())
            selected_expert = cast(ImpedanceExpert, self.experts[expert_idx])
            target_sel, gains_sel = selected_expert.params(latent)
            combined_output, parts = impedance_action(target_sel, gains_sel, task_state, scale)
            info = {
                "target": target_sel,
                "gains": gains_sel,
                "task_error": parts["task_error"],
                "pre_clip_u": parts["pre_clip_u"],
                "task_state": task_state,
                "expert_index": indices[:, 0],
            }
            return MoEOutput(
                combined_output=combined_output,
                routing_weights=weights,
                expert_indices=indices,
                balance_loss=router_out.balance_loss,
                gate_logits=router_out.gate_logits,
                sticky_loss=router_out.sticky_loss,
                info=info,
                clean_gate_logits=_clean_logits(router_out),
            )

        targets_all = torch.zeros(
            (batch_size, select, task_dim), dtype=latent.dtype, device=latent.device
        )
        gains_all = torch.zeros(
            (batch_size, select, error_dim), dtype=latent.dtype, device=latent.device
        )
        for expert_idx, expert_net in enumerate(self.experts):
            expert_net = cast(ImpedanceExpert, expert_net)
            batch_idx, k_idx = torch.where(indices == expert_idx)
            if batch_idx.numel() == 0:
                continue
            target_e, gains_e = expert_net.params(latent[batch_idx])
            targets_all[batch_idx, k_idx] = target_e.to(targets_all.dtype)
            gains_all[batch_idx, k_idx] = gains_e.to(gains_all.dtype)

        if select == 1:
            target_sel = targets_all[:, 0]
            gains_sel = gains_all[:, 0]
        else:
            target_sel, gains_sel = blend_impedance(targets_all, gains_all, weights)
        combined_output, parts = impedance_action(target_sel, gains_sel, task_state, scale)
        info = {
            # Attached (not detached): the Lipschitz loss differentiates
            # through "target". Consumers that persist diagnostics detach.
            "target": target_sel,
            "gains": gains_sel,
            "task_error": parts["task_error"],
            "pre_clip_u": parts["pre_clip_u"],
            "task_state": task_state,
            "expert_index": indices[:, 0],
        }
        return MoEOutput(
            combined_output=combined_output,
            routing_weights=weights,
            expert_indices=indices,
            balance_loss=router_out.balance_loss,
            gate_logits=router_out.gate_logits,
            sticky_loss=router_out.sticky_loss,
            info=info,
            clean_gate_logits=_clean_logits(router_out),
        )

"""TopKRouter: Sparse top-k expert router with load balancing."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

#: Cached 0.0 scalar per device for the sticky-loss default — the default is
#: requested every forward; a fresh device tensor per call is a small
#: allocation repeated over tens of thousands of steps. Read-only constant,
#: value/dtype identical to ``torch.tensor(0.0)``.
_ZERO_SCALARS: dict[torch.device, Tensor] = {}


def _zero_scalar(device: torch.device) -> Tensor:
    zero = _ZERO_SCALARS.get(device)
    if zero is None:
        zero = torch.zeros((), device=device)
        _ZERO_SCALARS[device] = zero
    return zero


class RouterOutput(NamedTuple):
    """Standardized output from the TopKRouter."""

    weights: Tensor  # (B, K) normalized top-k weights
    indices: Tensor  # (B, K) selected expert indices
    gate_logits: Tensor  # (B, E) raw logits over all experts
    balance_loss: Tensor  # scalar, auxiliary load-balancing loss
    sticky_loss: Tensor  # scalar, raw (unscaled) history-stickiness loss
    clean_gate_logits: Tensor | None = None  # (B, E) deterministic pre-exploration logits
    # ``clean_gate_logits`` carries the gate logits WITHOUT the training-time
    # exploration noise: the linear projection plus the learned history bias
    # when ``use_history`` is on. In evaluation (noise off) it equals
    # ``gate_logits``. The Stage 2 margin loss consumes it so training
    # noise never silently shifts the margin targets; diagnostics keep using
    # ``gate_logits`` (the actually-dispatched routing signal).


class TopKRouter(nn.Module):
    """Sparse top-k expert router with auxiliary load-balancing loss.

    Follows the Shazeer et al. (2017) / GShard-style top-k noisy-gating
    mechanism (not Switch Transformer, whose defining contribution is
    top-1 routing):
    1. Compute gate logits via linear projection.
    2. Add scaled Gaussian noise during training for exploration.
    3. Select the top-k experts.
    4. Normalize the top-k gate values via softmax.
    5. Compute a load-balancing loss to encourage equal expert utilization.

    Args:
        latent_dim: Dimension of the input latent vector.
        num_experts: Total number of experts (E) to route to.
        top_k: Number of experts to select per input (K). Must be in
            ``[1, num_experts]``.
        noise_std: Standard deviation of the routing noise added during training.
            If 0.0, routing is purely deterministic.
        balance_coeff: Multiplier for the auxiliary balance loss.
        normalize_input: If ``True``, L2-normalize the latent vector before
            the gate projection. Combined with unit-norm centroids loaded
            into ``gate_linear.weight`` (see the bootstrap), the gate logits
            become true cosine similarities between the latent and each
            centroid — the precision-residual prototype-router cells.
            ``False`` keeps raw dot products.
        use_history: If ``True`` (V2-C), a two-pass in-batch history bias is
            added: the first pass computes the gate logits, the previous
            in-trajectory step's top-1 expert is resolved (adjacent
            ``(trajectory_id, trajectory_position)`` pairs only), and the
            second pass adds a learned embedding of that choice to the
            logits. Samples without an adjacent previous step (trajectory
            starts, or batch-order breaks) use the dedicated "no history"
            embedding slot. Requires ``trajectory_id`` / ``trajectory_position``
            in the batch; without them the router degrades to the plain path.
            The stickiness loss (scaled by ``train.sticky_coeff``) is always
            emitted alongside it — ``0.0`` when no pairs exist or history is
            off.
        sticky_beta: Decay of the V2-E evaluation-time sticky EMA (default
            0.9). The EMA tracks the learned top-1 choice across steps and is
            the routing signal of the ``sticky`` eval mode; it lives only in
            evaluation state and is cleared by :meth:`reset_sticky_ema`.
    """

    def __init__(
        self,
        latent_dim: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1,
        balance_coeff: float = 0.01,
        normalize_input: bool = False,
        use_history: bool = False,
        sticky_beta: float = 0.9,
    ) -> None:
        super().__init__()
        if not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError(f"latent_dim must be a positive int, got {latent_dim!r}")
        if not isinstance(num_experts, int) or num_experts < 1:
            raise ValueError(f"num_experts must be a positive int, got {num_experts!r}")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive int, got {top_k!r}")
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot exceed num_experts ({num_experts}). "
                "The config routes to more experts than exist."
            )
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be >= 0.0, got {noise_std}")
        if balance_coeff < 0.0:
            raise ValueError(f"balance_coeff must be >= 0.0, got {balance_coeff}")

        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.balance_coeff = balance_coeff
        self.normalize_input = bool(normalize_input)
        self.use_history = bool(use_history)
        if not 0.0 <= sticky_beta < 1.0:
            raise ValueError(f"sticky_beta must be in [0, 1), got {sticky_beta}")
        self.sticky_beta = float(sticky_beta)

        self.gate_linear = nn.Linear(latent_dim, num_experts)

        # Linear layer to scale the noise per-input, following standard MoE practices
        if self.noise_std > 0.0:
            self.noise_linear = nn.Linear(latent_dim, num_experts)
        else:
            self.noise_linear = None

        # V2-C: previous-step history. Slot 0 is the "no previous step" sentinel
        # (trajectory start or non-adjacent batch order); slots 1..E hold the
        # previous top-1 expert (+1). The embedding is projected to an
        # additive logit bias, which preserves the cosine-semantics of the
        # gate weights when normalize_input=True.
        self.history_embedding: nn.Embedding | None
        self.history_proj: nn.Linear | None
        if self.use_history:
            self.history_embedding = nn.Embedding(num_experts + 1, latent_dim)
            self.history_proj = nn.Linear(latent_dim, num_experts, bias=False)
        else:
            self.history_embedding = None
            self.history_proj = None

        # V2-E: per-episode sticky-EMA evaluation state (never in the
        # state dict; cleared by reset_sticky_ema()).
        self._sticky_ema: Tensor | None = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize routing weights.

        Using normal initialization with a small std dev helps prevent
        all inputs from collapsing to a single expert at the start.
        """
        nn.init.normal_(self.gate_linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate_linear.bias)

        if self.noise_linear is not None:
            nn.init.normal_(self.noise_linear.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.noise_linear.bias)

        if self.history_embedding is not None:
            nn.init.normal_(self.history_embedding.weight, mean=0.0, std=0.02)
        if self.history_proj is not None:
            nn.init.normal_(self.history_proj.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # V2-E evaluation-time routing interventions
    # ------------------------------------------------------------------

    def reset_sticky_ema(self) -> None:
        """Clear the sticky-EMA evaluation state (start of a new episode)."""
        self._sticky_ema = None

    def uniform_selection(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        """Route with equal weight over ALL experts (ensemble baseline).

        Returns ``(weights, indices)`` of shape ``(B, E)`` / ``(B, E)`` —
        the deterministic counterpart of the offline uniform counterfactual
        (average of every expert's action, no gate signal at all).
        """
        B = latent.size(0)
        indices = torch.arange(
            self.num_experts, device=latent.device
        ).unsqueeze(0).expand(B, self.num_experts)
        weights = torch.full(
            (B, self.num_experts),
            1.0 / self.num_experts,
            device=latent.device,
        )
        return weights, indices

    def sticky_selection(self, gate_logits: Tensor) -> tuple[Tensor, Tensor]:
        """Route by a beta-EMA over the learned top-1 choice (V2-E sticky).

        The EMA is a single ``(E,)`` vector of expert scores updated every
        step as ``ema = beta*ema + (1-beta)*mean_onehot(argmax(probs))``;
        the current top-k of the EMA picks the experts and the softmax over
        those EMA scores sets their weights. At the first step (EMA unset)
        the EMA is initialized from the current batch, so the very first
        action equals the learned one. Batches larger than one update the
        EMA with the batch-mean one-hot (the rollout runner always steps
        with B=1, one episode at a time).
        """
        probs = F.softmax(gate_logits, dim=-1)  # (B, E)
        top1 = probs.argmax(dim=-1)  # (B,)
        onehot = F.one_hot(top1, num_classes=self.num_experts).float()  # (B, E)
        batch_ema = onehot.mean(dim=0)  # (E,)
        if self._sticky_ema is None:
            self._sticky_ema = batch_ema
        else:
            self._sticky_ema = (
                self.sticky_beta * self._sticky_ema + (1.0 - self.sticky_beta) * batch_ema
            )
        ema = self._sticky_ema.expand(gate_logits.size(0), -1)
        top_k_logits, top_k_indices = torch.topk(ema, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        return top_k_weights, top_k_indices

    def oracle_selection(self, phase_logits: Tensor, mapping: Tensor) -> tuple[Tensor, Tensor]:
        """Route by the phase-head oracle through the soft mapping M.

        ``mapping`` is the right-stochastic ``(P, E)`` matrix M (the
        model's persistent soft phase->expert mapping; must be validated
        nonzero by the caller). The expert distribution is
        ``M^T softmax(phase_logits)``; the top-k of that distribution are
        selected and re-normalized.
        """
        phase_probs = F.softmax(phase_logits, dim=-1)  # (B, P)
        expert_probs = torch.einsum("pe,bp->be", mapping, phase_probs)  # (B, E)
        top_k_logits, top_k_indices = torch.topk(expert_probs, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        return top_k_weights, top_k_indices

    def _resolve_previous_top1(
        self, top1: Tensor, trajectory_id: Tensor, trajectory_position: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Resolve each sample's previous in-trajectory top-1 expert.

        Two-pass in-batch semantics: samples are sorted by
        ``(trajectory_id, trajectory_position)``; a sample's previous step
        exists iff it shares the trajectory with the preceding sample AND
        that sample's position is exactly ``position - 1`` (adjacent pair).
        Returns ``(prev_top1, prev_valid)``: the previous expert index for
        paired samples (unset for the rest) and the pairing mask.
        """
        order = torch.argsort(
            trajectory_id * (trajectory_position.max() + 1) + trajectory_position,
            stable=True,
        )
        sorted_ids = trajectory_id[order]
        sorted_pos = trajectory_position[order]
        sorted_top1 = top1[order]

        adjacent = (sorted_ids[1:] == sorted_ids[:-1]) & (
            sorted_pos[1:] == sorted_pos[:-1] + 1
        )
        prev_top1 = torch.zeros_like(top1)
        prev_valid = torch.zeros_like(top1, dtype=torch.bool)
        if adjacent.any():
            prev_positions = order[1:][adjacent]
            prev_values = sorted_top1[:-1][adjacent]
            prev_top1[prev_positions] = prev_values
            prev_valid[prev_positions] = True
        return prev_top1, prev_valid

    def forward(
        self,
        latent: Tensor,
        trajectory_id: Tensor | None = None,
        trajectory_position: Tensor | None = None,
    ) -> RouterOutput:
        """Route inputs to top-k experts and compute balance loss.

        Args:
            latent: Tensor of shape (B, latent_dim).
            trajectory_id: Optional (B,) long tensor identifying each
                sample's source trajectory (V2-C history routing).
            trajectory_position: Optional (B,) long tensor with each
                sample's step position within its trajectory.

        Returns:
            RouterOutput containing weights, indices, logits, and balance loss.
        """
        # (B, E) raw gating logits
        if self.normalize_input:
            # Unit-norm latents + unit-norm centroid weights (set by the
            # bootstrap) => gate logits are true cosine similarities.
            latent = F.normalize(latent, p=2, dim=-1)
        gate_logits = self.gate_linear(latent)

        # Deterministic pre-exploration logits: everything except the
        # stochastic exploration term below. Reassignment via ``+``
        # creates new tensors, so this reference keeps tracking the linear
        # output while ``gate_logits`` advances through noise/history.
        clean_gate_logits = gate_logits

        # Add exploration noise during training
        if self.training and self.noise_std > 0.0 and self.noise_linear is not None:
            noise_logits = self.noise_linear(latent)
            # softplus ensures noise scaling is positive
            noise_scale = self.noise_std * F.softplus(noise_logits)
            # standard normal noise
            noise = torch.randn_like(gate_logits)
            gate_logits = gate_logits + noise_scale * noise

        # V2-C second pass: add the learned history bias of the previous
        # step's top-1 choice (first-pass logits decide the choice).
        prev_top1: Tensor | None = None
        sticky_loss = _zero_scalar(gate_logits.device)
        B = gate_logits.size(0)
        if (
            self.use_history
            and self.history_embedding is not None
            and self.history_proj is not None
            and trajectory_id is not None
            and trajectory_position is not None
        ):
            traj_id = trajectory_id
            traj_pos = trajectory_position
            if traj_id.dim() != 1 or traj_pos.dim() != 1 or traj_id.numel() != B:
                raise ValueError(
                    "trajectory_id/trajectory_position must be 1-D tensors of "
                    f"length B={B} when use_history=True"
                )
            first_pass_top1 = gate_logits.detach().argmax(dim=-1)  # (B,)
            prev_top1, prev_valid = self._resolve_previous_top1(
                first_pass_top1, traj_id, traj_pos
            )
            embed_idx = torch.where(prev_valid, prev_top1 + 1, 0)
            history_bias = self.history_proj(self.history_embedding(embed_idx))
            gate_logits = gate_logits + history_bias
            # The history bias is learned and deterministic given the batch,
            # so it belongs to the clean signal too. (Caveat: with
            # use_history=True the bias derives from the noisy first pass;
            # all final-matrix rows disable history, where clean is exact.)
            clean_gate_logits = clean_gate_logits + history_bias

            # Stickiness: -log p_t[top1_{t-1}] over samples with a previous
            # step. The final (second-pass) logits define p_t.
            if prev_valid.any():
                log_probs = F.log_softmax(gate_logits, dim=-1)
                sticky_loss = -log_probs[prev_valid].gather(
                    1, prev_top1[prev_valid].unsqueeze(-1)
                ).squeeze(-1).mean()

        # Get routing probabilities (B, E)
        routing_probs = F.softmax(gate_logits, dim=-1)

        # Select top-k experts
        # values: (B, K), indices: (B, K)
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)

        # Normalize top-k values to sum to 1
        # Re-compute softmax over just the top-k elements so that sum(weights) == 1
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Compute auxiliary balance loss
        # We compute this even during eval so metrics can track it if desired
        balance_loss = self._compute_balance_loss(routing_probs, gate_logits)

        return RouterOutput(
            weights=top_k_weights,
            indices=top_k_indices,
            gate_logits=gate_logits,
            balance_loss=balance_loss * self.balance_coeff,
            sticky_loss=sticky_loss,
            clean_gate_logits=clean_gate_logits,
        )

    def margin_loss_from_logits(
        self, gate_logits: Tensor, targets: Tensor, margin: float | None = None
    ) -> Tensor:
        """Large-margin loss from gate logits.

        Logit-space mirror of :meth:`PrototypeRouter.margin_loss_from_logits`:
        where the prototype form penalizes ``m - (d_j - d_y)`` on distances
        (smaller is better), this penalizes ``m - (l_y - l_j)`` on gate
        logits (larger is better)::

            L_margin = 1/|B| Σ_i Σ_{j≠y_i} max(0, m − (l_{iy_i} − l_{ij}))

        Callers must pass deterministic pre-exploration logits
        (:attr:`RouterOutput.clean_gate_logits`) so training-time
        exploration noise never silently shifts the margin targets. The
        operation preserves gradients to the gate weights.

        Args:
            gate_logits: ``(B, E)`` deterministic gate logits.
            targets: ``(B,)`` true regime ids.
            margin: Override for the locked Stage 2 margin (default ``0.5``,
                the ``train.margin.margin`` protocol value; the trainer
                always passes it explicitly in practice).

        Returns:
            Scalar margin loss tensor (same dtype as ``gate_logits``).
        """
        m = 0.5 if margin is None else float(margin)
        if m < 0.0:
            raise ValueError(f"margin must be >= 0.0, got {m}.")
        flat_targets = targets.reshape(-1).long()
        if gate_logits.ndim != 2 or gate_logits.size(0) != flat_targets.numel():
            raise ValueError(
                "gate_logits (B, E) and targets (B,) must share the batch size."
            )
        if gate_logits.size(-1) != self.num_experts:
            raise ValueError(
                f"gate_logits width {gate_logits.size(-1)} != num_experts {self.num_experts}."
            )
        if flat_targets.numel() == 0:
            return torch.zeros((), device=gate_logits.device, dtype=gate_logits.dtype)
        if bool((flat_targets < 0).any()) or bool((flat_targets >= self.num_experts).any()):
            raise ValueError("margin targets are out of range.")
        correct = gate_logits.float().gather(1, flat_targets.unsqueeze(-1))
        gaps = correct - gate_logits.float()
        margins = torch.clamp(m - gaps, min=0.0)
        eye = torch.zeros_like(margins).scatter_(1, flat_targets.unsqueeze(-1), 1.0)
        margins = margins * (1.0 - eye)
        return margins.sum(dim=-1).mean().to(gate_logits.dtype)

    def _compute_balance_loss(self, routing_probs: Tensor, gate_logits: Tensor) -> Tensor:
        """Compute the Switch Transformer auxiliary load-balancing loss.

        L_balance = E * sum(f_i * p_i) for i in 1..E
        where f_i is the fraction of items routed to expert i (based on top-1)
        and p_i is the mean routing probability for expert i.
        (The f_i * p_i auxiliary-loss formulation is Switch Transformer's;
        the surrounding top-k noisy-gating mechanism is Shazeer et al.
        (2017) / GShard lineage.)

        .. note:: Routing semantics — this loss is computed on the *top-1*
            hard assignment (``argmax`` of the gate logits, as in Switch
            Transformer), while the utilization diagnostics reported during
            validation/offline evaluation count *all* top-k assignments
            (see ``expert_utilization``). This is intentional: the loss
            penalizes the primary-choice imbalance that drives dead experts,
            whereas the reported metrics measure full dispatch load. The
            balance score therefore does NOT measure exactly what the
            optimization loss minimizes — interpret them side by side, not
            interchangeably.

        Args:
            routing_probs: (B, E) softmax probabilities
            gate_logits: (B, E) raw logits before softmax

        Returns:
            Scalar balance loss tensor.
        """
        B, E = gate_logits.shape

        # f_i: fraction of batch routed to each expert (based on primary choice)
        # We use top-1 for the balance loss calculation, as is standard.
        top_1_indices = gate_logits.argmax(dim=-1)  # (B,)

        # One-hot encoding of expert assignments (B, E)
        expert_mask = F.one_hot(top_1_indices, num_classes=E).float()

        # Mean fraction of tokens routed to each expert: (E,)
        f_i = expert_mask.mean(dim=0)

        # Mean probability assigned to each expert: (E,)
        p_i = routing_probs.mean(dim=0)

        # The loss encourages f_i and p_i to be uniform (1/E)
        balance_loss = E * torch.sum(f_i * p_i)

        return balance_loss

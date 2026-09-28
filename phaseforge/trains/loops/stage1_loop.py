"""Stage 1 Trainer: Phase-supervised generalist pretraining."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from phaseforge.models.base import ModelOutput
from phaseforge.trains.loops.base import (
    BaseTrainer,
    MetricValue,
    _PhaseAccumulator,
)
from phaseforge.trains.loops.label_contract import (
    LOCKED_NUM_CLASSES,
    describe_run,
    peek_loader_batch,
    resolve_label_field,
    resolve_label_tensor,
    resolve_supcon_label_field,
    validate_label_values,
)


def _phase_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    phase_weights: torch.Tensor | None,
    soft_target_eps: float,
) -> torch.Tensor:
    """Phase cross-entropy with optional label smoothing.

    ``soft_target_eps == 0`` delegates to ``F.cross_entropy``. For
    ``eps > 0`` each sample mixes its NLL with the uniform-NLL (label
    smoothing)::

        L_i = (1 - eps) * nll_i + eps * mean_c(-log p_i,c)

    Class weights (when set) are applied once per sample on the mixed loss,
    matching ``F.cross_entropy``'s ``weight`` semantics.
    """
    if soft_target_eps == 0.0:
        return F.cross_entropy(logits, targets, weight=phase_weights)
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    uniform_nll = -log_probs.mean(dim=-1)
    per_sample = (1.0 - soft_target_eps) * nll + soft_target_eps * uniform_nll
    if phase_weights is not None:
        per_sample = per_sample * phase_weights[targets]
    return per_sample.mean()


def _grad_cosine_similarity(
    loss_action: torch.Tensor,
    loss_phase: torch.Tensor,
    parameters: Iterable[nn.Parameter],
) -> torch.Tensor | None:
    """Cosine similarity between the action- and phase-loss gradients.

    Auxiliary-task conflict diagnostic (Du et al. 2018 gradient-cosine
    weighting; PCGrad's conflict definition): the angle between the two
    per-task gradient vectors over all trainable parameters. Sustained
    negative values indicate active gradient conflict between the action
    objective and the phase head; values near zero indicate the tasks are
    (locally) orthogonal. Parameters not reached by a loss contribute zero
    (their true gradient). Returns ``None`` when either gradient vector is
    degenerate (zero norm), which the caller must tolerate.

    Diagnostic only: computed with ``retain_graph=True`` and detached, so
    it never changes the optimizer update. The extra backward passes run
    only when ``train.grad_cosine`` is enabled.
    """
    if not (loss_action.requires_grad and loss_phase.requires_grad):
        return None
    params = [p for p in parameters if p.requires_grad]
    if not params:
        return None
    grads_action = torch.autograd.grad(
        loss_action, params, retain_graph=True, allow_unused=True
    )
    grads_phase = torch.autograd.grad(
        loss_phase, params, retain_graph=True, allow_unused=True
    )
    flat_a = torch.cat(
        [
            (g if g is not None else torch.zeros_like(p)).detach().flatten()
            for g, p in zip(grads_action, params)
        ]
    )
    flat_p = torch.cat(
        [
            (g if g is not None else torch.zeros_like(p)).detach().flatten()
            for g, p in zip(grads_phase, params)
        ]
    )
    norm_a = flat_a.norm()
    norm_p = flat_p.norm()
    if norm_a == 0.0 or norm_p == 0.0:
        return None
    return (flat_a @ flat_p) / (norm_a * norm_p)


class Stage1Trainer(BaseTrainer):
    """Trainer for Stage 1.

    Computes: L_total = L_action + λ_phase * L_phase
    Action loss is MSE. Phase loss is CrossEntropy.

    Also persists the phase-classification accuracy: micro ``phase_acc``
    and macro/balanced ``phase_balanced_acc`` (mean per-class recall), computed in the
    training loop and in validation. ``bc`` has no phase head, so its
    ``phase_logits`` are ``None`` and no phase fields are emitted — absence
    is honest, never a fabricated zero.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._train_phase_acc = _PhaseAccumulator(device=self.device)
        self._val_phase_acc = _PhaseAccumulator(device=self.device)

    def _phase_label_field(self) -> str:
        """Resolved Stage 1 phase-CE / diagnostic label field."""
        return resolve_label_field(
            self.train_cfg, "phase_label_field", cfg=self.cfg, purpose="Stage 1 phase CE"
        )

    def _supcon_label_field(self) -> str:
        """Resolved Stage 1 SupCon label field."""
        return resolve_supcon_label_field(self.train_cfg, cfg=self.cfg)

    def validate_label_contract(self) -> None:
        """Preflight: fail closed on missing/invalid label fields.

        Checks one sample batch from the train loader (and the val loader
        when present) before any optimizer step. Called at the start of
        :meth:`fit` so misconfigured method/task/seed combinations stop
        before training begins.
        """
        ctx = describe_run(self.cfg)
        # Peek without perturbing the sampler: the helper restores the
        # loader generator so epoch-1 shuffles bit-identically with or
        # without this preflight.
        train_batch = peek_loader_batch(self.train_loader)
        if train_batch is None:
            raise RuntimeError(
                f"Stage 1 label preflight: train loader is empty. "
                f"(method={ctx['method']}, task={ctx['task']}, seed={ctx['seed']})"
            )
        val_batch = (
            peek_loader_batch(self.val_loader) if self.val_loader is not None else None
        )
        needs_phase = True
        try:
            needs_phase = getattr(self.model, "phase_head", None) is not None
        except Exception:
            needs_phase = True
        # Models without a phase head (e.g. BC) skip the CE field unless
        # SupCon needs labels.
        supcon_cfg = self.train_cfg.get("supcon", None)
        supcon_enabled = bool(supcon_cfg.get("enabled", False)) if supcon_cfg is not None else False
        if needs_phase or supcon_enabled:
            phase_field = self._phase_label_field()
            for split, sample in (("train", train_batch), ("val", val_batch)):
                if sample is None:
                    continue
                if phase_field not in sample:
                    available = sorted(str(k) for k in sample.keys())
                    raise RuntimeError(
                        f"Stage 1 label preflight: label field {phase_field!r} "
                        f"is missing from the {split} batch. "
                        f"(method={ctx['method']}, task={ctx['task']}, split={split}, "
                        f"seed={ctx['seed']}, requested_field={phase_field}, "
                        f"available_keys={available})"
                    )
                # Range check against the phase head when present, else the
                # locked six-class contract (subset passes, e.g. unit tests
                # with 2-3 phases; out-of-range fails).
                head = getattr(self.model, "phase_head", None)
                head_phases = getattr(head, "num_phases", None) if head is not None else None
                try:
                    num_classes = (
                        int(head_phases) if head_phases is not None else LOCKED_NUM_CLASSES
                    )
                except (TypeError, ValueError):
                    num_classes = LOCKED_NUM_CLASSES
                validate_label_values(
                    sample[phase_field], num_classes,
                    cfg=self.cfg, field=phase_field,
                    purpose="Stage 1 label preflight", split=split,
                )
        if supcon_enabled:
            supcon_field = self._supcon_label_field()
            for split, sample in (("train", train_batch), ("val", val_batch)):
                if sample is None:
                    continue
                if supcon_field not in sample:
                    available = sorted(str(k) for k in sample.keys())
                    raise RuntimeError(
                        f"Stage 1 label preflight: SupCon label field {supcon_field!r} "
                        f"is missing from the {split} batch. "
                        f"(method={ctx['method']}, task={ctx['task']}, split={split}, "
                        f"seed={ctx['seed']}, requested_field={supcon_field}, "
                        f"available_keys={available})"
                    )

    def fit(self) -> None:
        """Run preflight label validation before the training lifecycle."""
        self.validate_label_contract()
        super().fit()

    def _effective_lambda_phase(self) -> float:
        """Effective λ(t) for the auxiliary phase loss at the current epoch.

        ``train.lambda_schedule.type``:
        - ``"constant"``: λ = ``lambda_phase`` for every epoch.
        - ``"linear"``: the multiplier anneals ``start -> end`` linearly
          over the run (``end <= start``, both in [0, 1]), so the phase head
          is shaped early and the action loss dominates late. A schedule is
          used instead of a gradient-conflict correction because the measured
          cos(∇L_action, ∇L_phase) ≈ 0 indicated no conflict to resolve.

        Returns 0.0 when the supervised-contrastive path is enabled with
        ``train.supcon.zero_ce`` (the default): SupCon then supplies the
        regime-alignment signal (``supcon_lambda * supcon_term`` is added
        to the total loss), so the phase cross-entropy weight is zeroed.
        """
        supcon_cfg = self.train_cfg.get("supcon") or {}
        if bool(supcon_cfg.get("enabled", False)):
            if bool(supcon_cfg.get("zero_ce", True)):
                return 0.0
        base = float(self.train_cfg.get("lambda_phase", 1.0))
        schedule = self.train_cfg.get("lambda_schedule") or {}
        sched_type = str(schedule.get("type", "constant"))
        if sched_type == "constant":
            return base
        if sched_type == "linear":
            start = float(schedule.get("start", 1.0))
            end = float(schedule.get("end", 0.0))
            if not 0.0 <= end <= start <= 1.0:
                raise ValueError(
                    "train.lambda_schedule.linear requires 0 <= end <= start <= 1, "
                    f"got start={start}, end={end}"
                )
            total = max(1, int(self.train_cfg.get("epochs", 1)))
            t = min(max(float(self.current_epoch), 0.0), float(total))
            multiplier = start + (end - start) * (t / total)
            return base * max(0.0, multiplier)
        raise ValueError(
            f"unknown train.lambda_schedule.type {sched_type!r} "
            "(expected 'constant' or 'linear')"
        )

    def _compute_loss(
        self, batch: dict[str, torch.Tensor], out: ModelOutput | None = None
    ) -> tuple[torch.Tensor, dict[str, MetricValue]]:
        if out is None:
            out = self.model(batch)

        target_action = batch["action"]  # (B, A) or (B, T, A)
        # Resolved label field: topology rows use "phase_topo", the Static
        # Rule comparison uses "phase". Fail-closed with
        # method/task/split/field context when absent or invalid.
        phase_field = self._phase_label_field()
        mask = batch.get("padding_mask")  # (B, T) boolean or None
        # Eagerly resolve the declared field so a misconfigured vocabulary
        # stops before any loss is computed. Models without a
        # phase head and without SupCon (e.g. BC) consume no phase labels:
        # absence is honest there, never fabricated.
        target_phase: torch.Tensor | None = None
        try:
            _num_classes_probe = (
                int(out.phase_logits.size(-1))
                if out is not None and getattr(out, "phase_logits", None) is not None
                else LOCKED_NUM_CLASSES
            )
            target_phase = resolve_label_tensor(
                batch, phase_field, cfg=self.cfg,
                purpose="Stage 1 phase CE",
                num_classes=_num_classes_probe,
            )
        except (RuntimeError, ValueError):
            # Re-raise when any phase consumer is active; tolerate absence
            # only for head-less, SupCon-less models.
            _supcon_on = False
            try:
                _sc = self.train_cfg.get("supcon", None)
                _supcon_on = bool(_sc.get("enabled", False)) if _sc is not None else False
            except Exception:
                _supcon_on = False
            _needs_phase = (
                (out is not None and getattr(out, "phase_logits", None) is not None)
                or _supcon_on
            )
            if _needs_phase:
                raise
            target_phase = None

        lambda_phase = self._effective_lambda_phase()
        # The raw phase loss is always computed (and always reported on the
        # curve) when the model has a phase head and the BASE weight is
        # positive, even under a schedule whose effective λ(t) has reached
        # zero — otherwise the val/loss_phase overfitting signal that the
        # schedule is supposed to suppress would be masked on the curve.
        base_lambda_positive = float(self.train_cfg.get("lambda_phase", 1.0)) > 0.0

        # Optional inverse-frequency class weights for the phase CE
        # (train.phase_class_weight="balanced" or "cui"). None preserves
        # plain cross-entropy.
        phase_weights: torch.Tensor | None = None
        pw = self.train_cfg.get("phase_weights")
        if out.phase_logits is not None and pw:
            num_classes = out.phase_logits.size(-1)
            pw_list = [float(x) for x in pw]
            if len(pw_list) != num_classes:
                import logging
                _logger = logging.getLogger(__name__)
                if len(pw_list) < num_classes:
                    _logger.warning(
                        "train.phase_weights has %d entries but phase head predicts %d classes. "
                        "Auto-padding remaining classes with 1.0.",
                        len(pw_list),
                        num_classes,
                    )
                    pw_list.extend([1.0] * (num_classes - len(pw_list)))
                else:
                    _logger.warning(
                        "train.phase_weights has %d entries but phase head predicts %d classes. "
                        "Truncating to %d classes.",
                        len(pw_list),
                        num_classes,
                        num_classes,
                    )
                    pw_list = pw_list[:num_classes]
            cached = getattr(self, "_cached_phase_weights", None)
            if cached is None or cached.device != self.device or len(cached) != len(pw_list):
                cached = torch.tensor(pw_list, dtype=torch.float32, device=self.device)
                self._cached_phase_weights = cached
            phase_weights = cached

        # Optional label smoothing on the phase targets: the head is
        # discouraged from saturating, which keeps the phase logits soft for
        # the downstream diagnostic cells that route on them. 0.0 (default)
        # preserves plain CE bit-for-bit.
        soft_target_eps = float(self.train_cfg.get("soft_target_eps", 0.0))
        if not 0.0 <= soft_target_eps <= 1.0:
            raise ValueError(
                "train.soft_target_eps must be in [0, 1], "
                f"got {soft_target_eps}"
            )

        if mask is not None:
            action_loss = F.mse_loss(out.action_pred, target_action, reduction="none")
            # Average only over valid steps
            action_loss = action_loss[mask].mean()
        else:
            action_loss = F.mse_loss(out.action_pred, target_action)

        phase_loss = torch.tensor(0.0, device=self.device)
        if out.phase_logits is not None and base_lambda_positive and lambda_phase > 0.0:
            logits = out.phase_logits
            # Range was already validated against this head width by the
            # eager resolve_label_tensor above (fail-closed); the
            # assert below only narrows the type, it never silently zeroes.
            assert target_phase is not None  # fail-closed above when head exists
            if mask is not None:
                # Reshape for CE: (B*T, num_classes) and (B*T,)
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = target_phase.view(-1)
                mask_flat = mask.view(-1)

                logits_valid = logits_flat[mask_flat]
                targets_valid = targets_flat[mask_flat]

                if len(targets_valid) > 0:
                    phase_loss = _phase_ce(
                        logits_valid,
                        targets_valid,
                        phase_weights,
                        soft_target_eps,
                    )
            else:
                phase_loss = _phase_ce(
                    logits, target_phase, phase_weights, soft_target_eps
                )

        total_loss = action_loss + lambda_phase * phase_loss

        # Supervised contrastive regime alignment.
        # Disabled by default (`train.supcon.enabled=false`): when absent or
        # disabled this block contributes exactly 0.0 and emits no metrics,
        # so legacy loss curves are bit-identical.
        supcon_cfg = self.train_cfg.get("supcon", None)
        supcon_enabled = bool(supcon_cfg.get("enabled", False)) if supcon_cfg is not None else False
        if supcon_enabled:
            from phaseforge.trains.losses.supcon import supcon_loss

            assert supcon_cfg is not None
            supcon_lambda = float(supcon_cfg.get("lambda_sc", 1.0))
            temperature = float(supcon_cfg.get("temperature", 0.07))
            # Resolved SupCon label field, with method/task/split/field
            # context on failure.
            label_field = self._supcon_label_field()
            if out.latent is None:
                raise RuntimeError(
                    "train.supcon.enabled=true but the model forward did not "
                    "expose latents (ModelOutput.latent is None). SupCon needs "
                    "the exact latents the action head consumed."
                )
            latents = out.latent
            regime_labels = resolve_label_tensor(
                batch, label_field, cfg=self.cfg,
                purpose="Stage 1 SupCon",
                num_classes=(
                    int(out.phase_logits.size(-1))
                    if getattr(out, "phase_logits", None) is not None
                    else LOCKED_NUM_CLASSES
                ),
            )
            if mask is not None:
                flat_latents = latents.view(-1, latents.size(-1))
                flat_labels = regime_labels.view(-1)
                keep = mask.view(-1).bool()
                latents_valid = flat_latents[keep]
                labels_valid = flat_labels[keep]
            else:
                latents_valid = latents.reshape(-1, latents.size(-1))
                labels_valid = regime_labels.reshape(-1)
            if labels_valid.numel() == 0:
                supcon_term = torch.tensor(0.0, device=self.device)
            else:
                supcon_term = supcon_loss(latents_valid, labels_valid, temperature)
            total_loss = total_loss + supcon_lambda * supcon_term
        else:
            supcon_lambda = 0.0
            supcon_term = torch.tensor(0.0, device=self.device)

        metrics = {
            # Keep scalar metrics on-device until the trainer actually logs or
            # aggregates them. Calling .item() for every metric on every CUDA
            # batch would force a host/device synchronization each step.
            "loss_total": total_loss.detach(),
            "loss_action": action_loss.detach(),
            "loss_phase": phase_loss.detach() if base_lambda_positive else 0.0,
            "lambda_phase": lambda_phase,
        }
        if supcon_enabled:
            metrics["loss_supcon"] = supcon_term.detach()
            metrics["supcon_lambda"] = supcon_lambda

        # Optional per-step diagnostic: cosine similarity between the action
        # and phase gradient vectors (auxiliary-task conflict detector, Du et
        # al. 2018 / PCGrad). Diagnostic only — off by default, never part of
        # the loss, and computed only during training (torch.is_grad_enabled
        # is False inside validation's no_grad context). Skipped when the
        # phase loss carries no gradient (no phase head, lambda=0, or an
        # empty masked batch).
        if (
            self.train_cfg.get("grad_cosine", False)
            and torch.is_grad_enabled()
            and phase_loss.requires_grad
        ):
            grad_cos = _grad_cosine_similarity(
                action_loss, phase_loss, self.model.parameters()
            )
            if grad_cos is not None:
                metrics["grad_cos_action_phase"] = grad_cos.detach()

        return total_loss, metrics

    def _on_train_batch_processed(
        self,
        batch: dict[str, torch.Tensor],
        out: ModelOutput,
        metrics: dict[str, MetricValue],
        n: int,
    ) -> None:
        if out.phase_logits is not None:
            # Diagnostics follow the resolved field; fall back to
            # the legacy key only when the configured field is absent here
            # (per-batch loss already failed closed when active).
            _field = self._phase_label_field()
            _targets = batch.get(_field, batch.get("phase"))
            if _targets is not None:
                self._train_phase_acc.update(
                    out.phase_logits, _targets, mask=batch.get("padding_mask")
                )

    def epoch_train_metrics(self) -> dict[str, float]:
        if not self._train_phase_acc.has_data:
            return {}
        acc, balanced = self._train_phase_acc.compute()
        return {
            "train/phase_acc": acc,
            "train/phase_balanced_acc": balanced,
        }

    def _reset_train_pool(self) -> None:
        self._train_phase_acc.reset()

    def _reset_validation_pool(self) -> None:
        self._val_phase_acc.reset()

    def _collect_validation_pool(
        self,
        batch: dict[str, torch.Tensor],
        out: ModelOutput,
        metrics: dict[str, MetricValue],
    ) -> None:
        if out.phase_logits is not None:
            _field = self._phase_label_field()
            _targets = batch.get(_field, batch.get("phase"))
            if _targets is not None:
                self._val_phase_acc.update(
                    out.phase_logits, _targets, mask=batch.get("padding_mask")
                )

    def _finalize_validation_pool(self, agg_metrics: dict[str, float]) -> None:
        if self._val_phase_acc.has_data:
            acc, balanced = self._val_phase_acc.compute()
            agg_metrics["val/phase_acc"] = acc
            agg_metrics["val/phase_balanced_acc"] = balanced

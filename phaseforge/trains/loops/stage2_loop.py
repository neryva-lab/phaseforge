"""Stage 2 Trainer: Bootstrapped MoE specialization."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from phaseforge.evaluations.metrics import expert_utilization, phase_alignment, routing_stability
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
    resolve_margin_label_field,
    validate_label_values,
)

logger = logging.getLogger(__name__)

#: Cached 0.0 scalar per device. The aux-loss defaults below are requested
#: every training step; constructing a fresh device tensor each time is a
#: small per-step allocation × tens of thousands of steps per run. The cached
#: tensor is a read-only constant (value and dtype identical to
#: ``torch.tensor(0.0)``) — consumers only read or multiply it.
_ZERO_SCALARS: dict[torch.device, torch.Tensor] = {}


def _zero_scalar(device: torch.device) -> torch.Tensor:
    zero = _ZERO_SCALARS.get(device)
    if zero is None:
        zero = torch.zeros((), device=device)
        _ZERO_SCALARS[device] = zero
    return zero


class Stage2Trainer(BaseTrainer):
    """Trainer for Stage 2.

    Computes: L_total = L_action + β_balance * L_balance
    Action loss is MSE. Balance loss is auxiliary.
    Optionally freezes the encoder.

    Every epoch also logs the routing diagnostics (balance-vs-NMI
    trajectory) over the validation set — ``val/phase_expert_nmi``,
    ``val/topk_balance_score``, ``val/topk_collapse_rate``,
    ``val/top1_balance_score``, ``val/top1_collapse_rate``,
    ``val/routing_entropy`` —
    so pseudo-balancing (balance high while NMI collapses to 0) is visible
    at training time instead of only in offline evaluation.

    For the final teacher diagnostic, the label-free routing accuracy (micro
    and macro/balanced) is computed over the SAME inference path that
    selects experts by the frozen phase head (``out.phase_logits`` is only
    non-None for that cell in Stage 2) — never from the GT-routed training
    path.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._val_routing_acc = _PhaseAccumulator(device=self.device)

    def _phase_label_field(self) -> str:
        """Resolved Stage 2 diagnostic label field."""
        return resolve_label_field(
            self.train_cfg, "phase_label_field", cfg=self.cfg, purpose="Stage 2 diagnostics"
        )

    def _margin_label_field(self) -> str:
        """Resolved Stage 2 margin-routing label field."""
        return resolve_margin_label_field(self.train_cfg, cfg=self.cfg)

    def _router_num_experts(self) -> int:
        """Expert count from the active router (margin validation width)."""
        try:
            router = getattr(getattr(self.model, "moe_layer", None), "router", None)
            if router is not None and getattr(router, "num_experts", None) is not None:
                return int(router.num_experts)
        except Exception:
            pass
        return LOCKED_NUM_CLASSES

    def validate_label_contract(self) -> None:
        """Preflight: fail closed on missing/invalid Stage 2 label fields.

        Checks the resolved ``phase_label_field`` (diagnostics / weighted
        losses) and, when the margin loss is enabled, ``margin.label_field``
        against one sample batch from each split before any optimizer step.
        """
        ctx = describe_run(self.cfg)
        # Peek without perturbing the sampler (see Stage 1 preflight).
        train_batch = peek_loader_batch(self.train_loader)
        if train_batch is None:
            raise RuntimeError(
                "Stage 2 label preflight: train loader is empty. "
                f"(method={ctx['method']}, task={ctx['task']}, seed={ctx['seed']})"
            )
        val_batch = (
            peek_loader_batch(self.val_loader) if self.val_loader is not None else None
        )
        phase_field = self._phase_label_field()
        for split, sample in (("train", train_batch), ("val", val_batch)):
            if sample is None:
                continue
            if phase_field not in sample:
                available = sorted(str(k) for k in sample.keys())
                raise RuntimeError(
                    f"Stage 2 label preflight: label field {phase_field!r} "
                    f"is missing from the {split} batch. "
                    f"(method={ctx['method']}, task={ctx['task']}, split={split}, "
                    f"seed={ctx['seed']}, requested_field={phase_field}, "
                    f"available_keys={available})"
                )
            validate_label_values(
                sample[phase_field], self._router_num_experts(),
                cfg=self.cfg, field=phase_field,
                purpose="Stage 2 label preflight", split=split,
            )
        margin_cfg = self.train_cfg.get("margin", None)
        margin_enabled = bool(margin_cfg.get("enabled", False)) if margin_cfg is not None else False
        if margin_enabled:
            margin_field = self._margin_label_field()
            for split, sample in (("train", train_batch), ("val", val_batch)):
                if sample is None:
                    continue
                if margin_field not in sample:
                    available = sorted(str(k) for k in sample.keys())
                    raise RuntimeError(
                        f"Stage 2 label preflight: margin label field {margin_field!r} "
                        f"is missing from the {split} batch. "
                        f"(method={ctx['method']}, task={ctx['task']}, split={split}, "
                        f"seed={ctx['seed']}, requested_field={margin_field}, "
                        f"available_keys={available})"
                    )
                validate_label_values(
                    sample[margin_field], self._router_num_experts(),
                    cfg=self.cfg, field=margin_field,
                    purpose="Stage 2 margin preflight", split=split,
                )

    def fit(self) -> None:
        """Override fit to handle encoder freezing and fine-tuning LR scaling."""
        from hydra.utils import instantiate

        # Preflight: resolved label fields stop misconfigured
        # method/task/seed combinations before any optimizer state exists.
        self.validate_label_contract()

        # Precedence: per-model models.freeze_encoder wins when key is present;
        # train.freeze_encoder otherwise; default True.
        models_cfg = (
            self.cfg.get("models")
            if hasattr(self, "cfg") and self.cfg is not None
            else None
        )
        if (
            models_cfg is not None
            and "freeze_encoder" in models_cfg
            and models_cfg.get("freeze_encoder") is not None
        ):
            freeze = bool(models_cfg.get("freeze_encoder"))
        elif self.train_cfg.get("freeze_encoder") is not None:
            freeze = bool(self.train_cfg.get("freeze_encoder"))
        else:
            freeze = True

        if freeze:
            if hasattr(self.model, "freeze_encoder"):
                self.model.freeze_encoder()
        elif hasattr(self.model, "unfreeze_encoder"):
            self.model.unfreeze_encoder()

        # Precedence: models.encoder_lr_scale wins if present, else
        # train.encoder_lr_scale, else 1.0.
        if (
            models_cfg is not None
            and "encoder_lr_scale" in models_cfg
            and models_cfg.get("encoder_lr_scale") is not None
        ):
            encoder_lr_scale = float(models_cfg.get("encoder_lr_scale"))
        elif self.train_cfg.get("encoder_lr_scale") is not None:
            encoder_lr_scale = float(self.train_cfg.get("encoder_lr_scale"))
        else:
            encoder_lr_scale = 1.0
        if not freeze and encoder_lr_scale != 1.0 and hasattr(self.model, "encoder"):
            base_lr = float(self.train_cfg.optimizer.get("lr", 1e-4))
            encoder_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
            other_params = [
                p for n, p in self.model.named_parameters()
                if not n.startswith("encoder.") and p.requires_grad
            ]
            param_groups = [
                {"params": other_params, "lr": base_lr},
                {"params": encoder_params, "lr": base_lr * encoder_lr_scale},
            ]
            self.optimizer = instantiate(
                self.train_cfg.optimizer, params=param_groups, _convert_="all"
            )
            logger.info(
                f"Stage 2 Fine-Tuning: Encoder trained with scaled LR "
                f"{base_lr * encoder_lr_scale:.2e} (scale={encoder_lr_scale}), "
                f"other parameters at base LR {base_lr:.2e}."
            )
        else:
            active_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = instantiate(self.train_cfg.optimizer, params=active_params)

        self.scheduler = instantiate(self.train_cfg.scheduler, optimizer=self.optimizer)

        # IS-path guardrail: λ_bal ≪ 1 so the balance term
        # cannot override the discovered regime structure. Warn once here
        # (not per batch) when any IS loss is enabled with a large balance
        # coefficient on the active router.
        is_term_keys = ("margin", "lipschitz", "gain_reg")
        is_path = any(
            bool((self.train_cfg.get(key, {}) or {}).get("enabled", False))
            for key in is_term_keys
        )
        if is_path:
            router = getattr(getattr(self.model, "moe_layer", None), "router", None)
            try:
                balance_coeff = float(getattr(router, "balance_coeff", 0.0))
            except (TypeError, ValueError):
                balance_coeff = 0.0
            if balance_coeff > 1e-3:
                logger.warning(
                    "IS-path losses are enabled with router balance_coeff=%.4g > 1e-3; "
                    "the balance term may override regime structure. Prefer λ_bal ≪ 1.",
                    balance_coeff,
                )

        active_params = [p for p in self.model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in active_params)
        logger.info(f"Stage 2 initialized. Trainable parameters: {n_params}")

        super().fit()

    def _teacher_kl_coeff(self) -> float:
        """TGR schedule: λ0 for the first half, then linear anneal to 0.

        The teacher signal shapes the router early and is faded out so the
        learned gate takes over in the second half. ``lambda0 <= 0`` (the
        default) disables the schedule entirely.
        """
        tcfg = self.train_cfg.get("teacher_routing") or {}
        lambda0 = float(tcfg.get("lambda0", 0.0))
        if lambda0 <= 0.0:
            return 0.0
        total = max(1, int(self.train_cfg.get("epochs", 1)))
        t = min(max(float(self.current_epoch), 0.0), float(total)) / total
        if t <= 0.5:
            return lambda0
        return lambda0 * 2.0 * (1.0 - t)

    def _compute_loss(
        self, batch: dict[str, torch.Tensor], out: ModelOutput | None = None
    ) -> tuple[torch.Tensor, dict[str, MetricValue]]:
        # Forward pass (reuse an existing output when provided, e.g. from the
        # validation loop, to avoid a double forward).
        if out is None:
            out = self.model(batch)

        target_action = batch["action"]  # (B, A) or (B, T, A)
        mask = batch.get("padding_mask")  # (B, T)
        # Resolved diagnostic label field: topology rows
        # use "phase_topo", the Static Rule comparison uses "phase".
        phase_field = self._phase_label_field()
        phase_weights = self.train_cfg.get("phase_weights", None)
        dim_weights = self.train_cfg.get("dim_weights", None)

        if dim_weights is not None or phase_weights is not None:
            sq_err = F.mse_loss(out.action_pred, target_action, reduction="none")
            if dim_weights is not None:
                dw_tensor = torch.as_tensor(dim_weights, dtype=torch.float32, device=self.device)
                if dw_tensor.numel() == 7 and target_action.size(-1) == 14:
                    dw_tensor = dw_tensor.repeat(2)
                sq_err = sq_err * dw_tensor
            if phase_weights is not None:
                # Fail-closed: a requested weighting with a missing/invalid
                # field stops instead of silently training unweighted.
                phase_ids = resolve_label_tensor(
                    batch, phase_field, cfg=self.cfg,
                    purpose="Stage 2 phase-weighted action loss",
                    num_classes=self._router_num_experts(),
                )
                weights_tensor = torch.as_tensor(
                    phase_weights, dtype=torch.float32, device=self.device
                )
                max_p = int(phase_ids.max().item()) if phase_ids.numel() > 0 else 0
                if weights_tensor.numel() <= max_p:
                    pad = torch.ones(
                        max_p + 1 - weights_tensor.numel(),
                        dtype=weights_tensor.dtype,
                        device=weights_tensor.device,
                    )
                    weights_tensor = torch.cat([weights_tensor, pad])
                sample_weights = weights_tensor[phase_ids.long()]
                if mask is not None:
                    sq_err = sq_err[mask]
                    sample_weights = sample_weights[mask]
                if sq_err.ndim > 1:
                    action_loss = (sq_err.mean(dim=-1) * sample_weights).mean()
                else:
                    action_loss = (sq_err * sample_weights).mean()
            elif mask is not None:
                action_loss = sq_err[mask].mean()
            else:
                action_loss = sq_err.mean()
        elif mask is not None:
            action_loss = F.mse_loss(out.action_pred, target_action, reduction="none")
            action_loss = action_loss[mask].mean()
        else:
            action_loss = F.mse_loss(out.action_pred, target_action)

        # Release Kinematics Auxiliary Loss.
        # When gripper is commanded to open during Place, penalizes lateral
        # command drift (x, y velocities) to enforce "stop-then-drop"
        # release stability. Robosuite convention: -1.0 is open
        # (releasing), +1.0 is closed (holding).
        # Disabled by default; when enabled it follows the resolved phase
        # field below (its own label-field contract) and never silently
        # falls back to a different vocabulary.
        release_loss = _zero_scalar(self.device)
        rel_cfg = self.train_cfg.get("release_loss", None)
        rel_enabled = bool(rel_cfg.get("enabled", False)) if rel_cfg is not None else False
        if rel_enabled and target_action.size(-1) >= 7:
            lambda_rel = float(rel_cfg.get("lambda_rel", 0.1))
            grip_threshold = float(rel_cfg.get("gripper_threshold", 0.0))
            rel_phase = resolve_label_tensor(
                batch, phase_field, cfg=self.cfg,
                purpose="Stage 2 release loss",
                num_classes=self._router_num_experts(),
            )
            if target_action.size(-1) >= 14:
                is_rel_0 = target_action[..., 6] < grip_threshold
                is_rel_1 = target_action[..., 13] < grip_threshold
                place_phase = int(rel_cfg.get("place_phase", 4))
                is_rel_0 = is_rel_0 & (rel_phase == place_phase)
                is_rel_1 = is_rel_1 & (rel_phase == place_phase)
                if mask is not None:
                    is_rel_0 = is_rel_0 & mask
                    is_rel_1 = is_rel_1 & mask
                loss_0 = (
                    (out.action_pred[is_rel_0, 0:2] ** 2).sum(dim=-1).mean()
                    if is_rel_0.any()
                    else _zero_scalar(self.device)
                )
                loss_1 = (
                    (out.action_pred[is_rel_1, 7:9] ** 2).sum(dim=-1).mean()
                    if is_rel_1.any()
                    else _zero_scalar(self.device)
                )
                release_loss = lambda_rel * (loss_0 + loss_1)
            else:
                is_releasing = target_action[..., 6] < grip_threshold
                place_phase = int(rel_cfg.get("place_phase", 4))
                is_releasing = is_releasing & (rel_phase == place_phase)
                if mask is not None:
                    is_releasing = is_releasing & mask
                if is_releasing.any():
                    lat_pred = out.action_pred[is_releasing, 0:2]
                    release_loss = lambda_rel * (lat_pred ** 2).sum(dim=-1).mean()

        balance_loss = out.aux_losses.get("balance", _zero_scalar(self.device))

        # Stickiness loss: -log p_t[top1_{t-1}], emitted by the router
        # when use_history=True. train.sticky_coeff scales it; 0.0 keeps the
        # loss bit-identical (0.0 * x == 0.0).
        sticky_loss = out.aux_losses.get("sticky", _zero_scalar(self.device))
        sticky_coeff = float(self.train_cfg.get("sticky_coeff", 0.0))

        # Teacher KL routing. Single source of truth is
        # models.teacher_routing.enabled (the same block the model's forward
        # reads to emit phase_logits in Stage 2); the trainer additionally
        # requires phase_logits present, so the teacher diagnostic (which
        # emits them through its own label-free eval path) never receives the
        # teacher KL. T = M^T softmax(phase_logits), detached; λ(t) = λ0 for
        # the first half of training, then linear anneal to 0.
        teacher_kl = _zero_scalar(self.device)
        teacher_lambda = 0.0
        models_cfg = self.cfg.get("models") or {}
        teacher_enabled = bool(
            (models_cfg.get("teacher_routing") or {}).get("enabled", False)
        )
        if (
            teacher_enabled
            and out.phase_logits is not None
            and out.gate_logits is not None
        ):
            require_soft_mapping = getattr(self.model, "require_soft_mapping", None)
            if require_soft_mapping is None:
                raise RuntimeError(
                    "models.teacher_routing.enabled=true but the model has no "
                    "require_soft_mapping() accessor — teacher KL needs the "
                    "soft phase->expert mapping M."
                )
            mapping = require_soft_mapping().detach()
            phase_probs = F.softmax(out.phase_logits, dim=-1)
            teacher_target = torch.einsum("pe,bp->be", mapping, phase_probs).detach()
            teacher_kl = F.kl_div(
                F.log_softmax(out.gate_logits, dim=-1),
                teacher_target,
                reduction="batchmean",
            )
            teacher_lambda = self._teacher_kl_coeff()

        total_loss = (
            action_loss
            + release_loss
            + balance_loss
            + sticky_coeff * sticky_loss
            + teacher_lambda * teacher_kl
        )

        # Large-margin prototype routing. Disabled by
        # default (`train.margin.enabled=false`): absent/disabled contributes
        # exactly 0.0 and emits no metrics, so legacy loss curves and the
        # exact metric-key assertion in test_compute_loss_accepts_reused_output
        # are bit-identical. Enabled only with a prototype router, whose
        # gate logits are negative distances (see PrototypeRouter).
        margin_cfg = self.train_cfg.get("margin", None)
        if margin_cfg is None:
            margin_enabled = False
        else:
            margin_enabled = bool(margin_cfg.get("enabled", False))
        if margin_enabled:
            assert margin_cfg is not None
            margin_lambda = float(margin_cfg.get("lambda_margin", 1.0))
            margin_value = float(margin_cfg.get("margin", 0.5))
            router = getattr(getattr(self.model, "moe_layer", None), "router", None)
            margin_fn = getattr(router, "margin_loss_from_logits", None)
            if margin_fn is None or out.gate_logits is None:
                raise RuntimeError(
                    "train.margin.enabled=true requires a prototype router "
                    "(or any router with the common margin_loss_from_logits "
                    "interface, e.g. TopKRouter) emitting gate "
                    "logits; the active model provides neither."
                )
            # Margin targets use deterministic pre-exploration
            # logits so training-time exploration noise (TopKRouter) never
            # silently shifts them. Prototype routers report identical
            # tensors for both; TopK routers report the pre-noise signal.
            # Diagnostics (entropy/NMI/utilization) intentionally keep using
            # out.gate_logits — the actually-dispatched routing signal.
            clean = getattr(out, "clean_gate_logits", None)
            logits_source = clean if clean is not None else out.gate_logits
            targets = resolve_label_tensor(
                batch, self._margin_label_field(), cfg=self.cfg,
                purpose="Stage 2 margin",
                num_classes=int(logits_source.size(-1)),
            )
            if mask is not None:
                flat_logits = logits_source.view(-1, logits_source.size(-1))
                flat_targets = targets.view(-1)
                keep = mask.view(-1).bool()
                margin_logits = flat_logits[keep]
                margin_targets = flat_targets[keep]
            else:
                margin_logits = logits_source
                margin_targets = targets
            if margin_targets.numel() == 0:
                loss_margin = _zero_scalar(self.device)
            else:
                loss_margin = margin_fn(margin_logits, margin_targets, margin_value)
            total_loss = total_loss + margin_lambda * loss_margin

        metrics = {
            # Defer .item() until the trainer logs/aggregates the metric so
            # the hot training loop does not synchronize CUDA every batch.
            "loss_total": total_loss.detach(),
            "loss_action": action_loss.detach(),
            "loss_balance": balance_loss.detach(),
            "loss_sticky": (sticky_coeff * sticky_loss).detach(),
            "loss_teacher_kl": (teacher_lambda * teacher_kl).detach(),
            "teacher_lambda": teacher_lambda,
        }
        if rel_enabled:
            metrics["loss_release"] = release_loss.detach()
        if self.train_cfg.get("log_dimension_mse", False):
            with torch.no_grad():
                diff_sq = (out.action_pred - target_action) ** 2
                if mask is not None:
                    diff_sq = diff_sq[mask]
                if diff_sq.ndim > 1 and diff_sq.size(-1) >= 7:
                    metrics["loss_action_pos"] = diff_sq[:, 0:3].mean().detach()
                    metrics["loss_action_rot"] = diff_sq[:, 3:6].mean().detach()
                    metrics["loss_action_grip"] = diff_sq[:, 6].mean().detach()
                    if phase_field in batch:
                        p_flat = batch[phase_field]
                        if mask is not None:
                            p_flat = p_flat[mask]
                        p4 = p_flat == 4
                        if p4.any():
                            metrics["loss_action_place"] = diff_sq[p4].mean().detach()
        if margin_enabled:
            metrics["loss_margin"] = loss_margin.detach()
            metrics["margin_lambda"] = margin_lambda

        # Local contraction + gain regularization.
        # Disabled by default (absent/disabled contributes exactly 0.0 and
        # emits no metrics). Enabled only with impedance experts, whose
        # forward exposes per-sample (target, gains, task_state) in
        # `out.info`; anything else fails closed.
        lip_cfg = self.train_cfg.get("lipschitz", None)
        lip_enabled = bool(lip_cfg.get("enabled", False)) if lip_cfg is not None else False
        if lip_enabled:
            from phaseforge.trains.losses.lipschitz import lip_penalty

            assert lip_cfg is not None
            info = out.info or {}
            try:
                lip_targets = info["target"]
                lip_states = info["task_state"]
                lip_experts = info["expert_index"]
            except KeyError as exc:
                raise RuntimeError(
                    "train.lipschitz.enabled=true requires impedance experts "
                    f"exposing per-sample diagnostics; missing info key {exc}."
                ) from exc
            lip_lambda = float(lip_cfg.get("lambda_lip", 0.1))
            lip_rho = float(lip_cfg.get("rho", 0.8))
            lip_pairs = int(lip_cfg.get("num_pairs", 256))
            lip_coord_slice = lip_cfg.get("coord_slice")
            coord_slice = (
                tuple(int(v) for v in lip_coord_slice)
                if lip_coord_slice is not None
                else None
            )
            loss_lip = lip_penalty(
                lip_targets,
                lip_states,
                lip_experts,
                trajectory_ids=batch.get("trajectory_id"),
                rho=lip_rho,
                num_pairs=lip_pairs,
                coord_slice=coord_slice,
            )
            total_loss = total_loss + lip_lambda * loss_lip
            metrics["loss_lip"] = loss_lip.detach()
            metrics["lip_lambda"] = lip_lambda

        gain_cfg = self.train_cfg.get("gain_reg", None)
        gain_enabled = bool(gain_cfg.get("enabled", False)) if gain_cfg is not None else False
        if gain_enabled:
            from phaseforge.trains.losses.lipschitz import gain_penalty

            assert gain_cfg is not None
            info = out.info or {}
            if "gains" not in info:
                raise RuntimeError(
                    "train.gain_reg.enabled=true requires impedance experts "
                    "exposing per-sample gains in `out.info`."
                )
            gain_lambda = float(gain_cfg.get("lambda_gain", 1.0e-4))
            gain_nominal = float(gain_cfg.get("kappa_nominal", 1.0))
            loss_gain = gain_penalty(info["gains"], kappa_nominal=gain_nominal)
            total_loss = total_loss + gain_lambda * loss_gain
            metrics["loss_gain"] = loss_gain.detach()
            metrics["gain_lambda"] = gain_lambda

        # Re-materialize the total after the optional terms above: the
        # metrics dict snapshotted it before they were added.
        metrics["loss_total"] = total_loss.detach()

        return total_loss, metrics

    @torch.inference_mode()
    def _validate(self) -> dict[str, float]:
        """Validation plus per-epoch routing diagnostics.

        Reuses the forward outputs for both the loss metrics and the routing
        metrics so validation stays a single forward pass per batch.

        Loss metrics are aggregated sample-weighted (see
        :meth:`BaseTrainer._batch_sample_count`) with on-device float64
        accumulation — bit-identical to the former Python-float arithmetic,
        materialized once at the epoch boundary instead of one host/device
        sync per tensor metric per batch. Routing tensors (expert indices,
        gate logits, phases, trajectory keys) are likewise collected on-device
        and moved to the host with a single ``torch.cat(...).cpu()`` per
        epoch.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        agg_sums: dict[str, torch.Tensor] = {}
        total_samples = 0
        self._val_routing_acc.reset()

        expert_indices_all: list[torch.Tensor] = []
        phases_all: list[torch.Tensor] = []
        gate_logits_all: list[torch.Tensor] = []
        traj_id_all: list[torch.Tensor] = []
        traj_pos_all: list[torch.Tensor] = []

        for batch in self.val_loader:
            batch = self._move_batch(batch)

            out = self.model(batch)
            _, metrics = self._compute_loss(batch, out=out)

            # Routing accuracy against the resolved diagnostic field when the
            # model emits phase_logits in Stage 2: the
            # teacher diagnostic (label-free eval path) and the teacher
            # path both qualify.
            if out.phase_logits is not None:
                _vfield = self._phase_label_field()
                _vt = batch.get(_vfield, batch.get("phase"))
                if _vt is not None:
                    self._val_routing_acc.update(
                        out.phase_logits, _vt, mask=batch.get("padding_mask")
                    )

            n = self._batch_sample_count(batch)
            if n == 0:
                continue
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    contribution = v.detach().to(device=self.device, dtype=torch.float64)
                else:
                    contribution = torch.tensor(
                        float(v), device=self.device, dtype=torch.float64
                    )
                acc = agg_sums.get(k)
                agg_sums[k] = contribution * n if acc is None else acc + contribution * n
            total_samples += n

            if out.expert_indices is not None:
                expert_indices_all.append(out.expert_indices.detach())
                # NMI diagnostics follow the resolved field.
                _nfield = self._phase_label_field()
                phases_all.append(batch.get(_nfield, batch["phase"]))
                if "trajectory_id" in batch:
                    traj_id_all.append(batch["trajectory_id"])
                    traj_pos_all.append(batch["trajectory_position"])
            if out.gate_logits is not None:
                gate_logits_all.append(out.gate_logits.detach())

        if total_samples == 0:
            return {}
        agg_metrics = {k: float((v / total_samples).item()) for k, v in agg_sums.items()}

        # Routing diagnostics: balance-vs-specialization trajectory.
        if expert_indices_all:
            expert_indices = torch.cat(expert_indices_all, dim=0).cpu()
            phases = torch.cat(phases_all, dim=0).cpu()

            agg_metrics["val/phase_expert_nmi"] = phase_alignment.phase_expert_nmi(
                phases, expert_indices
            )

            # Use the CONFIGURED expert count (gate-logit width), never the
            # largest observed index: an unused expert must still count as
            # dead, otherwise collapse/balance look healthy while an expert
            # never fires. Note the balance/collapse diagnostics count all
            # top-k assignments, whereas the training balance loss uses
            # top-1 assignments (documented in TopKRouter._compute_balance_loss).
            if gate_logits_all:
                num_experts = gate_logits_all[0].size(-1)
            else:
                num_experts = int(expert_indices.max().item()) + 1
            topk_fractions = expert_utilization.expert_utilization(expert_indices, num_experts)
            top1_fractions = expert_utilization.expert_utilization_top1(expert_indices, num_experts)
            agg_metrics["val/topk_balance_score"] = expert_utilization.expert_utilization_balance(
                topk_fractions
            )
            agg_metrics["val/top1_balance_score"] = expert_utilization.expert_utilization_balance(
                top1_fractions
            )
            agg_metrics["val/topk_collapse_rate"] = expert_utilization.collapse_rate(topk_fractions)
            agg_metrics["val/top1_collapse_rate"] = expert_utilization.collapse_rate(top1_fractions)

        if gate_logits_all:
            gate_logits = torch.cat(gate_logits_all, dim=0).cpu()
            agg_metrics["val/routing_entropy"] = routing_stability.routing_entropy(
                gate_logits, normalize=True
            ).item()

        # How often the top-1 expert changes between adjacent
        # in-trajectory steps (0.0 = perfectly sticky, 1.0 = flips every
        # step). Absent when the validation batches carry no trajectory keys.
        if traj_id_all:
            agg_metrics["val/routing_switch_rate"] = (
                routing_stability.routing_switch_rate(
                    expert_indices[:, 0],
                    torch.cat(traj_id_all, dim=0).cpu(),
                    torch.cat(traj_pos_all, dim=0).cpu(),
                )
            )

        if self._val_routing_acc.has_data:
            acc, balanced = self._val_routing_acc.compute()
            agg_metrics["val/routing_accuracy"] = acc
            agg_metrics["val/routing_balanced_accuracy"] = balanced

        logger.info(
            "Epoch %d routing diagnostics: NMI=%.4f topk_balance=%.4f "
            "top1_balance=%.4f topk_collapse=%.4f top1_collapse=%.4f entropy=%.4f "
            "switch_rate=%.4f",
            self.current_epoch,
            agg_metrics.get("val/phase_expert_nmi", float("nan")),
            agg_metrics.get("val/topk_balance_score", float("nan")),
            agg_metrics.get("val/top1_balance_score", float("nan")),
            agg_metrics.get("val/topk_collapse_rate", float("nan")),
            agg_metrics.get("val/top1_collapse_rate", float("nan")),
            agg_metrics.get("val/routing_entropy", float("nan")),
            agg_metrics.get("val/routing_switch_rate", float("nan")),
        )

        return agg_metrics

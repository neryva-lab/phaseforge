"""Abstract base trainer defining the training lifecycle."""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from phaseforge.models.base import BaseManipulationModel, ModelOutput

logger = logging.getLogger(__name__)

MetricValue = float | torch.Tensor


def _flatten_valid(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten logits/targets to ``(N, C)`` / ``(N,)``, dropping padding.

    Handles single-step ``(B, C)`` / ``(B,)`` and multi-step
    ``(B, T, C)`` / ``(B, T)`` shapes; without a mask nothing is dropped.
    """
    if mask is None:
        return logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.reshape(-1)
    mask_flat = mask.bool().reshape(-1)
    return logits_flat[mask_flat], targets_flat[mask_flat]


class _PhaseAccumulator:
    """Accumulates micro + macro/balanced phase accuracy on the device.

    Keeps per-class correct/count tensors on the training device and
    materializes scalars only in :meth:`compute` (at an epoch boundary), so
    the training loop never synchronizes CUDA per batch.
    """

    def __init__(self, device: torch.device, num_phases: int | None = None) -> None:
        self.device = device
        self.num_phases = num_phases
        self._correct: torch.Tensor | None = None
        self._total: torch.Tensor | None = None
        self._samples = 0

    def reset(self) -> None:
        self._correct = None
        self._total = None
        self._samples = 0

    @property
    def has_data(self) -> bool:
        return self._samples > 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        logits, targets = _flatten_valid(logits, targets, mask)
        if logits.numel() == 0:
            return
        targets = targets.long()
        if self.num_phases is None:
            self.num_phases = int(logits.size(-1))
        n_classes = self.num_phases
        if (targets >= n_classes).any() or (targets < 0).any():
            valid_idx = (targets >= 0) & (targets < n_classes)
            if not valid_idx.any():
                return
            logits = logits[valid_idx]
            targets = targets[valid_idx]
        if self._correct is None:
            self._correct = torch.zeros(n_classes, device=logits.device)
            self._total = torch.zeros(n_classes, device=logits.device)
        correct_acc = self._correct
        total_acc = self._total
        assert correct_acc is not None and total_acc is not None
        correct = (logits.argmax(dim=-1) == targets).float()
        correct_acc.scatter_add_(0, targets, correct)
        total_acc.scatter_add_(
            0,
            targets,
            torch.ones(targets.numel(), device=logits.device),
        )
        self._samples += targets.numel()

    def compute(self) -> tuple[float, float]:
        """Return ``(micro_accuracy, balanced_accuracy)`` as Python floats."""
        if not self.has_data:
            raise ValueError("_PhaseAccumulator.compute() with no data")
        correct_acc = self._correct
        total_acc = self._total
        assert correct_acc is not None and total_acc is not None
        micro = float((correct_acc.sum() / total_acc.sum()).item())
        recall = correct_acc / total_acc
        valid = total_acc > 0
        balanced = float(recall[valid].mean().item()) if valid.any() else float("nan")
        return micro, balanced


class BaseTrainer(ABC):
    """Abstract base class for all training loops.

    Defines a standard lifecycle:
    fit -> [on_train_start] -> loop epochs -> [on_epoch_start] -> train_epoch ->
    validate -> [on_epoch_end] -> end loop -> [on_train_end].

    Subclasses must implement _compute_loss.
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: BaseManipulationModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
    ) -> None:
        self.cfg = cfg
        self.train_cfg = cfg.train

        requested_device = str(cfg.project.get("device", "cuda"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "Device '%s' requested but CUDA is unavailable. Falling back to CPU.",
                requested_device,
            )
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epochs = self.train_cfg.epochs
        self.grad_clip_norm = self.train_cfg.grad_clip_norm
        self.log_every_n_steps = self.train_cfg.log_every_n_steps

        self.optimizer = instantiate(self.train_cfg.optimizer, params=self.model.parameters())
        self.scheduler = instantiate(self.train_cfg.scheduler, optimizer=self.optimizer)

        self.callbacks: list[Any] = []
        self.current_epoch = 0
        self.global_step = 0
        self.should_stop = False
        # On-device "every training loss this epoch was finite" flag. AND-ed
        # per batch without a host sync; materialized once per epoch in
        # :meth:`_check_epoch_loss_finite` (see the guard note in
        # :meth:`_train_epoch`).
        self._epoch_loss_finite: torch.Tensor | None = None
        # Checkpoint payload queued by resume(); applied at the start of the
        # next fit() so subclasses that re-create the optimizer (e.g. after
        # freezing) restore state into the final optimizer instance.
        self._resume_payload: dict[str, Any] | None = None

    def resume(self, checkpoint_path: str | Path) -> None:
        """Queue a checkpoint for restoration at the next :meth:`fit`.

        Restores model weights, optimizer/scheduler state, epoch, global
        step, RNG state (torch/numpy/random/cuda) and callback state so an
        interrupted run can continue from where it stopped (model state is
        exact; per-epoch DataLoader shuffle *order* is re-seeded from
        ``project.seed`` and may differ from an uninterrupted run — see
        :class:`~phaseforge.trains.callbacks.checkpointing.CheckpointCallback`).
        Applied in :meth:`fit` (after any optimizer re-instantiation by
        subclasses).

        Args:
            checkpoint_path: Path to a checkpoint saved by the
                :class:`~phaseforge.trains.callbacks.checkpointing.CheckpointCallback`.
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        required = {"model_state_dict", "optimizer_state_dict"}
        missing = required - set(ckpt.keys())
        if missing:
            raise ValueError(
                f"Resume checkpoint {path} is missing required keys: {sorted(missing)}"
            )
        self._resume_payload = ckpt
        logger.info(f"Queued checkpoint for resume: {path}")

    def _apply_resume_payload(self) -> None:
        """Restore the state recorded by :meth:`resume` into this trainer."""
        ckpt = self._resume_payload
        if ckpt is None:
            return
        self._resume_payload = None

        # Strict load: resuming into a different architecture must fail
        # loudly instead of silently leaving random weights.
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        self.current_epoch = int(ckpt.get("epoch", self.current_epoch))
        self.global_step = int(ckpt.get("global_step", self.global_step))
        self.should_stop = bool(ckpt.get("should_stop", self.should_stop))
        if hasattr(self.model, "stage") and ckpt.get("stage") is not None:
            setattr(self.model, "stage", int(ckpt["stage"]))

        rng = ckpt.get("rng_state")
        if isinstance(rng, dict):
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"].cpu())
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if rng.get("random") is not None:
                random.setstate(rng["random"])
            cuda_states = rng.get("cuda")
            if cuda_states and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cuda() for s in cuda_states])

        callbacks = ckpt.get("callbacks", {})
        for cb in self.callbacks:
            state = callbacks.get(type(cb).__name__)
            if state is not None and hasattr(cb, "load_state_dict"):
                cb.load_state_dict(state)

        logger.info(
            "Resumed training at epoch %d, global step %d.",
            self.current_epoch,
            self.global_step,
        )

    def add_callback(self, callback: Any) -> None:
        """Add a lifecycle callback."""
        self.callbacks.append(callback)

    def _trigger_callbacks(self, hook: str, **kwargs: Any) -> None:
        """Trigger a specific hook on all registered callbacks."""
        for cb in self.callbacks:
            method = getattr(cb, hook, None)
            if method:
                method(trainer=self, **kwargs)

    def _forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        """Run the model on a batch. Override to wrap/route the forward."""
        return self.model(batch)

    @abstractmethod
    def _compute_loss(
        self, batch: dict[str, torch.Tensor], out: ModelOutput | None = None
    ) -> tuple[torch.Tensor, dict[str, MetricValue]]:
        """Compute the loss for a single batch.

        ``out`` may be supplied by a caller that already ran the forward
        (validation reuses it to avoid a double pass); when ``None`` the
        trainer runs :meth:`_forward` itself.

        Returns:
            total_loss: The scalar loss tensor to backpropagate.
            metrics: A dictionary of metric floats to log.
        """

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Move tensor fields to the training device.

        ``pin_memory`` only helps when the host-to-CUDA copy is explicitly
        non-blocking. The CPU path remains unchanged, while CUDA batches can
        overlap their transfer with device work where the DataLoader provides
        pinned tensors.
        """
        non_blocking = self.device.type == "cuda"
        return {
            key: value.to(self.device, non_blocking=non_blocking)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }

    @staticmethod
    def _metric_to_float(value: MetricValue) -> float:
        """Materialize a metric only at a logging/aggregation boundary."""
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def _epoch_timing_dict(
        self, *, epoch_wall_seconds: float, train_seconds: float, steps_in_epoch: int
    ) -> dict[str, float | None]:
        """Assemble the per-epoch timing record persisted in curve rows."""
        return {
            "epoch_wall_seconds": float(epoch_wall_seconds),
            "train_steps_per_second": (
                float(steps_in_epoch / train_seconds) if train_seconds > 0.0 else float("nan")
            ),
            "steps_in_epoch": float(steps_in_epoch),
            "peak_gpu_memory_mb": self._peak_gpu_memory_mb(),
        }

    def epoch_timing(self) -> dict[str, float | None]:
        """Per-epoch timing for the most recently completed epoch."""
        return dict(self._epoch_timing)

    def parameter_counts(self) -> dict[str, int]:
        """Trainable and total parameter counts **after** freeze logic."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"trainable_params": int(trainable), "total_params": int(total)}

    def _peak_gpu_memory_mb(self) -> float | None:
        """Peak GPU memory this epoch, or ``None`` on CPU."""
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        peak = torch.cuda.max_memory_allocated(self.device)
        return float(peak / (1024.0 * 1024.0)) if peak > 0 else None

    def _on_train_batch_processed(
        self,
        batch: dict[str, torch.Tensor],
        out: ModelOutput,
        metrics: dict[str, MetricValue],
        n: int,
    ) -> None:
        """Trainer-level per-batch hook (default: no-op).

        Subclasses accumulate train-phase scalars here so the epoch-level
        values are exact without per-batch host/device syncs.
        """

    def epoch_train_metrics(self) -> dict[str, float]:
        """Extra per-epoch **training** scalars beyond the loss metrics.

        Stage 1 returns ``train/phase_acc`` + ``train/phase_balanced_acc``
        when the model has a phase head; base returns ``{}``.
        """
        return {}

    def fit(self) -> None:
        """Execute the full training loop."""
        # Restore resume state now: subclasses that re-create the optimizer
        # (e.g. after freezing the encoder) have done so by this point.
        self._apply_resume_payload()

        logger.info(f"Starting training for {self.epochs} epochs on {self.device}.")
        self._trigger_callbacks("on_train_start")

        epoch_pbar = None
        epoch_iter = range(self.current_epoch + 1, self.epochs + 1)
        if self.train_cfg.get("epoch_progressbar", False):
            from tqdm.auto import tqdm

            models_cfg = self.cfg.get("models")
            model_name = str(
                models_cfg.get("name", type(self.model).__name__)
                if models_cfg is not None
                else type(self.model).__name__
            )
            stage_name = self.train_cfg.get("stage", getattr(self.model, "stage", "?"))
            epoch_pbar = tqdm(
                epoch_iter,
                total=max(0, self.epochs - self.current_epoch),
                desc=f"{model_name} stage {stage_name}",
                unit="epoch",
                dynamic_ncols=True,
                leave=True,
                mininterval=1.0,
            )
            epoch_iter = epoch_pbar

        self._epoch_timing: dict[str, float | None] = {}

        try:
            for epoch in epoch_iter:
                self.current_epoch = epoch
                self._trigger_callbacks("on_epoch_start")

                if self.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(self.device)

                steps_before = self.global_step
                epoch_t0 = time.perf_counter()
                train_t0 = time.perf_counter()
                self._train_epoch()
                self._check_epoch_loss_finite()
                train_seconds = time.perf_counter() - train_t0
                steps_in_epoch = self.global_step - steps_before
                val_metrics = self._validate()
                epoch_wall_seconds = time.perf_counter() - epoch_t0

                self._epoch_timing = self._epoch_timing_dict(
                    epoch_wall_seconds=epoch_wall_seconds,
                    train_seconds=train_seconds,
                    steps_in_epoch=steps_in_epoch,
                )

                self._trigger_callbacks("on_epoch_end", val_metrics=val_metrics)

                if self.scheduler:
                    self.scheduler.step()

                if epoch_pbar is not None:
                    postfix: dict[str, str | int] = {"step": self.global_step}
                    if "loss_total" in val_metrics:
                        postfix["val_loss"] = f"{val_metrics['loss_total']:.4f}"
                    if "loss_action" in val_metrics:
                        postfix["val_action"] = f"{val_metrics['loss_action']:.4f}"
                    epoch_pbar.set_postfix(postfix)

                if self.should_stop:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break
        finally:
            if epoch_pbar is not None:
                epoch_pbar.close()

        self._trigger_callbacks("on_train_end")
        logger.info("Training complete.")

    def _train_epoch(self) -> None:
        self.model.train()
        self._reset_train_pool()
        self._epoch_loss_finite = None

        use_pbar = self.train_cfg.get("rich_progressbar", False)

        if use_pbar:
            try:
                from tqdm.rich import tqdm
            except ImportError:
                from tqdm import tqdm

            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {self.current_epoch}/{self.epochs} [Train]",
                leave=False,
            )
            iterable = pbar
        else:
            pbar = None
            iterable = self.train_loader

        for batch_idx, batch in enumerate(iterable):
            batch = self._move_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)

            # Subclasses implement the specific loss logic. The forward pass
            # is separated so the same outputs feed the loss, the per-batch
            # persistence hook, and subclass train-phase accumulators.
            out = self._forward(batch)
            loss, metrics = self._compute_loss(batch, out=out)
            n = self._batch_sample_count(batch)

            # Fail fast on non-finite losses — without a per-step host sync.
            # Converting a device tensor to bool (`if not torch.isfinite(loss)`)
            # synchronizes every step; instead the per-batch finiteness is
            # AND-reduced on-device here and materialized exactly once per
            # epoch in _check_epoch_loss_finite(), BEFORE validation, curve
            # persistence, and the checkpoint callback run — so a corrupted
            # epoch can never produce artifacts. Non-finite *gradients* are
            # still caught per step (clip_grad_norm_'s error_if_nonfinite,
            # or the explicit guard below when clipping is disabled), i.e.
            # before the optimizer applies them.
            finite = torch.isfinite(loss.detach())
            self._epoch_loss_finite = (
                finite
                if self._epoch_loss_finite is None
                else self._epoch_loss_finite & finite
            )

            loss.backward()

            if self.grad_clip_norm > 0:
                # clip_grad_norm_ computes one aggregate norm and can perform
                # the finite-value check in the same pass. Scanning every
                # parameter tensor separately creates many synchronization
                # points on CUDA.
                try:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip_norm,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as exc:
                    raise FloatingPointError(
                        f"Non-finite gradient at global step "
                        f"{self.global_step + 1}, epoch {self.current_epoch}. "
                        "Aborting to avoid corrupting the run."
                    ) from exc
            else:
                # Keep the explicit guard when clipping is disabled; there is
                # no aggregate norm call in that configuration. A single fused
                # check (one host sync) replaces a per-parameter loop (~one
                # sync per parameter); the offending parameter is named only
                # on the failure path.
                grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                if grads and not bool(
                    torch.stack([torch.isfinite(g).all() for g in grads]).all().item()
                ):
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and not bool(
                            torch.isfinite(param.grad).all().item()
                        ):
                            raise FloatingPointError(
                                f"Non-finite gradient in '{name}' at global step "
                                f"{self.global_step + 1}, epoch {self.current_epoch}. "
                                "Aborting to avoid corrupting the run."
                            )
                    raise FloatingPointError(
                        f"Non-finite gradient at global step "
                        f"{self.global_step + 1}, epoch {self.current_epoch}. "
                        "Aborting to avoid corrupting the run."
                    )

            self.optimizer.step()
            self.global_step += 1

            # Every batch feeds the persistence layer and subclass train-phase
            # accumulators so the persisted epoch means/accuracies are exact;
            # the metrics stay on-device to avoid per-batch syncs.
            self._on_train_batch_processed(batch, out, metrics, n)
            self._trigger_callbacks(
                "on_train_batch",
                batch=batch,
                out=out,
                metrics=metrics,
                n=n,
                step=self.global_step,
            )

            should_log = self.global_step % self.log_every_n_steps == 0
            if should_log:
                self._trigger_callbacks(
                    "on_train_step",
                    step=self.global_step,
                    metrics={k: self._metric_to_float(v) for k, v in metrics.items()},
                )

            if pbar is not None:
                postfix = {k: f"{self._metric_to_float(v):.4f}" for k, v in metrics.items()}
                postfix["loss"] = f"{loss.item():.4f}"
                pbar.set_postfix(postfix)

    def _check_epoch_loss_finite(self) -> None:
        """Abort the run if any training loss this epoch was non-finite.

        The per-batch finiteness flag is accumulated on-device (no host sync
        per step) and materialized exactly once per epoch here — before
        validation, curve persistence, and the checkpoint callback, so a
        corrupted epoch never produces artifacts (no checkpoint, no curve
        row, no ledger completion). With gradient clipping enabled,
        ``clip_grad_norm_(error_if_nonfinite=True)`` additionally aborts per
        step on non-finite gradients, before the optimizer applies them.
        """
        flag = self._epoch_loss_finite
        if flag is None:
            return
        if not bool(flag.item()):
            raise FloatingPointError(
                f"Non-finite training loss during epoch {self.current_epoch} "
                f"(global steps up to {self.global_step}). Aborting before "
                "any epoch-end artifact is written."
            )

    def _batch_sample_count(self, batch: dict[str, torch.Tensor]) -> int:
        """Number of valid samples a batch contributes to the losses.

        Uses the padding mask when present (``drop_last=False`` validation
        batches can be short), otherwise the batch size.
        """
        mask = batch.get("padding_mask")
        if mask is not None:
            return int(mask.sum().item())
        return batch["action"].shape[0]

    def _reset_train_pool(self) -> None:
        """Hook: subclass training accumulators reset per epoch."""

    def _reset_validation_pool(self) -> None:
        """Hook: subclass validation accumulators reset per epoch."""

    def _collect_validation_pool(
        self,
        batch: dict[str, torch.Tensor],
        out: ModelOutput,
        metrics: dict[str, MetricValue],
    ) -> None:
        """Hook: accumulate validation scalars for one batch."""

    def _finalize_validation_pool(self, agg_metrics: dict[str, float]) -> None:
        """Hook: fold accumulated validation scalars into the epoch metrics."""

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_samples = 0
        self._reset_validation_pool()

        # Loss metrics are accumulated on-device in float64 — the exact
        # arithmetic of the former Python-float accumulation (float32 ->
        # float64 is exact, adds and the final division keep their order) —
        # and materialized ONCE at the epoch boundary. The old per-batch
        # `.item()` cost one host/device synchronization per tensor metric
        # per batch (~340/epoch on the big tasks' validation sets).
        agg_sums: dict[str, torch.Tensor] = {}

        use_pbar = self.train_cfg.get("rich_progressbar", False)

        if use_pbar:
            try:
                from tqdm.rich import tqdm
            except ImportError:
                from tqdm import tqdm

            pbar = tqdm(
                self.val_loader, desc=f"Epoch {self.current_epoch}/{self.epochs} [Val]", leave=False
            )
            iterable = pbar
        else:
            pbar = None
            iterable = self.val_loader

        for batch in iterable:
            batch = self._move_batch(batch)

            # Reuse the forward outputs for the losses and the validation
            # accumulators (single forward pass per batch).
            out = self._forward(batch)
            _, metrics = self._compute_loss(batch, out=out)

            if pbar is not None:
                postfix = {k: f"{self._metric_to_float(v):.4f}" for k, v in metrics.items()}
                pbar.set_postfix(postfix)

            # Weight per-batch metrics by the number of valid samples so the
            # final (short) validation batch does not get the same weight as a
            # full batch (validation uses drop_last=False).
            n = self._batch_sample_count(batch)
            if n == 0:
                continue
            self._collect_validation_pool(batch, out, metrics)
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

        if total_samples > 0:
            agg_metrics = {k: float((v / total_samples).item()) for k, v in agg_sums.items()}
            self._finalize_validation_pool(agg_metrics)
            return agg_metrics
        return {}

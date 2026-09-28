"""Phase-aware collator for DataLoader."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class PhaseAwareCollator:
    """Custom ``collate_fn`` for DataLoader.

    For single-step samples this is equivalent to default collation. For
    temporal samples, pads to the maximum length in the batch and returns a
    boolean validity mask (``True`` means the timestep is real data).
    """

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Tensor]:
        # Detect if samples have a time dimension
        first = batch[0]
        has_time_dim = first["state"].ndim == 2  # (T, S)

        if not has_time_dim:
            # Single-step: simple stack
            out = {
                "state": torch.stack([b["state"] for b in batch]),  # (B, S)
                "action": torch.stack([b["action"] for b in batch]),  # (B, A)
                "phase": torch.stack([b["phase"] for b in batch]),  # (B,)
                "task_id": torch.stack(
                    [torch.as_tensor(b["task_id"], dtype=torch.long) for b in batch]
                ),  # (B,)
            }
            if "trajectory_id" in first:
                out["trajectory_id"] = torch.stack(
                    [torch.as_tensor(b["trajectory_id"], dtype=torch.long) for b in batch]
                )
            if "trajectory_position" in first:
                out["trajectory_position"] = torch.stack(
                    [torch.as_tensor(b["trajectory_position"], dtype=torch.long) for b in batch]
                )
            # Preserve alternate regime labels and clean ground truth when present
            for extra_key in ("phase_dynamic", "phase_rule", "phase_topo", "phase_gt_clean"):
                if extra_key in first:
                    out[extra_key] = torch.stack(
                        [torch.as_tensor(b[extra_key], dtype=torch.long) for b in batch]
                    )
            return out

        # Multi-step: pad to max length
        lengths = [b["state"].shape[0] for b in batch]
        max_T = max(lengths)
        B = len(batch)
        S = first["state"].shape[-1]
        A = first["action"].shape[-1]

        state_padded = torch.zeros(B, max_T, S)
        action_padded = torch.zeros(B, max_T, A)
        phase_padded = torch.zeros(B, max_T, dtype=torch.long)
        mask = torch.zeros(B, max_T, dtype=torch.bool)

        for i, (sample, T) in enumerate(zip(batch, lengths)):
            state_padded[i, :T] = sample["state"]
            action_padded[i, :T] = sample["action"]
            phase_padded[i, :T] = sample["phase"]
            mask[i, :T] = True

        out = {
            "state": state_padded,  # (B, max_T, S)
            "action": action_padded,  # (B, max_T, A)
            "phase": phase_padded,  # (B, max_T)
            "task_id": torch.stack(
                [torch.as_tensor(b["task_id"], dtype=torch.long) for b in batch]
            ),  # (B,)
            "padding_mask": mask,  # (B, max_T)
        }
        if "trajectory_id" in first:
            out["trajectory_id"] = torch.stack(
                [torch.as_tensor(b["trajectory_id"], dtype=torch.long) for b in batch]
            )
        if "trajectory_position" in first:
            out["trajectory_position"] = torch.stack(
                [torch.as_tensor(b["trajectory_position"], dtype=torch.long) for b in batch]
            )
        for extra_key in ("phase_dynamic", "phase_rule", "phase_topo", "phase_gt_clean"):
            if extra_key in first:
                # Pad like phase (long)
                extra_padded = torch.zeros(B, max_T, dtype=torch.long)
                for i, (sample, T) in enumerate(zip(batch, lengths)):
                    extra_padded[i, :T] = sample[extra_key]
                out[extra_key] = extra_padded
        return out

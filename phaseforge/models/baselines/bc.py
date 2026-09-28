"""Behavior Cloning baseline."""

from __future__ import annotations

from torch import Tensor

from phaseforge.models.base import BaseManipulationModel, ModelOutput
from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.encoder import StateEncoder


class BehaviorCloningModel(BaseManipulationModel):
    """Simple Behavior Cloning baseline.

    Encoder -> ActionHead. No phase head, no routing.
    """

    def __init__(
        self,
        encoder: StateEncoder,
        action_head: ActionHead,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.action_head = action_head

    def forward(self, batch: dict[str, Tensor]) -> ModelOutput:
        state = batch["state"]
        latent = self.encoder(state)
        action_pred = self.action_head(latent)

        return ModelOutput(
            action_pred=action_pred,
            phase_logits=None,
            routing_weights=None,
            expert_indices=None,
            gate_logits=None,
            latent=latent,
        )

    def get_action(self, state: Tensor) -> Tensor:
        latent = self.encoder(state)
        return self.action_head(latent)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

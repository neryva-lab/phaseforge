"""Factory registry: build models, data pipelines, and trainers via Hydra instantiate."""

from __future__ import annotations

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def build_model(cfg: DictConfig):
    """Instantiate the model from the models config subtree.

    The model _target_ must resolve to a subclass of BaseManipulationModel.

    .. note::
       Config keys used for run-metadata (``name``) are stripped before
       passing to the target constructor, because Hydra's ``instantiate``
       passes *all* config keys as keyword arguments.
    """
    from phaseforge.models.base import BaseManipulationModel

    model_cfg = OmegaConf.to_container(cfg.models, resolve=True)
    model_cfg.pop("name", None)
    # Training-only metadata carried in the model config (per-model override
    # for `train.freeze_encoder` and `train.encoder_lr_scale`); not model constructor arguments.
    model_cfg.pop("freeze_encoder", None)
    model_cfg.pop("encoder_lr_scale", None)
    model = instantiate(model_cfg)
    if not isinstance(model, BaseManipulationModel):
        raise TypeError(
            f"Model {type(model).__name__} does not implement BaseManipulationModel. "
            "Check the _target_ in your model config."
        )
    return model


def build_data_pipeline(cfg: DictConfig):
    """Instantiate and return the data pipeline state machine."""
    from phaseforge.data.ingestion.state_machine import DataPipelineStateMachine

    return DataPipelineStateMachine(cfg=cfg)


def build_trainer(cfg: DictConfig, model, train_loader, val_loader):
    """Instantiate the correct trainer based on cfg.train.stage."""
    stage = cfg.train.get("stage", 1)
    if stage == 1:
        from phaseforge.trains.loops.stage1_loop import Stage1Trainer

        return Stage1Trainer(cfg=cfg, model=model, train_loader=train_loader, val_loader=val_loader)
    elif stage == 2:
        from phaseforge.trains.loops.stage2_loop import Stage2Trainer

        return Stage2Trainer(cfg=cfg, model=model, train_loader=train_loader, val_loader=val_loader)
    else:
        raise ValueError(f"Unknown training stage: {stage}. Must be 1 or 2.")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Union, Dict
from omegaconf import OmegaConf
import torch

from config.cbm_trainer_config import CBMTrainerConfig
from config.dataset_config import DatasetConfig
from config.standard_trainer_config import StandardTrainerConfig
from models.architectures import CBMSequentialEfficientNetFCN
from models.architectures.cbm_ste_efficientNetFCN import CBMSTEEfficientNetFCN
from models.registries import (
    CRITERIONS_REGISTRY,
    OPTIMIZERS_REGISTRY,
    SCHEDULERS_REGISTRY,
)
from data_access.registry import DATASET_FACTORY_REGISTRY
from models.trainer.cbm_STE_trainer import CBMSTETrainer
from training_loop import set_reproducibility_seed


def load_config(
    config_path,
    configClass=CBMTrainerConfig,
    overrides: Union[List[str], Dict[str, Any], None] = None,
):
    """
    Load the configuration from a YAML file.
    """
    # Load YAML
    cfg_yaml = OmegaConf.load(config_path)
    cfg_structured = OmegaConf.structured(configClass)
    cfg = OmegaConf.merge(cfg_structured, cfg_yaml)
    cfg = OmegaConf.to_object(cfg)

    if overrides is not None:
        overrides_cfg = None
        if isinstance(overrides, list):
            # Create a DictConfig from a dot-list (e.g., ["model.hidden_size=512", "lr=0.001"])
            overrides_cfg = OmegaConf.from_dotlist(overrides)
        elif isinstance(overrides, dict):
            # Create a DictConfig from a standard dictionary
            overrides_cfg = OmegaConf.create(overrides)

        if overrides_cfg is not None:
            cfg = OmegaConf.merge(cfg, overrides_cfg)
            cfg = OmegaConf.to_object(cfg)
    print(cfg.device)

    cfg.resolve()
    return cfg


if __name__ == "__main__":

    config = load_config(
        Path("../files/configs/GTSRB_CBM_config_STE.yaml")
    )

    set_reproducibility_seed(seed=config.seed)

    # Dataset
    dataset_factory = DATASET_FACTORY_REGISTRY["gtsrb"](
        seed=config.seed, config=config.dataset
    ).set_dataloaders()

    train_loader = dataset_factory.train_dataloader
    val_loader = dataset_factory.val_dataloader
    test_loader = dataset_factory.test_dataloader

    # # Model
    model = CBMSTEEfficientNetFCN(config)

    
    predictor_trainer = CBMSTETrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    predictor_trainer.concept_lambda = 0.5

    predictor_trainer.train()

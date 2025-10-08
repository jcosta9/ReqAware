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



def load_config(config_path, configClass = CBMTrainerConfig, overrides: Union[List[str], Dict[str, Any], None] = None):
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


@dataclass
class STECBMConfig:
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_freq = 30
    output_dir = "experiments"
    dataset: dict = field(
        default_factory=lambda: {
            "name": "gtsrb",
            "n_labels": 43,
            "n_concepts": 43,
            "batch_size": 128,
            "num_workers": 4,
            "shuffle_dataset": True,
            "pin_memory": True,
            "data_path": Path("../../data/raw/GTSRB/converted"),
            "val_split": 0.2,
            "concepts_file": Path("../../data/raw/GTSRB/concepts/concepts_per_class.csv")
        }
    )
    trainer: dict = field(
        default_factory=lambda: {
            "log_dir": "logs",
            "checkpoint_dir": "models",
            "lr": 0.1,
            "concept_lambda": 1.0,  # Hyperparameter for concept regularization
            "epochs": 50,
            "lr_step": 10,
            "lr_gamma": 0.5,
            "early_stopping_patience": 10,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "freeze": False,
            "pretrained_weights": None,
            "criterion": CRITERIONS_REGISTRY["cross_entropy"],
            "concept_criterion": CRITERIONS_REGISTRY["bce_with_logits"],
            "optimizer": OPTIMIZERS_REGISTRY["adamw"],
            "scheduler": SCHEDULERS_REGISTRY["reduce_on_plateau"],
            "scheduler_params": {
                "mode": "min",
                "factor": 0.5,
                "patience": 5,
                "min_lr": 1e-5,
            },
        }
    )

if __name__ == "__main__":
    ste_config = STECBMConfig()

    overrides = [
        "device_no=1",
        "dataset.name=gtsrb",
        "dataset.n_labels=43",
        "dataset.data_path=../../data/raw/GTSRB/converted",
        "dataset.n_concepts=43",
        "dataset.concepts_file=../../data/raw/GTSRB/concepts/concepts_per_class.csv"
        ]


    dataset_config = load_config(
        Path("../files/configs/GTSRB_CBM_config_best_trial.yaml"),
        overrides=overrides
        )
    
    print(dataset_config.device)

    # Dataset
    dataset_factory = DATASET_FACTORY_REGISTRY['gtsrb'](
        seed=ste_config.seed, config=dataset_config.dataset
    ).set_dataloaders()

    train_loader = dataset_factory.train_dataloader
    val_loader = dataset_factory.val_dataloader
    test_loader = dataset_factory.test_dataloader

    # # Model
    model = CBMSTEEfficientNetFCN(dataset_config)

    criterion = ste_config.trainer['criterion']()
    optimizer = ste_config.trainer['optimizer'](
        model.parameters(),
        lr=ste_config.trainer['lr'],
        weight_decay=ste_config.trainer['weight_decay'],
    )
    scheduler = ste_config.trainer['scheduler'](
        optimizer, **ste_config.trainer['scheduler_params']
    )

    predictor_trainer = CBMSTETrainer(
            config=dataset_config.label_predictor,
            model=model,
            experiment_id=dataset_config.experiment_id,
            concept_predictor=model.concept_predictor,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=dataset_config.device,
        )
    
    predictor_trainer.concept_criterion = ste_config.trainer['concept_criterion']()
    predictor_trainer.concept_lambda = ste_config.trainer['concept_lambda']

    predictor_trainer.train()

    
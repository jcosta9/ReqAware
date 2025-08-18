from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from pathlib import Path
import torch
from omegaconf import MISSING

from data_access.registry import DATASET_FACTORY_REGISTRY
from models.registries import CRITERIONS_REGISTRY, OPTIMIZERS_REGISTRY, SCHEDULERS_REGISTRY

@dataclass
class DatasetConfig:
    dataset: str = "cifar10"
    n_labels: int = 10
    batch_size: int = 128
    num_workers: int = 4
    shuffle_dataset: bool = True
    pin_memory: bool = True
    data_path: Path = MISSING
    val_split: float = 0.8
    concepts_file: Optional[Path] = None

    def resolve(self):
        if self.dataset in DATASET_FACTORY_REGISTRY:
            self.dataset = DATASET_FACTORY_REGISTRY[self.dataset]
        else:
            raise ValueError(f"Unknown dataset {self.dataset}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path {self.data_path} does not exist")

        if self.concepts_file and not self.concepts_file.exists():
            raise FileNotFoundError(f"Concepts file {self.concepts_file} does not exist")

@dataclass
class TrainingConfig:
    lr: float = 0.001
    epochs: int = 10
    lr_step: int = 10
    lr_gamma: float = 0.5
    early_stopping_patience: int = 15
    momentum: float = 0.9
    weight_decay: float = 5e-4
    checkpoint_dir: Path = MISSING
    criterion: str = "cross_entropy"
    optimizer: str = "sgd"
    scheduler: str = "cosine_annealing"
    scheduler_params: Dict[str, Any] = field(default_factory=dict)

    def resolve(self):
        if self.criterion in CRITERIONS_REGISTRY:
            self.criterion = CRITERIONS_REGISTRY[self.criterion]()
        else:
            raise ValueError(f"Unknown criterion {self.criterion}")

        if self.optimizer in OPTIMIZERS_REGISTRY:
            self.optimizer = OPTIMIZERS_REGISTRY[self.optimizer]
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")
        
        if self.scheduler in SCHEDULERS_REGISTRY:
            self.scheduler = SCHEDULERS_REGISTRY[self.scheduler]
        else:
            raise ValueError(f"Unknown scheduler {self.scheduler}")

        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Data path {self.checkpoint_dir} does not exist")
        

    
@dataclass
class StandardTrainerConfig:
    seed: float = 42
    print_freq: int = 30
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: Path = "runs/default"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def resolve(self):
        self.dataset.resolve()
        self.training.resolve()

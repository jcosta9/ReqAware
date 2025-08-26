from dataclasses import dataclass, field
from typing import Dict, Any

from pathlib import Path
import torch
from omegaconf import MISSING

from config.dataset_config import DatasetConfig
from config.training_config import TrainingConfig
from models.registries import (
    CRITERIONS_REGISTRY,
    OPTIMIZERS_REGISTRY,
    SCHEDULERS_REGISTRY,
)


@dataclass
class StandardTrainerConfig:
    seed: int = 42
    print_freq: int = 30
    device: str = "cuda"
    device_no: int = 0
    log_dir: Path = "runs/default"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def resolve(self):
        self.device = f"cuda:{self.device_no}" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        self.dataset.resolve()
        self.training.resolve()

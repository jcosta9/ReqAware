from dataclasses import dataclass
from typing import Optional

from pathlib import Path
import torch
from omegaconf import MISSING

from data_access.registry import DATASET_FACTORY_REGISTRY

@dataclass
class DatasetConfig:
    name: str = "cifar10"
    n_labels: int = 10
    batch_size: int = 128
    num_workers: int = 4
    shuffle_dataset: bool = True
    pin_memory: bool = True
    data_path: Path = MISSING
    val_split: float = 0.8
    concepts_file: Optional[Path] = None

    def resolve(self):
        if self.name in DATASET_FACTORY_REGISTRY:
            self.factory = DATASET_FACTORY_REGISTRY[self.name]
        else:
            raise ValueError(f"Unknown dataset {self.dataset}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path {self.data_path} does not exist")

        if self.concepts_file and not self.concepts_file.exists():
            raise FileNotFoundError(
                f"Concepts file {self.concepts_file} does not exist"
            )
from dataclasses import dataclass
from typing import Optional

from pathlib import Path
from omegaconf import MISSING
import pandas as pd

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

    def resolve(self):
        if self.name in DATASET_FACTORY_REGISTRY:
            self.factory = DATASET_FACTORY_REGISTRY[self.name]
        else:
            raise ValueError(f"Unknown dataset {self.dataset}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path {self.data_path} does not exist")

        if self.n_labels <= 0:
            raise ValueError("Number of labels must be a positive integer")

        if self.val_split <= 0 or self.val_split >= 1:
            raise ValueError("Validation split must be between 0 and 1")

        self.extra_resolve()

    def extra_resolve(self):
        """
        Placeholder for any additional resolution logic that might be needed.
        This can be overridden in subclasses if specific datasets require extra steps.
        """
        pass


@dataclass
class ConceptDatasetConfig(DatasetConfig):
    n_concepts: int = 43  # Example for GTSRB, adjust as needed
    concepts_file: Optional[Path] = None

    def get_concept_map(self):
        if hasattr(self, "concept_map"):
            return self.concept_map
        else:
            raise AttributeError(
                "Concept map not loaded. Ensure concepts_file is provided and resolved."
            )

    @classmethod
    def from_parent(cls, parent: DatasetConfig, **kwargs):
        return cls(**parent.__dict__, **kwargs)

    def extra_resolve(self):
        super().extra_resolve()

        if self.concepts_file and not self.concepts_file.exists():
            raise FileNotFoundError(
                f"Concepts file {self.concepts_file} does not exist"
            )

        if self.n_concepts <= 0:
            raise ValueError("Number of concepts must be a positive integer")

        if self.concepts_file:
            self.concepts = pd.read_csv(self.concepts_file)
            self.concept_map = {
                col: idx for idx, col in enumerate(self.concepts.columns[2:])
            }  # Skip non-concept columns if any

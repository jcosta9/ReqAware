from dataclasses import dataclass, field
from typing import Dict, Any

from pathlib import Path
import torch
from omegaconf import MISSING

from config.standard_trainer_config import StandardTrainerConfig
from config.dataset_config import ConceptDatasetConfig
from config.training_config import ConceptTrainingConfig


@dataclass
class CBMTrainerConfig():
    seed: int = 42
    print_freq: int = 30
    device: str = "cuda"
    device_no: int = 0
    log_dir: Path = "runs/default"
    dataset: ConceptDatasetConfig = field(default_factory=ConceptDatasetConfig)
    concept_predictor: ConceptTrainingConfig = field(default_factory=ConceptTrainingConfig)
    label_predictor: ConceptTrainingConfig = field(default_factory=ConceptTrainingConfig)

    def resolve(self):
        self.device = f"cuda:{self.device_no}" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        self.dataset.resolve()
        self.concept_predictor.resolve()
        self.label_predictor.resolve()
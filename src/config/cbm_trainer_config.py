from dataclasses import dataclass, field
from typing import Dict, Any

from pathlib import Path
import torch
from omegaconf import MISSING

from config.standard_trainer_config import StandardTrainerConfig
from config.dataset_config import ConceptDatasetConfig
from config.training_config import ConceptTrainingConfig


@dataclass
class CBMTrainerConfig(StandardTrainerConfig):
    dataset: ConceptDatasetConfig = field(default_factory=ConceptDatasetConfig)
    training: ConceptTrainingConfig = field(default_factory=ConceptTrainingConfig)

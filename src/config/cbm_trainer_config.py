from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any

from pathlib import Path
import torch
from omegaconf import MISSING

from config.standard_trainer_config import StandardTrainerConfig
from config.dataset_config import ConceptDatasetConfig
from config.training_config import ConceptTrainingConfig


@dataclass
class CBMTrainerConfig:
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed: int = 42
    print_freq: int = 30
    device: str = "cuda"
    device_no: int = 0
    output_dir: Path = "experiments"
    dataset: ConceptDatasetConfig = field(default_factory=ConceptDatasetConfig)
    concept_predictor: ConceptTrainingConfig = field(
        default_factory=ConceptTrainingConfig
    )
    label_predictor: ConceptTrainingConfig = field(
        default_factory=ConceptTrainingConfig
    )

    def resolve(self):
        self.device = (
            f"cuda:{self.device_no}"
            if self.device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )

        try:
            self.output_dir = Path(self.output_dir) / self.experiment_id
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Directory '{self.output_dir}' created successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")
            

        self.dataset.resolve()
        self.concept_predictor.resolve(output_dir=self.output_dir, experiment_id=self.experiment_id)
        self.label_predictor.resolve(output_dir=self.output_dir, experiment_id=self.experiment_id)

        self.concept_predictor.fuzzy_loss.concept_map = self.dataset.get_concept_map()

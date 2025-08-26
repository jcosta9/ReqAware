from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from pathlib import Path
from omegaconf import MISSING

from models.registries import (
    CRITERIONS_REGISTRY,
    OPTIMIZERS_REGISTRY,
    SCHEDULERS_REGISTRY,
    CUSTOM_RULES_REGISTRY,
)


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
            self.criterion = CRITERIONS_REGISTRY[self.criterion]
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

        self.extra_resolve()

    def extra_resolve(self):
        """
        Placeholder for any additional resolution logic that might be needed.
        This can be overridden in subclasses if specific training require extra steps.
        """
        pass


@dataclass
class FuzzyLossConfig:
    use_fuzzy_loss: bool = False
    fuzzy_lambda: float = 0.5
    custom_rules: Optional[list] = None

    def resolve(self, parent_config):
        if not (0 <= self.fuzzy_lambda <= 1):
            raise ValueError("fuzzy_lambda must be between 0 and 1")
        if self.use_fuzzy_loss and (
            self.custom_rules is None or len(self.custom_rules) == 0
        ):
            raise ValueError(
                "custom_rules must be provided when use_fuzzy_loss is True"
            )
        if not self.use_fuzzy_loss and (
            self.custom_rules is not None and len(self.custom_rules) > 0
        ):
            raise ValueError(
                "custom_rules should be None or empty when use_fuzzy_loss is False"
            )

        resolved_rules = []
        for rule in self.custom_rules or []:
            if rule not in CUSTOM_RULES_REGISTRY:
                raise ValueError(
                    f"Unknown custom rule {rule} in fuzzy_loss configuration"
                )
            resolved_rules.append(CUSTOM_RULES_REGISTRY[rule])
        self.custom_rules = resolved_rules


@dataclass
class ConceptTrainingConfig(TrainingConfig):
    dropout: float = 0.5
    freeze_concept_predictor: bool = False
    concept_predictor_file: Optional[Path] = None
    fuzzy_loss: FuzzyLossConfig = field(default_factory=FuzzyLossConfig)

    def extra_resolve(self):
        if self.concept_predictor_file and not self.concept_predictor_file.exists():
            raise FileNotFoundError(
                f"Concept predictor file {self.concept_predictor_file} does not exist"
            )
        if not (0 <= self.dropout <= 1):
            raise ValueError("Dropout must be between 0 and 1")

        self.fuzzy_loss.resolve(self)

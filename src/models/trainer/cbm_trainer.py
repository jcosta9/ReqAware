import torch

from models.trainer.cbm_concept_predictor_trainer import CBMConceptPredictorTrainer
from models.trainer.cbm_label_predictor_trainer import CBMLabelPredictorTrainer
from models.trainer.standard_trainer import StandardTrainer


class CBMTrainer:
    def __init__(self, config, model, train_loader, val_loader, test_loader):
        """
        Initialize the CBMTrainer with model, data loaders, and training configuration.

        Parameters:
            config (Config): Configuration object containing training parameters and paths.
            model (torch.nn.Module): CBM Model containing a concept_predictor and a label_predictor.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            test_loader (DataLoader): DataLoader for the test dataset.
        """
        self.model = model
        self.config = config
        self.device = config.device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        if not hasattr(self.model, "concept_predictor"):
            raise ValueError("CBM Model must have a 'concept_predictor' attribute.")
        if not hasattr(self.model, "label_predictor"):
            raise ValueError("CBM Model must have a 'label_predictor' attribute.")

        self.concept_predictor = model.concept_predictor.to(self.device)
        self.label_predictor = model.label_predictor.to(self.device)

        self.concept_predictor_trainer = CBMConceptPredictorTrainer(
            config=config.concept_predictor,
            model=self.concept_predictor,
            experiment_id=config.experiment_id,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=config.device,
        )

        self.label_predictor_trainer = CBMLabelPredictorTrainer(
            config=config.label_predictor,
            model=self.label_predictor,
            experiment_id=config.experiment_id,
            concept_predictor=self.concept_predictor,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=config.device,
        )

    def train(self):
        """
        Train the concept and label predictors using the training data.
        """
        # TODO: check if concept predictor should be trained first
        # Train concept predictor
        if not self.config.concept_predictor.freeze:
            print("\n#### Training concept predictor...")
            self.concept_predictor = self.concept_predictor_trainer.train()

        if not self.config.label_predictor.freeze:
            print("\n\n#### Training label predictor...")
            self.label_predictor = self.label_predictor_trainer.train()

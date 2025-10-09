"""
This module provides code for loading the GTSRB dataset.

Author: Joao Paulo Costa de Araujo
Date: 2025-04-29
"""

import logging

import torch
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms

from data_access.data_factory import DatasetFactory
from data_access.concepts.ConceptAwareDataSet import ConceptAwareDataset


GTSRB_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

GTSRB_basic_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)


class GTSRBFactory(DatasetFactory):
    """Factory for generating GTSRB data."""

    def __init__(self, config, seed=42):
        """Builder method for GTSRBFactory.

        Args:
            config (Defaults): Configuration file with parameters.
        """
        super().__init__(config=config, seed=seed)

    def load_datasets(
        self,
        train_transform=GTSRB_transform,
        test_transform=GTSRB_basic_transform,
    ):
        """Load GTSRB dataset and splits the training set into training and validation subsets.

        Args:
            train_transform (callable, optional): Data transformations for training set.
                                                                Defaults to GTSRB_train_transform.
            test_transform (callable, optional): Data transformations for test set.
                                                                Defaults to GTSRB_train_transform.
        """
        logging.info(f"[DATA ACCESS] Loading GTSRB training dataset")
        full_train_dataset = ConceptAwareDataset(
            root_dir=self.config.data_path / "training",
            concepts_file=self.config.concepts_file if hasattr(self.config, "concepts_file") else None,
            transform=train_transform,
        )

        logging.info(f"[DATA ACCESS] Splitting training and validation datasets")
        val_size = int(len(full_train_dataset) * self.config.val_split)
        train_size = len(full_train_dataset) - val_size

        train_indices, val_indices = torch.utils.data.random_split(
            range(len(full_train_dataset)),
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices.indices)

        self.val_dataset = ConceptAwareDataset(
            root_dir=self.config.data_path / "training",
            concepts_file=self.config.concepts_file if hasattr(self.config, "concepts_file") else None,
            transform=test_transform,
        )
        self.val_dataset = torch.utils.data.Subset(self.val_dataset, val_indices.indices)

        logging.info(f"[DATA ACCESS] Loading GTSRB test dataset")
        self.test_dataset = ConceptAwareDataset(
            root_dir=self.config.data_path / "test",
            concepts_file=self.config.concepts_file if hasattr(self.config, "concepts_file") else None,
            transform=test_transform,
        )

        logging.info(
            f"[DATA ACCESS] Training dataset length: {len(self.train_dataset)}"
        )
        logging.info(
            f"[DATA ACCESS] Validation dataset length: {len(self.val_dataset)}"
        )
        logging.info(f"[DATA ACCESS] Test dataset length: {len(self.test_dataset)}")

        self.datasets_loaded = True

        return self

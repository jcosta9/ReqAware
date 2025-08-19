"""
This module provides code for loading the CIFAR10 dataset.

Author: Joao Paulo Costa de Araujo
Date: 2025-04-29
"""

import logging
import torch
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms

from data_access.data_factory import DatasetFactory


CIFAR10_train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

CIFAR10_test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


class CIFAR10Factory(DatasetFactory):
    """Factory for generating CIFAR10 data."""

    def __init__(self, config):
        """Builder method for CIFAR10Factory.

        Args:
            config (Defaults): Configuration file with parameters.
        """
        super().__init__(config)

    def load_datasets(
        self,
        train_transform=CIFAR10_train_transform,
        test_transform=CIFAR10_test_transform,
    ):
        """Load CIFAR-10 dataset and splits the training set into training and validation subsets.

        Args:
            train_transform (callable, optional): Data transformations for training set.
                                                                Defaults to CIFAR10_train_transform.
            test_transform (callable, optional): Data transformations for test set.
                                                                Defaults to CIFAR10_test_transform.
        """
        logging.info(f"[DATA ACCESS] Loading CIFAR-10 training dataset")
        full_train_dataset = torchvision.datasets.CIFAR10(
            root=self.config.data_path / self.config.name,
            train=True,
            download=True,
            transform=train_transform,
        )

        logging.info(f"[DATA ACCESS] Splitting training and validation datasets")
        val_size = int(len(full_train_dataset) * self.config.val_split)
        train_size = len(full_train_dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

        logging.info(f"[DATA ACCESS] Loading CIFAR-10 test dataset")
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.config.data_path / self.config.name,
            train=False,
            download=True,
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

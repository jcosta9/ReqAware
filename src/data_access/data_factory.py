"""Module Responsible for defining an abstract factory for datasets."""

from abc import ABC, abstractmethod
import logging
import torch
import numpy as np
import random


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DatasetFactory(ABC):
    """abstract factory for datasets."""

    def __init__(self, config):
        """Construct the DatasetFactory.

        Args:
            config: configuration file
        """
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.datasets_loaded = False
        self.dataloaders_set = False

        # For concept aware datasets
        self.concepts_file = (
            config.concepts_file if hasattr(config, "concepts_file") else None
        )

    @abstractmethod
    def load_datasets(self):
        """Loads datasets for training/testing models."""
        pass

    def get_datasets(self, load_if_none=True):
        """Calls the set_datasets method to load datasets."""
        logging.info(f"[DATA ACCESS] Loading datasets")

        if not self.datasets_loaded:
            logging.info(
                f"[DATA ACCESS] Datasets not loaded yet. Calling load_datasets()"
            )

            if not load_if_none:
                logging.info(
                    f"[DATA ACCESS] load_if_none is False. Not loading datasets."
                )
                return (None, None, None)

            self.load_datasets()

        return self.train_dataset, self.val_dataset, self.test_dataset

    def _wrap_dataloader(self, dataset):
        if dataset is None:
            raise ValueError("Dataset is None")

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=self.config.dataset.shuffle_dataset,
            num_workers=self.config.dataset.num_workers,
            pin_memory=self.config.dataset.pin_memory,
            worker_init_fn=seed_worker,  # Ensures reproducibility
            generator=torch.Generator().manual_seed(
                self.config.seed
            ),  # Ensures reproducibility
        )

    def set_dataloaders(self, load_if_none=True):
        """Wrap datasets into dataloaders after get_datasets() is called."""

        logging.info(f"[DATA ACCESS] Setting dataloaders")
        if not self.datasets_loaded:
            logging.info(
                f"[DATA ACCESS] Datasets not loaded yet. Calling get_datasets()"
            )

            if not load_if_none:
                logging.info(
                    f"[DATA ACCESS] load_if_none is False. Not loading dataloaders."
                )
                return self

            self.load_datasets()

        logging.info(f"[DATA ACCESS] Wrapping datasets into dataloaders")
        if self.train_dataset:
            self.train_dataloader = self._wrap_dataloader(self.train_dataset)
        if self.val_dataset:
            self.val_dataloader = self._wrap_dataloader(self.val_dataset)
        if self.test_dataset:
            self.test_dataloader = self._wrap_dataloader(self.test_dataset)

        logging.info(f"[DATA ACCESS] Dataloaders set successfully")

        self.dataloaders_set = True

        return self

    def get_dataloaders(self, load_if_none=True):
        """Returns the dataloaders for training, validation, and testing."""
        logging.info(f"[DATA ACCESS] Getting dataloaders")
        if not self.dataloaders_set:
            logging.info(
                f"[DATA ACCESS] Dataloaders not loaded yet. Calling set_dataloaders()"
            )

            if not load_if_none:
                logging.info(
                    f"[DATA ACCESS] load_if_none is False. Not loading dataloaders."
                )
                return (None, None, None)

            self.set_dataloaders(load_if_none)

        return self.train_dataloader, self.val_dataloader, self.test_dataloader

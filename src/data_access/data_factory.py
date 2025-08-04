"""Module Responsible for defining an abstract factory for datasets."""

from abc import ABC, abstractmethod
import torch


class DatasetFactory(ABC):
    """abstract factory for datasets."""

    def __init__(self, config):
        """Construct the DatasetFactory.

        Args:
            config (_type_): configuration file
        """
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

    @abstractmethod
    def get_datasets(self):
        """Return datasets for training/testing models."""
        pass

    def _wrap_dataloader(self, dataset):
        if dataset is None:
            raise ValueError("Dataset is None")

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def get_dataloaders(self):
        """Wrap datasets into dataloaders after get_datasets() is called."""
        if self.train_dataset:
            self.train_dataloader = self._wrap_dataloader(self.train_dataset)
        if self.val_dataset:
            self.val_dataloader = self._wrap_dataloader(self.val_dataset)
        if self.test_dataset:
            self.test_dataloader = self._wrap_dataloader(self.test_dataset)
        return self

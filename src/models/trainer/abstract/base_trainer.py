"""
Define the training pipeline including model training, validation, and checkpointing.

This module includes a Trainer class that manages the training process, along with
a decorator for setting model mode (train or eval) during different phases.
"""

from abc import ABC, abstractmethod
import os
import tqdm
import logging

import numpy as np
from sklearn.metrics import classification_report

import torch
from torch.utils.tensorboard import SummaryWriter

from models.utils import EarlyStopping


# TODO: Test methods in this file
class BaseTrainer(ABC):
    """
    Manage training and validation loops, model evaluation, and checkpointing.

    Attributes:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): Dataloader for training data.
        val_loader (DataLoader): Dataloader for validation data.
        config (Config): Configuration object containing training hyperparameters.
        device (torch.device): Device on which to run training.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        writer (SummaryWriter): TensorBoard writer for logging.
        best_val_accuracy (float): Best recorded validation accuracy.
    """

    def __init__(
        self,
        config,
        model,
        train_loader,
        val_loader,
        test_loader,
        log_dir=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Initialize the Trainer with model, data loaders, and training configuration.

        Parameters:
            model (torch.nn.Module): The model to be trained.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            config (Config): Configuration object containing training parameters and paths.
        """
        self.config = config
        self.device = device
        self.log_dir = log_dir
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.criterion = self.config.criterion()
        self.optimizer = self.config.optimizer(
            self.model.parameters(),
            lr=self.config.lr,
        )
        self.scheduler = self.config.scheduler(  # TODO: Flexible scheduler
            self.optimizer, **self.config.scheduler_params
        )
        self.writer = SummaryWriter(
            log_dir=self.log_dir
        )  # TODO: separate class for logging

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self.best_val_accuracy = 0.0

        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience
        )

        return self

    def validate(self, epoch):
        """
        Run a single validation epoch.

        This method wraps `_run_epoch` with evaluation mode enabled and disables gradients.

        Parameters:
            epoch (int): Current epoch number.

        Returns:
            float: Accuracy for this validation epoch.
        """
        return self.test(dataloader=self.val_loader, epoch=epoch, mode="val")

    def save_checkpoint(self, epoch, val_accuracy):
        """
        Save the model checkpoint if validation accuracy improves.

        Parameters:
            epoch (int): Current epoch number.
            val_accuracy (float): Validation accuracy of the current epoch.
        """
        self.early_stopping(val_accuracy, self.model)

        print(val_accuracy, self.best_val_accuracy)

        if self.early_stopping.early_stop:
            print("🛑 Early stopping triggered.")  # TODO: move prints to a logger
            return

        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy  # TODO: log best_val_accuracy
            path = self.config.checkpoint_dir / "best_model.pt"  # TODO: meaningful name
            torch.save(self.model.state_dict(), path)
            print(
                f"✅ Best model saved at epoch {epoch+1} — Accuracy: {val_accuracy:.4f}"
            )

    def load_best_model(self):
        """Load the best model from disk.

        Returns:
            torch.nn.Module: trained model
        """
        # TODO: change file name and store it
        # TODO: Option for passing filename
        print("🔁 Loading best model...")
        self.model.load_state_dict(
            torch.load(self.config.checkpoint_dir / "best_model.pt")
        )
        return self.model

    def train(self):
        """
        Run the full training loop across all configured epochs.

        For each epoch, this method performs training, validation, and checkpointing.
        """
        self.on_train_start()

        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")

            self.on_epoch_start(epoch)
            self._train_epoch(epoch)
            self.on_epoch_end(epoch, None)

            val_loss, val_accuracy = self.validate(epoch=epoch)
            self.save_checkpoint(epoch, val_accuracy)
            self.scheduler.step(val_loss)

            if self.early_stopping.early_stop:
                break

        self.on_train_end()
        self.writer.close()

        print("Final test evaluation")
        self.load_best_model()
        _, test_accuracy = self.test(dataloader=self.test_loader, mode="test")
        print(f"🎯 Final Test Accuracy: {test_accuracy:.4f}")

        return self.model, test_accuracy

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Run one epoch of training or validation.

        Parameters:
            epoch (int): Current epoch number.

        Returns:
            float: Accuracy over the dataset for this epoch.
        """
        pass

    @abstractmethod
    def test(self, dataloader, epoch=0, mode="val"):
        """
        Evaluate the model on the provided dataloader.

        Parameters:
            dataloader (DataLoader): DataLoader for the dataset to evaluate.
            mode (str): 'val' for validation or 'test' for final evaluation.

        Returns:
            tuple: Average loss and accuracy for the dataset.
        """
        pass

    # Optional hooks for more control over training lifecycle
    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_start(self, epoch):
        pass

    def on_epoch_end(self, epoch, metrics):
        pass

    def on_batch_start(self, batch_idx, loss):
        pass

    def on_batch_end(self, batch_idx, loss):
        pass

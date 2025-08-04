"""
Define the training pipeline including model training, validation, and checkpointing.

This module includes a Trainer class that manages the training process, along with
a decorator for setting model mode (train or eval) during different phases.
"""

import os
import tqdm
import logging

import numpy as np
from sklearn.metrics import classification_report

import torch
from torch.utils.tensorboard import SummaryWriter

from models.utils import EarlyStopping


# TODO: Test methods in this file
class StandardTrainer:
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

    def __init__(self, model, train_loader, val_loader, test_loader, config):
        """
        Initialize the Trainer with model, data loaders, and training configuration.

        Parameters:
            model (torch.nn.Module): The model to be trained.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            config (Config): Configuration object containing training parameters and paths.
        """
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = config.device

        self.criterion = config.criterion()
        self.optimizer = config.optimizer(
            self.model.parameters(),
            lr=config.lr,
        )
        self.scheduler = config.scheduler(  # TODO: Flexible scheduler
            self.optimizer,
            T_max=200,  # step_size=config.lr_step, gamma=config.lr_gamma
        )
        self.writer = SummaryWriter(
            log_dir=config.log_dir
        )  # TODO: separate class for logging

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.best_val_accuracy = 0.0

        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)

    def _train_epoch(self, epoch, dataloader):
        """
        Run one epoch of training or validation.

        Parameters:
            epoch (int): Current epoch number.
            dataloader (DataLoader): DataLoader for the current dataset.
            mode (str): 'train' or 'val', used for logging and context control.
            on_batch_end (Callable, optional): Function called after each batch
                                                (e.g., backward and step).

        Returns:
            float: Accuracy over the dataset for this epoch.
        """
        running_loss = 0.0
        correct, total = 0, 0
        STEPS = len(dataloader)

        with tqdm.trange(STEPS) as progress:
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()

                logging.debug(
                    "[MODEL] Compute Label predictor output using input images"
                )
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                logging.debug("[MODEL] Compute gradient and do SGD step")
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()  # * inputs.size(0)

                logging.debug("[MODEL] Progress bar")
                progress.colour = "green"
                progress.desc = (
                    f"Epoch: [{epoch + 1}/{self.config.epochs}][{batch_idx}/{len(dataloader)}]"
                    + f" | Baseline Loss {loss:.10f} "
                )
                # if batch_idx % self.config.print_freq == 0:
                #     logging.debug(f"[MODEL] {progress.desc}")
                #     print(f"[MODEL] {progress.desc}")
                progress.update(1)

            avg_loss = running_loss / len(dataloader)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            accuracy = correct / total

            self.writer.add_scalar("Loss/Train", avg_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", accuracy, epoch)
        print(
            f"Train | Epoch: [{epoch + 1}/{self.config.epochs}] \
                      Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
        )
        return accuracy

    def test(self, dataloader, mode="val"):
        self.model.eval()

        y_true = []
        y_pred = []

        loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Evaluating label_predictor on predicted concepts
                outputs = self.model(inputs)

                if mode == "val":
                    loss = self.criterion(outputs, labels)
                    loss += loss.item() * inputs.size(0)

                # Measuring Accuracy
                _, pred_labels = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred_labels == labels).sum().item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(pred_labels.cpu().numpy())

        accuracy = correct / total

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if mode == "test":
            report = f"Labels: \n {classification_report(y_true, y_pred)}"
            logging.info(report)
            print(report)
            return None, accuracy

        avg_loss = loss / len(dataloader.dataset)

        return avg_loss, accuracy

    def validate(self):
        """
        Run a single validation epoch.

        This method wraps `_run_epoch` with evaluation mode enabled and disables gradients.

        Parameters:
            epoch (int): Current epoch number.

        Returns:
            float: Accuracy for this validation epoch.
        """
        return self.test(dataloader=self.val_loader, mode="val")

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
            self.model.load_state_dict(self.early_stopping.best_model_wts)
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
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            self._train_epoch(epoch, self.train_loader)
            _, val_accuracy = self.validate()
            self.save_checkpoint(epoch, val_accuracy)
            self.scheduler.step()

            if self.early_stopping.early_stop:
                break

        self.writer.close()

        print("Final test evaluation")
        self.load_best_model()
        _, test_accuracy = self.test(self.test_loader)
        print(f"🎯 Final Test Accuracy: {test_accuracy:.4f}")

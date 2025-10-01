import tqdm
import logging

import numpy as np
from sklearn.metrics import classification_report

import torch
from torch.utils.tensorboard import SummaryWriter

from .abstract import BaseTrainer


class StandardTrainer(BaseTrainer):
    """
    Standard training loop for a model with training and validation phases.

    Inherits from BaseTrainer and implements the training logic.
    """

    def __init__(
        self,
        config,
        model,
        train_loader,
        val_loader,
        test_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):

        super().__init__(
            config, model, train_loader, val_loader, test_loader, device
        )

    def _train_epoch(self, epoch):
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
        STEPS = len(self.train_loader)

        with tqdm.trange(STEPS) as progress:
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
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
                    f"Epoch: [{epoch + 1}/{self.config.epochs}][{batch_idx}/{STEPS}]"
                    + f" | Baseline Loss {loss:.10f} "
                )
                # if batch_idx % self.config.print_freq == 0:
                #     logging.debug(f"[MODEL] {progress.desc}")
                #     print(f"[MODEL] {progress.desc}")
                progress.update(1)

                avg_loss = running_loss / STEPS

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

    @torch.no_grad()
    def test(self, dataloader, epoch=0, mode="val"):
        """
        Evaluate the model on the provided dataloader.

        Parameters:
            dataloader (DataLoader): DataLoader for the dataset to evaluate.
            mode (str): 'val' for validation or 'test' for final evaluation.

        Returns:
            tuple: Average loss and accuracy for the dataset.
        """
        self.model.eval()
        STEPS = len(dataloader)

        y_true = []
        y_pred = []

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        with tqdm.trange(STEPS, desc=f"{mode.title()} Evaluation") as progress:
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Evaluating label_predictor on predicted concepts
                outputs = self.model(inputs)

                if mode == "val":
                    loss = self.criterion(outputs, labels)
                    running_loss += loss.item()

                # Measuring Accuracy
                _, pred_labels = torch.max(outputs, 1)
                running_total += labels.size(0)
                running_correct += (pred_labels == labels).sum().item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(pred_labels.cpu().numpy())

                if mode == "val":
                    progress.desc = (
                        f"{mode.title()} [{batch_idx}/{STEPS}]"
                        + f" | Loss {loss:.10f} "
                    )
                    progress.update(1)

        accuracy = running_correct / running_total

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if mode == "test":
            report = f"Labels: \n {classification_report(y_true, y_pred)}"
            logging.info(report)
            self.writer.add_text(
                "Classification Report/Test", report, 0
            )
            return None, accuracy

        avg_loss = running_loss / STEPS

        self.writer.add_scalar(
            f"Loss/{mode.upper()}", avg_loss, epoch
        )
        self.writer.add_scalar(
            f"Accuracy/{mode.upper()}", accuracy, epoch
        )

        return avg_loss, accuracy

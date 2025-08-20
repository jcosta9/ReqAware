import tqdm
import logging

import numpy as np
from sklearn.metrics import classification_report

import torch
from torch.utils.tensorboard import SummaryWriter

from .abstract import BaseTrainer


class CBMConceptPredictorTrainer(BaseTrainer):
    """
    Standard training loop for a model with training and validation phases.

    Inherits from BaseTrainer and implements the training logic.
    """

    def __init__(self, 
                config,
                model, 
                train_loader, 
                val_loader, 
                test_loader,
                log_dir=None, 
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                ):
        
        super().__init__(config,
                            model, 
                            train_loader, 
                            val_loader, 
                            test_loader,
                            log_dir, 
                            device
                        )

    def compute_accuracy(self, outputs, concepts):
        """
        Compute the accuracy of the model's predictions.

        Parameters:
            outputs (torch.Tensor): Model outputs.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            float: Accuracy of the predictions.
        """
        probs = torch.sigmoid(outputs)
        predicted = (probs > 0.5).long()  #TODO: Threshold can be a config parameter
        correct = (predicted == concepts).sum().item()
        total = concepts.size(0) * concepts.size(1)
        return predicted, correct, total

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
        running_correct, running_total = 0, 0
        STEPS = len(self.train_loader)

        with tqdm.trange(STEPS) as progress:
            for batch_idx, (idx, inputs, (concepts, _)) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                concepts = concepts.to(self.device)

                self.optimizer.zero_grad()

                logging.debug(
                    "[MODEL] Compute Label predictor output using input images"
                )
                outputs = self.model(inputs)
                loss = self.criterion(outputs, concepts)

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

                _, correct, total = self.compute_accuracy(outputs, concepts)
                running_correct += correct
                running_total += total
            accuracy = running_correct / running_total

            self.writer.add_scalar("Loss/Train", avg_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", accuracy, epoch)
        print(
            f"Train | Epoch: [{epoch + 1}/{self.config.epochs}] \
                      Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
        )
        return accuracy

    @torch.no_grad()
    def test(self, dataloader, mode="val"):
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

        loss = 0.0
        running_correct = 0
        running_total = 0

        with tqdm.trange(STEPS, desc=f"{mode.title()} Evaluation") as progress:
            for batch_idx, (idx, inputs, (concepts, _)) in enumerate(self.train_loader):
                inputs, concepts = inputs.to(self.device), concepts.to(self.device)

                # Evaluating label_predictor on predicted concepts
                outputs = self.model(inputs)

                if mode == "val":
                    batch_loss = self.criterion(outputs, concepts).item()
                    loss += batch_loss * inputs.size(0)

                # Measuring Accuracy
                predicted, correct, total = self.compute_accuracy(outputs, concepts)
                running_correct += correct
                running_total += total
                
                
                y_true.extend(concepts.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        accuracy = running_correct / running_total

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if mode == "test":
            report = f"Labels: \n {classification_report(y_true, y_pred)}"
            logging.info(report)
            print(report)
            return None, accuracy

        avg_loss = loss / len(dataloader.dataset)

        return avg_loss, accuracy

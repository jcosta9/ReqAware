import tqdm
import logging

import numpy as np
from sklearn.metrics import classification_report

import torch

from models.loss import CustomFuzzyLoss
from .abstract import BaseTrainer


class CBMConceptPredictorTrainer(BaseTrainer):
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
        log_dir=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):

        super().__init__(
            config, model, train_loader, val_loader, test_loader, log_dir, device
        )

        self.criterion = CustomFuzzyLoss(
            config=self.config.fuzzy_loss, current_loss_fn=self.criterion
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
        predicted = (probs > 0.5).long()  # TODO: Threshold can be a config parameter
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
        global_step_base = epoch * STEPS

        self.writer.add_scalar('Learning Rate/Concept_Predictor', self.optimizer.param_groups[0]['lr'], epoch)

        with tqdm.trange(STEPS) as progress:
            for batch_idx, (idx, inputs, (concepts, _)) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                concepts = concepts.to(self.device)

                global_step = global_step_base + batch_idx

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

                # Log Batch loss: track loss on a per-batch basis.
                self.writer.add_scalar('Loss/Train_Batch/Concept_Predictor', loss.item(), global_step)
                if self.config.fuzzy_loss.use_fuzzy_loss:
                    self.writer.add_scalar('Loss/Fuzzy_Total', self.criterion.last_fuzzy_loss.item(), global_step)
                    for name, loss_val in self.criterion.last_individual_losses.items():
                        self.writer.add_scalar(f'FuzzyLoss/{name}', loss_val.item(), global_step)
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
                
                _, correct, total = self.compute_accuracy(outputs, concepts)
                running_correct += correct
                running_total += total

            avg_loss = running_loss / STEPS
            accuracy = running_correct / running_total

            self.writer.add_scalar("Loss/Train/Concept_Predictor", avg_loss, epoch)
            self.writer.add_scalar("Accuracy/Train/Concept_Predictor", accuracy, epoch)
            
            # Log weight histograms
            for name, param in self.model.named_parameters():
                if 'weight' in name or 'bias' in name:
                    self.writer.add_histogram(f'Concept_Predictor/{name}', param.clone().detach().cpu().numpy(), epoch)

                if param.grad is not None:
                    self.writer.add_histogram(f'Concept_Predictor/{name}_grad', param.grad.clone().detach().cpu().numpy(), epoch)
        
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
            report = f"{classification_report(y_true, y_pred)}"
            logging.info(report)
            self.writer.add_text("Classification Report/Test", report, 0)
            return None, accuracy

        avg_loss = loss / len(dataloader.dataset)

        self.writer.add_scalar(f"Loss/{mode.upper()}/Concept_Predictor", avg_loss, epoch)
        self.writer.add_scalar(f"Accuracy/{mode.upper()}/Concept_Predictor", accuracy, epoch)

        return avg_loss, accuracy

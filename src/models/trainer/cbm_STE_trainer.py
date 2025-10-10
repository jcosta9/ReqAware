import tqdm
import logging

import numpy as np
from sklearn.metrics import classification_report

import torch
from torch.utils.tensorboard import SummaryWriter

from models.loss.custom_fuzzy_loss import CustomFuzzyLoss

from .abstract import BaseTrainer


class CBMSTETrainer(BaseTrainer):
    """
    Standard training loop for a model with training and validation phases.

    Inherits from BaseTrainer and implements the training logic.
    """

    def __init__(
        self,
        config,
        experiment_id,
        model,
        concept_predictor,
        train_loader,
        val_loader,
        test_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):

        super().__init__(
            config,
            model,
            train_loader,
            val_loader,
            test_loader,
            device,
        )

        self.tag += "label_predictor"
        self.concept_predictor = concept_predictor.to(self.device)
        self.concepts_threshold = 0.5

        self.criterion = CustomFuzzyLoss(
            config=self.config.fuzzy_loss, current_loss_fn=self.criterion
        )

    def compute_accuracy(self, outputs, targets):
        """
        Compute the accuracy of the model's predictions.

        Parameters:
            outputs (torch.Tensor): Model outputs.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            float: Accuracy of the predictions.
        """
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        total = targets.size(0)
        return correct, total

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

        self.writer.add_scalar(
            "Learning Rate/Label_Predictor",
            self.optimizer.param_groups[0]["lr"],
            epoch,
        )

        if self.config.fuzzy_loss.use_fuzzy_loss:
            print(
                f"[MODEL] Using Fuzzy Loss with rules: {list(self.criterion.fuzzy_rules.keys())}"
            )
            running_loss_standard = 0.0
            running_loss_fuzzy = 0.0
            running_loss_individual = {
                name: 0.0 for name in self.criterion.fuzzy_rules.keys()
            }

        with tqdm.trange(STEPS) as progress:
            for batch_idx, (idx, inputs, (concepts, labels)) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                concepts = concepts.to(self.device)
                labels = labels.to(self.device)

                global_step = global_step_base + batch_idx

                self.optimizer.zero_grad()

                logging.debug(
                    "[MODEL] Compute Label predictor output using input images"
                )
                hard_concepts, pred_labels, concept_logits = self.model(inputs)
                
                # 1. Label Loss (L_label) - The original loss
                label_loss = self.criterion(pred_labels, labels)
                # 2. Concept Loss (L_concept)
                # Use logits and ground truth concepts (concepts)
                concept_loss = self.concept_criterion(concept_logits, concepts.float())
                # 3. Total Joint Loss (L_total)
                loss = label_loss + self.concept_lambda * concept_loss # Use self.LAMBDA for weight


                logging.debug("[MODEL] Compute gradient and do SGD step")
                loss.backward()

                self.optimizer.step()
                running_loss += loss.item()  # * inputs.size(0)

                if self.config.fuzzy_loss.use_fuzzy_loss:
                    running_loss_standard += self.criterion.last_standard_loss.item()
                    running_loss_fuzzy += self.criterion.last_fuzzy_loss.item()
                    for name, loss_val in self.criterion.last_individual_losses.items():
                        running_loss_individual[name] += loss_val.item()

                # Log Batch loss: track loss on a per-batch basis.
                self.writer.add_scalar(
                    "Loss/Train_Batch/Label_Predictor", loss.item(), global_step
                )
                if self.config.fuzzy_loss.use_fuzzy_loss:
                    self.writer.add_scalar(
                        "Loss/Train_batch/Concept_Predictor/Fuzzy/Standard",
                        self.criterion.last_standard_loss.item(),
                        global_step,
                    )
                    self.writer.add_scalar(
                        "Loss/Train_batch/Concept_Predictor/Fuzzy/Total",
                        self.criterion.last_fuzzy_loss.item(),
                        global_step,
                    )
                    for name, loss_val in self.criterion.last_individual_losses.items():
                        self.writer.add_scalar(
                            f"Loss/Train_batch/Concept_Predictor/Fuzzy/{name}",
                            loss_val.item(),
                            global_step,
                        )

                logging.debug("[MODEL] Progress bar")
                progress.colour = "green"
                progress.desc = (
                    f"Epoch: [{epoch + 1}/{self.config.epochs}][{batch_idx}/{STEPS}]"
                    + f" | Baseline Loss {loss:.10f} "
                )
                progress.update(1)

                correct, total = self.compute_accuracy(pred_labels, labels)
                running_correct += correct
                running_total += total

            avg_loss = running_loss / STEPS
            accuracy = running_correct / running_total

            self.writer.add_scalar("Loss/Train/Label_Predictor", avg_loss, epoch)
            self.writer.add_scalar("Accuracy/Train/Label_Predictor", accuracy, epoch)

            if self.config.fuzzy_loss.use_fuzzy_loss:
                avg_loss_standard = running_loss_standard / STEPS
                avg_loss_fuzzy = running_loss_fuzzy / STEPS
                self.writer.add_scalar(
                    "Loss/Train/Concept_Predictor/Fuzzy/Standard",
                    avg_loss_standard,
                    epoch,
                )
                self.writer.add_scalar(
                    "Loss/Train/Concept_Predictor/Fuzzy/Total",
                    avg_loss_fuzzy,
                    epoch,
                )
                for name, loss_val in running_loss_individual.items():
                    self.writer.add_scalar(
                        f"Loss/Train/Concept_Predictor/Fuzzy/{name}",
                        loss_val / STEPS,
                        epoch,
                    )
                    
            # Log weight histograms
            for name, param in self.model.named_parameters():
                if "weight" in name or "bias" in name:
                    if param.data.numel() > 0 and len(torch.unique(param.grad)) > 1:
                        try:
                            self.writer.add_histogram(
                                f"Label_Predictor/{name}",
                                param.clone().detach().cpu().numpy(),
                                epoch,
                            )
                        except ValueError as e:
                            print(param.clone().detach().cpu().numpy())
                            logging.warning(
                                f"Could not log histogram for {name} at epoch {epoch}: {e}"
                            )

                if param.grad is not None:
                    if param.grad.numel() > 0 and len(torch.unique(param.grad)) > 1:
                        try:
                            self.writer.add_histogram(
                                f"Label_Predictor/{name}_grad",
                                param.grad.clone().detach().cpu().numpy(),
                                epoch,
                            )
                        except ValueError as e:
                            print(param.grad.clone().detach().cpu().numpy())
                            logging.warning(
                                f"Could not log gradient histogram for {name} at epoch {epoch}: {e}"
                            )

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
            for batch_idx, (idx, inputs, (concepts, labels)) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                concepts = concepts.to(self.device)
                labels = labels.to(self.device)

                hard_concepts, pred_labels, concept_logits = self.model(inputs)

                if mode == "val":
                    # 1. Label Loss (L_label)
                    label_loss = self.criterion(pred_labels, labels)

                    # 2. Concept Loss (L_concept) - Use BCEWithLogitsLoss
                    concept_loss = self.concept_criterion(concept_logits, concepts.float())
                    
                    # 3. Total Loss
                    loss = label_loss + self.concept_lambda * concept_loss
                    
                    # Update running loss
                    running_loss += loss.item() * inputs.size(0)

                # Measuring Accuracy
                correct, total = self.compute_accuracy(pred_labels, labels)
                running_correct += correct
                running_total += total

                y_true.extend(labels.cpu().numpy())
                _, pred_labels = torch.max(pred_labels, 1)
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
            print(report)
            self.writer.add_text("Classification Report/Label_Predictor/Test", report, 0)
            return None, accuracy

        avg_loss = running_loss / running_total if running_total > 0 else 0.0 # Use running_total if scaled by batch size

        self.writer.add_scalar(
            f"Loss/{mode.upper()}/Label_Predictor", avg_loss, epoch
        )
        self.writer.add_scalar(
            f"Accuracy/{mode.upper()}/Label_Predictor", accuracy, epoch
        )

        return avg_loss, accuracy
    
    def get_predictions(self, dataloader):
        """
        Get predictions and ground truth labels from the dataloader.

        Parameters:
            dataloader (DataLoader): DataLoader for the dataset to evaluate.
        Returns:
            tuple: (predictions, ground_truth)"""
        
        self.model.eval()
        STEPS = len(dataloader)

        y_true = []
        y_pred = []

        loss = 0.0
        running_correct = 0
        running_total = 0

        with tqdm.trange(STEPS, desc="Getting predictions") as progress:
            for batch_idx, (idx, inputs, (_, labels)) in enumerate(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Evaluating label_predictor on predicted concepts
                pred_concepts = self.concept_predictor(inputs)
                pred_concepts = (
                    torch.sigmoid(pred_concepts) > self.concepts_threshold
                ).float()  # TODO: Threshold can be a config parameter

                pred_labels = self.model(pred_concepts)


                y_true.extend(labels.cpu().numpy())
                _, pred_labels = torch.max(pred_labels, 1)
                y_pred.extend(pred_labels.cpu().numpy())

                progress.update(1)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        return y_pred, y_true
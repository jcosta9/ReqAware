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
        experiment_id,
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
            device
        )

        self.tag += "concept_predictor"
        self.criterion = CustomFuzzyLoss(
            config=self.config.fuzzy_loss, current_loss_fn=self.criterion
        )

    def compute_accuracy(self, outputs, concepts):
        """
        Compute the accuracy of the model's predictions.

        Parameters:
            outputs (torch.Tensor): Model outputs.
            concepts (torch.Tensor): Ground truth concept labels.

        Returns:
            tuple: (predicted_concepts, correct_predictions, total_predictions, accuracy)
        """
        probs = torch.sigmoid(outputs)
        predicted = probs > 0.5  # TODO: Threshold can be a config parameter
        correct = (predicted == concepts).sum().item()
        total = concepts.size(0) * concepts.size(1)
        accuracy = correct / total if total > 0 else 0.0
        return predicted, correct, total, accuracy

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
            "Learning Rate/Concept_Predictor",
            self.optimizer.param_groups[0]["lr"],
            epoch,
        )

        if self.config.fuzzy_loss.use_fuzzy_loss:
            logging.info(
                f"[MODEL] Using Fuzzy Loss with rules: {list(self.criterion.fuzzy_rules.keys())}"
            )
            running_loss_standard = 0.0
            running_loss_fuzzy = 0.0
            running_loss_individual = {
                name: 0.0 for name in self.criterion.fuzzy_rules.keys()
            }

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

                if self.config.fuzzy_loss.use_fuzzy_loss:
                    running_loss_standard += self.criterion.last_standard_loss.item()
                    running_loss_fuzzy += self.criterion.last_fuzzy_loss.item()
                    for name, loss_val in self.criterion.last_individual_losses.items():
                        running_loss_individual[name] += loss_val.item()

                # Log Batch loss: track loss on a per-batch basis.
                self.writer.add_scalar(
                    "Loss/Train_batch/Concept_Predictor/Total", loss.item(), global_step
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
                # if batch_idx % self.config.print_freq == 0:
                #     logging.debug(f"[MODEL] {progress.desc}")
                #     print(f"[MODEL] {progress.desc}")
                progress.update(1)

                _, correct, total, _ = self.compute_accuracy(outputs, concepts)
                running_correct += correct
                running_total += total

            avg_loss = running_loss / STEPS
            accuracy = running_correct / running_total
            self.writer.add_scalar("Loss/Train/Concept_Predictor", avg_loss, epoch)
            self.writer.add_scalar("Accuracy/Train/Concept_Predictor", accuracy, epoch)

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

            for name, param in self.model.named_parameters():
                # Log weights and biases
                if "weight" in name or "bias" in name:
                    # Check if the tensor has at least one non-zero element
                    if param.data.numel() > 0 and len(torch.unique(param.grad)) > 1:
                        try:
                            self.writer.add_histogram(
                                f"Concept_Predictor/{name}",
                                param.clone().detach().cpu().numpy(),
                                epoch,
                            )
                        except ValueError as e:
                            print(param.clone().detach().cpu().numpy())
                            logging.warning(
                                f"Could not log histogram for {name} at epoch {epoch}: {e}"
                            )

                # Log gradients
                if param.grad is not None:
                    # Check if the gradient tensor has at least one non-zero element
                    if param.grad.numel() > 0 and len(torch.unique(param.grad)) > 1:
                        try:
                            self.writer.add_histogram(
                                f"Concept_Predictor/{name}_grad",
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
        print(len(dataloader.dataset))

        with tqdm.trange(STEPS, desc=f"{mode.title()} Evaluation") as progress:
            for batch_idx, (idx, inputs, (concepts, _)) in enumerate(dataloader):
                inputs, concepts = inputs.to(self.device), concepts.to(self.device)

                # Evaluating label_predictor on predicted concepts
                outputs = self.model(inputs)

                if mode == "val":
                    loss = self.criterion(outputs, concepts)
                    running_loss += loss.item()

                # Measuring Accuracy
                predicted, correct, total, _ = self.compute_accuracy(outputs, concepts)
                running_correct += correct
                running_total += total

                y_true.extend(concepts.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                
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
            report = f"{classification_report(y_true, y_pred)}"
            print(report)
            self.writer.add_text(
                "Classification Report/Concept_Predictor/Test", report, 0
            )
            return None, accuracy

        avg_loss = running_loss / STEPS

        self.writer.add_scalar(
            f"Loss/{mode.upper()}/Concept_Predictor", avg_loss, epoch
        )
        self.writer.add_scalar(
            f"Accuracy/{mode.upper()}/Concept_Predictor", accuracy, epoch
        )

        return avg_loss, accuracy

    @torch.no_grad()
    def get_predictions(self, dataloader):
        """
        Perform a complete forward pass on the dataloader and return predictions and ground truth.
        
        Parameters:
            dataloader (DataLoader): DataLoader containing the dataset to evaluate.
            
        Returns:
            tuple: (concept_predictions, concept_ground_truth, concept_probabilities)
                - concept_predictions (numpy.ndarray): Binary predictions for concepts (0 or 1)
                - concept_ground_truth (numpy.ndarray): Ground truth concept labels
                - concept_probabilities (numpy.ndarray): Probability scores for each concept
        """
        self.model.eval()
        
        # Lists to store predictions and ground truth
        all_logits = []
        all_predictions = []
        all_ground_truth = []
        all_probabilities = []
        
        # Process all batches
        with tqdm.tqdm(dataloader, desc="Getting predictions") as progress:
            for idx, inputs, (concepts, _) in progress:
                inputs = inputs.to(self.device)
                concepts = concepts.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Convert logits to probabilities
                probabilities = torch.sigmoid(outputs)
                
                # Get binary predictions using threshold
                predictions = (probabilities > 0.5).long()
                
                # Store batch results
                all_logits.append(outputs.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
                all_ground_truth.append(concepts.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
        
        # Concatenate results from all batches
        all_logits = np.vstack(all_logits)
        concept_predictions = np.vstack(all_predictions)
        concept_ground_truth = np.vstack(all_ground_truth)
        concept_probabilities = np.vstack(all_probabilities)
        
        return all_logits, concept_predictions, concept_ground_truth, concept_probabilities

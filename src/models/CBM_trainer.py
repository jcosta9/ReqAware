import os
import tqdm
import logging

import torch
from torch.utils.tensorboard import SummaryWriter

from .utils import EarlyStopping

class CBMTrainer():
    """
    CBM (Concept Bottleneck Model) training loop for a model with training and validation phases.

    Inherits from BaseTrainer and implements the training logic specific to CBM.
    """

    def __init__(self, config, model:torch.nn.Module=None, data_factory=None, concept_predictor:torch.nn.Module=None, label_predictor:torch.nn.Module=None, train_loader=None, val_loader=None, test_loader=None):
        super().__init__()
        self.phase = None  # Phase can be 'concept' or 'label'
        self.config = config
        self.device = config.device

        logging.info("[MODEL] Initializing CBM Trainer...")

        self.set_model(model, concept_predictor, label_predictor)            
        self.set_dataloaders(data_factory, train_loader, val_loader, test_loader)

        # TODO: change this. should load first concept, then label - if no_freeze
        self.set_optimizers()
        self.set_criterion()
        self.set_schedulers()

        self.writer = SummaryWriter(
            log_dir=config.log_dir
        )  # TODO: separate class for logging

        os.makedirs(config.model_checkpoint_dir, exist_ok=True)
        self.best_val_accuracy = 0.0

        self.concept_early_stopping = EarlyStopping(patience=config.concept_predictor_early_stopping_patience)
        self.label_early_stopping = EarlyStopping(patience=config.label_predictor_early_stopping_patience)
    
    def set_model(self, model, concept_predictor, label_predictor):
        """
        Check if the model and its predictors are correctly initialized.
        """
        logging.info("[MODEL] Checking model and predictors...")
        print(model)
        print(model.concept_predictor)
        print(model.label_predictor)
        self.model = model
        if model is None: # model given separately
            if concept_predictor is None or label_predictor is None:
                msg = "[MODEL] Model and predictors must be provided or initialized."
                logging.error(msg)
                raise ValueError(msg)
            logging.info("[MODEL] Model and predictors were given as parameters and are valid.")
            self.concept_predictor = concept_predictor
            self.label_predictor = label_predictor
        else:  # model given as a whole
            if not hasattr(model, 'concept_predictor') or not hasattr(model, 'label_predictor'):
                raise ValueError("Model must have concept_predictor and label_predictor attributes.")
            self.concept_predictor = model.concept_predictor
            self.label_predictor = model.label_predictor
        
        self.concept_predictor = self.concept_predictor.to(self.device)
        self.label_predictor = self.label_predictor.to(self.device)
    
    def set_dataloaders(self, data_factory, train_loader, val_loader, test_loader):
        """
        Set the data loaders for training, validation, and testing.

        Parameters:
            data_factory (DatasetFactory): Factory to create data loaders.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            test_loader (DataLoader): DataLoader for testing data.
        """
        if data_factory is None:
            if train_loader is None or val_loader is None or test_loader is None:
                raise ValueError("Data factory or individual data loaders must be provided.")
            logging.info("[MODEL] Using provided data loaders.")
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader

        else:
            if not hasattr(data_factory, 'train_dataloader') or not hasattr(data_factory, 'val_dataloader') or not hasattr(data_factory, 'test_dataloader'):
                raise ValueError("Data factory must have train_dataloader, val_dataloader, and test_dataloader attributes.")
            self.train_loader = data_factory.train_dataloader
            self.val_loader = data_factory.val_dataloader
            self.test_loader = data_factory.test_dataloader

        logging.info("[MODEL] Dataloaders set successfully.")

    def set_criterion(self):
        """
        Set the optimizers for the concept and label predictors.
        """
        if not hasattr(self.config, 'concept_predictor_criterion') or not hasattr(self.config, 'label_predictor_criterion'):
            msg = "[Model] CBM Config must have concept_criterion and label_criterion attributes."
            logging.error(msg)
            raise ValueError(msg)

        self.concept_criterion = self.config.concept_predictor_criterion()
        self.label_criterion = self.config.label_predictor_criterion()
    
    def set_optimizers(self):
        """
        Set the optimizers for the concept and label predictors.
        """
        
        if not hasattr(self.config, 'concept_predictor_optimizer') or not hasattr(self.config, 'label_predictor_optimizer'):
            msg = "[Model] CBM Config must have concept_optimizer and label_optimizer attributes."
            logging.error(msg)
            raise ValueError(msg)
        
        self.concept_predictor_optimizer = self.config.concept_predictor_optimizer(
            self.concept_predictor.parameters(),
            lr=self.config.concept_predictor_lr,
        )
        self.label_predictor_optimizer = self.config.label_predictor_optimizer(
            self.label_predictor.parameters(),
            lr=self.config.label_predictor_lr,
        )

    def set_schedulers(self):
        """
        Set the learning rate schedulers for the concept and label predictors.
        """
        if not hasattr(self.config, 'concept_predictor_scheduler') or not hasattr(self.config, 'label_predictor_scheduler'):
            msg = "[Model] CBM Config must have concept_scheduler and label_scheduler attributes."
            logging.error(msg)
            raise ValueError(msg)

        self.concept_predictor_scheduler = self.config.concept_predictor_scheduler(
            self.concept_predictor_optimizer,
            **self.config.concept_predictor_scheduler_params
        )
        self.Label_predictor_scheduler = self.config.label_predictor_scheduler(
            self.label_predictor_optimizer,
            **self.config.label_predictor_scheduler_params
        )

    def set_phase(self, phase):
        """
        Set the training phase for the CBM model.

        Parameters:
            phase (str): The phase to set, either 'concept' or 'label'.
        """
        if phase not in ["concept", "label"]:
            raise ValueError("Phase must be either 'concept' or 'label'.")
        self.phase = phase
        logging.info(f"[MODEL] Phase set to {self.phase}.")

    def get_phase(self):
        """
        Get the current training phase.

        Returns:
            str: The current phase, either 'concept' or 'label'.
        """
        return self.phase

    def _train_concept_epoch(self, epoch):
        """
        Run one epoch of training for the concept bottleneck model.

        Parameters:
            epoch (int): Current epoch number.

        Returns:
            float: Average loss over the dataset for this epoch.
        """
        running_loss = 0.0
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
                    + f" | CBM Loss {loss:.10f} "
                )
                progress.update(1)

        avg_loss = running_loss / STEPS
        return avg_loss

    def _train_label_epoch(self, epoch):
        """
        Run one epoch of training for the label predictor in CBM.
        """
        running_loss = 0.0
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
                    + f" | Label Loss {loss:.10f} "
                )
                progress.update(1)

        avg_loss = running_loss / STEPS
        return avg_loss
    

    def _train_epoch(self, epoch):
        """
        Run one epoch of training for the CBM model.

        Parameters:
            epoch (int): Current epoch number.

        Returns:
            float: Accuracy over the dataset for this epoch.
        """
        if self.phase == "concept":
            return self._train_concept_epoch(epoch)
        else:
            return self._train_label_epoch(epoch)

    def test(self, dataloader, mode="val"):
        """
        Evaluate the model on the provided dataloader.

        Parameters:
            dataloader (DataLoader): DataLoader for the dataset to evaluate.
            mode (str): 'val' for validation or 'test' for final evaluation.

        Returns:
            tuple: Average loss and accuracy for the dataset.
        """
        # self.model.eval()
        # running_loss = 0.0
        # correct, total = 0, 0

        # with torch.no_grad():
        #     for inputs, targets in dataloader:
        #         inputs, targets = inputs.to(self.device), targets.to(self.device)
        #         outputs = self.model(inputs)
        #         loss = self.criterion(outputs, targets)

        #         running_loss += loss.item() * inputs.size(0)
        #         _, predicted = outputs.max(1)
        #         total += targets.size(0)
        #         correct += predicted.eq(targets).sum().item()

        # avg_loss = running_loss / len(dataloader.dataset)
        # accuracy = correct / total
        # logging.info(f"[MODEL] {mode.capitalize()} Loss {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        # return avg_loss, accuracy

    def on_train_start(self):
        logging.info("[MODEL] Starting CBM training...")

    def on_train_end(self):
        logging.info("[MODEL] CBM training completed.")

    def on_epoch_start(self, epoch):
        logging.info(f"[MODEL] Starting epoch {epoch + 1}/{self.config.epochs}")

    def on_epoch_end(self, epoch, metrics):
        logging.info(f"[MODEL] End of epoch {epoch + 1}/{self.config.epochs}")
        if metrics:
            logging.info(f"[MODEL] Metrics: {metrics}")

    def on_batch_start(self, batch_idx, loss):
        logging.debug(f"[MODEL] Starting batch {batch_idx}, Loss: {loss:.4f}")

    def on_batch_end(self, batch_idx, loss):
        logging.debug(f"[MODEL] End of batch {batch_idx}, Loss: {loss:.4f}")
        self.writer.add_scalar("Loss/Train", loss, batch_idx)
        self.writer.flush()
        logging.info(f"[MODEL] Batch {batch_idx} completed with loss {loss:.4f}")
        self.on_batch_end(batch_idx, loss)
        self.writer.add_scalar("Loss/Train", loss, batch_idx)
        self.writer.flush()
        logging.info(f"[MODEL] Batch {batch_idx} completed with loss {loss:.4f}")

    def save_checkpoint(self, epoch, val_accuracy):
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': val_accuracy,
        }, checkpoint_path)
        logging.info(f"[MODEL] Checkpoint saved at {checkpoint_path}")
        self.writer.add_scalar("Accuracy/Val", val_accuracy, epoch)
        self.writer.flush()

    def load_best_model(self):
        best_checkpoint = max(self.config.checkpoint_dir.glob("*.pth"), key=lambda x: x.stat().st_mtime)
        checkpoint = torch.load(best_checkpoint)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"[MODEL] Best model loaded from {best_checkpoint}")
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"[MODEL] Optimizer state loaded from {best_checkpoint}")
        self.writer.add_scalar("Accuracy/Test", checkpoint['val_accuracy'], checkpoint['epoch'])
        self.writer.flush()
        logging.info(f"[MODEL] Test accuracy logged for epoch {checkpoint['epoch']}")
        return checkpoint['epoch'], checkpoint['val_accuracy']
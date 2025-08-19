"""
Define the EarlyStopping class to stop training when a model's performance does not improve.

This module includes an EarlyStopping class to keep track of the model performance, which
stores the best weights and raises a flag if the performance does not improve after some
predefined epochs.
"""


class EarlyStopping:
    """
    EarlyStopping class to stop training when a model's performance does not improve.

    EarlyStopping is a utility class to stop training when a model's performance on
    the validation set does not improve for a specified number of consecutive epochs
    (patience). It tracks the best validation score and restores the model's weights
    when the early stopping condition is met.

    Attributes:
        patience (int): The number of epochs with no improvement to wait before
                        stopping training.
        delta (float): Minimum change required to be considered an improvement
                        (default is 0).
        counter (int): Counter to track how many epochs have passed without improvement.
        best_score (float or None): The best validation score achieved during training.
        early_stop (bool): Flag to indicate whether early stopping should be triggered.
        best_model_wts (dict or None): Stores the model's state_dict when the best
                                        validation score is achieved.
    """

    def __init__(self, patience, delta=0):
        """
        Initialize the EarlyStopping object.

        Args:
            patience (int): The number of epochs with no improvement to wait before
                            stopping training.
            delta (float): Minimum change required to be considered an improvement
                            (default is 0).
        """
        self.patience = (
            patience  # Number of epochs with no improvement to wait before stopping
        )
        self.delta = delta  # Minimum change to qualify as an improvement
        self.counter = 0  # How many epochs without improvement
        self.best_score = None  # Best validation score
        self.early_stop = False  # Flag to indicate if training should stop
        self.best_model_wts = None  # Store best model weights

    def __call__(self, val_score, model):
        """
        Check whether early stopping criteria are met based on the current validation score.

        This method should be called after each validation epoch to track whether the validation
        score has improved. If the score improves, it resets the counter. If the score does not
        improve, it increments the counter. When the counter reaches the patience value, early
        stopping is triggered.

        Args:
            val_score (float): The current validation score (e.g., validation accuracy or loss).
            model (torch.nn.Module): The model whose weights are being tracked for saving.
        """
        # If this is the first epoch or the best score is improved
        if self.best_score is None or val_score > self.best_score + self.delta:
            self.best_score = val_score
            self.counter = 0
            self.best_model_wts = (
                model.state_dict()
            )  # Save model weights for later restoration
        else:
            self.counter += 1
            print(
                f"⚠️  No improvement. Early stop counter: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_weights(self, model):
        """
        Restores the model's weights from the best performing epoch.

        If early stopping is triggered, this method can be called to load the model's weights
        from the epoch with the best validation score (before training stopped due to no
        improvement).

        Args:
            model (torch.nn.Module): The model to which the best weights should be restored.
        """
        if self.best_model_wts is not None:
            model.load_state_dict(self.best_model_wts)

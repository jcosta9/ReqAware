import torch
from torch import nn

class CustomFuzzyLoss(nn.Module):
    def __init__(self, current_loss_fn, fuzzy_lambda=0.5):
        super().__init__()
        self.current_loss_fn = current_loss_fn()
        self.fuzzy_lambda = fuzzy_lambda

    def c1_should_be_close_to_c2(self, p1, p2):
        """
        Example fuzzy rule: p1 should be close to p2.
        This is a placeholder for your specific fuzzy logic implementation.
        """
        return torch.mean(torch.abs(p1 - p2))

    def forward(self, y_pred, y_true):
        # Calculate standard loss (e.g., Binary Cross-Entropy)
        standard_loss = self.current_loss_fn(y_pred, y_true)

        fuzzy_loss = self.c1_should_be_close_to_c2(y_pred[:, 0], y_pred[:, 1])

        # Combine the losses
        total_loss = standard_loss + self.fuzzy_lambda * fuzzy_loss
        return total_loss
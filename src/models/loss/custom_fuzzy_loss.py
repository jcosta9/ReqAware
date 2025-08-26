import torch
from torch import nn

class CustomFuzzyLoss(nn.Module):
    def __init__(self, config, current_loss_fn):
        super().__init__()
        self.config = config
        self.current_loss_fn = current_loss_fn
        self.fuzzy_lambda = config.fuzzy_lambda #TODO: make it learnable

        for rule_fn in self.config.custom_rules:
            print(f"Using custom fuzzy rule: {rule_fn.__name__}")

    def forward(self, y_pred, y_true):
        # Calculate standard loss (e.g., Binary Cross-Entropy)
        standard_loss = self.current_loss_fn(y_pred, y_true)

        # fuzzy_loss = self.c1_should_be_close_to_c2(y_pred[:, 0], y_pred[:, 1])

        if not self.config.use_fuzzy_loss:
            return standard_loss

        # Calculate total fuzzy loss by iterating through active rules
        total_fuzzy_loss = torch.tensor(0.0, device=y_pred.device)
        for rule_fn in self.config.custom_rules:
            rule_loss = rule_fn(y_pred, y_true)
            total_fuzzy_loss += rule_loss

        # Combine the losses
        total_loss = standard_loss + self.fuzzy_lambda * total_fuzzy_loss

        # Combine the losses
        # total_loss = standard_loss + self.fuzzy_lambda * fuzzy_loss
        return total_loss
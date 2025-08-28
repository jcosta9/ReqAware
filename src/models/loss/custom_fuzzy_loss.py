import torch
from torch import nn
from fuzzy_transformations import GodelTNorm, GodelAAggreation, GodelEAggregation


class CustomFuzzyLoss(nn.Module):
    def __init__(self, config, current_loss_fn):
        super().__init__()
        self.config = config
        self.current_loss_fn = current_loss_fn
        self.fuzzy_lambda = config.fuzzy_lambda  # TODO: make it learnable

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
            rule_loss = rule_fn(y_pred, y_true, self.config.concept_map)
            total_fuzzy_loss += rule_loss

        # Combine the losses
        total_loss = standard_loss + self.fuzzy_lambda * total_fuzzy_loss

        # Combine the losses
        # total_loss = standard_loss + self.fuzzy_lambda * fuzzy_loss
        return total_loss

class ExactlyOneShapeGodel(nn.Module):
    """A loss function resembling that a traffic sign has exactly one shape in the GTSRB using the Godel t-norm transformation."""
    def __init__(self, config, current_loss_fn, shape_indices):
        super().__init__()
        self.config = config
        self.current_loss_fn = current_loss_fn
        self.fuzzy_lambda = config.fuzzy_lambda
        self.shape_concept_indices = shape_indices

        for rule_fn in self.config.custom_rules:
            print(f"Using custom fuzzy rule: {rule_fn.__name__}")
    

    def forward(self, y_pred: torch.tensor, y_true: torch.tensor) -> torch.tensor:
        """Implementation of the exactly one shape loss using the Godel t-norm"""
        standard_loss = self.current_loss_fn(y_pred, y_true)
        exist_aggregation = torch.tensor(device=self.config.device + ":" + self.config.device_no)
        all_tnorm_values = []
        for i in range(len(self.shape_concept_indices)):
            # calculating the inner most t-norm aggreation where the shapes are unequal: (∀c): min_{j!=i}(1 - prob_j(c))
            inner_most_probs = torch.cat([y_pred[:, :i], y_pred[:, i+1:]], dim=1)
            inner_most_values = 1 - inner_most_probs
            inner_most_tnorm = GodelAAggreation(inner_most_values)
            # calculating the t-norm between the shape i and the innermost aggregation
            tnorm_value = GodelTNorm(y_pred[i] , inner_most_tnorm)
            all_tnorm_values.append(tnorm_value)
        
        # final exist aggregation step
        fuzzy_loss = 1 - GodelEAggregation(torch.tensor(all_tnorm_values))
        return standard_loss + self.fuzzy_lambda * fuzzy_loss

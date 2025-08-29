import torch
from torch import nn
from .abstract.fuzzy_loss import FuzzyLoss


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

class ExactlyOneShape(FuzzyLoss):
    def __init__(self, t_norm, t_conorm, implication, e_aggregation, a_aggregation, shape_indices):
        super().__init__(t_norm, t_conorm, implication, e_aggregation, a_aggregation)
        self.shape_indices = shape_indices

    def forward(self, y_pred: torch.tensor) -> torch.tensor:
        """Collects t-norm values using an explicit for loop."""

        # tricking around to fix batch size
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
        
        # Isolate the concept probabilities we're working with
        concept_probs = y_pred[:, self.shape_indices]
        batch_size, num_concepts = concept_probs.shape
        batch_losses = []

        for i in range(num_concepts):
            # The concept we are focusing on in this iteration
            other_concepts = torch.cat([concept_probs[:,:i], concept_probs[:,i+1:]], dim=1)
            # Negate them
            negated_other_concepts = 1.0 - other_concepts
            # Apply the universal aggregation (e.g., min) over the "other" concepts
            all_aggregation = self.a_aggregation(negated_other_concepts)
            # Apply the t-norm between the current concept and the aggregation
            t_norm = self.t_norm(concept_probs[:,i], all_aggregation)
            # Add the result for this iteration to our list
            batch_losses.append(t_norm.unsqueeze(1))

        batch_losses =  torch.cat(batch_losses, dim=1)
        exist_agg = self.e_aggregation(batch_losses)

        if y_pred.dim() == 1:
            batch_losses = batch_losses.squezze()
        return 1 - exist_agg

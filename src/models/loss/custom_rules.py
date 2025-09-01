import torch
from .abstract.fuzzy_loss import FuzzyLoss

class ExactlyOneShape(FuzzyLoss):
    def __init__(self, t_norm, t_conorm, e_aggregation, a_aggregation, shape_indices):
        super().__init__(t_norm, t_conorm, e_aggregation, a_aggregation)
        self.shape_indices = shape_indices

    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
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

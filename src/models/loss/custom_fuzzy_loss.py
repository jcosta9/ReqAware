import torch
from torch import nn
from .abstract.fuzzy_loss import FuzzyLoss
from .abstract.fuzzy_transformation_abstract import FuzzyTransformation
from typing import Dict, Type


class CustomFuzzyLoss(nn.Module):
    """This is a highly versatile class that constructs a custom loss function for a multi class multi label problem.
    The config can specify the transformation rules and define the custom domain-rules
    """

    def __init__(self, config: Dict, current_loss_fn: nn.Module) -> None:
        super().__init__()
        self.current_loss_fn = current_loss_fn
        self.fuzzy_rules = nn.ModuleDict()
        self.fuzzy_lambdas = {}
        self.use_fuzzy_loss = config.use_fuzzy_loss
        if self.use_fuzzy_loss:
            self._build_rules_from_config(config)
        self.last_standard_loss = torch.tensor(0.0)
        self.last_fuzzy_loss = torch.tensor(0.0)
        # Optional: store individual rule losses too
        self.last_individual_losses = {}

    def _build_rules_from_config(self, config: Dict):
        for rule_name, rule_config in config.rules.items():
            try:
                self.fuzzy_rules[rule_name] = rule_config.rule(
                    t_norm=rule_config.operators.t_norm,
                    t_conorm=rule_config.operators.t_conorm,
                    e_aggregation=rule_config.operators.e_aggregation,
                    a_aggregation=rule_config.operators.a_aggregation,
                    params=rule_config.params,
                )
                self.fuzzy_lambdas[rule_name] = rule_config.fuzzy_lambda
            except Exception as e:
                print(f"Error: {e}")
                raise Exception(f"Fuzzy rule {rule_name} could not be instantiated.")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = torch.sigmoid(y_pred)
        standard_loss = self.current_loss_fn(y_pred, y_true)
        # updating the loss to make it visible outside the class
        self.last_standard_loss = standard_loss.detach()

        total_fuzzy_loss = torch.tensor(0.0, device=y_pred.device)
        self.last_individual_losses.clear()
        for rule_name, rule_module in self.fuzzy_rules.items():
            # Calculate the loss for one rule (already averaged over the batch)
            rule_loss = rule_module(y_pred).mean()
            self.last_individual_losses[rule_name] = rule_loss.detach()

            # Weight it by its specific lambda and add to the total
            total_fuzzy_loss += self.fuzzy_lambdas[rule_name] * rule_loss

        self.last_fuzzy_loss = total_fuzzy_loss.detach()
        return standard_loss + total_fuzzy_loss

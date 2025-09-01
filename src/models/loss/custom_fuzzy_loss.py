import torch
from torch import nn
from .abstract.fuzzy_loss import FuzzyLoss
from .abstract.fuzzy_transformation_abstract import FuzzyTransformation
from typing import Dict, Type
# getting all the Godel transformations
from .fuzzy_transformations import GodelTNorm, GodelTConorm, GodelAAggregation, GodelEAggregation 
from .custom_rules import ExactlyOneShape

OPERATOR_MAP = {
    "godel_t_norm": GodelTNorm,
    "godel_t_conorm": GodelTConorm,
    "godel_a_aggregation": GodelAAggregation,
    "godel_e_aggregation": GodelEAggregation,
}

RULE_MAP = {
    "ExactlyOneShape": ExactlyOneShape,
}

def get_rule_class(name: str) -> Type[FuzzyLoss]:
    if name not in RULE_MAP:
        raise ValueError(f"Unknown rule class name: {name}")
    return RULE_MAP[name]

def get_operator(name: str) -> FuzzyTransformation:
    if name.lower() not in OPERATOR_MAP:
        raise ValueError(f"Unknown operator name: {name}")
    return OPERATOR_MAP[name.lower()]()


class CustomFuzzyLoss(nn.Module):
    """This is a highly versatile class that constructs a custom loss function for a multi class multi label problem. 
    The config can specify the transformation rules and define the custom domain-rules"""

    def __init__(self, config: Dict, current_loss_fn: nn.Module) -> None:
        super().__init__()
        self.current_loss_fn = current_loss_fn
        self.fuzzy_rules = nn.ModuleDict()
        self.lambdas = {}
        self.use_fuzzy_loss = config.get("use_fuzzy_loss", False)
        if self.use_fuzzy_loss:
            self._build_rules_from_config(config)

    def _build_rules_from_config(self, config: Dict):
        for rule_name, rule_config in config.get("rules", {}).items():
            rule_class_name = rule_config["class"]
            
            # Find the actual class object (e.g., ExactlyOneShape)
            RuleClass = get_rule_class(rule_class_name)
            # Build the operator objects for this specific rule
            operator_kwargs = {
                op_name: get_operator(op_impl)
                for op_name, op_impl in rule_config.get("operators", {}).items()
            }
            
            # Get other rule-specific parameters
            params = rule_config.get("params", {})
            
            # Instantiate the rule and store it
            self.fuzzy_rules[rule_name] = RuleClass(**operator_kwargs, **params)
            
            # Store its lambda
            self.lambdas[rule_name] = rule_config.get("lambda", 1.0)
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        standard_loss = self.current_loss_fn(y_pred, y_true)

        if not self.use_fuzzy_loss or not self.fuzzy_rules:
            return standard_loss

        total_fuzzy_loss = torch.tensor(0.0, device=y_pred.device)
        for rule_name, rule_module in self.fuzzy_rules.items():
            # Calculate the loss for one rule (already averaged over the batch)
            rule_loss = rule_module(y_pred).mean()
            
            # Weight it by its specific lambda and add to the total
            total_fuzzy_loss += self.lambdas[rule_name] * rule_loss

        return standard_loss + total_fuzzy_loss

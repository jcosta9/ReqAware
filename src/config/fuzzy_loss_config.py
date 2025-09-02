from dataclasses import dataclass, field
from typing import Dict, Optional

from models.registries import (
    CUSTOM_RULES_REGISTRY,
    TNORM_REGISTRY,
    TCONORM_REGISTRY,
    AAGGREGATION_REGISTRY,
    EAGGREGATION_REGISTRY,
)


@dataclass
class FuzzyLossOperators:
    t_norm: str = "godel_t_norm"
    t_conorm: str = "godel_t_conorm"
    a_aggregation: str = "godel_a_aggregation"
    e_aggregation: str = "godel_e_aggregation"

    def resolve(self):
        if self.t_norm in TNORM_REGISTRY:
            self.t_norm = TNORM_REGISTRY[self.t_norm]
        else:
            raise ValueError(f"Unknown TNORM {self.t_norm}")

        if self.t_conorm in TCONORM_REGISTRY:
            self.t_conorm = TCONORM_REGISTRY[self.t_conorm]
        else:
            raise ValueError(f"Unknown TCONORM {self.t_conorm}")

        if self.a_aggregation in AAGGREGATION_REGISTRY:
            self.a_aggregation = AAGGREGATION_REGISTRY[self.a_aggregation]
        else:
            raise ValueError(f"Unknown A Aggregation {self.a_aggregation}")

        if self.e_aggregation in EAGGREGATION_REGISTRY:
            self.e_aggregation = EAGGREGATION_REGISTRY[self.e_aggregation]
        else:
            raise ValueError(f"Unknown E Aggregation {self.e_aggregation}")


@dataclass
class FuzzyLossCustomRules:
    rule: str
    fuzzy_lambda: float = 0.5
    operators: FuzzyLossOperators = field(default_factory=FuzzyLossOperators)
    params: Dict = field(default_factory=dict)

    def resolve(self):
        if not (0 <= self.fuzzy_lambda <= 1):
            raise ValueError("fuzzy_lambda must be between 0 and 1")

        if self.rule in CUSTOM_RULES_REGISTRY:
            self.rule = CUSTOM_RULES_REGISTRY[self.rule]
        else:
            raise ValueError(f"Unknown Fuzzy Rule: {self.rule}")

        self.operators.resolve()


@dataclass
class FuzzyLossConfig:
    use_fuzzy_loss: bool = False
    rules: Optional[Dict[str, FuzzyLossCustomRules]] = None

    def resolve(self, parent_config):
        if self.use_fuzzy_loss and (self.rules is None or len(self.rules) == 0):
            raise ValueError(
                "custom_rules must be provided when use_fuzzy_loss is True"
            )
        if not self.use_fuzzy_loss and (self.rules is not None and len(self.rules) > 0):
            raise ValueError(
                "custom_rules should be None or empty when use_fuzzy_loss is False"
            )

        if self.rules is None:
            return

        resolved_rules = {}
        for rule_name, rule in self.rules.items():
            rule.resolve()
            resolved_rules[rule_name] = rule
        self.rules = resolved_rules

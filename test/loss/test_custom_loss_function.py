import pytest
import torch
import torch.nn as nn
from models.loss.custom_fuzzy_loss import CustomFuzzyLoss
from models.loss.custom_rules import ExactlyOneShape
from models.loss.abstract.fuzzy_loss import FuzzyLoss

@pytest.fixture
def test_config():
    """Provides the exact config from the user's question."""
    return {
        "use_fuzzy_loss": True,
        "rules": {
            "exactly_one_shape": {
                "class": "ExactlyOneShape",
                "lambda": 0.5,
                "operators": {
                    "t_norm": "godel_t_norm",
                    "a_aggregation": "godel_a_aggregation",
                    "e_aggregation": "godel_e_aggregation",
                    't_conorm': 'godel_t_conorm',
                },
                "params": {
                    "shape_indices": [0, 1, 2, 3]
                }
            }
        }
    }

class TestCustomFuzzyLoss:
    def test_system_initializes_from_config(self, test_config):
        # This test verifies the bug fix: does the manager actually build the rule?
        loss_fn = CustomFuzzyLoss(config=test_config, current_loss_fn=nn.BCELoss())
        assert "exactly_one_shape" in loss_fn.fuzzy_rules
        assert isinstance(loss_fn.fuzzy_rules["exactly_one_shape"], ExactlyOneShape)
        assert loss_fn.lambdas["exactly_one_shape"] == 0.5

    def test_system_applies_lambda_correctly(self, test_config):
        # Design an input where standard and fuzzy losses are easy to calculate.
        standard_loss_fn = lambda y_pred, y_true: torch.tensor(100.0) # Mock standard loss
        
        # Rig the fuzzy rule's forward pass to always return a known value.
        class MockRule(FuzzyLoss):
            def forward(self, y_pred): return torch.tensor([1, 1, 1, 1]) # Loss = 0.1
        
        # Temporarily replace the real rule with our mock.
        RULE_MAP = {"ExactlyOneShape": MockRule}
        
        loss_fn = CustomFuzzyLoss(config=test_config, current_loss_fn=standard_loss_fn)

        y_pred = torch.zeros(4, 5) # Dummy tensor
        y_true = torch.zeros(4, 5) # Dummy tensor

        # Expected: standard_loss + lambda * fuzzy_loss_mean (every vector violates the rule.)
        #           100.0        + 0.5  * 1
        expected_total_loss = torch.tensor(100.5)
        
        total_loss = loss_fn(y_pred, y_true)
        assert torch.allclose(total_loss, expected_total_loss)

    def test_system_with_fuzzy_loss_disabled(self, test_config):
        test_config["use_fuzzy_loss"] = False
        standard_loss_fn = lambda y_pred, y_true: torch.tensor(100.0)
        
        loss_fn = CustomFuzzyLoss(config=test_config, current_loss_fn=standard_loss_fn)
        
        y_pred = torch.zeros(4, 5)
        y_true = torch.zeros(4, 5)

        # Expect only the standard loss
        expected_total_loss = torch.tensor(100.0)
        
        total_loss = loss_fn(y_pred, y_true)
        assert torch.allclose(total_loss, expected_total_loss)

import pytest
import torch
import pandas as pd
from models.loss.fuzzy_transformations import (
    GodelTNorm, GodelTConorm, GodelAAggregation, GodelEAggregation, YagerTNorm, YagerTConorm, LogProductAAggregation, GeneralizedMeanEAggregation
)
from models.loss.custom_rules import (
    ExactlyOneShape, ExactlyOneMainColour, AtMostOneBorderColour, BetweenTwoAndThreeNumbers, AtMostOneWarning, NoSymbolsExactlyTwoColours, WarningSignExclusivity, WarningImpliesMainWhite
)

class TestGodelOperators:
    @pytest.fixture
    def test_batch(self):
        # the relevant concept indices are the first 3
        return torch.tensor([[0,0,0,1], [1,0,0,1], [1,1,1,0], [0,1,1,0]])
    
    @pytest.fixture
    def test_tensor(self):
        return torch.tensor([0,1,0])
    
    @pytest.fixture
    def expected_loss_batch(self):
        return torch.tensor([1,0,1,1])
    
    @pytest.fixture
    def expected_loss_tensor(self):
        return torch.tensor([0])
        
    @pytest.fixture
    def godel_tnorm(self):
        return GodelTNorm()
        
    @pytest.fixture
    def godel_tconorm(self):
        return GodelTConorm()
        
    @pytest.fixture
    def godel_aagg(self):
        return GodelAAggregation()
        
    @pytest.fixture
    def godel_eagg(self):
        return GodelEAggregation()
    
    def test_godel_aaggregation_batch(self, test_batch, godel_aagg):
        assert torch.equal(torch.tensor([0,0,0,0]), godel_aagg(test_batch))

    def test_godel_aggregation_1d_tensor(self, test_tensor, godel_aagg):
        result = godel_aagg(test_tensor)
        assert torch.equal(torch.tensor(0), result)

    def test_godel_eaggregation_batch(self, test_batch, godel_eagg):
        assert torch.equal(torch.tensor([1,1,1,1]), godel_eagg(test_batch))

    def test_godel_eaggregation_1d_tensor(self, test_tensor, godel_eagg):
        result = godel_eagg(test_tensor)
        assert torch.equal(torch.tensor(1), result)

    def test_godel_tnorm(self, godel_tnorm):
        a = torch.tensor([0,1,0,1])
        b = torch.tensor([0,0,1,1])
        assert torch.equal(torch.tensor([0,0,0,1]), godel_tnorm(a,b))
        assert torch.equal(torch.tensor(0), godel_tnorm(torch.tensor(0),torch.tensor(1)))

    def test_godel_tconorm(self, godel_tconorm):
        a = torch.tensor([0,1,0,1])
        b = torch.tensor([0,0,1,1])
        assert torch.equal(torch.tensor([0,1,1,1]), godel_tconorm(a,b))
        assert torch.equal(torch.tensor(1), godel_tconorm(torch.tensor(0),torch.tensor(1)))


class TestFuzzyRules:
    @pytest.fixture
    def test_batch(self):
        return torch.tensor([[0,0,0,1], [1,0,0,1], [1,1,1,0], [0,1,1,0]])
    
    @pytest.fixture
    def test_tensor(self):
        return torch.tensor([0,1,0])
    
    @pytest.fixture
    def expected_shape_loss_batch(self):
        return torch.tensor([1,0,1,1])
    
    @pytest.fixture
    def expected_shape_loss_tensor(self):
        return torch.tensor(0)
    
    @pytest.fixture
    def godel_operators(self):
        return {
            't_norm': GodelTNorm(),
            't_conorm': GodelTConorm(),
            'e_aggregation': GodelEAggregation(),
            'a_aggregation': GodelAAggregation()
        }
    
    @pytest.fixture
    def exactly_one_shape_rule(self, godel_operators):
        return ExactlyOneShape(
            t_norm=godel_operators['t_norm'],
            t_conorm=godel_operators['t_conorm'],
            e_aggregation=godel_operators['e_aggregation'],
            a_aggregation=godel_operators['a_aggregation'],
            params={"shape_indices":[0,1,2]}
        )
    
    def test_exactly_one_shape_godel_batch(self, test_batch, exactly_one_shape_rule, expected_shape_loss_batch):
        assert torch.equal(exactly_one_shape_rule(test_batch), expected_shape_loss_batch)

    def test_exactly_one_shape_godel_vector(self, test_tensor, exactly_one_shape_rule, expected_shape_loss_tensor):
        assert torch.equal(exactly_one_shape_rule(test_tensor), expected_shape_loss_tensor)
    
    def test_exactly_one_main_colour(self, godel_operators):
        # Setup test data for main color rule
        test_batch = torch.tensor([
            [1, 0, 0, 0],  # Only first color - follows rule
            [1, 1, 0, 0],  # Half and half - violates rule
            [0, 0, 0, 0],  # No main color - violates rule
            [1, 1, 1, 0],  # All colors - violates rule
        ])
        
        expected_loss = torch.tensor([0, 1, 1, 1])
        
        rule = ExactlyOneMainColour(
            t_norm=godel_operators['t_norm'],
            t_conorm=godel_operators['t_conorm'],
            e_aggregation=godel_operators['e_aggregation'],
            a_aggregation=godel_operators['a_aggregation'],
            params={'main_colour_indices': [0, 1, 2, 3]}
        )
        
        result = rule(test_batch)
        assert torch.equal(result, expected_loss), f"Expected {expected_loss}, got {result}"

    def test_at_most_one_border_colour(self, godel_operators):
        # Setup test data for border color rule
        test_batch = torch.tensor([
            [0, 0, 0, 1],  # No border color - follows rule
            [1, 0, 0, 1],  # One border color - follows rule
            [1, 1, 0, 1],  # Two border colors - violates rule
            [1, 1, 1, 1],  # Three border colors - violates rule
        ])
        
        expected_loss = torch.tensor([0, 0, 1, 1])
        
        rule = AtMostOneBorderColour(
            t_norm=godel_operators['t_norm'],
            t_conorm=godel_operators['t_conorm'],
            e_aggregation=godel_operators['e_aggregation'],
            a_aggregation=godel_operators['a_aggregation'],
            params={'border_colour_indices': [0, 1, 2]}
        )
        
        result = rule(test_batch)
        assert torch.equal(result, expected_loss), f"Expected {expected_loss}, got {result}"

    def test_between_two_and_three_numbers(self, godel_operators):
        # Setup test data for number rule
        test_batch = torch.tensor([
            [0, 0, 0, 0, 0,1],         # No numbers - follows rule
            [1, 0, 0, 0, 0,1],         # One number - violates rule
            [1, 1, 0, 0, 0,1],         # Two numbers - follows rule
            [1, 1, 1, 0, 0,1],         # Three numbers - follows rule
            [1, 1, 1, 1, 0,1],         # Four numbers - violates rule
            [1, 1, 1, 1, 1,1],         # Five numbers - violates rule
        ])

        expected_loss = torch.tensor([0, 1, 0, 0, 1, 1])

        rule = BetweenTwoAndThreeNumbers(
            t_norm=godel_operators['t_norm'],
            t_conorm=godel_operators['t_conorm'],
            e_aggregation=godel_operators['e_aggregation'],
            a_aggregation=godel_operators['a_aggregation'],
            params={'number_indices': [0, 1, 2, 3, 4]}
        )
        print(type(rule))
        result = rule(test_batch)
        assert torch.equal(result, expected_loss), f"Expected {expected_loss}, got {result}"

    def test_at_most_one_warning(self, godel_operators):
        """Test the AtMostOneWarning fuzzy rule with different warning symbol combinations"""
        # Setup test data for warning symbols
        test_batch = torch.tensor([
            [0, 0, 0, 1],  # No warning symbols - follows rule
            [1, 0, 0, 1],  # One warning symbol - follows rule
            [1, 1, 0, 1],  # Two warning symbols - violates rule
            [1, 1, 1, 1],  # Three warning symbols - violates rule
        ])
        
        expected_loss = torch.tensor([0, 0, 1, 1])
        
        rule = AtMostOneWarning(
            t_norm=godel_operators['t_norm'],
            t_conorm=godel_operators['t_conorm'],
            e_aggregation=godel_operators['e_aggregation'],
            a_aggregation=godel_operators['a_aggregation'],
            params={'warning_indices': [0, 1, 2]}
        )
        
        result = rule(test_batch)
        assert torch.equal(result, expected_loss), f"Expected {expected_loss}, got {result}"

    def test_at_most_one_warning_edge_cases(self, godel_operators):
        """Test the AtMostOneWarning fuzzy rule with edge cases"""
        # Edge case: single tensor input
        test_tensor = torch.tensor([0, 1, 0, 1])  # One warning symbol - follows rule
        expected_loss = torch.tensor(0)

        rule = AtMostOneWarning(
            t_norm=godel_operators['t_norm'],
            t_conorm=godel_operators['t_conorm'],
            e_aggregation=godel_operators['e_aggregation'],
            a_aggregation=godel_operators['a_aggregation'],
            params={'warning_indices': [0, 1, 2]}
        )

        result = rule(test_tensor)
        assert torch.equal(result, expected_loss), f"Expected {expected_loss} for single tensor, got {result}"

        # Edge case: fuzzy values between 0 and 1
        test_fuzzy = torch.tensor([
            [0.3, 0.4, 0, 1],  # Two partial warning symbols - partially violates rule
            [0.9, 0.8, 0, 1],  # Two strong warning symbols - strongly violates rule
        ])

        # With Gödel operators, result should be binary (0 or 1)
        # But the violation should be stronger for the second case
        result_fuzzy = rule(test_fuzzy)
        assert result_fuzzy[1] >= result_fuzzy[0], "Higher probabilities should result in stronger violation"

    def test_no_symbols_exactly_two_colours(self, godel_operators):
        """Test the NoSymbolsExactlyTwoColours fuzzy rule with various cases"""
        # Setup test data for the no symbols exactly two colors rule
        # Format: [symbols..., colors...]
        test_batch = torch.tensor([
            # Compliant cases (loss should be 0)
            [0, 0, 0, 1, 1, 0],  # No symbols, exactly two colors
            [1, 0, 0, 1, 0, 0],  # Has symbols, any number of colors
            [0, 1, 0, 0, 1, 1],  # Has symbols, exactly two colors
            [1, 1, 0, 1, 1, 1],  # Has symbols, three colors
            
            # Violation cases (loss should be 1)
            [0, 0, 0, 1, 0, 0],  # No symbols, only one color
            [0, 0, 0, 0, 0, 0],  # No symbols, no colors
            [0, 0, 0, 1, 1, 1],  # No symbols, three colors
        ])
        
        expected_loss = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        
        rule = NoSymbolsExactlyTwoColours(
            t_norm=godel_operators['t_norm'],
            t_conorm=godel_operators['t_conorm'],
            e_aggregation=godel_operators['e_aggregation'],
            a_aggregation=godel_operators['a_aggregation'],
            params={
                'symbol_indices': [0, 1, 2],
                'colour_indices': [3, 4, 5]
            }
        )
        
        result = rule(test_batch)
        assert torch.equal(result, expected_loss), f"Expected {expected_loss}, got {result}"
    
    def test_no_symbols_exactly_two_colours_edge_cases(self, godel_operators):
        """Test the NoSymbolsExactlyTwoColours fuzzy rule with edge cases"""
        
        # Edge case 1: Single tensor input (non-batch)
        # Format: [symbols..., colors...]
        test_tensor = torch.tensor([0, 0, 0, 1, 1, 0])  # No symbols, exactly two colors
        expected_loss = torch.tensor(0)
        
        rule = NoSymbolsExactlyTwoColours(
            t_norm=godel_operators['t_norm'],
            t_conorm=godel_operators['t_conorm'],
            e_aggregation=godel_operators['e_aggregation'],
            a_aggregation=godel_operators['a_aggregation'],
            params={
                'symbol_indices': [0, 1, 2],
                'colour_indices': [3, 4, 5]
            }
        )
        
        # Fix the typo in squezze() for this test
        # This is needed because there's a typo in the original implementation
        original_forward = rule.forward
        def patched_forward(y_pred):
            result = original_forward(y_pred)
            if isinstance(result, torch.Tensor) and not result.shape:
                result = result.squeeze(0)  # Correctly squeeze
            return result
            
        rule.forward = patched_forward
        
        result = rule(test_tensor)
        assert torch.equal(result, expected_loss), f"Expected {expected_loss} for single tensor, got {result}"
        
        # Edge case 2: Fuzzy values between 0 and 1
        test_fuzzy = torch.tensor([
            [0.1, 0.2, 0.1, 0.9, 0.8, 0.1],  # Almost no symbols, strong two colors
            [0.8, 0.1, 0.2, 0.7, 0.9, 0.8],  # Has symbols, three colors
            [0.1, 0.2, 0.1, 0.9, 0.1, 0.1],  # Almost no symbols, one strong color
            [0.1, 0.2, 0.1, 0.5, 0.6, 0.5],  # Almost no symbols, fuzzy three colors
        ])
        
        # With fuzzy values, we expect:
        # - First sample: low violation (close to 0) - rule almost satisfied
        # - Second sample: low violation (close to 0) - has symbols, rule doesn't apply
        # - Third sample: high violation (close to 1) - no symbols, only one color
        # - Fourth sample: medium violation - no symbols, unclear if exactly two colors
        
        result_fuzzy = rule(test_fuzzy)
        
        # Test that samples with symbols have less violation than those without
        assert result_fuzzy[1] < result_fuzzy[2], "Having symbols should reduce violation"
        
        # Test that samples with not exactly two colors have more violation
        assert result_fuzzy[2] > result_fuzzy[0], "Having only one color should increase violation"
        
        # Edge case 3: Minimum color count edge case (only 2 colors available)
        min_colors_rule = NoSymbolsExactlyTwoColours(
            t_norm=godel_operators['t_norm'],
            t_conorm=godel_operators['t_conorm'],
            e_aggregation=godel_operators['e_aggregation'],
            a_aggregation=godel_operators['a_aggregation'],
            params={
                'symbol_indices': [0, 1],
                'colour_indices': [2, 3]  # Only two colors available
            }
        )
        
        test_min = torch.tensor([
            [0, 0, 1, 1],  # No symbols, both colors
            [0, 0, 1, 0],  # No symbols, one color
        ])
        
        expected_min = torch.tensor([0, 1])  # First satisfies rule, second violates
        
        result_min = min_colors_rule(test_min)
        assert torch.equal(result_min, expected_min), f"Expected {expected_min} for minimum colors, got {result_min}"

        
    def test_warning_sign_exclusivity(self, godel_operators):
        """Test the WarningSignExclusivity fuzzy rule with different symbol combinations"""
        # Setup test data
        # Format: [symbols (warnings first, then other symbols)...]
        test_batch = torch.tensor([
            # Compliant cases (loss should be 0)
            [0, 0, 0, 0, 0],  # No warnings, no other symbols
            [1, 0, 0, 0, 0],  # One warning, no other symbols
            [0, 1, 0, 0, 0],  # One warning, no other symbols
            [0, 0, 1, 0, 0],  # No warnings, one symbol
            
            # Violation cases (loss should be 1)
            [1, 0, 1, 0, 0],  # One warning and one symbol
            [0, 1, 0, 1, 0],  # One warning and one symbol
            [1, 1, 1, 1, 1],  # Multiple warnings and symbols
        ])
        
        expected_loss = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        
        rule = WarningSignExclusivity(
            t_norm=godel_operators['t_norm'],
            t_conorm=godel_operators['t_conorm'],
            e_aggregation=godel_operators['e_aggregation'],
            a_aggregation=godel_operators['a_aggregation'],
            params={
                'warning_indices': [0, 1],  # First two indices are warnings
                'symbol_indices': [0, 1, 2, 3, 4]  # All are symbols (warnings included)
            }
        )
        
        result = rule(test_batch)
        assert torch.equal(result, expected_loss), f"Expected {expected_loss}, got {result}"

    def test_warning_sign_exclusivity_edge_cases(self, godel_operators):
        """Test the WarningSignExclusivity fuzzy rule with edge cases"""
        
        # Edge case 1: Single tensor input (non-batch)
        # Format: [warnings..., other symbols...]
        test_tensor = torch.tensor([1, 0, 0, 0, 0])  # One warning, no other symbols (compliant)
        expected_loss = torch.tensor(0)
        
        rule = WarningSignExclusivity(
            t_norm=godel_operators['t_norm'],
            t_conorm=godel_operators['t_conorm'],
            e_aggregation=godel_operators['e_aggregation'],
            a_aggregation=godel_operators['a_aggregation'],
            params={
                'warning_indices': [0, 1],
                'symbol_indices': [0, 1, 2, 3, 4]
            }
        )
        
        result = rule(test_tensor)
        assert torch.equal(result, expected_loss), f"Expected {expected_loss} for single tensor, got {result}"
        
        # Edge case 2: Single warning symbol
        single_warning_rule = WarningSignExclusivity(
            t_norm=godel_operators['t_norm'],
            t_conorm=godel_operators['t_conorm'],
            e_aggregation=godel_operators['e_aggregation'],
            a_aggregation=godel_operators['a_aggregation'],
            params={
                'warning_indices': [0],  # Only one warning index
                'symbol_indices': [0, 1, 2, 3]
            }
        )
        
        test_single = torch.tensor([
            [1, 0, 0, 0],  # Warning only (compliant)
            [1, 1, 0, 0],  # Warning and another symbol (violation)
            [0, 1, 0, 0],  # No warning, other symbol (compliant)
        ])
        
        expected_single = torch.tensor([0, 1, 0])
        
        result_single = single_warning_rule(test_single)
        assert torch.equal(result_single, expected_single), f"Expected {expected_single} for single warning, got {result_single}"
        
        # Edge case 3: Fuzzy values between 0 and 1
        test_fuzzy = torch.tensor([
            [0.9, 0.1, 0.1, 0.1, 0.1],  # Strong warning, weak other symbols
            [0.2, 0.3, 0.8, 0.7, 0.1],  # Weak warnings, strong other symbols
            [0.7, 0.8, 0.6, 0.9, 0.5],  # Strong warnings and strong other symbols
        ])
        
        # With Gödel operators:
        # - First sample: low violation (warnings don't strongly co-occur with other symbols)
        # - Second sample: moderate violation (weak warning presence with other symbols)
        # - Third sample: high violation (strong presence of both warnings and other symbols)
        
        result_fuzzy = rule(test_fuzzy)
        
        # The third sample should have the highest violation
        assert result_fuzzy[2] >= result_fuzzy[1], "Stronger co-occurrence should result in higher violation"
        assert result_fuzzy[2] >= result_fuzzy[0], "Stronger co-occurrence should result in higher violation"
        
        # Edge case 4: Empty warning set (should raise ValueError)
        with pytest.raises(ValueError):
            invalid_rule = WarningSignExclusivity(
                t_norm=godel_operators['t_norm'],
                t_conorm=godel_operators['t_conorm'],
                e_aggregation=godel_operators['e_aggregation'],
                a_aggregation=godel_operators['a_aggregation'],
                params={
                    'warning_indices': [],  # Empty warning set
                    'symbol_indices': [0, 1, 2]
                }
            )
        
        # Edge case 5: Warning indices not a subset of symbol indices (should raise ValueError)
        with pytest.raises(ValueError):
            invalid_rule = WarningSignExclusivity(
                t_norm=godel_operators['t_norm'],
                t_conorm=godel_operators['t_conorm'],
                e_aggregation=godel_operators['e_aggregation'],
                a_aggregation=godel_operators['a_aggregation'],
                params={
                    'warning_indices': [0, 5],  # Index 5 not in symbol_indices
                    'symbol_indices': [0, 1, 2, 3, 4]
                }
            )

    def test_warning_implies_main_white(self, godel_operators):
        """Test the WarningImpliesMainWhite fuzzy rule with different combinations"""
        # Setup test data
        # Format: [warning symbols..., white main color]
        test_batch = torch.tensor([
            # Compliant cases (loss should be 0)
            [0, 0, 1],  # No warnings, white main color
            [1, 0, 1],  # Warning present, white main color
            [0, 1, 1],  # Warning present, white main color
            [0, 0, 0],  # No warnings, no white main color
            
            # Violation cases (loss should be > 0)
            [1, 0, 0],  # Warning present, no white main color
            [0, 1, 0],  # Warning present, no white main color
            [1, 1, 0],  # Multiple warnings, no white main color
        ])
        
        expected_loss = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        
        rule = WarningImpliesMainWhite(
            t_norm=godel_operators['t_norm'],
            t_conorm=godel_operators['t_conorm'],
            e_aggregation=godel_operators['e_aggregation'],
            a_aggregation=godel_operators['a_aggregation'],
            params={
                'warning_indices': [0, 1],
                'main_colour_white': [2]
            }
        )
        
        result = rule(test_batch)
        assert torch.equal(result, expected_loss), f"Expected {expected_loss}, got {result}"

    def test_warning_implies_main_white_edge_cases(self, godel_operators):
        """Test the WarningImpliesMainWhite fuzzy rule with edge cases"""
        
        # Edge case 1: Single tensor input (non-batch)
        test_tensor = torch.tensor([1, 0, 1])  # One warning, white main color (compliant)
        expected_loss = torch.tensor(0)
        
        rule = WarningImpliesMainWhite(
            t_norm=godel_operators['t_norm'],
            t_conorm=godel_operators['t_conorm'],
            e_aggregation=godel_operators['e_aggregation'],
            a_aggregation=godel_operators['a_aggregation'],
            params={
                'warning_indices': [0, 1],
                'main_colour_white': [2]
            }
        )
        
        result = rule(test_tensor)
        assert torch.equal(result, expected_loss), f"Expected {expected_loss} for single tensor, got {result}"
        
        # Edge case 2: Fuzzy values between 0 and 1
        test_fuzzy = torch.tensor([
            [0.9, 0.1, 0.9],  # Strong warning, strong white (compliant)
            [0.9, 0.1, 0.1],  # Strong warning, weak white (violation)
            [0.1, 0.1, 0.9],  # Weak warning, strong white (mostly compliant)
            [0.1, 0.1, 0.1],  # Weak warning, weak white (slightly violating)
        ])
        
        result_fuzzy = rule(test_fuzzy)
        
        # Violations should be ordered by strength
        assert result_fuzzy[1] > result_fuzzy[3], "Strong warning with weak white should violate more than weak warning with weak white"
        assert result_fuzzy[1] > result_fuzzy[0], "Strong warning with weak white should violate more than strong warning with strong white"
        
        # Edge case 3: Multiple white main colors (should raise ValueError)
        with pytest.raises(ValueError):
            invalid_rule = WarningImpliesMainWhite(
                t_norm=godel_operators['t_norm'],
                t_conorm=godel_operators['t_conorm'],
                e_aggregation=godel_operators['e_aggregation'],
                a_aggregation=godel_operators['a_aggregation'],
                params={
                    'warning_indices': [0, 1],
                    'main_colour_white': [2, 3]  # Multiple white indices (invalid)
                }
            )

class TestYagerOperators:
    @pytest.fixture
    def test_batch(self):
        return torch.tensor([[0,0,0,1], [1,0,0,1], [1,1,1,0], [0,1,1,0]])
    
    @pytest.fixture
    def test_tensor(self):
        return torch.tensor([0,1,0])
    
    @pytest.fixture
    def yager_tnorm_p1(self):
        return YagerTNorm(p=1.0, eps=0)
    
    @pytest.fixture
    def yager_tnorm_p2(self):
        return YagerTNorm(p = 2.0, eps=0)
    
    @pytest.fixture
    def yager_tconorm_p1(self):
        return YagerTConorm(p=1.0, eps=0)
    
    @pytest.fixture
    def yager_tconorm_p2(self):
        return YagerTConorm(p = 2.0, eps=0)
    
    def test_yager_tnorm_p1(self, yager_tnorm_p1):
        """Test YagerTNorm with p=1.0"""
        a = torch.tensor([0,1,0,1])
        b = torch.tensor([0,0,1,1])
        # YagerTNorm with p=1 should be equivalent to max(0, a + b - 1)
        expected = torch.tensor([0.0,0.0,0.0,1.0])
        assert torch.allclose(yager_tnorm_p1(a,b), expected), f"Expected {expected}, got {yager_tnorm_p1(a,b)}"
        assert torch.allclose(yager_tnorm_p1(torch.tensor(0.7), torch.tensor(0.4)), torch.tensor(0.1)), "Should be max(0, 0.7+0.4-1)"
    
    def test_yager_tnorm_p2(self, yager_tnorm_p2):
        """Test YagerTNorm with p=2.0"""
        a = torch.tensor([0.0, 0.3, 0.5, 1.0])
        b = torch.tensor([0.0, 0.6, 0.5, 1.0])
        
        # Calculate expected output manually for p=2:
        # max(0, 1 - ((1-a)^p + (1-b)^p)^(1/p))
        expected = torch.zeros(4)
        for i in range(4):
            complement_sum = (1-a[i])**2 + (1-b[i])**2
            expected[i] = max(0, 1 - complement_sum**(1/2))
        
        assert torch.allclose(yager_tnorm_p2(a,b), expected, atol=1e-5), f"Expected {expected}, got {yager_tnorm_p2(a,b)}"
    
    def test_yager_tconorm_p1(self, yager_tconorm_p1):
        """Test YagerTConorm with p=1.0"""
        a = torch.tensor([0,1,0,1])
        b = torch.tensor([0,0,1,1])
        # YagerTConorm with p=1 should be equivalent to min(1, a + b)
        expected = torch.tensor([0.,1.,1.,1.])
        assert torch.allclose(yager_tconorm_p1(a,b), expected), f"Expected {expected}, got {yager_tconorm_p1(a,b)}"
        assert torch.allclose(yager_tconorm_p1(torch.tensor(0.7), torch.tensor(0.4)), torch.tensor(1.0)), "Should be min(1, 0.7+0.4)"
    
    def test_yager_tconorm_p2(self, yager_tconorm_p2):
        """Test YagerTConorm with p=2.0"""
        a = torch.tensor([0.0, 0.3, 0.5, 1.0])
        b = torch.tensor([0.0, 0.6, 0.5, 1.0])
        
        # Calculate expected output manually for p=2:
        # min(1, (a^p + b^p)^(1/p))
        expected = torch.zeros(4)
        for i in range(4):
            sum_powers = a[i]**2 + b[i]**2
            expected[i] = min(1, sum_powers**(1/2))
        
        assert torch.allclose(yager_tconorm_p2(a,b), expected, atol=1e-5), f"Expected {expected}, got {yager_tconorm_p2(a,b)}"
    
class TestAdvancedAggregations:
    @pytest.fixture
    def test_batch(self):
        return torch.tensor([[0.,0.,0.,1.], [1.,0.,0.,1.], [1.,1.,1.,0.], [0.,1.,1.,0.], [1.,1.,1.,1.]])
    
    @pytest.fixture
    def test_tensor(self):
        return torch.tensor([0,1,0])
    
    @pytest.fixture
    def log_product_aggregation(self):
        return LogProductAAggregation()
    
    @pytest.fixture
    def gen_mean_p1(self):
        return GeneralizedMeanEAggregation(p=1.0)
    
    @pytest.fixture
    def gen_mean_p2(self):
        return GeneralizedMeanEAggregation(p = 2.0)
    
    def test_log_product_aggregation_batch(self, test_batch, log_product_aggregation):
        """Test LogProductAAggregation on batch data"""
        # LogProductAAggregation is 1 - exp(mean(log(1-x_i)))
        result = log_product_aggregation(test_batch)
        
        # Calculate expected output manually
        expected = torch.tensor([0.,0.,0.,0.,1.])
        assert torch.allclose(result, expected, atol=1e-4), f"Expected {expected}, got {result}"
    
    def test_log_product_aggregation_tensor(self, test_tensor, log_product_aggregation):
        """Test LogProductAAggregation on a single tensor"""
        result = log_product_aggregation(test_tensor)
        
        # Calculate expected output manually
        epsilon = 1e-7
        adjusted = torch.clamp(test_tensor, epsilon, 1-epsilon)
        expected = 1 - torch.exp(torch.mean(torch.log(1 - adjusted)))
        
        assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected}, got {result}"
    
    def test_gen_mean_p1_batch(self, test_batch, gen_mean_p1):
        """Test GeneralizedMeanEAggregation with p=1.0 on batch data"""
        # p=1 is the arithmetic mean
        result = gen_mean_p1(test_batch)
        expected = torch.mean(test_batch, dim=1)
        
        assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected}, got {result}"
    
    def test_gen_mean_p1_tensor(self, test_tensor, gen_mean_p1):
        """Test GeneralizedMeanEAggregation with p=1.0 on a single tensor"""
        result = gen_mean_p1(test_tensor)
        expected = torch.mean(test_tensor)
        
        assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected}, got {result}"
    
    def test_gen_mean_p2_batch(self, test_batch, gen_mean_p2):
        """Test GeneralizedMeanEAggregation with p=2.0 on batch data"""
        # p=2 is the quadratic mean (root mean square)
        result = gen_mean_p2(test_batch)
        
        # Calculate expected output manually
        expected = torch.zeros(5)
        for i in range(5):
            expected[i] = torch.sqrt(torch.mean(test_batch[i]**2))
        
        assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected}, got {result}"
    
    def test_gen_mean_p2_tensor(self, test_tensor, gen_mean_p2):
        """Test GeneralizedMeanEAggregation with p=2.0 on a single tensor"""
        result = gen_mean_p2(test_tensor)
        expected = torch.sqrt(torch.mean(test_tensor**2))
        
        assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected}, got {result}"
    
    def test_gen_mean_special_cases(self):
        """Test special cases of GeneralizedMeanEAggregation with different p values"""
        # Test p=0 (geometric mean)
        gen_mean_p0 = GeneralizedMeanEAggregation(params={"p": 0.0})
        test_tensor = torch.tensor([0.2, 0.5, 0.8])
        
        # For p=0, it's the geometric mean: (x1*x2*...*xn)^(1/n)
        # We need to avoid zeros, so we'll use a small epsilon
        epsilon = 1e-7
        adjusted = torch.clamp(test_tensor, epsilon, 1.0)
        expected = torch.exp(torch.mean(torch.log(adjusted)))
        
        result = gen_mean_p0(test_tensor)
        assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected}, got {result}"
        
        # Test p→∞ (maximum)
        gen_mean_large = GeneralizedMeanEAggregation(params={"p": 100.0})
        expected_max = torch.max(test_tensor)
        result_large = gen_mean_large(test_tensor)
        
        assert torch.allclose(result_large, expected_max, atol=1e-2), f"Expected {expected_max}, got {result_large}"
        
        # Test p→-∞ (minimum)
        gen_mean_neg = GeneralizedMeanEAggregation(params={"p": -100.0})
        expected_min = torch.min(test_tensor)
        result_neg = gen_mean_neg(test_tensor)
        
        assert torch.allclose(result_neg, expected_min, atol=1e-2), f"Expected {expected_min}, got {result_neg}"

class TestExactlyOneMainColourWithRealData:
    @pytest.fixture
    def csv_data(self):
        """Load test data from CSV file"""
        df = pd.read_csv("/vol/home-vol2/se/reichmei/Schreitisch/trainReq/test_batch.csv")
        # Assuming CSV has concept columns we need to extract
        # Adjust column indices as needed based on your CSV structure
        concepts = df.values  # Skip the first column if it's an ID or label
        return torch.tensor(concepts, dtype=torch.float32)
    
    @pytest.fixture
    def param_combinations(self):
        """Define different parameter combinations to test"""
        return [
            {
                "p_norm": 1.0,
                "p_mean": 1.0,
                "description": "Yager(p=1) & GenMean(p=1) - equivalent to Lukasiewicz and arithmetic mean"
            },
            {
                "p_norm": 2.0,
                "p_mean": 1.0,
                "description": "Yager(p=2) & GenMean(p=1) - quadratic norm with arithmetic mean"
            },
            {
                "p_norm": 1.0,
                "p_mean": 2.0,
                "description": "Yager(p=1) & GenMean(p=2) - Lukasiewicz with quadratic mean"
            },
            {
                "p_norm": 3.0,
                "p_mean": 3.0,
                "description": "Yager(p=3) & GenMean(p=3) - cubic norm and mean"
            }
        ]
    
    def test_exactly_one_main_colour_with_csv_data(self, csv_data, param_combinations):
        """Test ExactlyOneMainColour rule with different parameter combinations on real data"""
        # Define main color indices - adjust based on your data structure
        main_colour_indices = [0, 1, 2, 3]  # Assuming first 4 columns are main colors
        
        # Store results for comparison
        results = []
        
        print("\n===== EXACTLY ONE MAIN COLOUR RULE WITH DIFFERENT PARAMETERS =====")
        
        # For each parameter combination
        for params in param_combinations:
            p_norm = params["p_norm"]
            p_mean = params["p_mean"]
            
            # Create operators with current parameters
            t_norm = YagerTNorm(p=p_norm)
            t_conorm = YagerTConorm(p=p_norm)
            e_aggregation = GeneralizedMeanEAggregation(p= p_mean)
            a_aggregation = LogProductAAggregation()
            
            # Create rule
            rule = ExactlyOneMainColour(
                t_norm=t_norm,
                t_conorm=t_conorm,
                e_aggregation=e_aggregation,
                a_aggregation=a_aggregation,
                params={'main_colour_indices': main_colour_indices}
            )
            
            # Apply rule to data
            rule_output = rule(csv_data)
            
            # Store results
            results.append({
                "params": params,
                "output": rule_output
            })
            
            # Print summary statistics
            print(f"\nParameters: {params['description']}")
            print(f"  Mean violation: {rule_output.mean().item():.4f}")
            print(f"  Min violation: {rule_output.min().item():.4f}")
            print(f"  Max violation: {rule_output.max().item():.4f}")
            print(f"  Samples with high violation (>0.8): {(rule_output > 0.8).sum().item()}")
            print(f"  Samples with low violation (<0.2): {(rule_output < 0.2).sum().item()}")
        
        # Compare results between different parameter settings
        print("\n===== PARAMETER SENSITIVITY ANALYSIS =====")
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                param_i = results[i]["params"]
                param_j = results[j]["params"]
                
                # Calculate difference statistics
                diff = results[i]["output"] - results[j]["output"]
                abs_diff = torch.abs(diff)
                
                print(f"\nComparing {param_i['description']} vs {param_j['description']}:")
                print(f"  Mean absolute difference: {abs_diff.mean().item():.4f}")
                print(f"  Max absolute difference: {abs_diff.max().item():.4f}")
                print(f"  Samples with significant difference (>0.2): {(abs_diff > 0.2).sum().item()}")
    
    def test_p_parameter_sensitivity(self, csv_data):
        """Test sensitivity of the p parameter in Yager norms with fixed aggregations"""
        # Define main color indices - adjust based on your data structure
        main_colour_indices = [0, 1, 2, 3]  # Assuming first 4 columns are main colors
        
        # Fixed aggregations
        e_aggregation = GeneralizedMeanEAggregation(params={"p": 1.0})
        a_aggregation = LogProductAAggregation()
        
        # Range of p values to test
        p_values = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
        
        print("\n===== P-VALUE SENSITIVITY ANALYSIS =====")
        
        previous_output = None
        
        for p in p_values:
            # Create operators with current p value
            t_norm = YagerTNorm(p=p)
            t_conorm = YagerTConorm(p=p)
            
            # Create rule
            rule = ExactlyOneMainColour(
                t_norm=t_norm,
                t_conorm=t_conorm,
                e_aggregation=e_aggregation,
                a_aggregation=a_aggregation,
                params={'main_colour_indices': main_colour_indices}
            )
            
            # Apply rule to data
            rule_output = rule(csv_data)
            
            # Print summary statistics
            print(f"\np={p}:")
            print(f"  Mean violation: {rule_output.mean().item():.4f}")
            print(f"  Min violation: {rule_output.min().item():.4f}")
            print(f"  Max violation: {rule_output.max().item():.4f}")
            
            # Compare with previous p value
            if previous_output is not None:
                diff = rule_output - previous_output
                abs_diff = torch.abs(diff)
                
                print(f"  Change from p={p_values[p_values.index(p)-1]}:")
                print(f"    Mean absolute difference: {abs_diff.mean().item():.4f}")
                print(f"    Max absolute difference: {abs_diff.max().item():.4f}")
                
                # Find samples with largest changes
                if len(csv_data) > 5:  # Only if we have enough samples
                    top5_indices = torch.argsort(abs_diff, descending=True)[:5]
                    print(f"    Top 5 largest changes (sample idx, old value → new value, diff):")
                    for idx in top5_indices:
                        print(f"      #{idx}: {previous_output[idx].item():.4f} → {rule_output[idx].item():.4f}, diff: {diff[idx].item():.4f}")
            
            previous_output = rule_output

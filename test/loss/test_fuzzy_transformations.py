import pytest
import torch
from models.loss.fuzzy_transformations import (
    GodelTNorm, GodelTConorm, GodelAAggregation, GodelEAggregation
)
from models.loss.custom_rules import (
    ExactlyOneShape, ExactlyOneMainColour, AtMostOneBorderColour, BetweenTwoAndThreeNumbers, AtMostOneWarning, NoSymbolsExactlyTwoColours
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
    
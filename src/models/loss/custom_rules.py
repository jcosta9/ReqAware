import torch
from .abstract.fuzzy_loss import FuzzyLoss
from .abstract.fuzzy_transformation_abstract import Tnorm, Tconorm, Aggregation


class ExactlyOneShape(FuzzyLoss):
    def __init__(
        self,
        t_norm: Tnorm,
        t_conorm: Tconorm,
        e_aggregation: Aggregation,
        a_aggregation: Aggregation,
        params: dict,
    ):
        super().__init__(t_norm, t_conorm, e_aggregation, a_aggregation)

        self.shape_indices = params.get("shape_indices", {})
        if len(self.shape_indices) == 0:
            raise ValueError(
                "ExactlyOneShape fuzzy rule requires a list of shape indices as params"
            )

    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Collects t-norm values using an explicit for loop."""

        # Track if we need to squeeze at the end
        was_unsqueezed = False
        # tricking around to fix batch size
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
            was_unsqueezed = True

        # Isolate the concept probabilities we're working with
        concept_probs = y_pred[:, self.shape_indices]
        batch_size, num_concepts = concept_probs.shape
        batch_losses = []

        for i in range(num_concepts):
            # The concept we are focusing on in this iteration
            other_concepts = torch.cat(
                [concept_probs[:, :i], concept_probs[:, i + 1 :]], dim=1
            )
            # Negate them
            negated_other_concepts = 1.0 - other_concepts
            # Apply the universal aggregation (e.g., min) over the "other" concepts
            all_aggregation = self.a_aggregation(negated_other_concepts)
            # Apply the t-norm between the current concept and the aggregation
            t_norm = self.t_norm(concept_probs[:, i], all_aggregation)
            # Add the result for this iteration to our list
            batch_losses.append(t_norm.unsqueeze(1))

        batch_losses = torch.cat(batch_losses, dim=1)
        exist_agg = self.e_aggregation(batch_losses)

        if was_unsqueezed:
            exist_agg = exist_agg.squeeze(0)
        return 1 - exist_agg

class ExactlyOneMainColour(FuzzyLoss):
    """This class implements the ExactlyOneMainColour fuzzy rule."""
    def __init__(
        self,
        t_norm: Tnorm,
        t_conorm: Tconorm,
        e_aggregation: Aggregation,
        a_aggregation: Aggregation,
        params: dict,
    ):
        super().__init__(t_norm, t_conorm, e_aggregation, a_aggregation)

        self.main_colour_indices = params.get("main_colour_indices", {})
        if len(self.main_colour_indices) == 0:
            raise ValueError(
                "ExactlyOneMainColour fuzzy rule requires a list of main colour indices as params"
            )

    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Collects t-norm values using an explicit for loop."""

        # Track if we need to squeeze at the end
        was_unsqueezed = False
        # tricking around to fix batch size
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
            was_unsqueezed = True

        # Isolate the concept probabilities we're working with
        concept_probs = y_pred[:, self.main_colour_indices]
        batch_size, num_concepts = concept_probs.shape
        batch_losses = []

        for i in range(num_concepts):
            # The concept we are focusing on in this iteration
            other_concepts = torch.cat(
                [concept_probs[:, :i], concept_probs[:, i + 1 :]], dim=1
            )
            # Negate them
            negated_other_concepts = 1.0 - other_concepts
            # Apply the universal aggregation (e.g., min) over the "other" concepts
            all_aggregation = self.a_aggregation(negated_other_concepts)
            # Apply the t-norm between the current concept and the aggregation
            t_norm = self.t_norm(concept_probs[:, i], all_aggregation)
            # Add the result for this iteration to our list
            batch_losses.append(t_norm.unsqueeze(1))

        batch_losses = torch.cat(batch_losses, dim=1)
        exist_agg = self.e_aggregation(batch_losses)

        if was_unsqueezed:
            exist_agg = exist_agg.squeeze(0)
        return 1 - exist_agg

class AtMostOneBorderColour(FuzzyLoss):
    """This class implements the ExactlyOneMainColour fuzzy rule."""
    def __init__(
        self,
        t_norm: Tnorm,
        t_conorm: Tconorm,
        e_aggregation: Aggregation,
        a_aggregation: Aggregation,
        params: dict,
    ):
        super().__init__(t_norm, t_conorm, e_aggregation, a_aggregation)

        self.border_colour_indices = params.get("border_colour_indices", {})
        if len(self.border_colour_indices) == 0:
            raise ValueError(
                "AtMostOneBorderColour fuzzy rule requires a list of main colour indices as params"
            )

    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Collects t-norm values using an explicit for loop."""

        # Track if we need to squeeze at the end
        was_unsqueezed = False
        # tricking around to fix batch size
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
            was_unsqueezed = True

        # Isolate the concept probabilities we're working with
        concept_probs = y_pred[:, self.border_colour_indices]
        batch_size, num_concepts = concept_probs.shape
        batch_losses = []

        for i in range(num_concepts):
            other_concepts = torch.cat(
                [concept_probs[:, :i], concept_probs[:, i + 1 :]], dim=1
            )
            # creating the pairs between current concept and other concepts
            pairs = []
            for j in range(other_concepts.shape[1]):
                # unsqueezing the tensors to allow for concate later
                tensor_i = concept_probs[:, i].unsqueeze(1) 
                tensor_j = other_concepts[:, j].unsqueeze(1)
                t_norm = self.t_norm(tensor_i, tensor_j)
                negation = 1 - t_norm
                pairs.append(negation)
            pairs = torch.cat(pairs, dim=1)
            batch_losses.append(pairs)
        # Apply the universal aggregation (e.g., godel) over the "other" concepts
        batch_losses = torch.cat(batch_losses, dim=1)
        all_aggregation = self.a_aggregation(batch_losses)

        # accounting for single tensor inputs
        if was_unsqueezed:
            all_aggregation = all_aggregation.squeeze(0)
        return 1 - all_aggregation

class BetweenTwoAndThreeNumbers(FuzzyLoss):
    """This class implements the """
    def __init__(
        self,
        t_norm: Tnorm,
        t_conorm: Tconorm,
        e_aggregation: Aggregation,
        a_aggregation: Aggregation,
        params: dict,
    ):
        super().__init__(t_norm, t_conorm, e_aggregation, a_aggregation)

        self.number_indices = params.get("number_indices", {})
        if len(self.number_indices) == 0:
            raise ValueError(
                "BetweenTwoAndThreeNumbers fuzzy rule requires a list of main colour indices as params"
            )
    
    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        """A conjunction of a negation with exactly one number and four or more numbers."""

        # Track if we need to squeeze at the end
        was_unsqueezed = False
        # tricking around to fix batch size in case a tensor of size (n,) is passed
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
            was_unsqueezed = True

        concept_probs = y_pred[:, self.number_indices]
        batch_size, num_concepts = concept_probs.shape

        # calculating the four numbers loss and skipping the calculation if we have less than 4 numbers
        if num_concepts < 4:
            exist_agg_at_most_four = torch.zeros(batch_size)
        else:
            indices = torch.combinations(torch.arange(num_concepts, device=y_pred.device), r=4)
            # this tensor has the shape (batch, combinations, 4)
            combinations_probs = concept_probs[:, indices]
            # we want to calculate the t-norm on each of those combinations out of 4, hence we need to recursevly apply the t-norm
            # this is sadly incredible ineffecient but I have not found another way
            recursions = combinations_probs[:, :, 0]
            for i in range(1, 4):
                recursions = self.t_norm(recursions, combinations_probs[:, :, i])
            # this gives us the result of all the t-norms over each combination of 4 numbers
            exist_agg_at_most_four = self.e_aggregation(recursions)

        # calculating the exactly one number
        exactly_one_number = []

        for i in range(num_concepts):
            # The concept we are focusing on in this iteration
            other_concepts = torch.cat(
                [concept_probs[:, :i], concept_probs[:, i + 1 :]], dim=1
            )
            # Negate them
            negated_other_concepts = 1.0 - other_concepts
            # Apply the universal aggregation (e.g., min) over the "other" concepts
            all_aggregation = self.a_aggregation(negated_other_concepts)
            # Apply the t-norm between the current concept and the aggregation
            t_norm = self.t_norm(concept_probs[:, i], all_aggregation)
            # Add the result for this iteration to our list
            exactly_one_number.append(t_norm.unsqueeze(1))

        exactly_one_number = torch.cat(exactly_one_number, dim=1)
        exist_agg_one_number = self.e_aggregation(exactly_one_number)

        result = self.t_conorm(exist_agg_at_most_four, exist_agg_one_number)
        if was_unsqueezed:
            result = result.squeeze(0)
        
        return result

class AtMostOneWarning(FuzzyLoss):
    """Defines the at most one warning symbol loss. This works exactly the same as at most one border colour."""

    def __init__(self, t_norm: Tnorm, t_conorm: Tconorm, e_aggregation: Aggregation, a_aggregation: Aggregation, params: dict):
        super().__init__(t_norm, t_conorm, e_aggregation, a_aggregation)

        self.warning_indices = params.get("warning_indices", {})
        if len(self.warning_indices) == 0:
            raise ValueError(
                "ExactlyOneMainColour fuzzy rule requires a list of main colour indices as params"
            )
        
    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:

        # Track if we need to squeeze at the end
        was_unsqueezed = False
        # tricking around to fix batch size
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
            was_unsqueezed = True

        # Isolate the concept probabilities we're working with
        concept_probs = y_pred[:, self.warning_indices]
        batch_size, num_concepts = concept_probs.shape
        batch_losses = []

        for i in range(num_concepts):
            other_concepts = torch.cat(
                [concept_probs[:, :i], concept_probs[:, i + 1 :]], dim=1
            )
            # creating the pairs between current concept and other concepts
            pairs = []
            for j in range(other_concepts.shape[1]):
                # unsqueezing the tensors to allow for concate later
                tensor_i = concept_probs[:, i].unsqueeze(1) 
                tensor_j = other_concepts[:, j].unsqueeze(1)
                t_norm = self.t_norm(tensor_i, tensor_j)
                negation = 1 - t_norm
                pairs.append(negation)
            pairs = torch.cat(pairs, dim=1)
            batch_losses.append(pairs)
        # Apply the universal aggregation (e.g., godel) over the "other" concepts
        batch_losses = torch.cat(batch_losses, dim=1)
        all_aggregation = self.a_aggregation(batch_losses)

        # accounting for single tensor inputs
        if was_unsqueezed:
            all_aggregation = all_aggregation.squeeze(0)
            
        return 1 - all_aggregation

class NoSymbolsExactlyTwoColours(FuzzyLoss):
    """This class implements the rule: If a sign has no symbols it has exactly two colours. This is a concept group relation.
    To calculate the loss we penalize each input if: It has no symbols and one or less colours or three or more symbols."""

    def __init__(self, t_norm: Tnorm, t_conorm: Tconorm, e_aggregation: Aggregation, a_aggregation: Aggregation, params: dict):
        super().__init__(t_norm, t_conorm, e_aggregation, a_aggregation)

        self.symbol_indices = params.get("symbol_indices", {})
        self.colour_indices = params.get("colour_indices", {})
        if (len(self.symbol_indices) == 0) or (len(self.colour_indices) == 0):
            raise ValueError(
                "NoSymbolsExactlyTwoColours fuzzy rule requires a list of symbol and colour indices as params."
            )
        
    def no_symbols(self, y_pred):
        symbol_probs = y_pred[:, self.symbol_indices]
        negation = 1 - symbol_probs
        no_symbols_aggregation = self.a_aggregation(negation)
        return no_symbols_aggregation    
    
    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        # Track if we need to squeeze at the end
        was_unsqueezed = False
        # tricking around to fix batch size in case a tensor of size (n,) is passed
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
            was_unsqueezed = True
        
        # ---- calculating has no symbols.
        no_symbols = self.no_symbols(y_pred)

        # ---- calculating zero or one colours from at least two colours
        colour_probs = y_pred[:, self.colour_indices]
        batch_size, num_colours = colour_probs.shape
        # vectorised implementation of the negation of has two colours
        # in the first step we get all the combinatorial indices
        indices = torch.triu_indices(num_colours, num_colours, offset=1) # gets all row and col indices from matrix diag that is offset by one
        concepts_i = colour_probs[:, indices[0]]
        concepts_j = colour_probs[:, indices[1]]

        pairwise_violations = self.t_norm(concepts_i, concepts_j)
        # is there are two distinct colours present this becomes zero (in the godel case)
        pairwise_negation = 1 - pairwise_violations
        has_zero_or_one_colour = self.a_aggregation(pairwise_negation)

        # ----- calculating has three or more colours
        # calculating the three colour loss and skipping the calculation if we have less than 3 colours
        if num_colours < 3:
            has_three_or_more = torch.zeros(batch_size)
        else:
            indices = torch.combinations(torch.arange(num_colours, device=y_pred.device), r=3)
            combinations_probs = colour_probs[:, indices]
            recursions = combinations_probs[:, :, 0]
            for i in range(1, 3):
                recursions = self.t_norm(recursions, combinations_probs[:, :, i])
            # this gives us the result of all the t-norms over each combination of 3 numbers
            has_three_or_more = self.e_aggregation(recursions)

        # ----- putting it all together via no t-conorm(zero_or_one, three_or_more)
        not_exactly_two_colours = self.t_conorm(has_three_or_more, has_zero_or_one_colour)
        batch_loss = self.t_norm(no_symbols, not_exactly_two_colours)

        if was_unsqueezed:
            batch_loss =  batch_loss.squeeze(0)
        
        return batch_loss

class WarningSignExclusivity(FuzzyLoss):
    """This class calculates the warning sign exclusivity. If there is a warning sign detected there can be no other signs.
    Calculating the loss for this one is an all aggregation over all t-norm pairs between a warning symbol and all other symbols."""


    def __init__(self, t_norm: Tnorm, t_conorm: Tconorm, e_aggregation: Aggregation, a_aggregation: Aggregation, params: dict):
        super().__init__(t_norm, t_conorm, e_aggregation, a_aggregation)

        self.warning_indices = params.get("warning_indices", {})
        self.symbol_indices = params.get("symbol_indices", {})

        if (len(self.warning_indices) ==0) or (len(self.symbol_indices) == 0) or not (set(self.warning_indices) <= set(self.symbol_indices)):
            raise ValueError("WarningSignExclusivity requires a set of warning and symbol indices and warning needs to be a subset of symbols.")
        
    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        # Track if we need to squeeze at the end
        was_unsqueezed = False
        # tricking around to fix batch size in case a tensor of size (n,) is passed
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
            was_unsqueezed = True
        
        symbol_probs = y_pred[:, self.symbol_indices]
        batch_size, num_symbols = symbol_probs.shape

        indices = torch.triu_indices(len(self.warning_indices),num_symbols, offset=1)
        concepts_i = symbol_probs[:, indices[0]]
        concepts_j = symbol_probs[:, indices[1]]

        pairwise_violations = self.t_norm(concepts_i, concepts_j)
        has_warning_and_other_symbol = self.e_aggregation(pairwise_violations)

        if was_unsqueezed:
            has_warning_and_other_symbol = has_warning_and_other_symbol.squeeze()

        return has_warning_and_other_symbol

class WarningImpliesMainWhite(FuzzyLoss):
    """This class implements the implication that is there is a warning symbol the main colour will be white."""

    def __init__(self, t_norm: Tnorm, t_conorm: Tconorm, e_aggregation: Aggregation, a_aggregation: Aggregation, params: dict):
        super().__init__(t_norm, t_conorm, e_aggregation, a_aggregation)

        self.warning_indices = params.get("warning_indices", {})
        self.main_colour_white = params.get("main_colour_white", {})

        if len(self.warning_indices) == 0 or len(self.main_colour_white) != 1:
            raise ValueError("There must be warning indices and exactly one index ")
        
    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        
        # Track if we need to squeeze at the end
        was_unsqueezed = False
        # tricking around to fix batch size in case a tensor of size (n,) is passed
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
            was_unsqueezed = True

        main_colour_white = y_pred[:, self.main_colour_white[0]]
        warning_symbols = y_pred[:, self.warning_indices]

        exist_aggregation = self.e_aggregation(warning_symbols)
        negation_of_main = 1 - main_colour_white

        pairwise = self.t_norm(exist_aggregation, negation_of_main)

        if was_unsqueezed:
            pairwise = pairwise.squeeze()

        return pairwise

class WarningImpliesBorderRed(FuzzyLoss):
    """This class implements the implication that is there is a warning symbol the main colour will be white."""

    def __init__(self, t_norm: Tnorm, t_conorm: Tconorm, e_aggregation: Aggregation, a_aggregation: Aggregation, params: dict):
        super().__init__(t_norm, t_conorm, e_aggregation, a_aggregation)

        self.warning_indices = params.get("warning_indices", {})
        self.border_red_index = params.get("border_red_index", {})

        if len(self.warning_indices) == 0 or len(self.border_red_index) != 1:
            raise ValueError("There must be warning indices and exactly one index ")
        
    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        
        # Track if we need to squeeze at the end
        was_unsqueezed = False
        # tricking around to fix batch size in case a tensor of size (n,) is passed
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
            was_unsqueezed = True

        border_red = y_pred[:, self.border_red_index[0]]
        warning_symbols = y_pred[:, self.warning_indices]

        exist_aggregation = self.e_aggregation(warning_symbols)
        negation_of_main = 1 - border_red

        pairwise = self.t_norm(exist_aggregation, negation_of_main)

        if was_unsqueezed:
            pairwise = pairwise.squeeze()

        return pairwise

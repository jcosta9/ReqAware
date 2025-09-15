from .abstract.fuzzy_transformation_abstract import Tnorm, Tconorm, Aggregation
import torch


class GodelTNorm(Tnorm):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.min(a, b)


class GodelTConorm(Tconorm):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.max(a, b)


class GodelEAggregation(Aggregation):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 2:
            agg, _ = torch.max(inputs, dim=1)
        elif inputs.dim() == 1:
            agg = torch.max(inputs)
        else:
            raise ValueError("Behaiviour for more dims not defined.")
        return agg


class GodelAAggregation(Aggregation):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 2:
            agg, _ = torch.min(inputs, dim=1)
        elif inputs.dim() == 1:
            agg = torch.min(inputs)
        else:
            raise ValueError("Behaiviour for more dims not defined.")
        return agg

class ProductTNorm(Tnorm):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b

class ProductTConorm(Tconorm):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b - (a * b)

class YagerTNorm(Tnorm):
    """
    Implements the Yager T-Norm.
    Hyperparameter 'p' controls the strictness. Must be tuned.
    """
    def __init__(self, p: float, eps: float = 1e-6):
        super().__init__()
        if p <= 0:
            raise ValueError("Parameter p must be greater than 0.")
        self.p = p
        self.eps = eps # For numerical stability to avoid division by zero 

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        base = torch.pow(1.0 - a, self.p) + torch.pow(1.0 - b, self.p)
        root = torch.pow(base + self.eps, 1.0 / self.p)
        result = 1.0 - root
        return torch.relu(result)
    
class YagerTConorm(Tconorm):
    """
    Implements the Yager T-Conorm, the dual of the Yager T-Norm.
    Hyperparameter 'p' should match the corresponding T-Norm.
    """
    def __init__(self, p: float, eps: float = 1e-6):
        super().__init__()
        if p <= 0:
            raise ValueError("Parameter p must be greater than 0.")
        self.p = p
        self.eps = eps

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        base = torch.pow(a, self.p) + torch.pow(b, self.p)
        root = torch.pow(base + self.eps, 1.0 / self.p)
        return torch.clamp(root, max=1.0)

class LogProductAAggregation(Aggregation):
    """
    Implements the Log-Product Aggregation for the universal quantifier.
    Note: The output is not in [0, 1], but in [-inf, 0]. The loss (1 - truth)
    becomes a positive value to be minimized.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps # To prevent log(0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # We expect inputs of shape (batch_size, num_instances)
        # We aggregate over the num_instances dimension.
        if inputs.dim() < 2:
             raise ValueError("LogProductAggregation expects at least a 2D tensor [batch, instances]")
        
        # Adding epsilon for numerical stability
        stable_inputs = inputs + self.eps
        log_product_agg =  torch.sum(torch.log(stable_inputs), dim=-1)
        raw = - log_product_agg

        return 1 - torch.tanh(raw)
    
class GeneralizedMeanEAggregation(Aggregation):
    """
    Implements the Generalized Mean Aggregation for the existential quantifier.
    Hyperparameter 'p' controls the focus on larger values. Must be tuned.
    """
    def __init__(self, p: float, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() < 2:
             raise ValueError("GeneralizedMeanAggregation expects at least a 2D tensor [batch, instances]")
        
        # Mean of powers
        mean_of_powers = torch.mean(torch.pow(inputs, self.p), dim=-1)
        
        # Root of mean
        return torch.pow(mean_of_powers + self.eps, 1.0 / self.p)
    
class ProductAAggregation(Aggregation):
    """This class implements the all aggregation based on the product t-norm."""
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() < 2:
            raise ValueError("Product aggregation expects at least a 2D tensor [batch, instances]")
        return torch.prod(inputs,dim=1)

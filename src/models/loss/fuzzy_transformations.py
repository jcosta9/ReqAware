from .abstract.fuzzy_transformation import Tnorm, Tconorm, Implication, Aggregation
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
    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 2:
            agg, _ = torch.min(inputs, dim=1)
        elif inputs.dim() == 1:
            agg = torch.min(inputs)
        else:
            raise ValueError("Behaiviour for more dims not defined.")
        return agg

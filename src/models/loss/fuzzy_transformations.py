from abstract.fuzzy_transformation import Tnorm, Tconorm, Implication, Aggregation
import torch

class GodelTNorm(Tnorm):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.min(a, b)

class GodelTConorm(Tconorm):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.max(a, b)
    
class GodelEAggregation(Aggregation):
    def forward(self, *inputs) -> torch.tensor:
        return torch.max(inputs)

class GodelAAggreation(Aggregation):
    def forward(self, *inputs) -> torch.tensor:
        return torch.min(inputs)

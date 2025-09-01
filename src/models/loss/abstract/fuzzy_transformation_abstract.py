import torch
from torch import nn
from abc import ABC, abstractmethod

class FuzzyTransformation(nn.Module, ABC):
    """Abstract base class for Fuzzy transformation. Intented to implement the open close principals."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pass

class Tnorm(FuzzyTransformation):
    """Abstract class for all conjunctive 'and' fuzzy logic transformations, called t-norm"""
    pass

class Tconorm(FuzzyTransformation):
    """Abstract class for all conjunctive 'or' fuzzy logic transformations, called t-conorm"""
    pass

class Aggregation(FuzzyTransformation):
    """Abstract class for all aggregation opperations in fuzzy logic"""
    pass


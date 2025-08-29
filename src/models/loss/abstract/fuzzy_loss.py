import torch
from torch import nn
from abc import ABC
from .fuzzy_transformation import Tnorm, Tconorm, Implication, Aggregation

class FuzzyLoss(nn.Module, ABC):
    """Abstract class implementation of a Fuzzy Loss, with the differnt transformation operations as input.
    This class is set up in a way that enables easy plug and play with the different fuzzy losses."""
    def __init__(self, t_norm: Tnorm, t_conorm: Tconorm, implication: Implication, e_aggregation: Aggregation, a_aggregation: Aggregation):
        super().__init__()
        self.t_norm = t_norm
        self.t_conorm = t_conorm
        self.implication = implication
        self.e_aggregation = e_aggregation
        self.a_aggregation = a_aggregation

    def forward(self, y_pred):
        """This forward assumes that we only need the predicted label since the fuzzy loss is indepndent of the true vector."""
        pass

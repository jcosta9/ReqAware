"""
Class for training a CBM with an EfficientNet Backbone and a FCNN label predictor

Copyright (C) 2024  Joao Paulo Costa de Araujo

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; If not, see <http://www.gnu.org/licenses/>.
"""

import os
import logging

import torch
from torch import nn

from .FCSoftmax import FCSoftmax
from .EfficientNetv2 import EfficientNetv2


class CBMSequentialEfficientNetFCN(nn.Module):
    def __init__(self, config, concepts_threshold=0.5):
        super(CBMSequentialEfficientNetFCN, self).__init__()

        self.config = config
        self.conceps_threshold = concepts_threshold

        self.concept_predictor = EfficientNetv2(
            n_labels=self.config.dataset.n_concepts
        )

        self.label_predictor = FCSoftmax(
            input_dim=self.config.dataset.n_concepts,
            num_classes=self.config.dataset.n_labels,
            dropout=self.config.label_predictor.dropout,
        )

    def to(self, device):
        self.concept_predictor.to(device)
        self.label_predictor.to(device)
        return self

    def forward(self, x):
        concepts = self.concept_predictor(x)
        pred_concepts = (
                    torch.sigmoid(concepts) > self.concepts_threshold
                ).float() 
        labels = self.label_predictor(pred_concepts)
        return pred_concepts, labels
    
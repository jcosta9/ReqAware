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
    def __init__(self, config):
        super(CBMSequentialEfficientNetFCN, self).__init__()

        self.config = config

        self.concept_predictor = EfficientNetv2(self.config)

        self.label_predictor = FCSoftmax(
            input_dim=self.config.dataset.n_concepts,
            num_classes=self.config.dataset.n_labels,
            dropout=self.config.training.dropout,
        )

        # if self.config.freeze_concept_predictor:
        #     # TODO: check if model exists
        #     logging.info(
        #         f"Loading concept predictor model: {self.config.concept_predictor_file}"
        #     )
        #     filename_concept = os.path.join(
        #         self.config.models_path, self.config.concept_predictor_file
        #     )
        #     self.concept_predictor.load_state_dict(torch.load(filename_concept))

        # if self.config.freeze_label_predictor:
        #     # TODO: check if model exists
        #     logging.info(
        #         f"Loading label predictor model: {self.config.label_predictor_file}"
        #     )
        #     filename_label = os.path.join(
        #         self.config.models_path, self.config.label_predictor_file
        #     )
        #     self.label_predictor.load_state_dict(torch.load(filename_label))

    def forward(self, x):
        concepts = self.concept_predictor(x)
        labels = self.label_predictor(concepts)
        return concepts, labels

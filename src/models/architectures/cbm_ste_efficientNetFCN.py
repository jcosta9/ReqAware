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


class CBMSTEEfficientNetFCN(nn.Module):
    def __init__(self, config, concepts_threshold=0.5):
        super(CBMSTEEfficientNetFCN, self).__init__()

        self.config = config
        self.concepts_threshold = concepts_threshold

        self.concept_predictor = EfficientNetv2(n_labels=self.config.dataset.n_concepts)

        self.label_predictor = FCSoftmax(
            input_dim=self.config.dataset.n_concepts,
            num_classes=self.config.dataset.n_labels,
            dropout=self.config.label_predictor.dropout,
        )

        if self.config.label_predictor.freeze:
            print("[CBMSTEEfficientNetFCN] Freezing label predictor weights.")
            for param in self.label_predictor.parameters():
                param.requires_grad = False
        
        if self.config.label_predictor.pretrained_weights is not None:
            pretrained_path = os.path.join(
                self.config.label_predictor.pretrained_weights
            )
            print(
                f"[CBMSTEEfficientNetFCN] Loading pretrained weights for label predictor from {pretrained_path}"
            )
            self.label_predictor.load_state_dict(
                torch.load(pretrained_path, map_location="cpu")
            )

        if self.config.concept_predictor.freeze:
            print("[CBMSTEEfficientNetFCN] Freezing label predictor weights.")
            for param in self.concept_predictor.parameters():
                param.requires_grad = False
        
        if self.config.concept_predictor.pretrained_weights is not None:
            pretrained_path = os.path.join(
                self.config.concept_predictor.pretrained_weights
            )
            print(
                f"[CBMSTEEfficientNetFCN] Loading pretrained weights for label predictor from {pretrained_path}"
            )
            self.concept_predictor.load_state_dict(
                torch.load(pretrained_path, map_location="cpu")
            )

    def to(self, device):
        self.concept_predictor.to(device)
        self.label_predictor.to(device)
        return self

    def forward(self, x):
        concept_logits = self.concept_predictor(x)

        soft_concepts = torch.sigmoid(concept_logits)
        hard_concepts = (soft_concepts > self.concepts_threshold).float()
        ste_concepts = hard_concepts.detach() + 0.001 * (soft_concepts - hard_concepts)

        pred_labels = self.label_predictor(ste_concepts)

        return hard_concepts, pred_labels, concept_logits

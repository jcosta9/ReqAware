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
        # 1. Get raw logits from the concept predictor
        concept_logits = self.concept_predictor(x)
        
        # 2. Soft concept scores (Continuous and Differentiable)
        soft_concepts = torch.sigmoid(concept_logits) 
        
        # 3. Hard concept predictions (Binary and Non-Differentiable)
        # This is for the forward pass *only* to enforce the hard bottleneck
        hard_concepts = (soft_concepts > self.concepts_threshold).float() 
        
        # 4. Apply STE for backpropagation:
        # We start with the hard_concepts (for the forward pass).
        # We subtract the hard_concepts (which has no gradient).
        # We add the soft_concepts (which has the correct gradient).
        # The equation for the output of this line is hard_concepts + (soft_concepts - hard_concepts)
        # 
        # Fwd: hard_concepts (Correct for bottleneck)
        # Bwd: gradient of (soft_concepts) (Correct for backprop)
        ste_concepts = hard_concepts.detach() + (soft_concepts - hard_concepts)
        
        # 5. Pass the STE concepts to the label predictor
        pred_labels = self.label_predictor(ste_concepts)
        
        # Return the actual hard concepts for logging/metrics
        return hard_concepts, pred_labels, concept_logits 
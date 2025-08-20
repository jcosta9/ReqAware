"""
Class for training a Baseline EfficientNet

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

import logging

from torch import nn
from torchvision import models


class EfficientNetv2(nn.Module):
    def __init__(self, n_labels, device):
        super(EfficientNetv2, self).__init__()

        self.predictor = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT
        )

        for params in self.predictor.parameters():
            params.requires_grad = True

        self.predictor.classifier[1] = nn.Linear(
            in_features=1280, out_features=n_labels
        )
        self.predictor.to(device)

    def forward(self, x):
        return self.predictor(x)

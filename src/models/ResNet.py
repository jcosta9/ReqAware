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

import torch
import torch.nn as nn
from torchvision import models


class ResNetCifar10(nn.Module):
    """Resnet enhanced for Cifar10"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.predictor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        for params in self.predictor.parameters():
            params.requires_grad = True

        in_features = self.predictor.fc.in_features

        self.predictor.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.predictor.maxpool = nn.Identity()  # Remove maxpool
        self.predictor.fc = torch.nn.Linear(
            in_features=in_features, out_features=self.config.n_labels
        )

        self.predictor.to(self.config.device)

    def forward(self, x):
        return self.predictor(x)

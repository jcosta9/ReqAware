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

from sklearn.metrics import classification_report
import torch
from torchvision import models

from models import StandardTrainer
from models.abstract import ModelWrapper


class EfficientNetv2(ModelWrapper):
    def __init__(self, config):
        self.config = config
        self.predictor = None
        self.trainer = None

        self.predictor = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT
        )
        for params in self.predictor.parameters():
            params.requires_grad = True
        self.predictor.classifier[1] = torch.nn.Linear(
            in_features=1280, out_features=self.config.n_labels
        )
        self.predictor.to(self.config.device)

        self.trainer = StandardTrainer(
            config=self.config,
            device=self.config.device,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
            lr=0.001,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
            patience=config.patience,
        )

    def train_model(self, train_loader, val_loader):
        self.trainer.train(self.predictor, train_loader, val_loader)

    def evaluate_model(self, test_loader):
        y_true, y_pred = self.trainer.test(self.predictor, test_loader)

        class_report_baseline = classification_report(y_true, y_pred)
        logging.info("Classification Report (Baseline):")
        logging.info(class_report_baseline)

        return None, None, y_true, y_pred

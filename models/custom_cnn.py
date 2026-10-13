"""
Custom Lightweight Convolutional Neural Network for Spina Bifida Detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SpinaBifidaCNN(nn.Module):
    """
    Lightweight 4-block ConvNet optimized for ultrasound image classification.
    Includes Monte Carlo Dropout layers for epistemic uncertainty estimation.
    """
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.4):
        super(SpinaBifidaCNN, self).__init__()
        
        # Block 1 (224 -> 112)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2 (112 -> 56)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3 (56 -> 28)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Block 4 (28 -> 14)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier Head with Dropout for MC-Dropout
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=dropout_rate / 2)
        self.fc2 = nn.Linear(128, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts latent feature embedding vector."""
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.gap(x)
        features = torch.flatten(x, 1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        x = self.dropout1(features)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits

    def enable_mc_dropout(self):
        """Forces dropout layers active during evaluation for Monte Carlo sampling."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

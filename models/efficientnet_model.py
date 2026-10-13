"""
EfficientNet-B0 Architecture for High-Efficiency Ultrasound Classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

class SpinaBifidaEfficientNet(nn.Module):
    """
    EfficientNet-B0 transfer learning architecture.
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout_rate: float = 0.4):
        super(SpinaBifidaEfficientNet, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b0(weights=weights)
        
        self.features = base.features
        self.avgpool = base.avgpool
        
        in_features = base.classifier[1].in_features
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(in_features, 256)
        self.dropout2 = nn.Dropout(p=dropout_rate / 2)
        self.fc2 = nn.Linear(256, num_classes)

    def get_target_layer_for_gradcam(self) -> nn.Module:
        return self.features[-1]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
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
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

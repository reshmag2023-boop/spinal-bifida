"""
ResNet Transfer Learning Architecture with Monte Carlo Dropout and Grad-CAM Hook.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from typing import Optional

class SpinaBifidaResNet(nn.Module):
    """
    ResNet-18 / ResNet-50 backbone with customized classification head
    and Monte Carlo Dropout enabled for uncertainty quantification.
    """
    def __init__(
        self, 
        backbone: str = "resnet18", 
        num_classes: int = 2, 
        pretrained: bool = True,
        dropout_rate: float = 0.4
    ):
        super(SpinaBifidaResNet, self).__init__()
        self.backbone_name = backbone
        
        if backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features
        else:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            in_features = base.fc.in_features
            
        # Backbone layers
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        
        self.avgpool = base.avgpool
        
        # Custom head with active dropout
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(in_features, 256)
        self.bn_head = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(p=dropout_rate / 2)
        self.fc2 = nn.Linear(256, num_classes)

    def get_target_layer_for_gradcam(self) -> nn.Module:
        """Returns the final convolutional block for Grad-CAM activation."""
        return self.layer4[-1]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        x = self.dropout1(features)
        x = F.relu(self.bn_head(self.fc1(x)))
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits

    def enable_mc_dropout(self):
        """Forces dropout layers active during evaluation."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

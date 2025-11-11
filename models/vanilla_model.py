#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Vanilla CNN models without attention mechanisms for DR classification
Supports: ResNet-50, DenseNet-121, EfficientNet-B0
"""
import torch
import torch.nn as nn
from torchvision import models


class APTOSVanillaNet(nn.Module):
    """
    Baseline CNN without attention mechanism
    """
    def __init__(self, backbone='resnet50', dropout_rate=0.4, num_classes=5):
        super(APTOSVanillaNet, self).__init__()
        self.backbone = backbone

        # Load backbone model
        if backbone == 'efficientnet_b0':
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.features = base_model.features
            final_features = 1280

        elif backbone == 'densenet121':
            base_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            self.features = base_model.features
            final_features = 1024

        elif backbone == 'resnet50':
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            final_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_features, final_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(final_features // 2, final_features // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(final_features // 4, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Check input for numerical issues
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        x = self.features(x)

        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        x = self.global_pool(x)
        out = self.classifier(x)

        if torch.isnan(out).any() or torch.isinf(out).any():
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

        return out


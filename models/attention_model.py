#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
CNN models WITH dual attention mechanisms for DR classification
Implements CBAM-style spatial and channel attention
"""
import torch
import torch.nn as nn
from torchvision import models


class DualAttentionModule(nn.Module):
    """
    Dual Attention Module combining Spatial and Channel Attention
    """
    def __init__(self, in_channels, reduction_ratio=16, dropout_rate=0.1):
        super(DualAttentionModule, self).__init__()

        # Spatial Attention - learns "where" to focus
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, reduction_ratio, kernel_size=1),
            nn.GroupNorm(num_groups=min(4, reduction_ratio), num_channels=reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(reduction_ratio, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Channel Attention - learns "what" features are important
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduction_ratio, kernel_size=1),
            nn.GroupNorm(num_groups=min(4, reduction_ratio), num_channels=reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Learnable residual weighting
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        identity = x

        # Apply spatial attention
        spatial_att = self.spatial_attention(x)
        spatial_out = x * spatial_att

        # Apply channel attention
        channel_att = self.channel_attention(spatial_out)
        output = spatial_out * channel_att

        # Residual connection with learnable weighting
        output = self.residual_weight * output + (1 - self.residual_weight) * identity

        return output, spatial_att


class APTOSAttentionNet(nn.Module):
    """
    CNN with Dual Attention mechanism for DR classification
    """
    def __init__(self, backbone='resnet50', dropout_rate=0.4, num_classes=5,
                 use_attention=True, attention_reduction=16):
        super(APTOSAttentionNet, self).__init__()
        self.backbone = backbone
        self.use_attention = use_attention

        # Load backbone
        if backbone == 'efficientnet_b0':
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.features = base_model.features
            final_features = 1280
            self.attention_positions = [4, 6] if use_attention else []

        elif backbone == 'densenet121':
            base_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            self.features = base_model.features
            final_features = 1024
            self.attention_positions = [5, 7] if use_attention else []

        elif backbone == 'resnet50':
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            final_features = 2048
            self.attention_positions = [3, 4] if use_attention else []
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Get channel dimensions at attention positions
        self.channel_dims = self._get_channel_dimensions()

        # Initialize attention modules
        self.attention_modules = nn.ModuleDict()
        if use_attention:
            for pos in self.attention_positions:
                in_channels = self.channel_dims[pos]
                self.attention_modules[f'attention_{pos}'] = DualAttentionModule(
                    in_channels=in_channels,
                    reduction_ratio=attention_reduction,
                    dropout_rate=dropout_rate * 0.5
                )

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

        self.attention_maps = {}
        self._initialize_weights()

    def _get_channel_dimensions(self):
        """Determine channel dimensions at each layer position"""
        channel_dims = {}
        dummy_input = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            x = dummy_input
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i in self.attention_positions:
                    channel_dims[i] = x.shape[1]

        return channel_dims

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        self.attention_maps.clear()

        # Forward through features with attention
        for i, layer in enumerate(self.features):
            x = layer(x)

            # Apply attention at specified positions
            if self.use_attention and i in self.attention_positions:
                attention_key = f'attention_{i}'
                if attention_key in self.attention_modules:
                    x, spatial_att = self.attention_modules[attention_key](x)
                    self.attention_maps[f'spatial_attention_{i}'] = spatial_att.detach()

            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        x = self.global_pool(x)
        out = self.classifier(x)

        if torch.isnan(out).any() or torch.isinf(out).any():
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

        return out

    def get_attention_maps(self):
        """Return stored attention maps for visualization"""
        return self.attention_maps


"""Student model architecture definition for AgriRobust Phase 3."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class AgriStudentMobileNetV3(nn.Module):
    """Compact mobile-oriented MobileNetV3-Small student model for plant disease diagnosis.

    Architecture Backbone: MobileNetV3-Small (ImageNet-1K pretrained weights)
    Classifier Head: Linear(1024, num_classes)
    Total Parameters: ~1.56M (5.59% of ConvNeXt-Tiny Teacher)
    """

    def __init__(
        self,
        num_classes: int = 38,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = "mobilenet_v3_small"
        self.weights_name = "MobileNet_V3_Small_Weights.IMAGENET1K_V1" if pretrained else "None"

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = mobilenet_v3_small(weights=weights)

        # Retain features and pooling
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # Reconstruct classification head
        in_features = base_model.classifier[0].in_features  # 576
        hidden_features = base_model.classifier[0].out_features  # 1024
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial and pooled feature representations before the final classifier."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits


def build_student_model(
    num_classes: int = 38,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> AgriStudentMobileNetV3:
    """Instantiate and return the frozen specification MobileNetV3-Small student model."""
    return AgriStudentMobileNetV3(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )

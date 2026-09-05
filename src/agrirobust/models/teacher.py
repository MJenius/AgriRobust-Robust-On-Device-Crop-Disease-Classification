"""Teacher model architecture definition for AgriRobust Phase 2."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class AgriTeacherConvNeXt(nn.Module):
    """High-capacity ConvNeXt-Tiny Teacher model for agricultural plant disease diagnosis.

    Architecture Backbone: ConvNeXt-Tiny (ImageNet-1K pretrained weights)
    Classifier Head: LayerNorm -> Linear(768, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 38,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = "convnext_tiny"
        self.weights_name = "ConvNeXt_Tiny_Weights.IMAGENET1K_V1" if pretrained else "None"

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = convnext_tiny(weights=weights)

        # Extract features and average pooling
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # Reconstruct classification head
        in_features = base_model.classifier[2].in_features  # 768
        self.norm = base_model.classifier[0]  # LayerNorm2d
        self.flatten = base_model.classifier[1]  # Flatten
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(in_features, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial and pooled feature representations before the final linear layer."""
        x = self.features(x)
        x = self.avgpool(x)
        x = self.norm(x)
        x = self.flatten(x)
        return x

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feat = self.extract_features(x)
        dropped = self.dropout(feat)
        logits = self.classifier(dropped)
        if return_features:
            return logits, feat
        return logits


def build_teacher_model(
    num_classes: int = 38,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> AgriTeacherConvNeXt:
    """Instantiate and return the frozen specification ConvNeXt-Tiny teacher model."""
    return AgriTeacherConvNeXt(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )

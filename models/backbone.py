import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MobileNetV3SmallBackbone(nn.Module):
    """MobileNetV3-Small 多尺度特征提取骨干网络。"""

    def __init__(self, pretrained: bool = True, input_size: int = 224, out_dim: int = 128):
        super().__init__()
        try:
            weights = (
                models.MobileNet_V3_Small_Weights.DEFAULT
                if pretrained
                else None
            )
            backbone = models.mobilenet_v3_small(weights=weights)
        except AttributeError:
            backbone = models.mobilenet_v3_small(pretrained=pretrained)

        self.features = backbone.features
        self.stage_indices = [4, 7, 11]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        fused_dim = self._infer_fused_dim(input_size=input_size)
        self.projection = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim),
        )

    def _infer_fused_dim(self, input_size: int) -> int:
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, input_size, input_size)
            dummy_features = []
            x = dummy_input
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i in self.stage_indices:
                    pooled = self.avgpool(x)
                    flattened = self.flatten(pooled)
                    dummy_features.append(flattened)
            return torch.cat(dummy_features, dim=1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale_features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.stage_indices:
                pooled = self.avgpool(x)
                flattened = self.flatten(pooled)
                multi_scale_features.append(flattened)
        fused_feature = torch.cat(multi_scale_features, dim=1)
        output = self.projection(fused_feature)
        return F.normalize(output, p=2, dim=1)

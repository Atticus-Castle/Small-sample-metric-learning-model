import torch
import torch.nn as nn


class MultiScaleFusion(nn.Module):
    """通用多尺度融合模块：拼接 + MLP 投影。"""

    def __init__(self, in_dims, hidden_dim: int = 256, out_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        fused_dim = sum(in_dims)
        self.projection = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, features):
        if not isinstance(features, (list, tuple)):
            raise TypeError("features 必须是 list 或 tuple")
        fused = torch.cat(features, dim=1)
        return self.projection(fused)

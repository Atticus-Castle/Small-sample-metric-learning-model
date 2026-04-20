import torch
import torch.nn as nn


class PrototypicalNetwork(nn.Module):
    def __init__(self, feature_extractor: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        support_features = self.feature_extractor(support_images)
        query_features = self.feature_extractor(query_images)

        unique_labels = torch.unique(support_labels)
        prototypes = []
        for label in unique_labels:
            class_features = support_features[support_labels == label]
            prototypes.append(class_features.mean(dim=0))
        prototypes = torch.stack(prototypes, dim=0)

        distances = torch.cdist(query_features, prototypes, p=2)
        return -distances

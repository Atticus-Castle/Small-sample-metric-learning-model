import torch
import torch.nn.functional as F


def prototypical_loss(logits: torch.Tensor, query_labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, query_labels)

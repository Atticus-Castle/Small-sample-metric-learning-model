import torch


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def precision_recall_f1_from_preds(
    preds: torch.Tensor, labels: torch.Tensor, num_classes: int
):
    """返回 macro Precision/Recall/F1。"""
    eps = 1e-12
    precisions = []
    recalls = []
    f1s = []

    for cls in range(num_classes):
        pred_pos = preds == cls
        label_pos = labels == cls

        tp = (pred_pos & label_pos).sum().item()
        fp = (pred_pos & (~label_pos)).sum().item()
        fn = ((~pred_pos) & label_pos).sum().item()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_precision = sum(precisions) / len(precisions)
    macro_recall = sum(recalls) / len(recalls)
    macro_f1 = sum(f1s) / len(f1s)
    return macro_precision, macro_recall, macro_f1

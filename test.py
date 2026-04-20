import torch

from config import Config
from datasets.railway_dataset import RailwayDataset
from models.backbone import MobileNetV3SmallBackbone
from models.proto_net import PrototypicalNetwork
from train import build_transform
from utils.metrics import precision_recall_f1_from_preds
from utils.sampler import sample_episode


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = RailwayDataset(cfg.test_root, transform=build_transform(cfg.image_size, False))

    feature_extractor = MobileNetV3SmallBackbone(
        pretrained=False, input_size=cfg.image_size, out_dim=cfg.feature_dim
    ).to(device)
    model = PrototypicalNetwork(feature_extractor).to(device)
    model.load_state_dict(torch.load(cfg.best_model_path, map_location=device))

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for _ in range(cfg.test_episodes):
            s_imgs, s_lbls, q_imgs, q_lbls = sample_episode(
                test_dataset, cfg.n_way, cfg.k_shot, cfg.n_query, allow_replacement=True
            )
            logits = model(s_imgs.to(device), s_lbls.to(device), q_imgs.to(device))
            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(q_lbls.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    test_acc = (all_preds == all_labels).float().mean().item()
    precision, recall, f1 = precision_recall_f1_from_preds(
        all_preds, all_labels, num_classes=cfg.n_way
    )

    print(f"Test Acc over {cfg.test_episodes} episodes: {test_acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro):    {recall:.4f}")
    print(f"F1-score (macro):  {f1:.4f}")


if __name__ == "__main__":
    main()

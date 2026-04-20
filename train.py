import torch
from torchvision import transforms

from config import Config
from datasets.railway_dataset import RailwayDataset
from models.backbone import MobileNetV3SmallBackbone
from models.proto_net import PrototypicalNetwork
from utils.loss import prototypical_loss
from utils.metrics import accuracy_from_logits
from utils.sampler import sample_episode


def train_one_epoch(model, dataset, optimizer, cfg: Config, device: torch.device):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for _ in range(cfg.episodes_per_epoch):
        s_imgs, s_lbls, q_imgs, q_lbls = sample_episode(
            dataset, cfg.n_way, cfg.k_shot, cfg.n_query
        )
        s_imgs = s_imgs.to(device)
        s_lbls = s_lbls.to(device)
        q_imgs = q_imgs.to(device)
        q_lbls = q_lbls.to(device)

        logits = model(s_imgs, s_lbls, q_imgs)
        loss = prototypical_loss(logits, q_lbls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, q_lbls)

    return total_loss / cfg.episodes_per_epoch, total_acc / cfg.episodes_per_epoch


@torch.no_grad()
def evaluate(model, dataset, cfg: Config, device: torch.device):
    model.eval()
    total_acc = 0.0
    for _ in range(cfg.test_episodes):
        s_imgs, s_lbls, q_imgs, q_lbls = sample_episode(
            dataset, cfg.n_way, cfg.k_shot, cfg.n_query, allow_replacement=True
        )
        s_imgs = s_imgs.to(device)
        s_lbls = s_lbls.to(device)
        q_imgs = q_imgs.to(device)
        q_lbls = q_lbls.to(device)
        logits = model(s_imgs, s_lbls, q_imgs)
        total_acc += accuracy_from_logits(logits, q_lbls)
    return total_acc / cfg.test_episodes


def build_transform(image_size: int, train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = RailwayDataset(cfg.train_root, transform=build_transform(cfg.image_size, True))
    test_dataset = RailwayDataset(cfg.test_root, transform=build_transform(cfg.image_size, False))

    feature_extractor = MobileNetV3SmallBackbone(
        pretrained=cfg.pretrained, input_size=cfg.image_size, out_dim=cfg.feature_dim
    ).to(device)
    model = PrototypicalNetwork(feature_extractor).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.step_size, gamma=cfg.gamma
    )

    best_test_acc = 0.0
    for epoch in range(cfg.num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_dataset, optimizer, cfg, device)
        test_acc = evaluate(model, test_dataset, cfg, device)
        scheduler.step()
        print(
            f"Epoch {epoch + 1:03d}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), cfg.best_model_path)

    torch.save(model.state_dict(), cfg.final_model_path)
    print(f"Best Test Acc: {best_test_acc:.4f}")


if __name__ == "__main__":
    main()

from dataclasses import dataclass


@dataclass
class Config:
    train_root: str = "dataset/Train"
    test_root: str = "dataset/Test"
    image_size: int = 224
    pretrained: bool = True
    feature_dim: int = 128

    n_way: int = 2
    k_shot: int = 5
    n_query: int = 10
    episodes_per_epoch: int = 100
    test_episodes: int = 100
    num_epochs: int = 50

    lr: float = 1e-3
    step_size: int = 20
    gamma: float = 0.5

    best_model_path: str = "best_prototypical_net.pth"
    final_model_path: str = "prototypical_net.pth"

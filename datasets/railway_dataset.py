import glob
import os
from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RailwayDataset(Dataset):
    def __init__(self, data_root: str, transform: Optional[transforms.Compose] = None):
        self.class_folders = sorted(glob.glob(os.path.join(data_root, "*")))
        self.class_names = [os.path.basename(f) for f in self.class_folders]
        self.img_paths: List[str] = []
        self.labels: List[int] = []
        self.class_to_indices: Dict[int, List[int]] = {}

        valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for label, folder in enumerate(self.class_folders):
            imgs = []
            for ext in valid_exts:
                imgs.extend(glob.glob(os.path.join(folder, ext)))
            start_idx = len(self.img_paths)
            self.img_paths.extend(sorted(imgs))
            self.labels.extend([label] * len(imgs))
            self.class_to_indices[label] = list(range(start_idx, start_idx + len(imgs)))

        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img) if self.transform else img
        return img, self.labels[idx]

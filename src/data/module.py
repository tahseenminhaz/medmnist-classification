"""`MedMNISTDataModule` that implements the DataModule protocol from `src/schemas.py`.
"""
from pathlib import Path

import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader

from src.data.transforms import get_train_transforms, get_eval_transforms


class MedMNISTDataModule:
    def __init__(self, config: dict):
        self.config = config
        self.dataset_name = config["data"].get("dataset")
        self.download_root = config["data"].get("download_root", "data")
        self.batch_size = config["data"].get("batch_size", 64)
        self.num_workers = config["data"].get("num_workers", 1)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        Path(self.download_root).mkdir(parents=True, exist_ok=True)

    def setup(self, stage: str = None) -> None:
        data_class = getattr(medmnist, INFO[self.dataset_name]["python_class"])

        train_transform = get_train_transforms(self.config)
        eval_transform = get_eval_transforms(self.config)

        self.train_dataset = data_class(
            split="train", transform=train_transform, download=True, root=self.download_root
        )
        self.val_dataset = data_class(
            split="val", transform=eval_transform, download=True, root=self.download_root
        )
        self.test_dataset = data_class(
            split="test", transform=eval_transform, download=True, root=self.download_root
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_transforms(self):
        return get_eval_transforms(self.config)
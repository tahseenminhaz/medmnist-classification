"""Training entrypoint that loads config, builds data/model/trainer, runs training + validation."""

from pathlib import Path
import yaml
import torch

from src.data.module import MedMNISTDataModule
from src.training.model import ModelFactory
from src.training.trainer import Trainer
from medmnist import INFO


def main():
    config = yaml.safe_load(Path("configs/train_config.yaml").read_text())

    dm = MedMNISTDataModule(config)
    dm.prepare_data()
    dm.setup()

    info = INFO[config["data"]["dataset"]]
    n_channels = info["n_channels"]
    n_classes = len(info["label"])
    label_names = list(info["label"].values()) if isinstance(info["label"], dict) else info["label"]

    model = ModelFactory().build_model(config, n_channels, n_classes)

    d = config["training"].get("device", "auto")
    device = torch.device(d if d != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    trainer = Trainer(config, model, device)

    print(f"Training on {device} — {config['data']['dataset']} ({n_channels}ch, {n_classes} classes)")
    trainer.train(dm.train_dataloader(), dm.val_dataloader(), label_names=label_names)


if __name__ == "__main__":
    main()

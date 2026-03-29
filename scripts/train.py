"""Training entrypoint that loads config, builds data/model/trainer, runs training + validation."""

import subprocess
from pathlib import Path
import yaml
import torch
import mlflow

from src.data.module import MedMNISTDataModule
from src.training.model import ModelFactory
from src.training.trainer import Trainer
from medmnist import INFO


def main():
    config = yaml.safe_load(Path("configs/train_config.yaml").read_text())

    mlflow.set_experiment(config["data"]["dataset"])

    with mlflow.start_run():
        mlflow.log_params({
            "model": config["model"]["name"],
            "pretrained": config["model"].get("pretrained", False),
            "lr": config["training"]["lr"],
            "epochs": config["training"]["epochs"],
            "weight_decay": config["training"]["weight_decay"],
            "batch_size": config["data"]["batch_size"],
            "augmentation.random_horizontal_flip": config["data"].get("augmentation", {}).get("random_horizontal_flip", False),
            "augmentation.random_rotation": config["data"].get("augmentation", {}).get("random_rotation", 0),
        })

        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            mlflow.set_tag("git_commit", commit)
        except Exception:
            pass

        mlflow.log_artifact("configs/train_config.yaml")

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

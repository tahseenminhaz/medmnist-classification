"""Evaluate a trained model on the test split and report metrics."""

from pathlib import Path

import numpy as np
import torch
import yaml
from medmnist import INFO
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from src.data.module import MedMNISTDataModule
from src.training.model import ModelFactory
from src.utils import save_confusion_matrix


def main():
    config = yaml.safe_load(Path("configs/serve_config.yaml").read_text())
    train_config = yaml.safe_load(Path("configs/train_config.yaml").read_text())

    info = INFO[config["data"]["dataset"]]
    n_channels = info["n_channels"]
    n_classes = len(info["label"])
    label_names = list(info["label"].values()) if isinstance(info["label"], dict) else info["label"]

    model = ModelFactory().build_model(config, n_channels, n_classes)
    checkpoint = config["model"]["checkpoint"]
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    dm = MedMNISTDataModule(train_config)
    dm.prepare_data()
    dm.setup()
    test_loader = dm.test_dataloader()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.squeeze().long()

            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)

            all_labels.append(labels.numpy())
            all_preds.append(outputs.argmax(1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    cm = confusion_matrix(y_true, y_pred)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 (macro): {f1:.4f}")
    print(f"Test AUC (macro): {auc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")

    model_name = config["model"]["name"]
    out_dir = Path("logs") / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(cm, label_names, out_dir / "test_confusion_matrix.png", title="Test Confusion Matrix")
    print(f"\nConfusion matrix saved to {out_dir / 'test_confusion_matrix.png'}")


if __name__ == "__main__":
    main()

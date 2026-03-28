"""Training and validation loop implementing TrainerAPI protocol."""

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


class Trainer:
    def __init__(self, config: dict, model: nn.Module, device: torch.device):
        self.config = config
        self.model = model.to(device)
        self.device = device

        train_config = config.get("training", {})
        self.epochs = train_config.get("epochs", 10)
        self.lr = train_config.get("lr", 1e-3)
        self.weight_decay = train_config.get("weight_decay", 1e-4)
        model_name = config["model"]["name"]
        self.log_dir = Path(train_config.get("log_dir", "logs")) / model_name

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_loader: DataLoader, val_loader: DataLoader, label_names=None):
        self.label_names = label_names
        self.log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.log_dir / "metrics.csv"
        best_f1 = 0.0
        checkpoint_path = self.log_dir / "best_model.pth"

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "epoch", "train_loss", "train_acc",
                "val_loss", "val_acc", "val_auc", "val_f1",
            ])
            writer.writeheader()

            for epoch in range(self.epochs):
                train_loss, train_acc = self._train_one_epoch(train_loader)
                val_metrics = self.validate(val_loader)

                if val_metrics["f1"] > best_f1:
                    best_f1 = val_metrics["f1"]
                    torch.save(self.model.state_dict(), checkpoint_path)

                row = {
                    "epoch": epoch + 1,
                    "train_loss": f"{train_loss:.4f}",
                    "train_acc": f"{train_acc:.4f}",
                    "val_loss": f"{val_metrics['loss']:.4f}",
                    "val_acc": f"{val_metrics['accuracy']:.4f}",
                    "val_auc": f"{val_metrics['auc']:.4f}",
                    "val_f1": f"{val_metrics['f1']:.4f}",
                }
                writer.writerow(row)
                f.flush()

                print(
                    f"Epoch {epoch+1}/{self.epochs} — "
                    f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f}  "
                    f"val_loss: {val_metrics['loss']:.4f}  val_acc: {val_metrics['accuracy']:.4f}  "
                    f"val_auc: {val_metrics['auc']:.4f}  val_f1: {val_metrics['f1']:.4f}"
                )

        cm = val_metrics["confusion_matrix"]
        print("\nFinal confusion matrix:\n", cm)

        label_names = self.label_names
        self._save_confusion_matrix_image(cm, label_names)

        print(f"\nLogs saved to {csv_path}")
        print(f"Best model saved to {checkpoint_path} (val_f1: {best_f1:.4f})")
        return val_metrics

    def _save_confusion_matrix_image(self, cm, label_names=None):
        n = cm.shape[0]
        if label_names is None:
            label_names = [str(i) for i in range(n)]

        fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, shrink=0.8)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(label_names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        fig.savefig(self.log_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)

    def _train_one_epoch(self, loader: DataLoader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in loader:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device).squeeze().long()

            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        return total_loss / total, correct / total

    def validate(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device).squeeze().long()

                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * imgs.size(0)
                probs = torch.softmax(outputs, dim=1)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(outputs.argmax(1).cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        y_prob = np.concatenate(all_probs)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        cm = confusion_matrix(y_true, y_pred)

        return {
            "loss": total_loss / len(y_true),
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "confusion_matrix": cm,
        }

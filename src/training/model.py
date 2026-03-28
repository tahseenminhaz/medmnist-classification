"""Model factory implementing ModelFactory protocol."""

import torch
import torch.nn as nn
from pathlib import Path
from torchvision.models import resnet18


class SimpleCNN(nn.Module):
    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _build_resnet18(n_channels: int, n_classes: int, pretrained: bool = False, weights_dir: str = "weights"):
    local_path = Path(weights_dir) / "resnet18-pretrained.pth"
    if pretrained and local_path.exists():
        model = resnet18(weights=None)
        model.load_state_dict(torch.load(local_path, map_location="cpu", weights_only=True))
    elif pretrained:
        model = resnet18(weights="IMAGENET1K_V1")
    else:
        model = resnet18(weights=None)
    if n_channels != 3:
        model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


class ModelFactory:
    MODELS = {"simple_cnn", "resnet18"}

    def build_model(self, config: dict, n_channels: int, n_classes: int):
        model_name = config.get("model", {}).get("name", "simple_cnn")
        pretrained = config.get("model", {}).get("pretrained", False)

        if model_name == "simple_cnn":
            return SimpleCNN(n_channels, n_classes)
        elif model_name == "resnet18":
            weights_dir = config.get("model", {}).get("weights_dir", "weights")
            return _build_resnet18(n_channels, n_classes, pretrained=pretrained, weights_dir=weights_dir)
        else:
            raise ValueError(f"Unknown model '{model_name}'. Choose from {self.MODELS}")

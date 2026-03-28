import torch

from src.training.model import ModelFactory


def test_simple_cnn_forward():
    config = {"model": {"name": "simple_cnn"}}
    model = ModelFactory().build_model(config, n_channels=3, n_classes=8)
    x = torch.randn(2, 3, 28, 28)
    out = model(x)
    assert out.shape == (2, 8)


def test_resnet18_forward():
    config = {"model": {"name": "resnet18", "pretrained": False}}
    model = ModelFactory().build_model(config, n_channels=3, n_classes=8)
    x = torch.randn(2, 3, 28, 28)
    out = model(x)
    assert out.shape == (2, 8)


def test_resnet18_single_channel():
    config = {"model": {"name": "resnet18", "pretrained": False}}
    model = ModelFactory().build_model(config, n_channels=1, n_classes=5)
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    assert out.shape == (2, 5)


def test_unknown_model_raises():
    config = {"model": {"name": "nonexistent"}}
    try:
        ModelFactory().build_model(config, n_channels=3, n_classes=8)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

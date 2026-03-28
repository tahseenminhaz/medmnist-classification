import torch
from src.data.module import MedMNISTDataModule
from medmnist import INFO


config = {
    "data": {
        "dataset": "bloodmnist",
        "download_root": "data",
        "batch_size": 100,
        "num_workers": 0,
        "n_channels": INFO["bloodmnist"]["n_channels"],
        "augmentation": {},
        "download": False,
    }
}

def test_datamodule_basic_iteration():
    dm = MedMNISTDataModule(config)
    dm.prepare_data()
    dm.setup()
    imgs, labels = next(iter(dm.train_dataloader()))
    assert imgs.ndim == 4
    assert labels.ndim in (1, 2)
    if labels.ndim == 2:
        assert labels.shape[1] == 1
    assert imgs.shape[0] <= config["data"]["batch_size"]
    assert imgs.dtype == torch.float32

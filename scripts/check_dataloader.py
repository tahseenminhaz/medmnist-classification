from src.data.module import MedMNISTDataModule

config = {
    "data": {
        "dataset": "bloodmnist",
        "download_root": "data",
        "batch_size": 16,
        "num_workers": 0,
        "n_channels": 3,
        "augmentation": {},
        "download": False,
    }
}

    
def main():
    dm = MedMNISTDataModule(config)
    dm.prepare_data()
    dm.setup()

    imgs, labels = next(iter(dm.train_dataloader()))
    print(f"Train batch - images shape: {imgs.shape}, labels shape: {labels.shape}")

    imgs, labels = next(iter(dm.val_dataloader()))
    print(f"Validation batch - images shape: {imgs.shape}, labels shape: {labels.shape}")

    imgs, labels = next(iter(dm.test_dataloader()))
    print(f"Test batch - images shape: {imgs.shape}, labels shape: {labels.shape}")


if __name__ == "__main__":
    main()
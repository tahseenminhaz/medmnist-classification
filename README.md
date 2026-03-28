# MedMNIST Blood Cell Classification

End-to-end ML system for medical image classification on **BloodMNIST** (8 blood cell types, 28×28 RGB).

## Repository Structure

```
├── configs/
│   └── train_config.yaml        # Experiment configuration
│   └── serve_config.yaml        # Serving configuration, will be populated later
├── src/
│   ├── schemas.py               # Protocol contracts (DataModule, ModelFactory, TrainerAPI)
│   ├── data/
│   │   ├── module.py            # MedMNISTDataModule that loads, splits, serves dataloaders
│   │   └── transforms.py        # Config-driven augmentation and preprocessing
│   └── training/
│       ├── model.py             # SimpleCNN, ResNet-18, and ModelFactory. More models can be added.
│       └── trainer.py           # Training loop, metrics, logging, checkpoints
├── scripts/
│   └── check_train.py           # Training entrypoint
├── tests/
│   └── data_test.py             # Data pipeline tests. Other tests will be added soon.
├── Dockerfile
└── pyproject.toml
```

## Setup

```bash
# Dev container (recommended) : open in VS Code → "Reopen in Container"
```

## Training

```bash
python scripts/check_train.py
```

Training reads `configs/train_config.yaml`. Outputs go to `logs/<model_name>/`:
- `metrics.csv` logs per-epoch train loss/acc, val loss/acc/AUC/F1
- `confusion_matrix.png` provides final validation predictions
- `best_model.pth` : checkpoint saved at the epoch with the highest val F1


## Architecture Decisions

- **Protocol-based contracts** : components are independently replaceable without inheritance coupling. `src/schemas.py` defines `DataModule`, `ModelFactory`, and `TrainerAPI` as Python Protocols. Trainer, data module, and model factory know nothing about each other.

- **Config-driven augmentation** : Single YAML controls the full experiment (dataset, augmentation, model, hyperparameters, device), keeping experiment changes out of source code. Also easier to couple with MLOps tools.

- **Model Selection**: `Resnet-18` is used for its strong transfer learning ability in small medical images. A lightweight `SimpleCNN` is available as an alternative.

## Tradeoffs

- No learning rate scheduler : fixed LR works for short runs; a cosine or step scheduler would help for longer training.
- No early stopping : would add for larger experiments to avoid wasted compute.
- Normalization uses fixed 0.5/0.5 rather than data specific statistics

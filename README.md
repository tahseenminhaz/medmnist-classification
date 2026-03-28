# MedMNIST Blood Cell Classification

End-to-end ML system for medical image classification on **BloodMNIST** (8 blood cell types, 28×28 RGB).

## Repository Structure

```
├── configs/
│   ├── train_config.yaml        # Experiment configuration
│   └── serve_config.yaml        # Serving configuration (checkpoint, host, port)
├── src/
│   ├── schemas.py               # Protocol contracts (DataModule, ModelFactory, TrainerAPI)
│   ├── data/
│   │   ├── module.py            # MedMNISTDataModule that loads, splits, serves dataloaders
│   │   └── transforms.py        # Config-driven augmentation and preprocessing
│   ├── training/
│   │   ├── model.py             # SimpleCNN, ResNet-18, and ModelFactory. More models can be added.
│   │   └── trainer.py           # Training loop, metrics, logging, checkpoints
│   └── serving/
│       ├── app.py               # FastAPI app to predict
│       └── static/
│           └── index.html       # Classifier UI
├── scripts/
│   ├── train.py           # Training entrypoint
│   ├── evaulate.py           # Evaluate entrypoint
│   └── serve.py                 # Serving entrypoint
├── tests/
│   └── data_test.py             # Data pipeline tests
│   └── model_test.py             # Data pipeline tests
├── Dockerfile
├── docker-compose.yaml
└── pyproject.toml
```

## Setup

```bash
# Dev container (recommended) : open in VS Code → "Reopen in Container"
```

## Training

```bash
python scripts/train.py
```

Training reads `configs/train_config.yaml`. Outputs go to `logs/<model_name>/`:
- `train_log.csv` logs per-epoch train loss/acc, val loss/acc/AUC/F1
- `confusion_matrix.png` provides final validation predictions
- `best_model.pth` : checkpoint saved at the epoch with the highest val F1

## Serving

Start the inference server:

```bash
python scripts/serve.py
```

- **UI**: http://localhost:8000 — upload an image, see prediction and probabilities

Configuration is in `configs/serve_config.yaml` (model checkpoint path, host, port).

## Docker

```bash
# First train
docker compose run train

# Then serve (http://localhost:8000)
docker compose up serve
```

## Architecture Decisions

- **Protocol-based contracts** : components are independently replaceable without inheritance coupling. `src/schemas.py` defines `DataModule`, `ModelFactory`, and `TrainerAPI` as Python Protocols. Trainer, data module, and model factory know nothing about each other.

- **Config-driven augmentation** : Single YAML controls the full experiment (dataset, augmentation, model, hyperparameters, device), keeping experiment changes out of source code. Also easier to couple with MLOps tools.

- **Model Selection**: `Resnet-18` is used for its strong transfer learning ability in small medical images. A lightweight `SimpleCNN` is available as an alternative.

## Tradeoffs

- No learning rate scheduler : fixed LR works for short runs; a cosine or step scheduler would help for longer training.
- No early stopping : would add for larger experiments to avoid wasted compute.
- Normalization uses fixed 0.5/0.5 rather than data specific statistics

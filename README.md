# MedMNIST Blood Cell Classification

End-to-end ML system for medical image classification on **BloodMNIST**.

## Prerequisites

Ensure your environment meets these requirements before starting:

* **Operating System**: Linux, macOS, or Windows (WSL2 recommended).
* **Docker Desktop**: Recommended for consistent environment reproduction. [Docker Desktop](https://www.docker.com/products/docker-desktop/)
* **Hardware**: 
    * *Recommended*: NVIDIA GPU with [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) for accelerated training.


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

## Quick Start 

The fastest way to get the system running without local dependency management is via Docker:

```bash
# 1. Train the model
# Downloads data, runs the pipeline, and saves the best checkpoint
docker compose run train

# 2. Launch the inference API & Web UI
docker compose up serve
```


## Local Development

### VS Code & Dev Containers (Recommended)

- Install VS Code.
- Install the "Dev Containers" extension in VS Code.
- Open this folder in VS Code.
- Reopen in Container: A pop-up should appear in the bottom right. Click "Reopen in Container".
- Alternatively: Press F1, type "Dev Containers: Rebuild and Reopen in Container".
- Result: VS Code will build the image from the Dockerfile.

### Manual Installation (No Docker)

If you prefer to run the system natively without Docker or VS Code:

**Set up a Virtual Environment**:
```bash
# Create the environment
python -m venv .venv

# Activate it (Windows):
.venv\Scripts\activate

# Activate it (macOS/Linux):
source .venv/bin/activate
```

## Training

```bash
python scripts/train.py
```

Training reads `configs/train_config.yaml`. Outputs go to `logs/<model_name>/`:
- `train_log.csv` logs per-epoch train loss/acc, val loss/acc/AUC/F1
- `val_confusion_matrix.png` provides final validation predictions
- `best_model.pth` : checkpoint saved at the epoch with the highest val F1

## Evaluate

```bash
python scripts/evaluate.py
```

Evaluate reads `configs/serve_config.yaml`. 
- `test_confusion_matrix.png` provides final validation predictions

## Serving

Start the inference server:

```bash
python scripts/serve.py
```

- **UI**: http://localhost:8000 — upload an image, see prediction and probabilities. Sample images are provided under ```sample/```

Configuration is in `configs/serve_config.yaml` (model checkpoint path, host, port).


## Architecture Decisions

- **Protocol-based contracts** : components are independently replaceable without inheritance coupling. `src/schemas.py` defines `DataModule`, `ModelFactory`, and `TrainerAPI` as Python Protocols. Trainer, data module, and model factory know nothing about each other.

- **Config-driven augmentation** : Single YAML controls the full experiment (dataset, augmentation, model, hyperparameters, device), keeping experiment changes out of source code. Also easier to couple with MLOps tools.

- **Model Selection**: `Resnet-18` is used for its strong transfer learning ability in small medical images. A lightweight `SimpleCNN` is available as an alternative.

## Tradeoffs

- No learning rate scheduler : fixed LR works for short runs; a cosine or step scheduler would help for longer training.
- No early stopping : would add for larger experiments to avoid wasted compute.
- Normalization uses fixed 0.5/0.5 rather than data specific statistics

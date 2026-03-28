"""FastAPI serving layer for trained MedMNIST models."""

from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
from medmnist import INFO
from PIL import Image

from src.data.transforms import get_eval_transforms
from src.training.model import ModelFactory

app = FastAPI(title="MedMNIST Serving")

STATIC_DIR = Path(__file__).parent / "static"

model = None
transform = None
label_names = None
device = None


def load_model(config: dict):
    global model, transform, label_names, device

    device = torch.device("cpu")

    info = INFO[config["data"]["dataset"]]
    n_channels = info["n_channels"]
    n_classes = len(info["label"])
    label_names = list(info["label"].values()) if isinstance(info["label"], dict) else info["label"]

    model = ModelFactory().build_model(config, n_channels, n_classes)
    checkpoint = config["model"]["checkpoint"]
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()

    transform = get_eval_transforms(config)


@app.get("/", response_class=HTMLResponse)
def ui():
    return (STATIC_DIR / "index.html").read_text()


@app.post("/predict")
async def predict(file: UploadFile):
    image_bytes = await file.read()
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze()

    pred_idx = probs.argmax().item()
    return {
        "predicted_class": label_names[pred_idx],
        "predicted_index": pred_idx,
        "probabilities": {name: round(probs[i].item(), 4) for i, name in enumerate(label_names)},
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

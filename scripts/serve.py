"""Serving entrypoint that loads config, initializes model, starts uvicorn."""

from pathlib import Path

import uvicorn
import yaml

from src.serving.app import app, load_model


def main():
    config = yaml.safe_load(Path("configs/serve_config.yaml").read_text())
    load_model(config)

    server_config = config.get("server", {})
    uvicorn.run(app, host=server_config.get("host", "0.0.0.0"), port=server_config.get("port", 8000))


if __name__ == "__main__":
    main()

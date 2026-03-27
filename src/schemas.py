"""Typed contracts and simple dataclasses for components. These may change.
"""
from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable


@dataclass
class DatasetInfo:
    n_channels: int
    n_classes: int
    label_names: List[str]
    task: str = "classification"


@runtime_checkable
class DataModule(Protocol):
    def prepare_data(self) -> None: ...

    def setup(self, stage: str = None) -> None: ...

    def train_dataloader(self): ...

    def val_dataloader(self): ...

    def test_dataloader(self): ...

    def get_transforms(self): ...


@runtime_checkable
class ModelFactory(Protocol):
    def build_model(self, config: dict, n_channels: int, n_classes: int): ...


@runtime_checkable
class TrainerAPI(Protocol):
    def train(self, *args, **kwargs): ...

    def validate(self, *args, **kwargs): ...

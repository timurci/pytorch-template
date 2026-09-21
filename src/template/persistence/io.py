from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl
import torch


def load_table(path: Path | str) -> pl.DataFrame:
    return pl.read_csv(path)


def save_table(frame: pl.DataFrame, path: Path | str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(destination)


def save_checkpoint(state_dict: Mapping[str, Any], path: Path | str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict), destination)


def load_checkpoint(path: Path | str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint at {path} is not a state dict")
    return payload

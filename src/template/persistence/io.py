import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl
import torch


def save_table(frame: pl.DataFrame, path: Path | str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(destination)


def save_checkpoint(state_dict: Mapping[str, Any], path: Path | str) -> None:
    """Atomically write a mapping. The caller supplies the path and contents."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(state_dict), temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_checkpoint(path: Path | str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint at {path} is not a mapping")
    return payload

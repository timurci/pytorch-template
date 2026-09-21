"""Shared wiring helpers for the CLI entrypoints."""

import json
import logging
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from typing import Any

import mlflow
import torch

from template.cli.config import (
    ExperimentConfig,
    MLflowTrackerConfig,
    StdoutTrackerConfig,
    TrackerConfig,
)
from template.data import TableDataset
from template.tracking.protocol import ExperimentTracker


def configure_logging() -> None:
    """Send `"template"` INFO records to stdout, message-only."""
    logger = logging.getLogger("template")
    logger.setLevel(logging.INFO)
    if not any(
        isinstance(handler, logging.StreamHandler)
        for handler in logger.handlers
    ):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


@contextmanager
def tracking(
    configs: Sequence[TrackerConfig],
    *,
    total_steps: int | None = None,
) -> Iterator[list[ExperimentTracker]]:
    """Build trackers, opening an MLflow run when one is configured."""
    with ExitStack() as stack:
        for config in configs:
            if isinstance(config, MLflowTrackerConfig):
                if config.tracking_uri is not None:
                    mlflow.set_tracking_uri(config.tracking_uri)
                mlflow.set_experiment(config.experiment_name)
                stack.enter_context(
                    mlflow.start_run(run_name=config.run_name)
                )
        yield [_build_tracker(config, total_steps) for config in configs]


def _build_tracker(
    config: TrackerConfig, total_steps: int | None
) -> ExperimentTracker:
    if isinstance(config, StdoutTrackerConfig):
        return config.build(total_steps=total_steps)
    return config.build()


def flatten_params(config: ExperimentConfig) -> dict[str, object]:
    return _flatten(config.model_dump(mode="json"))


def feature_count(dataset: TableDataset) -> int:
    return int(dataset[0]["features"].shape[0])


def _flatten(
    values: Mapping[str, Any], prefix: str = ""
) -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, prefix=f"{name}."))
        elif isinstance(value, list):
            flat[name] = json.dumps(value)
        else:
            flat[name] = value
    return flat

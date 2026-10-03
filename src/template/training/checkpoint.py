"""Checkpoint strategies over one training-state record.

The trainer builds the record once per completed epoch and passes it, with
that epoch's metrics, to every strategy. A strategy writes synchronously
during the call. The record's tensors alias the live parameters, so holding
it past the call observes later updates. Metrics are not part of the
record: a resumed run does not restore them.

The CLI opens files. `restore_training_state` loads an already-read
recovery record into objects the caller built.
"""

import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, TypedDict

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from template.persistence import load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)


class TrainerState(TypedDict):
    """What a run needs in order to continue after a finished epoch."""

    epoch: int
    model: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any] | None


class CheckpointStrategy(Protocol):
    """Retention rule for one checkpoint file.

    `state` is the training state just captured. `metrics` holds the floats
    logged for that epoch (`train/loss`, and `val/loss` when validation ran).
    Implementations write before returning, or not at all.
    """

    def update(self, state: TrainerState, metrics: Mapping[str, float]) -> None: ...


class RecoveryCheckpoint:
    """Writes the whole training state so a crashed run can continue.

    `epochs` is the run's total, the same value passed to `Trainer.train`.
    `every` writes when the count of completed epochs is a multiple of
    `every`. The final epoch is always written. Omit `every` and only the
    final epoch is written.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        every: int | None = None,
        epochs: int,
    ) -> None:
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if every is not None and every < 1:
            raise ValueError("every must be at least 1")
        self._path = Path(path)
        self._every = every
        self._epochs = epochs

    def update(self, state: TrainerState, metrics: Mapping[str, float]) -> None:
        del metrics
        completed = state["epoch"] + 1
        on_grid = self._every is not None and completed % self._every == 0
        if on_grid or completed == self._epochs:
            save_checkpoint(state, self._path)
            logger.info("saved checkpoint to %s", self._path)


class BestCheckpoint:
    """Keeps the epoch with the lowest loss seen so far.

    Uses `val/loss` when that key is present, otherwise `train/loss`. An
    equal score keeps the earlier file. `score` seeds the memory when a
    run is resumed; a fresh strategy starts with none.
    """

    def __init__(self, path: Path | str, *, score: float | None = None) -> None:
        if score is not None and not math.isfinite(score):
            raise ValueError("score must be finite")
        self._path = Path(path)
        self._score = score

    def update(self, state: TrainerState, metrics: Mapping[str, float]) -> None:
        metric = _loss(metrics)
        if not math.isfinite(metric):
            return
        if self._score is not None and metric >= self._score:
            return
        save_checkpoint(
            {
                "epoch": state["epoch"],
                "model": state["model"],
                "score": metric,
            },
            self._path,
        )
        self._score = metric
        logger.info("saved checkpoint to %s", self._path)


def load_training_state(path: Path | str) -> TrainerState:
    """Read a recovery checkpoint. The caller built the model already."""
    source = Path(path)
    payload = load_checkpoint(source)
    epoch = payload.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise TypeError(f"checkpoint at {source} has no completed epoch")
    scheduler = payload.get("scheduler", _MISSING)
    if scheduler is _MISSING:
        raise TypeError(f"checkpoint at {source} has no scheduler field")
    if scheduler is not None and not isinstance(scheduler, dict):
        raise TypeError(f"checkpoint at {source} has a malformed scheduler")
    return {
        "epoch": epoch,
        "model": _mapping(payload, "model", source),
        "optimizer": _mapping(payload, "optimizer", source),
        "scheduler": scheduler,
    }


def load_model_weights(path: Path | str) -> dict[str, Any]:
    """Read the model weights from a recovery or best checkpoint."""
    source = Path(path)
    return _mapping(load_checkpoint(source), "model", source)


def load_best_score(path: Path | str) -> float:
    """Read the score stored in a best checkpoint."""
    source = Path(path)
    score = load_checkpoint(source).get("score")
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise TypeError(f"checkpoint at {source} has no score")
    if not math.isfinite(score):
        raise ValueError(f"checkpoint at {source} has a non-finite score")
    return float(score)


def inference_checkpoint(recovery: Path, best: Path | None) -> Path:
    """Prefer the best file when it is present, otherwise the recovery file."""
    if best is not None and best.is_file():
        return best
    return recovery


def restore_training_state(
    state: TrainerState,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
) -> int:
    """Load a recovery record into a freshly built model, optimizer, and scheduler.

    Returns the epoch index training should continue from. The scheduler is
    loaded before the optimizer: building a scheduler resets the learning
    rate, and loading the optimizer afterwards puts the decayed rate back.
    """
    model.load_state_dict(state["model"])
    saved = state["scheduler"]
    if scheduler is None:
        if saved is not None:
            raise ValueError(
                "recovery checkpoint has scheduler state, but no scheduler was provided"
            )
    elif not isinstance(saved, dict):
        raise ValueError(
            "recovery checkpoint has no scheduler state, but a scheduler was provided"
        )
    else:
        scheduler.load_state_dict(saved)
    optimizer.load_state_dict(state["optimizer"])
    return state["epoch"] + 1


def _loss(metrics: Mapping[str, float]) -> float:
    if "val/loss" in metrics:
        return metrics["val/loss"]
    if "train/loss" in metrics:
        return metrics["train/loss"]
    raise KeyError("best checkpoint needs val/loss or train/loss")


def _mapping(payload: Mapping[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint at {path} has no {key}")
    return value


_MISSING = object()

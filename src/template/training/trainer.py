from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader

from template.data import Batch
from template.features import ProcessingPipeline, Stage
from template.tracking.protocol import ExperimentTracker


class _ClassMetrics(NamedTuple):
    precision: tuple[float, ...]
    recall: tuple[float, ...]
    f1: tuple[float, ...]


class _EpochMetrics(NamedTuple):
    loss: float
    accuracy: float
    classes: _ClassMetrics
    grad_norm: float | None = None


class Trainer:
    """Runs train/val epochs and fans metrics out to trackers.

    The model must already live on `device`; the trainer only moves batches.
    It is expected to return per-class logits `(N, C)`, used for the loss and
    for per-class precision/recall/F1. Classes with no predictions or no
    support log `0.0` so trackers never see NaN; macros are unweighted means
    over all classes.

    An optional `ProcessingPipeline` runs on each batch after the move to
    `device`: training batches at stage `train`, validation batches at stage
    `eval` (so seeded validation-time augmentation is a tagging choice in
    the pipeline, not trainer code). Processed features and labels replace
    the batch — metrics therefore describe the processed data. Each batch
    gets its own RNG stream seeded from `(seed, stage, epoch, batch)`, so a
    seeded run reproduces exactly.

    An optional `LRScheduler` is stepped once after each epoch (after
    validation, when there is a validation loader) with no arguments, the
    same place PyTorch's own examples call `step()`. Its unit is that call:
    `step_size`, milestones, and `T_max` count epochs. `None` leaves the
    learning rate constant. The scheduler must wrap this trainer's
    optimizer. `ReduceLROnPlateau` is rejected — `step` requires a metric —
    and iteration-level schedules are out of scope for the same reason:
    this loop has one cadence. Each train epoch logs `train/lr`, the first
    param group's rate during that epoch, before the scheduler steps.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        *,
        scheduler: LRScheduler | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        if isinstance(scheduler, ReduceLROnPlateau):
            raise TypeError(
                "ReduceLROnPlateau.step() requires a metric; "
                "Trainer steps a scheduler once per epoch with no arguments"
            )
        if scheduler is not None and scheduler.optimizer is not optimizer:
            raise ValueError("scheduler must wrap the trainer's optimizer")
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._device = torch.device(device)

    def train(
        self,
        train_loader: DataLoader[Batch],
        val_loader: DataLoader[Batch] | None = None,
        *,
        epochs: int,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        trackers: Sequence[ExperimentTracker] = (),
        processing: ProcessingPipeline | None = None,
        seed: int = 0,
        track_gradients: bool = False,
    ) -> None:
        for epoch in range(epochs):
            train_metrics = self._run_train_epoch(
                train_loader,
                loss_fn,
                processing=processing,
                seed=seed,
                epoch=epoch,
                track_gradients=track_gradients,
            )
            self._log_metrics(train_metrics, trackers, step=epoch)
            if val_loader is not None:
                val_metrics = self._run_val_epoch(
                    val_loader,
                    loss_fn,
                    processing=processing,
                    seed=seed,
                    epoch=epoch,
                )
                self._log_metrics(val_metrics, trackers, step=epoch)
            if self._scheduler is not None:
                self._scheduler.step()

    def _run_train_epoch(
        self,
        loader: DataLoader[Batch],
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        *,
        processing: ProcessingPipeline | None = None,
        seed: int = 0,
        epoch: int = 0,
        track_gradients: bool = False,
    ) -> dict[str, float]:
        self._model.train()
        metrics = _run_epoch(
            self._model,
            loader,
            loss_fn,
            self._device,
            optimizer=self._optimizer,
            processing=processing,
            stage="train",
            seed=seed,
            epoch=epoch,
            track_gradients=track_gradients,
        )
        logged = _metric_entries("train", metrics)
        logged["train/lr"] = _learning_rate(self._optimizer)
        if metrics.grad_norm is not None:
            logged["train/grad_norm"] = metrics.grad_norm
        return logged

    def _run_val_epoch(
        self,
        loader: DataLoader[Batch],
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        *,
        processing: ProcessingPipeline | None = None,
        seed: int = 0,
        epoch: int = 0,
    ) -> dict[str, float]:
        self._model.eval()
        with torch.no_grad():
            metrics = _run_epoch(
                self._model,
                loader,
                loss_fn,
                self._device,
                processing=processing,
                stage="eval",
                seed=seed,
                epoch=epoch,
            )
        return _metric_entries("val", metrics)

    def _log_metrics(
        self,
        metrics: Mapping[str, float],
        trackers: Sequence[ExperimentTracker],
        *,
        step: int | None = None,
    ) -> None:
        for tracker in trackers:
            tracker.log_metrics(metrics, step=step)


def _metric_entries(split: str, metrics: _EpochMetrics) -> dict[str, float]:
    logged = {
        f"{split}/loss": metrics.loss,
        f"{split}/accuracy": metrics.accuracy,
    }
    classes = metrics.classes
    for index, (precision, recall, f1) in enumerate(
        zip(classes.precision, classes.recall, classes.f1)
    ):
        logged[f"{split}/precision_{index}"] = precision
        logged[f"{split}/recall_{index}"] = recall
        logged[f"{split}/f1_{index}"] = f1
    logged[f"{split}/precision_macro"] = _mean(classes.precision)
    logged[f"{split}/recall_macro"] = _mean(classes.recall)
    logged[f"{split}/f1_macro"] = _mean(classes.f1)
    return logged


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values) if values else 0.0


def _learning_rate(optimizer: Optimizer) -> float:
    """Rate used for this epoch's updates: the first param group."""
    return float(optimizer.param_groups[0]["lr"])


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[Batch],
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: torch.device,
    optimizer: Optimizer | None = None,
    *,
    processing: ProcessingPipeline | None = None,
    stage: Stage = "train",
    seed: int = 0,
    epoch: int = 0,
    track_gradients: bool = False,
) -> _EpochMetrics:
    total = 0.0
    correct = 0
    count = 0
    grad_norm_total = 0.0
    batches = 0
    true_positive: list[Tensor] = []
    false_positive: list[Tensor] = []
    false_negative: list[Tensor] = []
    for index, batch in enumerate(loader):
        moved: Batch = {
            "features": batch["features"].to(device),
            "targets": batch["targets"].to(device),
        }
        if "source_indices" in batch:
            moved["source_indices"] = batch["source_indices"].to(device)
        if processing is not None:
            moved = processing.process(
                moved,
                stage=stage,
                rng=_batch_rng(seed, stage, epoch, index),
            )
        features = moved["features"]
        targets = moved["targets"]
        if optimizer is not None:
            optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, targets)
        if optimizer is not None:
            loss.backward()
            if track_gradients:
                grads = [
                    param.grad
                    for param in model.parameters()
                    if param.grad is not None
                ]
                grad_norm_total += float(torch.nn.utils.get_total_norm(grads))
            optimizer.step()
        predictions = logits.argmax(dim=1)
        n_classes = logits.shape[1]
        batch_tp = torch.bincount(
            targets[predictions == targets], minlength=n_classes
        )
        batch_fp = torch.bincount(
            predictions[predictions != targets], minlength=n_classes
        )
        batch_fn = torch.bincount(
            targets[predictions != targets], minlength=n_classes
        )
        true_positive.append(batch_tp)
        false_positive.append(batch_fp)
        false_negative.append(batch_fn)
        batch_size = targets.shape[0]
        total += loss.item() * batch_size
        correct += predictions.eq(targets).sum().item()
        count += batch_size
        batches += 1
    grad_norm = grad_norm_total / batches if track_gradients else None
    if count == 0:
        empty = _ClassMetrics(precision=(), recall=(), f1=())
        return _EpochMetrics(
            loss=0.0, accuracy=0.0, classes=empty, grad_norm=grad_norm
        )
    classes = _class_metrics(
        torch.stack(true_positive).sum(dim=0),
        torch.stack(false_positive).sum(dim=0),
        torch.stack(false_negative).sum(dim=0),
    )
    return _EpochMetrics(
        loss=total / count,
        accuracy=correct / count,
        classes=classes,
        grad_norm=grad_norm,
    )


_STAGE_OFFSET: dict[Stage, int] = {"train": 0, "eval": 1, "predict": 2}


def _batch_rng(seed: int, stage: Stage, epoch: int, index: int) -> torch.Generator:
    """Per-(seed, stage, epoch, batch) CPU generator.

    CPU-side so one seed reproduces the same draws on every device; steps
    move drawn tensors to the batch's device.
    """
    value = (
        seed * 1_000_003 + epoch * 10_007 + index * 101 + _STAGE_OFFSET[stage]
    ) % (2**63)
    return torch.Generator().manual_seed(value)


def _class_metrics(
    true_positive: Tensor,
    false_positive: Tensor,
    false_negative: Tensor,
) -> _ClassMetrics:
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    return _ClassMetrics(
        precision=tuple(precision.tolist()),
        recall=tuple(recall.tolist()),
        f1=tuple(f1.tolist()),
    )


def _safe_ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    positive = denominator > 0
    safe = torch.where(positive, denominator, torch.ones_like(denominator))
    return torch.where(
        positive,
        numerator / safe,
        torch.zeros_like(numerator, dtype=torch.float32),
    )

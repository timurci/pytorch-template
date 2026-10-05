"""Smoke tests: training loop end-to-end on random tensors."""

import math
from collections.abc import Mapping

import pytest
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, Dataset

from template.data import Batch
from template.features.tensor import GaussianNoise, TensorPipeline
from template.models import MLPClassifier
from template.training import Trainer


class _DictDataset(Dataset[Batch]):
    def __init__(self, features: Tensor, targets: Tensor) -> None:
        self._features = features
        self._targets = targets

    def __len__(self) -> int:
        return self._features.shape[0]

    def __getitem__(self, index: int) -> Batch:
        return {
            "features": self._features[index],
            "targets": self._targets[index],
        }


class _RecordingTracker:
    def __init__(self) -> None:
        self.metrics: list[dict[str, float]] = []

    def log_params(self, params: Mapping[str, object]) -> None:
        return None

    def log_metrics(
        self, metrics: Mapping[str, float], *, step: int | None = None
    ) -> None:
        self.metrics.append(dict(metrics))


class _CountingStep:
    """Test double: counts how many batches it processed."""

    def __init__(self) -> None:
        self.calls = 0

    def process(self, batch: Batch, rng: torch.Generator) -> Batch:
        self.calls += 1
        return batch


def _loader(rows: int = 64, features: int = 8) -> DataLoader[Batch]:
    generator = torch.Generator().manual_seed(0)
    dataset = _DictDataset(
        torch.randn(rows, features, generator=generator),
        torch.randint(0, 2, (rows,), generator=generator),
    )
    return DataLoader(dataset, batch_size=16)


def _trainer() -> Trainer:
    model = MLPClassifier(8, hidden_size=16, hidden_depth=1, n_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return Trainer(model, optimizer, device="cpu")


def test_train_epoch_with_processing_produces_finite_metrics() -> None:
    tracker = _RecordingTracker()
    pipeline = TensorPipeline([(GaussianNoise(std=0.01), {"train"})])
    _trainer().train(
        _loader(),
        _loader(),
        epochs=2,
        loss_fn=nn.CrossEntropyLoss(),
        trackers=[tracker],
        tensor_pipeline=pipeline,
        seed=42,
    )
    assert len(tracker.metrics) == 4  # train + val per epoch
    for metrics in tracker.metrics:
        assert all(math.isfinite(value) for value in metrics.values())


def test_train_epoch_without_processing() -> None:
    tracker = _RecordingTracker()
    _trainer().train(
        _loader(),
        epochs=1,
        loss_fn=nn.CrossEntropyLoss(),
        trackers=[tracker],
    )
    assert len(tracker.metrics) == 1  # train only, no val loader


def test_validation_runs_eval_tagged_steps() -> None:
    train_step = _CountingStep()
    eval_step = _CountingStep()
    pipeline = TensorPipeline(
        [(train_step, {"train"}), (eval_step, {"eval"})]
    )
    _trainer().train(
        _loader(),
        _loader(),
        epochs=1,
        loss_fn=nn.CrossEntropyLoss(),
        tensor_pipeline=pipeline,
    )
    # 64 rows / batch 16: four train batches and four validation batches.
    assert train_step.calls == 4
    assert eval_step.calls == 4


def test_without_scheduler_learning_rate_stays_constant() -> None:
    model = MLPClassifier(8, hidden_size=16, hidden_depth=1, n_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    tracker = _RecordingTracker()
    Trainer(model, optimizer).train(
        _loader(),
        epochs=2,
        loss_fn=nn.CrossEntropyLoss(),
        trackers=[tracker],
    )
    assert tracker.metrics[0]["train/lr"] == pytest.approx(0.01)
    assert tracker.metrics[1]["train/lr"] == pytest.approx(0.01)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_scheduler_steps_once_per_epoch_after_validation() -> None:
    model = MLPClassifier(8, hidden_size=16, hidden_depth=1, n_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    tracker = _RecordingTracker()
    Trainer(model, optimizer, scheduler=scheduler).train(
        _loader(),
        _loader(),
        epochs=2,
        loss_fn=nn.CrossEntropyLoss(),
        trackers=[tracker],
    )
    # Train metrics are interleaved with validation: the logged rate is the
    # one used for that epoch's updates, and validation does not step again.
    assert tracker.metrics[0]["train/lr"] == pytest.approx(0.1)
    assert "train/lr" not in tracker.metrics[1]
    assert tracker.metrics[2]["train/lr"] == pytest.approx(0.01)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)


def test_plateau_scheduler_is_rejected() -> None:
    model = MLPClassifier(8, hidden_size=16, hidden_depth=1, n_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = ReduceLROnPlateau(optimizer)
    with pytest.raises(TypeError, match="ReduceLROnPlateau"):
        Trainer(model, optimizer, scheduler=scheduler)


def test_scheduler_must_wrap_the_trainers_optimizer() -> None:
    model = MLPClassifier(8, hidden_size=16, hidden_depth=1, n_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    other = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(other, step_size=1)
    with pytest.raises(ValueError, match="wrap the trainer's optimizer"):
        Trainer(model, optimizer, scheduler=scheduler)

"""Smoke tests: training loop end-to-end on random tensors."""

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from template.data import Batch
from template.features import GaussianNoise, ProcessingPipeline
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
    pipeline = ProcessingPipeline([(GaussianNoise(std=0.01), {"train"})])
    _trainer().train(
        _loader(),
        _loader(),
        epochs=2,
        loss_fn=nn.CrossEntropyLoss(),
        trackers=[tracker],
        processing=pipeline,
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
    pipeline = ProcessingPipeline(
        [(train_step, {"train"}), (eval_step, {"eval"})]
    )
    _trainer().train(
        _loader(),
        _loader(),
        epochs=1,
        loss_fn=nn.CrossEntropyLoss(),
        processing=pipeline,
    )
    # 64 rows / batch 16: four train batches and four validation batches.
    assert train_step.calls == 4
    assert eval_step.calls == 4

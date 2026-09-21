"""Smoke tests: training loop end-to-end on random tensors."""

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from template.data import Batch
from template.features import GaussianNoiseAugmenter, IdentityAugmenter
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


def test_train_epoch_with_augmenter_produces_finite_metrics() -> None:
    tracker = _RecordingTracker()
    _trainer().train(
        _loader(),
        _loader(),
        epochs=2,
        loss_fn=nn.CrossEntropyLoss(),
        trackers=[tracker],
        augmenter=GaussianNoiseAugmenter(std=0.01),
    )
    assert len(tracker.metrics) == 4  # train + val per epoch
    for metrics in tracker.metrics:
        assert all(math.isfinite(value) for value in metrics.values())


def test_train_epoch_without_augmenter() -> None:
    tracker = _RecordingTracker()
    _trainer().train(
        _loader(),
        epochs=1,
        loss_fn=nn.CrossEntropyLoss(),
        trackers=[tracker],
    )
    assert len(tracker.metrics) == 1  # train only, no val loader


def test_identity_augmenter_satisfies_label_invariant() -> None:
    features = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    augmented = IdentityAugmenter().augment(features, targets)
    assert torch.equal(augmented.features, features)
    assert torch.equal(augmented.targets, targets)
    assert torch.equal(
        augmented.targets, targets[augmented.source_indices]
    )


def test_gaussian_noise_augmenter_preserves_shape_and_labels() -> None:
    features = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    augmented = GaussianNoiseAugmenter(std=0.1).augment(features, targets)
    assert augmented.features.shape == features.shape
    assert not torch.equal(augmented.features, features)
    assert torch.equal(augmented.targets, targets[augmented.source_indices])

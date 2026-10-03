"""Checkpoint strategies, training-state restore, and checkpoint I/O."""

import math
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

from template.cli.config import ExperimentConfig
from template.data import Batch
from template.models import MLPClassifier
from template.persistence import load_checkpoint, save_checkpoint
from template.training import (
    BestCheckpoint,
    RecoveryCheckpoint,
    Trainer,
    TrainerState,
    inference_checkpoint,
    load_best_score,
    load_model_weights,
    load_training_state,
    restore_training_state,
)


class _DictDataset(Dataset[Batch]):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        self._features = features
        self._targets = targets

    def __len__(self) -> int:
        return self._features.shape[0]

    def __getitem__(self, index: int) -> Batch:
        return {
            "features": self._features[index],
            "targets": self._targets[index],
        }


def _loader() -> DataLoader[Batch]:
    generator = torch.Generator().manual_seed(0)
    dataset = _DictDataset(
        torch.randn(32, 4, generator=generator),
        torch.randint(0, 2, (32,), generator=generator),
    )
    return DataLoader(dataset, batch_size=16)


def _model() -> MLPClassifier:
    return MLPClassifier(4, hidden_size=8, hidden_depth=1, n_classes=2)


def _state(epoch: int, *, scheduler: dict[str, object] | None = None) -> TrainerState:
    model = nn.Linear(1, 1)
    optimizer = AdamW(model.parameters(), lr=0.1)
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler,
    }


def _metrics(**values: float) -> dict[str, float]:
    return values


def test_save_checkpoint_replaces_a_file_without_leaving_a_temp(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "model.pt"
    save_checkpoint({"weight": torch.tensor(1)}, path)
    assert path.is_file()
    assert list(path.parent.glob(".*tmp")) == []
    assert load_checkpoint(path)["weight"].item() == 1


def test_best_checkpoint_keeps_the_lower_validation_loss(tmp_path: Path) -> None:
    path = tmp_path / "best.pt"
    strategy = BestCheckpoint(path)
    strategy.update(_state(0), _metrics(**{"train/loss": 0.1, "val/loss": 0.5}))
    strategy.update(_state(1), _metrics(**{"train/loss": 0.01, "val/loss": 0.4}))
    strategy.update(_state(2), _metrics(**{"train/loss": 0.0, "val/loss": 0.4}))
    strategy.update(_state(3), _metrics(**{"train/loss": 0.0, "val/loss": 0.2}))

    saved = load_checkpoint(path)
    assert set(saved) == {"epoch", "model", "score"}
    assert saved["epoch"] == 3
    assert saved["score"] == pytest.approx(0.2)
    assert load_best_score(path) == pytest.approx(0.2)


def test_best_checkpoint_uses_training_loss_without_validation(tmp_path: Path) -> None:
    path = tmp_path / "best.pt"
    strategy = BestCheckpoint(path)
    strategy.update(_state(0), _metrics(**{"train/loss": 0.5}))
    strategy.update(_state(1), _metrics(**{"train/loss": 0.7}))
    assert load_checkpoint(path)["epoch"] == 0


def test_best_checkpoint_keeps_a_seeded_score(tmp_path: Path) -> None:
    path = tmp_path / "best.pt"
    save_checkpoint({"epoch": 1, "model": {"w": torch.tensor(1)}, "score": 0.2}, path)
    strategy = BestCheckpoint(path, score=0.2)
    strategy.update(_state(4), _metrics(**{"val/loss": 0.3}))
    assert load_checkpoint(path)["epoch"] == 1
    strategy.update(_state(5), _metrics(**{"val/loss": 0.1}))
    assert load_checkpoint(path)["epoch"] == 5


def test_best_checkpoint_ignores_non_finite_loss(tmp_path: Path) -> None:
    path = tmp_path / "best.pt"
    strategy = BestCheckpoint(path)
    strategy.update(_state(0), _metrics(**{"train/loss": math.nan}))
    assert not path.exists()


def test_best_checkpoint_requires_a_loss() -> None:
    strategy = BestCheckpoint("unused.pt")
    with pytest.raises(KeyError, match="val/loss or train/loss"):
        strategy.update(_state(0), {})


def test_recovery_checkpoint_writes_the_grid_and_the_final_epoch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recovery.pt"
    strategy = RecoveryCheckpoint(path, every=5, epochs=10)
    strategy.update(_state(3), {})
    assert not path.exists()
    strategy.update(_state(4), {})
    assert load_training_state(path)["epoch"] == 4
    strategy.update(_state(9), {})
    saved = load_training_state(path)
    assert saved["epoch"] == 9
    assert set(load_checkpoint(path)) == {
        "epoch",
        "model",
        "optimizer",
        "scheduler",
    }


def test_recovery_checkpoint_without_every_writes_only_the_final_epoch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recovery.pt"
    strategy = RecoveryCheckpoint(path, epochs=3)
    strategy.update(_state(0), {})
    assert not path.exists()
    strategy.update(_state(2), {})
    assert load_training_state(path)["epoch"] == 2


def test_training_state_is_captured_after_the_scheduler_steps() -> None:
    model = _model()
    optimizer = AdamW(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    trainer = Trainer(model, optimizer, scheduler=scheduler)
    seen: list[tuple[int, int, Mapping[str, float]]] = []

    class _Recording:
        def update(self, state: TrainerState, metrics: Mapping[str, float]) -> None:
            scheduler_state = state["scheduler"]
            assert scheduler_state is not None
            seen.append((state["epoch"], int(scheduler_state["last_epoch"]), metrics))

    with pytest.raises(RuntimeError, match="after an epoch completes"):
        _ = trainer.state
    trainer.train(
        _loader(),
        _loader(),
        epochs=2,
        loss_fn=nn.CrossEntropyLoss(),
        checkpoints=[_Recording()],
    )
    assert [epoch for epoch, _, _ in seen] == [0, 1]
    assert seen[0][1] == 1
    assert seen[1][1] == 2
    assert "train/loss" in seen[0][2]
    assert "val/loss" in seen[0][2]
    assert trainer.state["epoch"] == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)


def test_start_epoch_skips_completed_epochs() -> None:
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    seen: list[int] = []

    class _Recording:
        def update(self, state: TrainerState, metrics: Mapping[str, float]) -> None:
            del metrics
            seen.append(state["epoch"])

    Trainer(model, optimizer).train(
        _loader(),
        epochs=3,
        start_epoch=1,
        loss_fn=nn.CrossEntropyLoss(),
        checkpoints=[_Recording()],
    )
    assert seen == [1, 2]


def test_start_epoch_past_the_end_is_rejected() -> None:
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="start_epoch"):
        Trainer(model, optimizer).train(
            _loader(),
            epochs=2,
            start_epoch=3,
            loss_fn=nn.CrossEntropyLoss(),
        )


def test_restore_training_state_reloads_weights_moments_and_learning_rate(
    tmp_path: Path,
) -> None:
    model = _model()
    optimizer = AdamW(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    path = tmp_path / "recovery.pt"
    Trainer(model, optimizer, scheduler=scheduler).train(
        _loader(),
        epochs=2,
        loss_fn=nn.CrossEntropyLoss(),
        checkpoints=[RecoveryCheckpoint(path, every=1, epochs=2)],
    )
    resumed_model = _model()
    resumed_optimizer = AdamW(resumed_model.parameters(), lr=0.1)
    resumed_scheduler = StepLR(resumed_optimizer, step_size=1, gamma=0.1)
    start_epoch = restore_training_state(
        load_training_state(path),
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
    )
    assert start_epoch == 2
    for name, value in model.state_dict().items():
        assert torch.allclose(value, resumed_model.state_dict()[name])
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )
    assert resumed_scheduler.last_epoch == scheduler.last_epoch
    assert _moment(resumed_optimizer, "exp_avg").allclose(_moment(optimizer, "exp_avg"))


def test_restore_rejects_a_scheduler_mismatch() -> None:
    model = nn.Linear(1, 1)
    optimizer = AdamW(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="no scheduler was provided"):
        restore_training_state(
            _state(0, scheduler={"last_epoch": 1}),
            model=model,
            optimizer=optimizer,
            scheduler=None,
        )
    with pytest.raises(ValueError, match="no scheduler state"):
        restore_training_state(
            _state(0, scheduler=None),
            model=model,
            optimizer=optimizer,
            scheduler=StepLR(optimizer, step_size=1),
        )


def test_inference_checkpoint_prefers_an_existing_best_file(tmp_path: Path) -> None:
    recovery = tmp_path / "recovery.pt"
    best = tmp_path / "best.pt"
    save_checkpoint({"epoch": 1, "model": {"w": torch.tensor([1.0])}}, recovery)
    assert inference_checkpoint(recovery, best) == recovery
    assert inference_checkpoint(recovery, None) == recovery
    save_checkpoint(
        {"epoch": 0, "model": {"w": torch.tensor([2.0])}, "score": 0.4},
        best,
    )
    chosen = inference_checkpoint(recovery, best)
    assert chosen == best
    assert load_model_weights(chosen)["w"].item() == pytest.approx(2.0)


def test_checkpoint_cardinality_is_enforced() -> None:
    def training(checkpoints: list[dict[str, object]]) -> dict[str, object]:
        return {
            "data": {
                "train": {"kind": "csv", "path": "data/train.csv"},
                "test": {"kind": "csv", "path": "data/test.csv"},
                "target": {
                    "column": "label",
                    "mapping": {"negative": 0, "positive": 1},
                },
            },
            "model": {"kind": "mlp"},
            "training": {"checkpoints": checkpoints},
        }

    with pytest.raises(ValidationError, match="exactly one recovery"):
        ExperimentConfig.model_validate(training([]))
    with pytest.raises(ValidationError, match="exactly one recovery"):
        ExperimentConfig.model_validate(
            training(
                [
                    {"kind": "recovery", "path": "a.pt"},
                    {"kind": "recovery", "path": "b.pt"},
                ]
            )
        )
    with pytest.raises(ValidationError, match="at most one best"):
        ExperimentConfig.model_validate(
            training(
                [
                    {"kind": "recovery", "path": "a.pt"},
                    {"kind": "best", "path": "b.pt"},
                    {"kind": "best", "path": "c.pt"},
                ]
            )
        )
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            training([{"kind": "recovery", "path": "a.pt", "every": 0}])
        )
    with pytest.raises(ValidationError, match="paths must differ"):
        ExperimentConfig.model_validate(
            training(
                [
                    {"kind": "recovery", "path": "outputs/model.pt"},
                    {"kind": "best", "path": "./outputs/model.pt"},
                ]
            )
        )


def _moment(optimizer: AdamW, name: str) -> torch.Tensor:
    for slot in optimizer.state.values():
        value = slot.get(name)
        if isinstance(value, torch.Tensor):
            return value
    raise AssertionError(f"{name} missing from optimizer state")

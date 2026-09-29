"""Tests for the feature processing layer: order, stages, seeding, provenance."""

import pytest
import torch

from template.data import Batch, TableSchema
from template.features import (
    GaussianNoise,
    LogScaleByCap,
    ProcessingPipeline,
    ScaleByCap,
)
from template.features.protocol import ProcessingStep


def _batch(rows: int = 4) -> Batch:
    return {
        "features": torch.tensor(
            [[20.0, 1.0], [30.0, 2.0], [40.0, 3.0], [50.0, 4.0]][:rows]
        ),
        "targets": torch.tensor([0, 1, 0, 1][:rows]),
    }


class _RecordingStep:
    """Test double: logs its turn and applies `x * mul + add`."""

    def __init__(
        self,
        name: str,
        log: list[str],
        *,
        mul: float = 1.0,
        add: float = 0.0,
    ) -> None:
        self.name = name
        self._log = log
        self._mul = mul
        self._add = add

    def process(self, batch: Batch, rng: torch.Generator) -> Batch:
        self._log.append(self.name)
        features = batch["features"] * self._mul + self._add
        return {**batch, "features": features}


def test_stack_order_is_just_list_order() -> None:
    # pre -> aug -> pre: each step sees its predecessor's output, in order.
    log: list[str] = []
    first = _RecordingStep("first", log, mul=2.0)
    second = _RecordingStep("second", log, add=3.0)
    third = _RecordingStep("third", log, mul=0.5)
    pipeline = ProcessingPipeline(
        [(first, {"train"}), (second, {"train"}), (third, {"train"})]
    )
    out = pipeline.process(
        _batch(), stage="train", rng=torch.Generator().manual_seed(0)
    )
    assert log == ["first", "second", "third"]
    # ((20 * 2) + 3) * 0.5 = 21.5 — only this order produces it.
    assert out["features"][0].tolist() == pytest.approx([21.5, 2.5])


def test_stage_tags_select_steps() -> None:
    log: list[str] = []
    train_only = _RecordingStep("train", log)
    eval_only = _RecordingStep("eval", log)
    everywhere = _RecordingStep("all", log)
    pipeline = ProcessingPipeline(
        [
            (train_only, {"train"}),
            (eval_only, {"eval"}),
            (everywhere, {"train", "eval", "predict"}),
        ]
    )
    rng = torch.Generator().manual_seed(0)
    pipeline.process(_batch(), stage="train", rng=rng)
    assert log == ["train", "all"]
    pipeline.process(_batch(), stage="eval", rng=rng)
    assert log == ["train", "all", "eval", "all"]
    pipeline.process(_batch(), stage="predict", rng=rng)
    assert log == ["train", "all", "eval", "all", "all"]


def test_eval_augmentation_is_seeded_and_reproducible() -> None:
    pipeline = ProcessingPipeline(
        [(GaussianNoise(std=1.0), {"train", "eval"})]
    )
    first = pipeline.process(
        _batch(), stage="eval", rng=torch.Generator().manual_seed(7)
    )["features"]
    second = pipeline.process(
        _batch(), stage="eval", rng=torch.Generator().manual_seed(7)
    )["features"]
    other = pipeline.process(
        _batch(), stage="eval", rng=torch.Generator().manual_seed(8)
    )["features"]
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    # predict skips train/eval-tagged noise entirely.
    clean = pipeline.process(
        _batch(), stage="predict", rng=torch.Generator().manual_seed(7)
    )["features"]
    assert torch.equal(clean, _batch()["features"])


def test_gaussian_noise_preserves_rows_labels_and_provenance() -> None:
    batch = _batch()
    pipeline = ProcessingPipeline([(GaussianNoise(std=0.5), {"train"})])
    out = pipeline.process(
        batch, stage="train", rng=torch.Generator().manual_seed(0)
    )
    assert out["features"].shape == batch["features"].shape
    assert not torch.equal(out["features"], batch["features"])
    assert torch.equal(out["targets"], batch["targets"])
    assert torch.equal(out["source_indices"], torch.arange(4))
    assert "source_indices" not in batch  # input left untouched


def test_steps_pass_untouched_fields_through() -> None:
    from typing import cast

    extra = torch.tensor([9, 9, 9, 9])
    batch = cast(Batch, {**_batch(), "group": extra})
    pipeline = ProcessingPipeline([(ScaleByCap(slice(0, 1), 100.0), {"train"})])
    out = cast(
        dict[str, torch.Tensor],
        pipeline.process(
            batch, stage="train", rng=torch.Generator().manual_seed(0)
        ),
    )
    assert out["group"] is extra


def test_scale_and_log_scale_hit_only_their_block() -> None:
    schema = TableSchema(feature_columns=("age", "income"))
    batch = _batch()
    pipeline = ProcessingPipeline(
        [
            (ScaleByCap(schema.feature_slice("age"), 100.0), {"train"}),
            (LogScaleByCap(schema.feature_slice("income"), 10.0), {"train"}),
        ]
    )
    out = pipeline.process(
        batch, stage="train", rng=torch.Generator().manual_seed(0)
    )
    assert out["features"][:, 0].tolist() == pytest.approx(
        [0.2, 0.3, 0.4, 0.5]
    )
    assert out["features"][:, 1].tolist() == pytest.approx(
        [0.0, 0.30103, 0.47712, 0.60206], abs=1e-4
    )


def test_scale_validates_bounds() -> None:
    with pytest.raises(ValueError):
        ScaleByCap(slice(0, 1), cap=1.0, floor=2.0)
    with pytest.raises(ValueError):
        LogScaleByCap(slice(0, 1), cap=1.0)
    with pytest.raises(ValueError):
        GaussianNoise(std=0.0)


def test_pipeline_step_contract_is_one_method() -> None:
    # A step is exactly `process(batch, rng) -> batch`; nothing else.
    step: ProcessingStep = ScaleByCap(slice(0, 1), 100.0)
    out = step.process(_batch(1), torch.Generator().manual_seed(0))
    assert out["features"].shape == (1, 2)

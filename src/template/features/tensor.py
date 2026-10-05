"""Tensor processing: `Batch`-to-`Batch` steps applied on the target device.

Tensor processing is the second half of preprocessing (`features.raw` is the
first). The trainer and the inference loop apply a `TensorPipeline` to each
batch *after* moving it to the training device, so every step runs vectorized
on the accelerator and draws from a seeded RNG.

Tensor processing is the preferred home for feature transforms: keep work in
raw processing (`features.raw`) only when the raw form is required (e.g.
one-hot expansion of strings) or the work must happen once at read time
rather than every epoch.

Step invariants:

- Steps never mutate their input; they return a `Batch` (cloning on write).
  Fields a step does not touch pass through unchanged.
- Deterministic steps ignore `rng`; stochastic steps MUST draw from `rng`
  (never the global RNG) so a seeded run reproduces exactly. This makes
  validation-time augmentation a tagging choice, not a code change: tag a
  stochastic step for the `eval` stage and it runs there, reproducibly.
- Steps that change the row count keep `source_indices` and `targets`
  aligned with `features` (label-preserving steps satisfy
  `targets == input_targets[source_indices]`).

The schemas at the bottom are the declarative form of these steps: `kind`
picks the implementation in YAML, `build()` resolves column names to
feature-tensor blocks through a `BlockResolver` (structurally satisfied by
`data.TableSchema`), so the steps themselves stay tensor-only.
"""

import math
from collections.abc import Collection, Sequence
from typing import Annotated, Literal, Protocol, Self

import torch
from pydantic import Field, model_validator
from torch import Generator

from template.config_pattern import ConfigModel
from template.data import Batch

Stage = Literal["train", "eval", "predict"]


class TensorStep(Protocol):
    """One processing step over batched tensors."""

    def process(self, batch: Batch, rng: Generator) -> Batch: ...


class BlockResolver(Protocol):
    """Resolves a feature column name to its block of the feature tensor.

    Satisfied structurally by `data.TableSchema`, so tensor-processing config
    can name columns while this layer keeps its `Batch`-only edge into `data`
    — the same no-import trick as the raw readers.
    """

    def feature_slice(self, column: str) -> slice: ...


class ScaleByCap:
    """Scales one feature block to `(x - floor) / (cap - floor)`.

    The block is a `TableSchema.feature_slice` resolved at composition, so
    the step is tensor-only and sees no column names.
    """

    def __init__(self, block: slice, cap: float, floor: float = 0.0) -> None:
        if cap <= floor:
            raise ValueError("cap must be greater than floor")
        self._block = block
        self._cap = cap
        self._floor = floor

    def process(self, batch: Batch, rng: Generator) -> Batch:
        features = batch["features"].clone()
        block = features[:, self._block]
        features[:, self._block] = (block - self._floor) / (
            self._cap - self._floor
        )
        return {**batch, "features": features}


class LogScaleByCap:
    """Scales one feature block by `log(clamp(x, min=1)) / log(cap)`."""

    def __init__(self, block: slice, cap: float) -> None:
        if cap <= 1:
            raise ValueError("cap must be greater than 1")
        self._block = block
        self._cap = cap

    def process(self, batch: Batch, rng: Generator) -> Batch:
        features = batch["features"].clone()
        block = features[:, self._block]
        features[:, self._block] = block.clamp(min=1.0).log() / math.log(
            self._cap
        )
        return {**batch, "features": features}


class GaussianNoise:
    """Adds iid Gaussian noise to all features; rows and labels preserved.

    Draws from the supplied `rng` (never the global RNG), so a seeded
    pipeline reproduces exactly. Tag it `train` only (the default) for
    ordinary augmentation, or also `eval` for seeded validation-time
    augmentation.
    """

    def __init__(self, std: float) -> None:
        if std <= 0:
            raise ValueError("std must be positive")
        self._std = std

    def process(self, batch: Batch, rng: Generator) -> Batch:
        features = batch["features"]
        # Drawn on CPU from the pipeline's CPU generator and moved, so one
        # seed reproduces the same noise on every device.
        noise = torch.randn(
            features.shape, generator=rng, dtype=features.dtype
        ).to(features.device)
        return {**batch, "features": features + noise * self._std}


class TensorPipeline:
    """Ordered, stage-tagged steps applied to one batch at a time.

    Order is construction order: interleaving deterministic preprocessing
    with stochastic augmentation (`pre -> aug -> pre`) is just list order —
    there is no separate preprocessing and augmentation phase. Each step
    runs in the stages it is tagged with (`train`, `eval`, `predict`), and
    every step sees the same `rng`, so one seed reproduces the whole pass:
    deterministic steps ignore it, stochastic steps draw from it.

    The pipeline guarantees `source_indices` on its output — absent input
    provenance becomes `arange(N)`; steps that change rows keep it aligned.
    """

    def __init__(
        self, steps: Sequence[tuple[TensorStep, Collection[Stage]]]
    ) -> None:
        self._steps = [
            (step, frozenset(stages)) for step, stages in steps
        ]

    def process(self, batch: Batch, *, stage: Stage, rng: Generator) -> Batch:
        features = batch["features"]
        if "source_indices" not in batch:
            batch = {
                **batch,
                "source_indices": torch.arange(
                    features.shape[0], device=features.device
                ),
            }
        for step, stages in self._steps:
            if stage in stages:
                batch = step.process(batch, rng)
        return batch


class ScaleByCapConfig(ConfigModel):
    kind: Literal["scale_by_cap"] = "scale_by_cap"
    column: str
    cap: float
    floor: float = 0.0
    stages: list[Stage] = Field(default=["train", "eval", "predict"], min_length=1)

    @model_validator(mode="after")
    def _validate_cap(self) -> Self:
        if self.cap <= self.floor:
            raise ValueError("cap must be greater than floor")
        return self

    def build(self, resolve: BlockResolver) -> tuple[TensorStep, frozenset[Stage]]:
        step = ScaleByCap(resolve.feature_slice(self.column), self.cap, self.floor)
        return step, frozenset(self.stages)


class LogScaleByCapConfig(ConfigModel):
    kind: Literal["log_scale_by_cap"] = "log_scale_by_cap"
    column: str
    cap: float
    stages: list[Stage] = Field(default=["train", "eval", "predict"], min_length=1)

    @model_validator(mode="after")
    def _validate_cap(self) -> Self:
        if self.cap <= 1:
            raise ValueError("cap must be greater than 1")
        return self

    def build(self, resolve: BlockResolver) -> tuple[TensorStep, frozenset[Stage]]:
        step = LogScaleByCap(resolve.feature_slice(self.column), self.cap)
        return step, frozenset(self.stages)


class GaussianNoiseConfig(ConfigModel):
    kind: Literal["gaussian_noise"] = "gaussian_noise"
    std: float = Field(gt=0)
    stages: list[Stage] = Field(default=["train"], min_length=1)

    def build(self, resolve: BlockResolver) -> tuple[TensorStep, frozenset[Stage]]:
        return GaussianNoise(self.std), frozenset(self.stages)


TensorStepConfig = Annotated[
    ScaleByCapConfig | LogScaleByCapConfig | GaussianNoiseConfig,
    Field(discriminator="kind"),
]


class TensorPipelineConfig(ConfigModel):
    """One ordered tensor-processing step list (config `tensor.steps`).

    Preprocessing and augmentation interleave: stacking is list order
    (`pre -> aug -> pre` needs no phases); each step names the stages it runs
    in. Deterministic steps default to every stage (predict-time scaling is
    part of the feature definition), stochastic steps to `train` only — tag
    them `eval` for seeded validation-time augmentation. The built pipeline
    is applied on the target device during training and inference; put
    transforms here in preference to raw processing (`features.raw`) when the
    raw form is not required.
    """

    steps: list[TensorStepConfig] = Field(default_factory=list)

    def build(self, resolve: BlockResolver) -> TensorPipeline:
        return TensorPipeline([step.build(resolve) for step in self.steps])

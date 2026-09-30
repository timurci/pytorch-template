"""Pydantic schemas for `features`' processing steps and their pipeline.

Column names in these schemas are declarations; `build` resolves them to
feature-tensor blocks through a `BlockResolver` (`features.protocol`,
structurally satisfied by `data.TableSchema`) — the same no-import trick
as the raw readers — so this layer keeps its `Batch`-only edge into
`data` and the steps stay tensor-only.
"""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from template.config_pattern import ConfigModel
from template.features.noise import GaussianNoise
from template.features.pipeline import ProcessingPipeline
from template.features.protocol import BlockResolver, ProcessingStep, Stage
from template.features.scale import LogScaleByCap, ScaleByCap


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

    def build(self, resolve: BlockResolver) -> tuple[ProcessingStep, frozenset[Stage]]:
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

    def build(self, resolve: BlockResolver) -> tuple[ProcessingStep, frozenset[Stage]]:
        step = LogScaleByCap(resolve.feature_slice(self.column), self.cap)
        return step, frozenset(self.stages)


class GaussianNoiseConfig(ConfigModel):
    kind: Literal["gaussian_noise"] = "gaussian_noise"
    std: float = Field(gt=0)
    stages: list[Stage] = Field(default=["train"], min_length=1)

    def build(self, resolve: BlockResolver) -> tuple[ProcessingStep, frozenset[Stage]]:
        return GaussianNoise(self.std), frozenset(self.stages)


ProcessingStepConfig = Annotated[
    ScaleByCapConfig | LogScaleByCapConfig | GaussianNoiseConfig,
    Field(discriminator="kind"),
]


class ProcessingConfig(ConfigModel):
    """One ordered step list: preprocessing and augmentation interleave.

    Stacking is list order (`pre -> aug -> pre` needs no phases); each step
    names the stages it runs in. Deterministic steps default to every
    stage (predict-time scaling is part of the feature definition),
    stochastic steps to `train` only — tag them `eval` for seeded
    validation-time augmentation.
    """

    steps: list[ProcessingStepConfig] = Field(default_factory=list)

    def build(self, resolve: BlockResolver) -> ProcessingPipeline:
        return ProcessingPipeline([step.build(resolve) for step in self.steps])

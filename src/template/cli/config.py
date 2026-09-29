"""Pydantic schemas for the YAML experiment config shared by both CLIs.

A single config file describes the whole experiment: data paths and column
roles, feature processing, model architecture, optimizer/loss, training
loop, and inference output. `template-train` and `template-infer` validate
the same file and each use their own sections.

This module is the composition root of the template: it is the only place
that knows both the config file format and the library constructors.
Library layers never see config objects; every schema builds plain library
objects via `build()`. Column names are config-level declarations; they
resolve to feature-tensor blocks (`TableSchema.feature_slice`) when the
processing pipeline is built, so processing steps stay tensor-only.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal, Self

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn
from torch.nn import Parameter
from torch.optim import SGD, Adam, AdamW, Optimizer

from template.data import MapValues, OneHot, TableSchema
from template.features import (
    GaussianNoise,
    LogScaleByCap,
    ProcessingPipeline,
    ProcessingStep,
    ScaleByCap,
    Stage,
)
from template.tracking import MLflowTracker, NullTracker, StdoutTracker
from template.tracking.protocol import ExperimentTracker


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OneHotEncodingConfig(_Config):
    kind: Literal["one_hot"] = "one_hot"
    column: str
    levels: list[str] | None = None
    levels_path: Path | None = None

    def build(self) -> OneHot:
        return OneHot(self.levels, levels_path=self.levels_path)


FeatureEncodingConfig = Annotated[
    OneHotEncodingConfig,
    Field(discriminator="kind"),
]


class TargetConfig(_Config):
    column: str
    mapping: dict[str, int]

    @model_validator(mode="after")
    def _validate_mapping(self) -> Self:
        if not self.mapping:
            raise ValueError("target mapping must not be empty")
        if len(set(self.mapping.values())) != len(self.mapping):
            raise ValueError("target mapping must be injective")
        return self

    def build(self) -> MapValues:
        return MapValues(self.mapping)


class DataConfig(_Config):
    train_path: Path
    test_path: Path
    id_column: str = "id"
    target: TargetConfig
    metadata: list[str] = []
    exclude: list[str] = []
    encodings: list[FeatureEncodingConfig] = []

    def build_schema(
        self, columns: Sequence[str], *, include_target: bool = True
    ) -> TableSchema:
        """Column names + declared roles/encodings -> `TableSchema`.

        Features are what remains after metadata, target, and exclusions —
        declaring exclusions is enough. `include_target=False` (inference)
        drops the target column: absent from unlabeled data, excluded when
        present.
        """
        metadata = list(dict.fromkeys([*self.metadata, self.id_column]))
        exclude = list(self.exclude)
        target: str | None = self.target.column
        target_encoder: MapValues | None = self.target.build()
        if not include_target:
            if target in columns and target not in exclude:
                exclude.append(target)
            target = None
            target_encoder = None
        return TableSchema.from_columns(
            columns,
            target=target,
            metadata=metadata,
            exclude=exclude,
            feature_encoders={
                item.column: item.build() for item in self.encodings
            },
            target_encoder=target_encoder,
        )


class ScaleByCapConfig(_Config):
    kind: Literal["scale_by_cap"] = "scale_by_cap"
    column: str
    cap: float
    floor: float = 0.0
    stages: list[Stage] = ["train", "eval", "predict"]

    def build(
        self, schema: TableSchema
    ) -> tuple[ProcessingStep, frozenset[Stage]]:
        step = ScaleByCap(
            schema.feature_slice(self.column), self.cap, self.floor
        )
        return step, frozenset(self.stages)


class LogScaleByCapConfig(_Config):
    kind: Literal["log_scale_by_cap"] = "log_scale_by_cap"
    column: str
    cap: float
    stages: list[Stage] = ["train", "eval", "predict"]

    def build(
        self, schema: TableSchema
    ) -> tuple[ProcessingStep, frozenset[Stage]]:
        step = LogScaleByCap(schema.feature_slice(self.column), self.cap)
        return step, frozenset(self.stages)


class GaussianNoiseConfig(_Config):
    kind: Literal["gaussian_noise"] = "gaussian_noise"
    std: float = Field(gt=0)
    stages: list[Stage] = ["train"]

    def build(
        self, schema: TableSchema
    ) -> tuple[ProcessingStep, frozenset[Stage]]:
        return GaussianNoise(self.std), frozenset(self.stages)


ProcessingStepConfig = Annotated[
    ScaleByCapConfig | LogScaleByCapConfig | GaussianNoiseConfig,
    Field(discriminator="kind"),
]


class ProcessingConfig(_Config):
    """One ordered step list: preprocessing and augmentation interleave.

    Stacking is list order (`pre -> aug -> pre` needs no phases); each step
    names the stages it runs in. Deterministic steps default to every
    stage (predict-time scaling is part of the feature definition),
    stochastic steps to `train` only — tag them `eval` for seeded
    validation-time augmentation.
    """

    steps: list[ProcessingStepConfig] = []

    def build(self, schema: TableSchema) -> ProcessingPipeline:
        return ProcessingPipeline(
            [step.build(schema) for step in self.steps]
        )


class NullTrackerConfig(_Config):
    kind: Literal["null"] = "null"

    def build(self) -> ExperimentTracker:
        return NullTracker()


class StdoutTrackerConfig(_Config):
    kind: Literal["stdout"] = "stdout"
    every_n_steps: int = Field(default=1, ge=1)

    def build(self, *, total_steps: int | None = None) -> ExperimentTracker:
        return StdoutTracker(self.every_n_steps, total_steps=total_steps)


class MLflowTrackerConfig(_Config):
    kind: Literal["mlflow"] = "mlflow"
    experiment_name: str = "template"
    run_name: str | None = None
    tracking_uri: str | None = None

    def build(self) -> ExperimentTracker:
        return MLflowTracker()


TrackerConfig = Annotated[
    NullTrackerConfig | StdoutTrackerConfig | MLflowTrackerConfig,
    Field(discriminator="kind"),
]


class ModelConfig(_Config):
    hidden_size: int = Field(default=128, ge=1)
    hidden_depth: int = Field(default=2, ge=0)
    n_classes: int = Field(default=2, ge=2)
    dropout: float = Field(default=0.0, ge=0, lt=1)


class OptimizerConfig(_Config):
    kind: Literal["adam", "adamw", "sgd"] = "adamw"
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    momentum: float = Field(default=0.0, ge=0)

    def build(self, parameters: Iterable[Parameter]) -> Optimizer:
        if self.kind == "adam":
            return Adam(
                parameters, lr=self.lr, weight_decay=self.weight_decay
            )
        if self.kind == "adamw":
            return AdamW(
                parameters, lr=self.lr, weight_decay=self.weight_decay
            )
        return SGD(
            parameters,
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
        )


class LossConfig(_Config):
    kind: Literal["cross_entropy"] = "cross_entropy"
    class_weights: Literal["balanced"] | None = None

    def resolve_weights(
        self, class_counts: Mapping[int, int]
    ) -> Tensor | None:
        if self.class_weights is None:
            return None
        return _balanced_weights(class_counts)

    def build(self, class_counts: Mapping[int, int]) -> nn.CrossEntropyLoss:
        return nn.CrossEntropyLoss(weight=self.resolve_weights(class_counts))


def _balanced_weights(class_counts: Mapping[int, int]) -> Tensor:
    indices = sorted(class_counts)
    if indices != list(range(len(indices))):
        raise ValueError(
            "class weights require contiguous class indices starting at 0, "
            f"got {indices}"
        )
    counts = [class_counts[index] for index in indices]
    if any(count == 0 for count in counts):
        raise ValueError("class weights require every class to have samples")
    total = sum(counts)
    weights = [total / (len(counts) * count) for count in counts]
    return torch.tensor(weights, dtype=torch.float32)


class TrainingConfig(_Config):
    epochs: int = Field(default=10, gt=0)
    batch_size: int = Field(default=4096, gt=0)
    val_fraction: float | None = Field(default=0.1, gt=0, lt=1)
    seed: int = 42
    shuffle: bool = True
    device: str = "auto"
    trackers: list[TrackerConfig] = [StdoutTrackerConfig()]
    track_gradients: bool = False
    checkpoint_path: Path


class InferenceConfig(_Config):
    batch_size: int = Field(default=8192, gt=0)
    save: bool = True
    report: bool = True
    output_path: Path = Path("outputs/predictions.csv")


class ExperimentConfig(_Config):
    data: DataConfig
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    model: ModelConfig
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    training: TrainingConfig
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    @model_validator(mode="after")
    def _validate_target_classes(self) -> Self:
        values = set(self.data.target.mapping.values())
        expected = set(range(self.model.n_classes))
        if values != expected:
            raise ValueError(
                "target mapping values must be exactly "
                f"0..{self.model.n_classes - 1}, got {sorted(values)}"
            )
        return self


def load_config(path: Path | str) -> ExperimentConfig:
    source = Path(path)
    raw = yaml.safe_load(source.read_text())
    if not isinstance(raw, dict):
        raise TypeError(f"config at {source} must be a YAML mapping")
    return ExperimentConfig.model_validate(raw)

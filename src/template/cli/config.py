"""Pydantic schema for the shared YAML experiment config and its loader.

One config file describes the whole experiment: data sources and column
roles, feature processing, model architecture, optimizer/scheduler/loss,
training loop, and inference output. `template-train` and `template-infer`
validate the same file and each use their own sections.

This module is the composition root's own schema: `ExperimentConfig`
assembles the layers' config schemas (`template.<layer>.config` — each
lives with the class it builds) and adds the sections no library layer
owns (which sources feed the run, optimizer/scheduler/loss over `torch`
objects, `training`, `inference`) plus the cross-object invariants. The
entrypoints do the wiring: they call each schema's `build()` and pass plain
objects between the layers. Loading is `load_config` — the only place the file is
read. The pattern and its extension recipes: docs/config-pattern.md.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal, Self

import torch
import yaml
from pydantic import Field, model_validator
from torch import Tensor, nn
from torch.nn import Parameter
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LRScheduler,
    MultiStepLR,
    StepLR,
)

from template.config_pattern import ConfigModel, default_kind
from template.data import TableSchema
from template.features.raw import RawPipelineConfig, TargetConfig
from template.features.tensor import TensorPipelineConfig
from template.models.config import MLPModelConfig, ModelConfig
from template.persistence.config import RawSourceConfig
from template.tracking.config import StdoutTrackerConfig, TrackerConfig
from template.training.config import (
    BestCheckpointConfig,
    CheckpointConfig,
    RecoveryCheckpointConfig,
)


class DataConfig(ConfigModel):
    """Where the rows come from and what each column is.

    `train` / `test` pick raw readers by `kind`; roles build the
    `TableSchema` over the raw pipeline's output columns.
    """

    train: RawSourceConfig
    test: RawSourceConfig
    id_column: str = "id"
    target: TargetConfig
    metadata: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)

    def build_schema(
        self, columns: Sequence[str], *, include_target: bool = True
    ) -> TableSchema:
        """Processed column names + declared roles -> `TableSchema`.

        Features are what remains after metadata, target, and exclusions —
        declaring exclusions is enough. `include_target=False` (inference)
        drops the target column: absent from unlabeled data, excluded when
        present.
        """
        metadata = list(dict.fromkeys([*self.metadata, self.id_column]))
        exclude = list(self.exclude)
        target: str | None = self.target.column
        target_encoder = self.target.build()
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
            target_encoder=target_encoder,
        )


class AdamConfig(ConfigModel):
    kind: Literal["adam"] = "adam"
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)

    def build(self, parameters: Iterable[Parameter]) -> Optimizer:
        return Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)


class AdamWConfig(ConfigModel):
    kind: Literal["adamw"] = "adamw"
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)

    def build(self, parameters: Iterable[Parameter]) -> Optimizer:
        return AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay)


class SgdConfig(ConfigModel):
    kind: Literal["sgd"] = "sgd"
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    momentum: float = Field(default=0.0, ge=0)

    def build(self, parameters: Iterable[Parameter]) -> Optimizer:
        return SGD(
            parameters,
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
        )


OptimizerConfig = Annotated[
    AdamConfig | AdamWConfig | SgdConfig,
    Field(discriminator="kind"),
    default_kind("adamw"),
]


class StepLRConfig(ConfigModel):
    kind: Literal["step"] = "step"
    step_size: int = Field(gt=0)
    gamma: float = Field(default=0.1, gt=0)

    def build(self, optimizer: Optimizer) -> LRScheduler:
        return StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)


class MultiStepLRConfig(ConfigModel):
    kind: Literal["multistep"] = "multistep"
    milestones: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)
    gamma: float = Field(default=0.1, gt=0)

    @model_validator(mode="after")
    def _milestones_increase(self) -> Self:
        if list(self.milestones) != sorted(set(self.milestones)):
            raise ValueError(
                "milestones must be strictly increasing, "
                f"got {list(self.milestones)}"
            )
        return self

    def build(self, optimizer: Optimizer) -> LRScheduler:
        return MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.gamma
        )


class ExponentialLRConfig(ConfigModel):
    kind: Literal["exponential"] = "exponential"
    gamma: float = Field(gt=0)

    def build(self, optimizer: Optimizer) -> LRScheduler:
        return ExponentialLR(optimizer, gamma=self.gamma)


class CosineAnnealingLRConfig(ConfigModel):
    kind: Literal["cosine"] = "cosine"
    t_max: int = Field(gt=0)
    eta_min: float = Field(default=0.0, ge=0)

    def build(self, optimizer: Optimizer) -> LRScheduler:
        return CosineAnnealingLR(
            optimizer, T_max=self.t_max, eta_min=self.eta_min
        )


SchedulerConfig = Annotated[
    StepLRConfig
    | MultiStepLRConfig
    | ExponentialLRConfig
    | CosineAnnealingLRConfig,
    Field(discriminator="kind"),
]


class CrossEntropyLossConfig(ConfigModel):
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


LossConfig = Annotated[
    CrossEntropyLossConfig,
    Field(discriminator="kind"),
    default_kind("cross_entropy"),
]


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


def _default_trackers() -> list[TrackerConfig]:
    return [StdoutTrackerConfig()]


class TrainingConfig(ConfigModel):
    epochs: int = Field(default=10, gt=0)
    batch_size: int = Field(default=4096, gt=0)
    val_fraction: float | None = Field(default=0.1, gt=0, lt=1)
    seed: int = 42
    shuffle: bool = True
    device: str = "auto"
    trackers: list[TrackerConfig] = Field(default_factory=_default_trackers)
    track_gradients: bool = False
    checkpoints: list[CheckpointConfig]

    @model_validator(mode="after")
    def _one_recovery_at_most_one_best(self) -> Self:
        recoveries = sum(
            isinstance(item, RecoveryCheckpointConfig)
            for item in self.checkpoints
        )
        bests = sum(
            isinstance(item, BestCheckpointConfig) for item in self.checkpoints
        )
        if recoveries != 1:
            raise ValueError(
                "training.checkpoints requires exactly one recovery entry"
            )
        if bests > 1:
            raise ValueError(
                "training.checkpoints allows at most one best entry"
            )
        best = self.best()
        if best is not None:
            recovery_path = self.recovery().path.resolve()
            best_path = best.path.resolve()
            if recovery_path == best_path:
                raise ValueError(
                    "training.checkpoints recovery and best paths must "
                    f"differ, both resolve to {recovery_path}"
                )
        return self

    def recovery(self) -> RecoveryCheckpointConfig:
        for item in self.checkpoints:
            if isinstance(item, RecoveryCheckpointConfig):
                return item
        raise RuntimeError("recovery checkpoint is required")

    def best(self) -> BestCheckpointConfig | None:
        for item in self.checkpoints:
            if isinstance(item, BestCheckpointConfig):
                return item
        return None


class InferenceConfig(ConfigModel):
    batch_size: int = Field(default=8192, gt=0)
    save: bool = True
    report: bool = True
    output_path: Path = Path("outputs/predictions.csv")


class ExperimentConfig(ConfigModel):
    data: DataConfig
    raw: RawPipelineConfig = Field(default_factory=RawPipelineConfig)
    tensor: TensorPipelineConfig = Field(default_factory=TensorPipelineConfig)
    model: ModelConfig
    optimizer: OptimizerConfig = Field(default_factory=AdamWConfig)
    scheduler: SchedulerConfig | None = None
    loss: LossConfig = Field(default_factory=CrossEntropyLossConfig)
    training: TrainingConfig
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    @model_validator(mode="after")
    def _validate_target_classes(self) -> Self:
        if isinstance(self.model, MLPModelConfig):
            values = set(self.data.target.mapping.values())
            expected = set(range(self.model.n_classes))
            if values != expected:
                raise ValueError(
                    "target mapping values must be exactly "
                    f"0..{self.model.n_classes - 1}, got {sorted(values)}"
                )
        return self


def load_config(path: Path | str) -> ExperimentConfig:
    """The one load path: YAML file -> validated `ExperimentConfig`."""
    source = Path(path)
    raw = yaml.safe_load(source.read_text())
    if not isinstance(raw, dict):
        raise TypeError(f"config at {source} must be a YAML mapping")
    return ExperimentConfig.model_validate(raw)

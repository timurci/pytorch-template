"""Raw processing: steps over named raw columns, before any tensor exists.

Raw processing is the first half of preprocessing (`features.tensor` is the
second). A raw step receives a `polars` frame and returns one — same rows,
in the same order — so it sees column names and raw values (strings, tokens,
ids). Feature selection, renaming, deriving columns, and encoding are all raw
steps; they run once per read, on the CPU, inside the dataset adapter.

Raw processing is declared state: steps are built from config, never fit from
the data they process, so train and inference process identically. Prefer
tensor processing (`features.tensor`) for any transform that can wait — it is
vectorized over a batch and runs on the target device.

A pipeline knows its output columns without reading rows: each step maps the
input column names to its output names, so column roles and the feature width
are resolved from declarations alone.

The schemas at the bottom are the declarative form of these steps: `kind`
picks the implementation in YAML, `build()` constructs the plain step. The
target's declared class mapping (`TargetConfig`) is a role-level encoding
applied at tensorization, so it stays with the column roles in `data`.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal, Protocol, Self

import polars as pl
import torch
from pydantic import Field, model_validator
from torch import Tensor

from template.config_pattern import ConfigModel


class RawStep(Protocol):
    """One raw-processing step: a frame in, a frame out, column names visible."""

    def columns(self, inputs: Sequence[str]) -> tuple[str, ...]:
        """The output column names, from the input names — no rows read.

        Declaring the layout up front keeps the feature width known before
        any row is read; `process` must emit exactly these columns.
        """
        ...

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Transform `frame`; the row count and row order are preserved."""
        ...


class OneHot:
    """Expands a categorical column into one float column per declared level.

    The source column is replaced by `f"{column}={level}"` columns, so the
    expansion is selection-friendly: each level is an ordinary (numeric)
    column that roles, exclusions, or later steps can name. Levels are
    declared state — inline, or from an artifact file with one level per line
    (for vocabularies too large for a config) — so train and inference
    process identically. Values outside the levels raise `ValueError`.
    """

    def __init__(
        self,
        column: str,
        levels: Sequence[str] | None = None,
        *,
        levels_path: Path | str | None = None,
    ) -> None:
        if (levels is None) == (levels_path is None):
            raise ValueError("declare exactly one of levels or levels_path")
        if levels_path is not None:
            levels = [
                line
                for line in Path(levels_path).read_text().splitlines()
                if line
            ]
        if not levels:
            raise ValueError("levels must not be empty")
        if len(set(levels)) != len(levels):
            raise ValueError("levels must be unique")
        self._column = column
        self._levels = tuple(levels)

    def columns(self, inputs: Sequence[str]) -> tuple[str, ...]:
        if self._column not in inputs:
            raise ValueError(f"not an input column: {self._column!r}")
        return tuple(name for name in inputs if name != self._column) + tuple(
            self._output_name(level) for level in self._levels
        )

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        values = frame.get_column(self._column)
        uncovered = sorted(
            set(values.unique().to_list()) - set(self._levels)
        )
        if uncovered:
            raise ValueError(
                f"values {uncovered} not covered by levels "
                f"for column {self._column!r}"
            )
        expanded = [
            (values == level).cast(pl.Float32).alias(self._output_name(level))
            for level in self._levels
        ]
        return frame.drop(self._column).with_columns(expanded)

    def _output_name(self, level: str) -> str:
        return f"{self._column}={level}"


class DropColumns:
    """Drops the named columns: raw, name-based feature selection."""

    def __init__(self, columns: Sequence[str]) -> None:
        if not columns:
            raise ValueError("columns must not be empty")
        self._columns = tuple(dict.fromkeys(columns))

    def columns(self, inputs: Sequence[str]) -> tuple[str, ...]:
        missing = [name for name in self._columns if name not in inputs]
        if missing:
            raise ValueError(f"unknown columns to drop: {missing}")
        return tuple(name for name in inputs if name not in self._columns)

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.drop(list(self._columns))


class RawPipeline:
    """Ordered raw steps applied to one frame at a time.

    Construction order is execution order. The output columns are resolved
    statically from the input names and each step's declaration, so a caller
    can build the column roles (and the model's input width) without reading
    a row.
    """

    def __init__(
        self, steps: Sequence[RawStep], columns: Sequence[str]
    ) -> None:
        self._steps = tuple(steps)
        output = tuple(columns)
        for step in steps:
            output = step.columns(output)
        if len(set(output)) != len(output):
            raise ValueError(
                f"raw pipeline produces duplicate columns: {output}"
            )
        self._output_columns = output

    @property
    def output_columns(self) -> tuple[str, ...]:
        return self._output_columns

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        for step in self._steps:
            frame = step.process(frame)
        return frame


class TargetEncoder(Protocol):
    """Turns the raw target column into one long column."""

    def encode(self, values: pl.Series) -> Tensor:
        """`(len(values),)` long tensor."""
        ...


class MapValues:
    """Maps raw target values to declared class indices (injective)."""

    def __init__(self, mapping: Mapping[str, int]) -> None:
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("mapping must be injective")
        self._mapping = dict(mapping)

    def encode(self, values: pl.Series) -> Tensor:
        try:
            return torch.tensor(
                [self._mapping[value] for value in values.to_list()],
                dtype=torch.long,
            )
        except KeyError as error:
            raise ValueError(
                f"value {error.args[0]!r} not covered by mapping "
                f"for column {values.name!r}"
            ) from None


def encode_target(values: pl.Series, encoder: TargetEncoder | None) -> Tensor:
    """Encode the target column: through `encoder`, or numeric passthrough."""
    if encoder is not None:
        return encoder.encode(values)
    return torch.tensor(values.to_list(), dtype=torch.long)


class OneHotConfig(ConfigModel):
    kind: Literal["one_hot"] = "one_hot"
    column: str
    levels: list[str] | None = None
    levels_path: Path | None = None

    @model_validator(mode="after")
    def _validate_levels_source(self) -> Self:
        if (self.levels is None) == (self.levels_path is None):
            raise ValueError("declare exactly one of levels or levels_path")
        if self.levels is not None:
            if not self.levels:
                raise ValueError("levels must not be empty")
            if len(set(self.levels)) != len(self.levels):
                raise ValueError("levels must be unique")
        return self

    def build(self) -> OneHot:
        return OneHot(self.column, self.levels, levels_path=self.levels_path)


class DropColumnsConfig(ConfigModel):
    kind: Literal["drop_columns"] = "drop_columns"
    columns: list[str] = Field(min_length=1)

    def build(self) -> DropColumns:
        return DropColumns(self.columns)


RawStepConfig = Annotated[
    OneHotConfig | DropColumnsConfig,
    Field(discriminator="kind"),
]


class RawPipelineConfig(ConfigModel):
    """One ordered list of raw-processing steps (config `raw.steps`).

    Steps run in list order, per read, over the named raw columns. `build`
    takes the raw column names so the pipeline resolves its output columns
    statically. Prefer tensor processing (`tensor.steps`) for any transform
    that does not need the raw form.
    """

    steps: list[RawStepConfig] = Field(default_factory=list)

    def build(self, columns: Sequence[str]) -> RawPipeline:
        return RawPipeline([step.build() for step in self.steps], columns)


class TargetConfig(ConfigModel):
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

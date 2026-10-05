"""The batch contract, the validated table, and the torch-side dataset adapter.

Raw rows are processed by a `RawPipeline` (`features.raw`) through
`ProcessedSource`, then a `TableSchema` declares what each processed column
is: roles only, over numeric columns. `ValidatedTable` enforces that
declaration when rows are read, and `ValidatedSource` applies it to a raw
row source. `TableDataset` is the one torch-aware adapter here: it stacks
the numeric feature columns into a `Batch` and maps the target. Tensor
processing (`features.tensor`) happens later, on the target device. Splits
and target statistics are index-based helpers so they work identically for
eager and lazy sources.
"""

import random
from collections.abc import Sequence
from typing import NotRequired, TypedDict

import polars as pl
import torch
from torch import Tensor
from torch.utils.data import Dataset

from template.data.schema import TableSchema
from template.data.source import DataSource, RawTableSource
from template.features.raw import RawPipeline, encode_target


class Batch(TypedDict):
    """The unit of data the training loop consumes: `DataLoader[Batch]`.

    `features` is a float tensor `(N, F)` and `targets`, when present, holds
    `(N,)` long class indices. `TableDataset` yields single rows `(F,)` /
    `()` that the default DataLoader collate stacks into this shape; targets
    are absent for unlabeled (inference) data. `source_indices`, present
    after feature processing, maps each row to its pre-processing row —
    provenance for row-changing steps and for consumers that need sample
    identity (e.g. contrastive pairs).

    This TypedDict is the data contract of the template: the training layer
    is written against it, and dataset implementations in this layer produce
    it. Tasks whose samples need a different shape redefine it here.
    """

    features: Tensor
    targets: NotRequired[Tensor]
    source_indices: NotRequired[Tensor]


class ValidatedTable:
    """A polars frame checked against a declared `TableSchema`.

    Construction enforces the schema's roles: every feature, target, and
    metadata column is present, feature columns are numeric (raw processing
    already encoded them), and the target is numeric unless a
    `target_encoder` maps it. Extra columns are ignored, so sources can
    carry auxiliary columns (e.g. an id column) through untouched. This is
    the polars side of the source seam; the torch side (tensorization)
    happens at ingestion, inside `TableDataset`.
    """

    def __init__(self, frame: pl.DataFrame, schema: TableSchema) -> None:
        required = [*schema.feature_columns]
        if schema.target_column is not None:
            required.append(schema.target_column)
        required.extend(schema.metadata_columns)
        _require_columns(frame, required)
        non_numeric = [
            name
            for name in schema.feature_columns
            if not frame.schema[name].is_numeric()
        ]
        if non_numeric:
            raise TypeError(
                "non-numeric feature columns need a raw step that encodes "
                f"them: {sorted(non_numeric)}"
            )
        if (
            schema.target_column is not None
            and schema.target_encoder is None
            and not frame.schema[schema.target_column].is_numeric()
        ):
            raise TypeError(
                f"non-numeric target column needs an encoder: "
                f"{schema.target_column!r}"
            )
        self._frame = frame
        self._schema = schema

    @property
    def frame(self) -> pl.DataFrame:
        return self._frame

    @property
    def schema(self) -> TableSchema:
        return self._schema


class ValidatedSource:
    """`DataSource[ValidatedTable]`: raw rows validated against the schema.

    Adapter from the raw side of the seam (`RawTableSource`) to the
    validated side: every read wraps the raw rows in a `ValidatedTable`, so
    raw sources stay schema-agnostic and validation is applied exactly once
    at the boundary.
    """

    def __init__(self, raw: RawTableSource, schema: TableSchema) -> None:
        self._raw = raw
        self._schema = schema

    @property
    def schema(self) -> TableSchema:
        return self._schema

    def count(self) -> int:
        return self._raw.count()

    def read(self, indices: Sequence[int]) -> ValidatedTable:
        return ValidatedTable(self._raw.read(list(indices)), self._schema)


class ProcessedSource:
    """`RawTableSource` with a `RawPipeline` applied to every read.

    Adapter from a raw source to its raw-processed form: `columns` is the
    pipeline's declared output, so the schema (column roles and the model's
    input width) resolves from declarations before any row is read, and
    every read returns the processed frame that `ValidatedSource` then
    checks.
    """

    def __init__(self, raw: RawTableSource, pipeline: RawPipeline) -> None:
        self._raw = raw
        self._pipeline = pipeline

    @property
    def columns(self) -> tuple[str, ...]:
        return self._pipeline.output_columns

    def count(self) -> int:
        return self._raw.count()

    def read(self, indices: Sequence[int]) -> pl.DataFrame:
        return self._pipeline.process(self._raw.read(list(indices)))


class TableDataset(Dataset[Batch]):
    """Torch adapter over a `DataSource[ValidatedTable]`.

    Owns exactly the torch-side work: stacking the numeric feature columns
    in `feature_columns` order into a `Batch`'s `features` tensor (`F` =
    `len(feature_columns)`). `__len__` is the source's count; the feature
    count comes from the declared schema, so no row is ever read to discover
    it. `__getitem__` is the per-item fallback; `__getitems__` fetches a
    whole batch's rows with one source read, and the DataLoader uses it when
    present.
    """

    def __init__(self, source: DataSource[ValidatedTable]) -> None:
        self._source = source

    def __len__(self) -> int:
        return self._source.count()

    def __getitem__(self, index: int) -> Batch:
        return self._samples(self._source.read([index]))[0]

    def __getitems__(self, indices: list[int]) -> list[Batch]:
        return self._samples(self._source.read(indices))

    def _samples(self, table: ValidatedTable) -> list[Batch]:
        schema = table.schema
        frame = table.frame
        features = torch.stack(
            [
                torch.tensor(
                    frame.get_column(name).to_list(), dtype=torch.float32
                )
                for name in schema.feature_columns
            ],
            dim=1,
        )
        rows: list[Batch] = [
            {"features": features[index]} for index in range(frame.height)
        ]
        if schema.target_column is not None:
            targets = encode_target(
                frame.get_column(schema.target_column),
                schema.target_encoder,
            )
            for row, target in zip(rows, targets, strict=True):
                row["targets"] = target
        return rows


def partition_indices(
    count: int, val_fraction: float, *, seed: int
) -> tuple[list[int], list[int]]:
    """Seed-random train/validation partition of `range(count)`.

    Returns `(train_indices, val_indices)`; the two sets are disjoint and
    together cover all rows. Works for any source, eager or lazy, because
    it partitions indices, not frames.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1, exclusive")
    n_val = int(count * val_fraction)
    if n_val == 0 or n_val == count:
        raise ValueError("split would leave an empty partition")
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return order[n_val:], order[:n_val]


def class_counts(
    source: DataSource[ValidatedTable], indices: Sequence[int]
) -> dict[int, int]:
    """Target distribution of the rows at `indices`, read through the port."""
    schema = source.schema
    if schema.target_column is None:
        raise ValueError("source has no target column")
    frame = source.read(indices).frame
    targets = encode_target(
        frame.get_column(schema.target_column), schema.target_encoder
    )
    counts = torch.bincount(targets)
    return {index: int(value) for index, value in enumerate(counts.tolist())}


def _require_columns(frame: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

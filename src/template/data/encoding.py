"""Raw column values to tensors: the encoders ingestion applies per column.

Encoding happens where raw form exists — at read time, before any tensor
is built — so feature processing never sees strings, tokens, or column
names. Encoders are declared state (inline values or pre-fit artifacts
such as a vocab file), never fit from the data they encode.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import polars as pl
import torch
from torch import Tensor


class FeatureEncoder(Protocol):
    """Turns one raw feature column into `width` float columns."""

    width: int

    def encode(self, values: pl.Series) -> Tensor:
        """`(len(values), width)` float32 tensor."""
        ...


class TargetEncoder(Protocol):
    """Turns the raw target column into one long column."""

    def encode(self, values: pl.Series) -> Tensor:
        """`(len(values),)` long tensor."""
        ...


class OneHot:
    """One-hot expansion of a categorical column.

    Levels are declared state — inline, or from an artifact file with one
    level per line (for vocabularies too large for a config) — so train and
    inference encode identically. Values outside the levels raise
    `ValueError`.
    """

    def __init__(
        self,
        levels: Sequence[str] | None = None,
        *,
        levels_path: Path | str | None = None,
    ) -> None:
        if (levels is None) == (levels_path is None):
            raise ValueError(
                "declare exactly one of levels or levels_path"
            )
        if levels_path is not None:
            levels = [
                line
                for line in Path(levels_path)
                .read_text()
                .splitlines()
                if line
            ]
        if not levels:
            raise ValueError("levels must not be empty")
        if len(set(levels)) != len(levels):
            raise ValueError("levels must be unique")
        self._levels = tuple(levels)
        self._index = {level: index for index, level in enumerate(levels)}
        self.width = len(self._levels)

    def encode(self, values: pl.Series) -> Tensor:
        try:
            indices = [
                self._index[value] for value in values.to_list()
            ]
        except KeyError as error:
            raise ValueError(_uncovered(values, error.args[0], "levels")) from None
        rows = torch.zeros(len(values), self.width)
        if indices:
            rows[
                torch.arange(len(values)), torch.tensor(indices)
            ] = 1.0
        return rows


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
                _uncovered(values, error.args[0], "mapping")
            ) from None


def encode_feature(values: pl.Series, encoder: FeatureEncoder | None) -> Tensor:
    """Encode one feature column: through `encoder`, or numeric passthrough."""
    if encoder is not None:
        return encoder.encode(values)
    return torch.tensor(values.to_list(), dtype=torch.float32).reshape(
        len(values), 1
    )


def encode_target(values: pl.Series, encoder: TargetEncoder | None) -> Tensor:
    """Encode the target column: through `encoder`, or numeric passthrough."""
    if encoder is not None:
        return encoder.encode(values)
    return torch.tensor(values.to_list(), dtype=torch.long)


def _uncovered(values: pl.Series, value: object, label: str) -> str:
    return (
        f"value {value!r} not covered by {label} "
        f"for column {values.name!r}"
    )

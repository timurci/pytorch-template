from collections.abc import Sequence

import polars as pl

from template.features._columns import require_columns
from template.features.protocol import Transform


class OneHot(Transform):
    def __init__(self, column: str, levels: Sequence[str]) -> None:
        if not levels:
            raise ValueError("levels must not be empty")
        if len(set(levels)) != len(levels):
            raise ValueError("levels must be unique")
        self._column = column
        self._levels = list(levels)

    def apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        require_columns(frame, [self._column])
        unknown = set(frame.get_column(self._column).unique().to_list()) - set(
            self._levels
        )
        if unknown:
            raise ValueError(
                f"unknown values in {self._column!r}: {sorted(unknown, key=str)}"
            )
        dummies = [
            (pl.col(self._column) == level)
            .cast(pl.Int8)
            .alias(self._dummy_name(level))
            for level in self._levels
        ]
        return frame.with_columns(dummies).drop(self._column)

    def revert(self, frame: pl.DataFrame) -> pl.DataFrame:
        dummies = [self._dummy_name(level) for level in self._levels]
        require_columns(frame, dummies)
        sums = frame.select(pl.sum_horizontal(dummies)).to_series()
        if any(value != 1 for value in sums.to_list()):
            raise ValueError(
                f"one-hot columns for {self._column!r} must have exactly "
                "one active level per row"
            )
        restored = pl.coalesce(
            pl.when(pl.col(name) == 1).then(pl.lit(level))
            for name, level in zip(dummies, self._levels, strict=True)
        )
        return frame.with_columns(restored.alias(self._column)).drop(dummies)

    def _dummy_name(self, level: str) -> str:
        return f"{self._column}__{level}"

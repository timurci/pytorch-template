import math

import polars as pl

from template.features._columns import require_columns
from template.features.protocol import Transform


class ScaleByCap(Transform):
    def __init__(
        self,
        column: str,
        cap: float,
        floor: float = 0.0,
    ) -> None:
        if cap <= floor:
            raise ValueError("cap must be greater than floor")
        self._column = column
        self._cap = cap
        self._floor = floor

    def apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        require_columns(frame, [self._column])
        scaled = (pl.col(self._column) - self._floor) / (self._cap - self._floor)
        return frame.with_columns(scaled.alias(self._column))

    def revert(self, frame: pl.DataFrame) -> pl.DataFrame:
        require_columns(frame, [self._column])
        restored = (
            pl.col(self._column) * (self._cap - self._floor) + self._floor
        )
        return frame.with_columns(restored.alias(self._column))


class LogScaleByCap(Transform):
    def __init__(self, column: str, cap: float) -> None:
        if cap <= 1:
            raise ValueError("cap must be greater than 1")
        self._column = column
        self._cap = cap

    def apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        require_columns(frame, [self._column])
        scaled = (
            pl.col(self._column).clip(lower_bound=1.0).log()
            / math.log(self._cap)
        )
        return frame.with_columns(scaled.alias(self._column))

    def revert(self, frame: pl.DataFrame) -> pl.DataFrame:
        require_columns(frame, [self._column])
        restored = (pl.col(self._column) * math.log(self._cap)).exp()
        return frame.with_columns(restored.alias(self._column))

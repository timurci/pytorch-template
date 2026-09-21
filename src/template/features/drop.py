from collections.abc import Sequence

import polars as pl

from template.features._columns import require_columns
from template.features.protocol import Transform


class DropColumns(Transform):
    def __init__(self, columns: Sequence[str]) -> None:
        self._columns = list(columns)

    def apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        require_columns(frame, self._columns)
        return frame.drop(self._columns)

    def revert(self, frame: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError(
            "DropColumns is not invertible: dropped columns cannot be restored"
        )

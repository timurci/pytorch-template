from collections.abc import Collection, Mapping

import polars as pl

from template.features._columns import require_columns
from template.features.protocol import Transform


class MapValues(Transform):
    def __init__(self, column: str, mapping: Mapping[str, int]) -> None:
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("mapping must be injective")
        self._column = column
        self._mapping = dict(mapping)
        self._inverse = {value: key for key, value in mapping.items()}

    def apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        _require_known(frame, self._column, self._mapping, "mapping")
        return frame.with_columns(
            pl.col(self._column).replace_strict(self._mapping)
        )

    def revert(self, frame: pl.DataFrame) -> pl.DataFrame:
        _require_known(frame, self._column, self._inverse, "inverse mapping")
        return frame.with_columns(
            pl.col(self._column).replace_strict(self._inverse)
        )


def _require_known(
    frame: pl.DataFrame,
    column: str,
    known: Collection[object],
    label: str,
) -> None:
    require_columns(frame, [column])
    unknown = set(frame.get_column(column).unique().to_list()) - set(known)
    if unknown:
        raise ValueError(
            f"values not covered by {label} for {column!r}: "
            f"{sorted(unknown, key=str)}"
        )

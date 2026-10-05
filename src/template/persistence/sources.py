"""Raw table readers: the whole file in memory, rows served by index.

Raw sources are deliberately unvalidated: they return plain frames, and
raw processing, roles, and validation are applied at the composition seam
(`features.raw` + `data.TableSchema` / `data.ValidatedSource`). Validation
here would couple the readers to the schema's declarations.

Lazy readers (e.g. `pl.scan_csv`) are a documented pattern, not shipped:
a polars lazy scan re-executes its query on every `read`, so per-batch
reads only pay off on indexed sources such as a database.
"""

from collections.abc import Sequence
from pathlib import Path

import polars as pl


class CsvSource:
    """Reads a whole CSV at construction; serves row slices by index."""

    def __init__(self, path: Path | str) -> None:
        self._frame = pl.read_csv(path)

    @property
    def columns(self) -> Sequence[str]:
        return self._frame.columns

    def count(self) -> int:
        return self._frame.height

    def read(self, indices: Sequence[int]) -> pl.DataFrame:
        return self._frame[list(indices)]


class ParquetSource:
    """Reads a whole Parquet file at construction; serves row slices."""

    def __init__(self, path: Path | str) -> None:
        self._frame = pl.read_parquet(path)

    @property
    def columns(self) -> Sequence[str]:
        return self._frame.columns

    def count(self) -> int:
        return self._frame.height

    def read(self, indices: Sequence[int]) -> pl.DataFrame:
        return self._frame[list(indices)]

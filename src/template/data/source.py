"""The data-source port: rows by index, raw and validated.

`RawTableSource` serves unvalidated frames — the raw side of the seam, for
readers that know nothing about raw processing or roles. `DataSource` serves
rows in the validated form its consumer declares; adapters apply validation
at the boundary (e.g. `data.table.ValidatedSource`).
"""

from collections.abc import Sequence
from typing import Protocol, TypeVar

import polars as pl

from template.data.schema import TableSchema

T_co = TypeVar("T_co", covariant=True)


class DataSource(Protocol[T_co]):
    """Serves rows of a table by index, in the form `T_co`.

    `schema` declares the minimal guaranteed structure of what `read`
    returns; `count` is the number of addressable rows; `read` returns the
    rows at `indices` (any subset, any order, validated against `schema`).

    Random access is mandatory: the map-style dataset built on this port
    needs index-addressable rows, so pure streams are out of scope for the
    port and must adapt (e.g. by materializing a buffer) at the boundary.
    """

    @property
    def schema(self) -> TableSchema: ...

    def count(self) -> int: ...

    def read(self, indices: Sequence[int]) -> T_co: ...


class RawTableSource(Protocol):
    """Unvalidated rows by index: the raw side of the same seam."""

    @property
    def columns(self) -> Sequence[str]: ...

    def count(self) -> int: ...

    def read(self, indices: Sequence[int]) -> pl.DataFrame: ...

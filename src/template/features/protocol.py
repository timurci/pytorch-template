from typing import Protocol

import polars as pl


class Transform(Protocol):
    def apply(self, frame: pl.DataFrame) -> pl.DataFrame: ...

    def revert(self, frame: pl.DataFrame) -> pl.DataFrame: ...

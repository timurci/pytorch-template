from collections.abc import Sequence

import polars as pl


def require_columns(frame: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

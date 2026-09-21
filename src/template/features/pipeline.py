from collections.abc import Sequence

import polars as pl

from template.features.protocol import Transform


class Pipeline(Transform):
    def __init__(self, steps: Sequence[Transform]) -> None:
        self._steps = list(steps)

    def apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        for step in self._steps:
            frame = step.apply(frame)
        return frame

    def revert(self, frame: pl.DataFrame) -> pl.DataFrame:
        for step in reversed(self._steps):
            frame = step.revert(frame)
        return frame

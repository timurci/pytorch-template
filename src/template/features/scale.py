"""Deterministic per-block scaling steps."""

import math

from torch import Generator

from template.data import Batch


class ScaleByCap:
    """Scales one feature block to `(x - floor) / (cap - floor)`.

    The block is a `TableSchema.feature_slice` resolved at composition, so
    the step is tensor-only and sees no column names.
    """

    def __init__(self, block: slice, cap: float, floor: float = 0.0) -> None:
        if cap <= floor:
            raise ValueError("cap must be greater than floor")
        self._block = block
        self._cap = cap
        self._floor = floor

    def process(self, batch: Batch, rng: Generator) -> Batch:
        features = batch["features"].clone()
        block = features[:, self._block]
        features[:, self._block] = (block - self._floor) / (
            self._cap - self._floor
        )
        return {**batch, "features": features}


class LogScaleByCap:
    """Scales one feature block by `log(clamp(x, min=1)) / log(cap)`."""

    def __init__(self, block: slice, cap: float) -> None:
        if cap <= 1:
            raise ValueError("cap must be greater than 1")
        self._block = block
        self._cap = cap

    def process(self, batch: Batch, rng: Generator) -> Batch:
        features = batch["features"].clone()
        block = features[:, self._block]
        features[:, self._block] = block.clamp(min=1.0).log() / math.log(
            self._cap
        )
        return {**batch, "features": features}

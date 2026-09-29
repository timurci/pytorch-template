"""Stochastic tensor noise: the template's augmentation step."""

import torch
from torch import Generator

from template.data import Batch


class GaussianNoise:
    """Adds iid Gaussian noise to all features; rows and labels preserved.

    Draws from the supplied `rng` (never the global RNG), so a seeded
    pipeline reproduces exactly. Tag it `train` only (the default) for
    ordinary augmentation, or also `eval` for seeded validation-time
    augmentation.
    """

    def __init__(self, std: float) -> None:
        if std <= 0:
            raise ValueError("std must be positive")
        self._std = std

    def process(self, batch: Batch, rng: Generator) -> Batch:
        features = batch["features"]
        # Drawn on CPU from the pipeline's CPU generator and moved, so one
        # seed reproduces the same noise on every device.
        noise = torch.randn(
            features.shape, generator=rng, dtype=features.dtype
        ).to(features.device)
        return {**batch, "features": features + noise * self._std}

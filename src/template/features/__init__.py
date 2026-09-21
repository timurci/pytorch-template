"""Reversible preprocessing transforms and train-time augmentation.

Two distinct concepts live here:

- `Transform` / `Pipeline`: offline, deterministic, reversible Polars frame
  preprocessing, applied once before the dataset is built.
- `Augmenter`: train-time, batch-level tensor augmentation applied by the
  trainer on the training device; may change the row count and returns
  provenance indices alongside the augmented batch.

No file I/O, no task-specific column names.
"""

from template.features.augment import (
    AugmentedBatch,
    Augmenter,
    GaussianNoiseAugmenter,
    IdentityAugmenter,
)
from template.features.drop import DropColumns
from template.features.map_values import MapValues
from template.features.onehot import OneHot
from template.features.pipeline import Pipeline
from template.features.protocol import Transform
from template.features.scale import LogScaleByCap, ScaleByCap

__all__ = [
    "AugmentedBatch",
    "Augmenter",
    "DropColumns",
    "GaussianNoiseAugmenter",
    "IdentityAugmenter",
    "LogScaleByCap",
    "MapValues",
    "OneHot",
    "Pipeline",
    "ScaleByCap",
    "Transform",
]

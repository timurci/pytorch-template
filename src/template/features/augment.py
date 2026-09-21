"""Train-time batch augmentation.

Augmentation is distinct from preprocessing (`Transform`): it runs inside the
training loop, once per training batch, on the training device, and may change
the number of rows. Implementations return labels aligned with the augmented
rows plus the provenance of each row (`source_indices`), so the batch the
trainer feeds to the model stays self-contained.
"""

from typing import NamedTuple, Protocol

import torch
from torch import Tensor


class AugmentedBatch(NamedTuple):
    """Output of augmenting one training batch.

    features: float tensor `(M, F)` of augmented rows; `M` may differ from
        the input batch size `N` (e.g. oversampling duplicates rows).
    targets: tensor `(M,)` of labels aligned with the augmented rows.
    source_indices: long tensor `(M,)`; `source_indices[j]` is the row of the
        *original* batch that produced augmented row `j` (provenance).

    Label-preserving augmenters MUST satisfy
    `targets == input_targets[source_indices]`: they may reorder, duplicate,
    or drop rows, but never change a row's label. Label-transforming
    augmenters (e.g. mixup soft labels) break this invariant on purpose and
    must be paired with a loss that accepts their target format.
    """

    features: Tensor
    targets: Tensor
    source_indices: Tensor


class Augmenter(Protocol):
    """Train-time, batch-level feature augmentation.

    The trainer applies the augmenter to each training batch, on the training
    device, before the forward pass. It is never applied to validation data.
    `targets` is provided so class-aware strategies (e.g. oversampling a
    minority class) can decide which rows to emit; the returned labels must
    align with the returned rows. See `AugmentedBatch`.
    """

    def augment(self, features: Tensor, targets: Tensor) -> AugmentedBatch: ...


class IdentityAugmenter:
    """Pass-through: returns the batch unchanged."""

    def augment(self, features: Tensor, targets: Tensor) -> AugmentedBatch:
        return AugmentedBatch(
            features=features,
            targets=targets,
            source_indices=torch.arange(
                features.shape[0], device=features.device
            ),
        )


class GaussianNoiseAugmenter:
    """Adds iid Gaussian noise to features; labels and row count preserved.

    `std` is the noise standard deviation, in the same units as the (already
    preprocessed) features.
    """

    def __init__(self, std: float) -> None:
        if std <= 0:
            raise ValueError("std must be positive")
        self._std = std

    def augment(self, features: Tensor, targets: Tensor) -> AugmentedBatch:
        noise = torch.randn_like(features) * self._std
        return AugmentedBatch(
            features=features + noise,
            targets=targets,
            source_indices=torch.arange(
                features.shape[0], device=features.device
            ),
        )

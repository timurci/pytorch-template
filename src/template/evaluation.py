"""Binary-classification analysis over class-1 probabilities.

The shipped inference contract is the class-1 probability, so this module
scores exactly that signal against binary labels: a thresholded confusion
matrix with its derived rates, plus the threshold-free ranking and
calibration scores (`roc_auc`, `average_precision`, `brier`, `log_loss`).

It is a leaf, like `config_pattern`: `torch` only, no sibling-layer imports,
no config, no I/O. The entrypoint owns reading the checkpoint and writing
the report artifact; the training loop keeps its own multiclass metrics.

`roc_auc` and `average_precision` are NaN when only one class is present —
a ranking score of a single-class sample is undefined, not zero.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

_EPSILON = 1e-7


@dataclass(frozen=True)
class Confusion:
    """The four counts of one probability threshold."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def rows(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )

    @property
    def accuracy(self) -> float:
        return _ratio(self.true_positive + self.true_negative, self.rows)

    @property
    def precision(self) -> float:
        return _ratio(
            self.true_positive, self.true_positive + self.false_positive
        )

    @property
    def recall(self) -> float:
        return _ratio(
            self.true_positive, self.true_positive + self.false_negative
        )

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        return _ratio(2 * precision * recall, precision + recall)


@dataclass(frozen=True)
class BinaryMetrics:
    """One operating point plus the threshold-free scores of the same rows."""

    rows: int
    positives: int
    negatives: int
    threshold: float
    confusion: Confusion
    roc_auc: float
    average_precision: float
    brier: float
    log_loss: float


def confusion(
    probabilities: Tensor,
    targets: Tensor,
    *,
    threshold: float = 0.5,
) -> Confusion:
    """Count rows at one threshold; a row is positive at `probability >= t`."""
    predicted, expected = _aligned_binary(probabilities, targets)
    positive = (predicted >= threshold).to(torch.float64)
    return Confusion(
        true_positive=int((positive * expected).sum().item()),
        false_positive=int((positive * (1 - expected)).sum().item()),
        true_negative=int(((1 - positive) * (1 - expected)).sum().item()),
        false_negative=int(((1 - positive) * expected).sum().item()),
    )


def binary_metrics(
    probabilities: Tensor,
    targets: Tensor,
    *,
    threshold: float = 0.5,
) -> BinaryMetrics:
    """Thresholded counts plus threshold-free scores of one prediction set."""
    predicted, expected = _aligned_binary(probabilities, targets)
    positives = int(expected.sum().item())
    return BinaryMetrics(
        rows=int(expected.numel()),
        positives=positives,
        negatives=int(expected.numel()) - positives,
        threshold=threshold,
        confusion=confusion(predicted, expected, threshold=threshold),
        roc_auc=_roc_auc(predicted, expected),
        average_precision=_average_precision(predicted, expected),
        brier=float(((predicted - expected) ** 2).mean().item()),
        log_loss=_log_loss(predicted, expected),
    )


def _aligned_binary(
    probabilities: Tensor, targets: Tensor
) -> tuple[Tensor, Tensor]:
    """float64 `(probability, label)` pair, validated at the boundary."""
    if probabilities.shape != targets.shape:
        raise ValueError(
            "probabilities and targets must have the same shape, got "
            f"{tuple(probabilities.shape)} and {tuple(targets.shape)}"
        )
    if probabilities.numel() == 0:
        raise ValueError("no rows to evaluate")
    if not bool(
        torch.isin(
            targets, torch.tensor([0, 1], dtype=targets.dtype)
        ).all()
    ):
        raise ValueError("targets must be class indices 0 or 1")
    return probabilities.to(torch.float64), targets.to(torch.float64)


def _roc_auc(probabilities: Tensor, targets: Tensor) -> float:
    """Rank-based AUC (Mann-Whitney U), ties sharing their average rank."""
    positives = targets.sum()
    negatives = probabilities.numel() - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _average_ranks(probabilities)
    rank_sum = ranks[targets == 1].sum()
    return float(
        (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    )


def _average_ranks(values: Tensor) -> Tensor:
    """1-based ascending ranks, ties receiving the mean of their ranks."""
    _, inverse = torch.unique(values, return_inverse=True)
    counts = torch.bincount(inverse).to(torch.float64)
    last = counts.cumsum(dim=0)
    first = last - counts + 1
    return ((first + last) / 2)[inverse]


def _average_precision(probabilities: Tensor, targets: Tensor) -> float:
    """Step-interpolated precision-recall AUC of the positive class."""
    positives = targets.sum()
    negatives = probabilities.numel() - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(probabilities, descending=True, stable=True)
    ordered = targets[order]
    true_positive = ordered.cumsum(dim=0)
    false_positive = (1 - ordered).cumsum(dim=0)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / positives
    previous = torch.cat([torch.zeros(1, dtype=recall.dtype), recall[:-1]])
    return float(((recall - previous) * precision).sum())


def _log_loss(probabilities: Tensor, targets: Tensor) -> float:
    clamped = probabilities.clamp(_EPSILON, 1 - _EPSILON)
    return float(
        (-(targets * clamped.log() + (1 - targets) * (1 - clamped).log()))
        .mean()
        .item()
    )


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0

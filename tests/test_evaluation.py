"""Binary evaluation metrics over class-1 probabilities."""

import math

import pytest
import torch

from template.evaluation import binary_metrics, confusion


def _probabilities(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def _targets(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long)


def test_confusion_counts_the_operating_point() -> None:
    metrics = confusion(
        _probabilities([0.9, 0.1, 0.6, 0.4]),
        _targets([1, 0, 0, 1]),
    )
    assert metrics.true_positive == 1
    assert metrics.false_positive == 1
    assert metrics.true_negative == 1
    assert metrics.false_negative == 1
    assert metrics.rows == 4
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)


def test_threshold_moves_the_operating_point() -> None:
    probabilities = _probabilities([0.6, 0.6])
    targets = _targets([1, 0])
    low = confusion(probabilities, targets, threshold=0.5)
    assert (low.true_positive, low.false_positive) == (1, 1)
    high = confusion(probabilities, targets, threshold=0.7)
    assert (high.true_negative, high.false_negative) == (1, 1)


def test_perfect_separation_scores_one() -> None:
    metrics = binary_metrics(
        _probabilities([0.1, 0.2, 0.8, 0.9]), _targets([0, 0, 1, 1])
    )
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.average_precision == pytest.approx(1.0)
    assert metrics.confusion.accuracy == 1.0
    assert metrics.confusion.f1 == 1.0


def test_reversed_scores_score_zero() -> None:
    metrics = binary_metrics(
        _probabilities([0.9, 0.8, 0.2, 0.1]), _targets([0, 0, 1, 1])
    )
    assert metrics.roc_auc == pytest.approx(0.0)


def test_tied_scores_share_their_average_rank() -> None:
    metrics = binary_metrics(
        _probabilities([0.5, 0.5, 0.9]), _targets([1, 0, 1])
    )
    # The tied pair splits its credit: 3 of 4 positive/negative pairs.
    assert metrics.roc_auc == pytest.approx(0.75)


def test_ranking_scores_match_a_worked_example() -> None:
    metrics = binary_metrics(
        _probabilities([0.1, 0.4, 0.35, 0.8]), _targets([0, 0, 1, 1])
    )
    assert metrics.roc_auc == pytest.approx(0.75)
    assert metrics.average_precision == pytest.approx(5 / 6)


def test_calibration_scores_are_known_values() -> None:
    metrics = binary_metrics(_probabilities([0.75, 0.25]), _targets([1, 0]))
    assert metrics.brier == pytest.approx(0.0625)
    assert metrics.log_loss == pytest.approx(-math.log(0.75), rel=1e-5)


def test_a_single_class_has_undefined_ranking_scores() -> None:
    metrics = binary_metrics(_probabilities([0.1, 0.9]), _targets([1, 1]))
    assert math.isnan(metrics.roc_auc)
    assert math.isnan(metrics.average_precision)
    assert metrics.confusion.precision == 1.0
    assert metrics.confusion.recall == pytest.approx(0.5)


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="no rows"):
        confusion(torch.empty(0), torch.empty(0, dtype=torch.long))


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="same shape"):
        confusion(_probabilities([0.5]), _targets([0, 1]))


def test_non_binary_targets_are_rejected() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        confusion(_probabilities([0.5, 0.5]), _targets([0, 2]))

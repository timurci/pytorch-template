"""`template-report`: evaluate a trained checkpoint over `data.test`.

A separate invocation, not a training side effect: it scores whatever
checkpoint is on disk, so it re-runs after a metric change or against the
recovery file instead of the best one — no retraining. `--checkpoint`
overrides the default rule (best file when it exists, otherwise recovery).

The shipped scope is binary classification, matching the class-1 probability
the inference contract emits; extend `evaluation.py` and this module for
your task.
"""

import argparse
import logging
import math
from collections.abc import Sequence
from pathlib import Path

from torch import Tensor

from template.cli.config import ExperimentConfig, load_config
from template.cli.predict import Predictions, predict
from template.cli.runtime import configure_logging
from template.evaluation import BinaryMetrics, binary_metrics, confusion

logger = logging.getLogger(__name__)

_SWEEP = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parse_args(argv)
    config = load_config(args.config)
    _require_binary(config)

    predictions = predict(config, with_labels=True, checkpoint=args.checkpoint)
    targets = predictions.targets
    if targets is None:  # `with_labels=True` always retains the target
        raise RuntimeError("predictions carry no targets")
    metrics = binary_metrics(
        predictions.probabilities, targets, threshold=config.report.threshold
    )
    _log_summary(metrics)

    path = config.report.output_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render(config, predictions, targets, metrics))
    logger.info("saved report to %s", path)


def _require_binary(config: ExperimentConfig) -> None:
    classes = set(config.data.target.mapping.values())
    if classes != {0, 1}:
        raise ValueError(
            "template-report evaluates binary classification (target mapping "
            f"classes 0 and 1), got classes {sorted(classes)}; extend "
            "evaluation.py and the report for your task"
        )


def _log_summary(metrics: BinaryMetrics) -> None:
    logger.info(
        "rows=%s positive=%s negative=%s",
        metrics.rows,
        metrics.positives,
        metrics.negatives,
    )
    logger.info(
        "roc_auc=%s average_precision=%s brier=%s log_loss=%s",
        _format(metrics.roc_auc),
        _format(metrics.average_precision),
        _format(metrics.brier),
        _format(metrics.log_loss),
    )
    point = metrics.confusion
    logger.info(
        "threshold=%s accuracy=%s precision=%s recall=%s f1=%s",
        metrics.threshold,
        _format(point.accuracy),
        _format(point.precision),
        _format(point.recall),
        _format(point.f1),
    )


def _render(
    config: ExperimentConfig,
    predictions: Predictions,
    targets: Tensor,
    metrics: BinaryMetrics,
) -> str:
    labels = {index: name for name, index in config.data.target.mapping.items()}
    negative, positive = labels[0], labels[1]
    point = metrics.confusion
    lines = [
        f"# Report: {config.data.target.column}",
        "",
        f"- checkpoint: `{predictions.checkpoint}`",
        (
            f"- rows: {metrics.rows} ({metrics.negatives} {negative}, "
            f"{metrics.positives} {positive})"
        ),
        f"- operating threshold: {metrics.threshold}",
        "",
        "## Threshold-free metrics",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| roc_auc | {_format(metrics.roc_auc)} |",
        f"| average_precision | {_format(metrics.average_precision)} |",
        f"| brier | {_format(metrics.brier)} |",
        f"| log_loss | {_format(metrics.log_loss)} |",
        "",
        f"## Operating point (threshold {metrics.threshold})",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| accuracy | {_format(point.accuracy)} |",
        f"| precision | {_format(point.precision)} |",
        f"| recall | {_format(point.recall)} |",
        f"| f1 | {_format(point.f1)} |",
        "",
        f"| | actual {negative} | actual {positive} |",
        "| --- | --- | --- |",
        (
            f"| predicted {negative} | {point.true_negative} (TN) | "
            f"{point.false_negative} (FN) |"
        ),
        (
            f"| predicted {positive} | {point.false_positive} (FP) | "
            f"{point.true_positive} (TP) |"
        ),
        "",
        "## Threshold sweep",
        "",
        "| threshold | precision | recall | f1 | predicted positive |",
        "| --- | --- | --- | --- | --- |",
    ]
    for threshold in _SWEEP:
        sweep = confusion(
            predictions.probabilities, targets, threshold=threshold
        )
        predicted_positive = sweep.true_positive + sweep.false_positive
        lines.append(
            f"| {threshold} | {sweep.precision:.4f} | {sweep.recall:.4f} | "
            f"{sweep.f1:.4f} | {predicted_positive} |"
        )
    return "\n".join(lines) + "\n"


def _format(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.4f}"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="template-report",
        description="Evaluate a trained checkpoint over data.test.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="path to the experiment YAML config",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint to evaluate (default: best file when present, "
        "otherwise the recovery file)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()

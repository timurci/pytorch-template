"""`template-infer`: run the trained classifier over a YAML experiment config."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import torch

from template.cli.config import load_config
from template.cli.predict import predict
from template.cli.runtime import configure_logging
from template.persistence import save_table

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parse_args(argv)
    config = load_config(args.config)

    predictions = predict(config, with_labels=False)
    probabilities = predictions.probabilities

    if config.inference.report:
        _report(len(predictions.ids), probabilities)

    if config.inference.save:
        frame = pl.DataFrame(
            {
                config.data.id_column: predictions.ids,
                config.data.target.column: probabilities.tolist(),
            }
        )
        save_table(frame, config.inference.output_path)
        logger.info("saved predictions to %s", config.inference.output_path)


def _report(rows: int, probabilities: torch.Tensor) -> None:
    logger.info("rows=%s", rows)
    if probabilities.numel() == 0:
        return
    logger.info(
        "probability mean=%.6f min=%.6f max=%.6f",
        probabilities.mean().item(),
        probabilities.min().item(),
        probabilities.max().item(),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="template-infer",
        description="Run inference from a YAML config and trained checkpoint.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="path to the experiment YAML config",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()

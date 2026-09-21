"""`template-infer`: run the trained classifier over a YAML experiment config."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader

from template.cli.config import load_config
from template.cli.runtime import configure_logging, feature_count
from template.data import TableDataset
from template.models import MLPClassifier
from template.persistence import load_checkpoint, load_table, save_table

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parse_args(argv)
    config = load_config(args.config)

    frame = load_table(config.data.test_path)
    ids = frame.get_column(config.data.id_column)
    features = config.pipeline.build().apply(frame)
    dataset = TableDataset.from_frame(features)

    model = MLPClassifier(
        feature_count(dataset),
        config.model.hidden_size,
        config.model.hidden_depth,
        config.model.n_classes,
        config.model.dropout,
    )
    model.load_state_dict(load_checkpoint(config.training.checkpoint_path))
    model.eval()

    loader = DataLoader(
        dataset, batch_size=config.inference.batch_size, shuffle=False
    )
    batches: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["features"])
            batches.append(torch.softmax(logits, dim=1)[:, 1])
    probabilities = torch.cat(batches) if batches else torch.empty(0)

    if config.inference.report:
        _report(len(ids), probabilities)

    if config.inference.save:
        predictions = pl.DataFrame(
            {
                config.data.id_column: ids,
                config.data.target.column: probabilities.tolist(),
            }
        )
        save_table(predictions, config.inference.output_path)
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


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
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

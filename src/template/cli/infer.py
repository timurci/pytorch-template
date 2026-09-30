"""`template-infer`: run the trained classifier over a YAML experiment config."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader

from template.cli.config import load_config
from template.cli.runtime import configure_logging
from template.data import TableDataset, ValidatedSource
from template.persistence import load_checkpoint, save_table

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parse_args(argv)
    config = load_config(args.config)

    raw = config.data.test.build()
    schema = config.data.build_schema(raw.columns, include_target=False)
    source = ValidatedSource(raw, schema)
    # Ids are metadata: they ride through ingestion untouched (row order is
    # preserved), so processing is free to reshape the feature blocks.
    ids = source.read(range(source.count())).frame.get_column(
        config.data.id_column
    )
    processing = config.processing.build(schema)

    dataset = TableDataset(source)
    model = config.model.build(schema.feature_width)
    model.load_state_dict(load_checkpoint(config.training.checkpoint_path))
    model.eval()

    loader = DataLoader(
        dataset, batch_size=config.inference.batch_size, shuffle=False
    )
    rng = torch.Generator().manual_seed(config.training.seed)
    batches: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            processed = processing.process(batch, stage="predict", rng=rng)
            logits = model(processed["features"])
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

"""`template-train`: train the classifier over a YAML experiment config."""

import argparse
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from template.cli.config import load_config
from template.cli.runtime import (
    configure_logging,
    flatten_params,
    resolve_device,
    tracking,
)
from template.data import (
    TableDataset,
    ValidatedSource,
    class_counts,
    partition_indices,
)
from template.models import MLPClassifier
from template.persistence import CsvSource, save_checkpoint
from template.training import Trainer

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parse_args(argv)
    config = load_config(args.config)
    device = resolve_device(config.training.device)

    raw = CsvSource(config.data.train_path)
    schema = config.data.build_schema(raw.columns)
    # The resolved feature set is loud on purpose: a stray or leaky column
    # shows up here (and in the tracked params) instead of in the model.
    logger.info(
        "features: %s (width %d)", schema.feature_columns, schema.feature_width
    )
    source = ValidatedSource(raw, schema)

    if config.training.val_fraction is None:
        train_indices = list(range(source.count()))
        val_indices = None
    else:
        train_indices, val_indices = partition_indices(
            source.count(),
            config.training.val_fraction,
            seed=config.training.seed,
        )

    train_counts = class_counts(source, train_indices)
    _log_class_distribution("train", train_counts)
    if val_indices is not None:
        _log_class_distribution("val", class_counts(source, val_indices))
    weights = config.loss.resolve_weights(train_counts)
    if weights is not None:
        logger.info(
            "loss class weights: %s",
            [round(value, 4) for value in weights.tolist()],
        )

    dataset = TableDataset(source)
    train_subset = Subset(dataset, train_indices)
    val_subset = (
        Subset(dataset, val_indices) if val_indices is not None else None
    )

    generator = torch.Generator().manual_seed(config.training.seed)
    train_loader = DataLoader(
        train_subset,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle,
        generator=generator,
    )
    val_loader = (
        DataLoader(
            val_subset, batch_size=config.training.batch_size, shuffle=False
        )
        if val_subset is not None
        else None
    )

    processing = config.processing.build(schema)
    model = MLPClassifier(
        schema.feature_width,
        config.model.hidden_size,
        config.model.hidden_depth,
        config.model.n_classes,
        config.model.dropout,
    ).to(device)
    optimizer = config.optimizer.build(model.parameters())
    loss_fn = config.loss.build(train_counts).to(device)

    rows = (
        f"{len(train_subset)}+{len(val_subset)}"
        if val_subset is not None
        else f"{len(train_subset)} (no validation)"
    )
    logger.info(
        "device=%s rows=%s epochs=%s", device, rows, config.training.epochs
    )
    with tracking(
        config.training.trackers, total_steps=config.training.epochs
    ) as trackers:
        params = flatten_params(config)
        if weights is not None:
            params["loss.class_weights_effective"] = [
                round(value, 6) for value in weights.tolist()
            ]
        for tracker in trackers:
            tracker.log_params(params)
        Trainer(model, optimizer, device=device).train(
            train_loader,
            val_loader,
            epochs=config.training.epochs,
            loss_fn=loss_fn,
            trackers=trackers,
            processing=processing,
            seed=config.training.seed,
            track_gradients=config.training.track_gradients,
        )

    save_checkpoint(model.state_dict(), config.training.checkpoint_path)
    logger.info("saved checkpoint to %s", config.training.checkpoint_path)


def _log_class_distribution(name: str, counts: Mapping[int, int]) -> None:
    total = sum(counts.values())
    if total == 0:
        logger.info("%s target distribution: empty", name)
        return
    summary = ", ".join(
        f"{label}={count} ({count / total:.2%})"
        for label, count in sorted(counts.items())
    )
    logger.info("%s target distribution: %s", name, summary)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="template-train",
        description="Train the classifier from a YAML config.",
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

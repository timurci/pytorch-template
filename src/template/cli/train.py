"""`template-train`: train the classifier from a YAML experiment config."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from template.cli.config import load_config
from template.cli.runtime import (
    configure_logging,
    feature_count,
    flatten_params,
    resolve_device,
    tracking,
)
from template.data import TableDataset
from template.features import MapValues
from template.models import MLPClassifier
from template.persistence import load_table, save_checkpoint
from template.training import Trainer

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parse_args(argv)
    config = load_config(args.config)
    device = resolve_device(config.training.device)

    frame = load_table(config.data.train_path)
    features = config.pipeline.build().apply(frame)
    target = MapValues(config.data.target.column, config.data.target.mapping)
    frame = target.apply(features)

    dataset = TableDataset.from_frame(
        frame, target_column=config.data.target.column
    )
    train_dataset, val_dataset = (
        (dataset, None)
        if config.training.val_fraction is None
        else dataset.split(
            config.training.val_fraction, seed=config.training.seed
        )
    )
    _log_class_distribution("train", train_dataset)
    if val_dataset is not None:
        _log_class_distribution("val", val_dataset)
    class_counts = train_dataset.class_counts()
    weights = config.loss.resolve_weights(class_counts)
    if weights is not None:
        logger.info(
            "loss class weights: %s",
            [round(value, 4) for value in weights.tolist()],
        )

    generator = torch.Generator().manual_seed(config.training.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle,
        generator=generator,
    )
    val_loader = (
        DataLoader(
            val_dataset, batch_size=config.training.batch_size, shuffle=False
        )
        if val_dataset is not None
        else None
    )

    model = MLPClassifier(
        feature_count(train_dataset),
        config.model.hidden_size,
        config.model.hidden_depth,
        config.model.n_classes,
        config.model.dropout,
    ).to(device)
    optimizer = config.optimizer.build(model.parameters())
    loss_fn = config.loss.build(class_counts).to(device)
    augmenter = (
        config.training.augmenter.build()
        if config.training.augmenter is not None
        else None
    )

    rows = (
        f"{len(train_dataset)}+{len(val_dataset)}"
        if val_dataset is not None
        else f"{len(train_dataset)} (no validation)"
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
            augmenter=augmenter,
            track_gradients=config.training.track_gradients,
        )

    save_checkpoint(model.state_dict(), config.training.checkpoint_path)
    logger.info("saved checkpoint to %s", config.training.checkpoint_path)


def _log_class_distribution(name: str, dataset: TableDataset) -> None:
    counts = dataset.class_counts()
    total = sum(counts.values())
    summary = ", ".join(
        f"{label}={count} ({count / total:.2%})"
        for label, count in sorted(counts.items())
    )
    logger.info("%s target distribution: %s", name, summary)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
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

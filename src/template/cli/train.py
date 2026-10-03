"""`template-train`: train the classifier over a YAML experiment config."""

import argparse
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from template.cli.config import ExperimentConfig, load_config
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
from template.training import (
    CheckpointStrategy,
    Trainer,
    load_best_score,
    load_training_state,
    restore_training_state,
)
from template.training.config import BestCheckpointConfig, RecoveryCheckpointConfig

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parse_args(argv)
    config = load_config(args.config)
    device = resolve_device(config.training.device)

    raw = config.data.train.build()
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
    model = config.model.build(schema.feature_width).to(device)
    optimizer = config.optimizer.build(model.parameters())
    scheduler = (
        config.scheduler.build(optimizer)
        if config.scheduler is not None
        else None
    )
    loss_fn = config.loss.build(train_counts).to(device)
    start_epoch = 0
    best_score = None
    if args.resume:
        start_epoch = _resume(config, model, optimizer, scheduler)
        best = config.training.best()
        if best is not None and best.path.is_file():
            best_score = load_best_score(best.path)
    checkpoints = _build_checkpoints(config, score=best_score)

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
        Trainer(model, optimizer, scheduler=scheduler, device=device).train(
            train_loader,
            val_loader,
            epochs=config.training.epochs,
            start_epoch=start_epoch,
            loss_fn=loss_fn,
            trackers=trackers,
            checkpoints=checkpoints,
            processing=processing,
            seed=config.training.seed,
            track_gradients=config.training.track_gradients,
        )


def _resume(
    config: ExperimentConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> int:
    path = config.training.recovery().path
    if not path.is_file():
        raise FileNotFoundError(f"recovery checkpoint not found at {path}")
    start_epoch = restore_training_state(
        load_training_state(path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    if start_epoch > config.training.epochs:
        raise ValueError(
            "recovery checkpoint completed epoch "
            f"{start_epoch - 1}, past training.epochs "
            f"({config.training.epochs})"
        )
    logger.info("resuming at epoch %s", start_epoch)
    return start_epoch


def _build_checkpoints(
    config: ExperimentConfig, *, score: float | None
) -> list[CheckpointStrategy]:
    built: list[CheckpointStrategy] = []
    for item in config.training.checkpoints:
        match item:
            case RecoveryCheckpointConfig():
                built.append(item.build(epochs=config.training.epochs))
            case BestCheckpointConfig():
                built.append(item.build(score=score))
    return built


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from the recovery checkpoint",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()

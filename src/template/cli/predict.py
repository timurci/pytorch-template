"""Shared evaluation path for the entrypoints.

`template-infer` and `template-report` both run the trained model over
`data.test`; this module owns that path so the two stay identical — the raw
source, the raw pipeline, the schema (target retained or dropped), the model
and its checkpoint, and one seeded pass at stage `predict`.

The shipped output contract is the class-1 probability.
`Predictions.targets` is present only when the caller retained the target
(`with_labels=True`) — unlabeled data has no target to read.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from template.cli.config import ExperimentConfig
from template.data import ProcessedSource, TableDataset, ValidatedSource
from template.training import inference_checkpoint, load_model_weights

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Predictions:
    """Row-aligned output of one pass over `data.test`."""

    ids: pl.Series
    probabilities: Tensor
    targets: Tensor | None
    checkpoint: Path


def predict(
    config: ExperimentConfig,
    *,
    with_labels: bool,
    checkpoint: Path | None = None,
) -> Predictions:
    """Score `data.test` with the configured model at stage `predict`.

    `checkpoint=None` keeps the inference rule: the best file when it exists,
    otherwise the recovery file. The pass is seeded from `training.seed`, so
    a seeded run reproduces exactly.
    """
    raw = config.data.test.build()
    pipeline = config.raw.build(raw.columns)
    processed_source = ProcessedSource(raw, pipeline)
    schema = config.data.build_schema(
        processed_source.columns, include_target=with_labels
    )
    source = ValidatedSource(processed_source, schema)
    # Ids are metadata: they ride through ingestion untouched (row order is
    # preserved), so tensor processing is free to reshape the feature blocks.
    ids = source.read(range(source.count())).frame.get_column(
        config.data.id_column
    )
    tensor_pipeline = config.tensor.build(schema)

    dataset = TableDataset(source)
    model = config.model.build(schema.feature_width)
    if checkpoint is None:
        best = config.training.best()
        checkpoint = inference_checkpoint(
            config.training.recovery().path,
            None if best is None else best.path,
        )
    model.load_state_dict(load_model_weights(checkpoint))
    logger.info("loaded checkpoint from %s", checkpoint)
    model.eval()

    loader = DataLoader(
        dataset, batch_size=config.inference.batch_size, shuffle=False
    )
    rng = torch.Generator().manual_seed(config.training.seed)
    probability_batches: list[Tensor] = []
    target_batches: list[Tensor] = []
    with torch.no_grad():
        for batch in loader:
            processed = tensor_pipeline.process(batch, stage="predict", rng=rng)
            logits = model(processed["features"])
            probability_batches.append(torch.softmax(logits, dim=1)[:, 1])
            if "targets" in processed:
                target_batches.append(processed["targets"])

    probabilities = (
        torch.cat(probability_batches)
        if probability_batches
        else torch.empty(0)
    )
    if not with_labels:
        targets = None
    else:
        targets = (
            torch.cat(target_batches)
            if target_batches
            else torch.empty(0, dtype=torch.long)
        )
    return Predictions(
        ids=ids,
        probabilities=probabilities,
        targets=targets,
        checkpoint=checkpoint,
    )

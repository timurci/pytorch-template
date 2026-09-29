"""The ordered processing pipeline: stage-tagged steps, one pass per batch."""

from collections.abc import Collection, Sequence

import torch
from torch import Generator

from template.data import Batch
from template.features.protocol import ProcessingStep, Stage


class ProcessingPipeline:
    """Ordered, stage-tagged steps applied to one batch at a time.

    Order is construction order: interleaving deterministic preprocessing
    with stochastic augmentation (`pre -> aug -> pre`) is just list order —
    there is no separate preprocessing and augmentation phase. Each step
    runs in the stages it is tagged with (`train`, `eval`, `predict`), and
    every step sees the same `rng`, so one seed reproduces the whole pass:
    deterministic steps ignore it, stochastic steps draw from it.

    The pipeline guarantees `source_indices` on its output — absent input
    provenance becomes `arange(N)`; steps that change rows keep it aligned.
    """

    def __init__(
        self, steps: Sequence[tuple[ProcessingStep, Collection[Stage]]]
    ) -> None:
        self._steps = [
            (step, frozenset(stages)) for step, stages in steps
        ]

    def process(self, batch: Batch, *, stage: Stage, rng: Generator) -> Batch:
        features = batch["features"]
        if "source_indices" not in batch:
            batch = {
                **batch,
                "source_indices": torch.arange(
                    features.shape[0], device=features.device
                ),
            }
        for step, stages in self._steps:
            if stage in stages:
                batch = step.process(batch, rng)
        return batch

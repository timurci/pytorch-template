"""The feature processing step contract: `Batch` in, `Batch` out.

Feature processing is the one place data changes during a run, and it is
tensor-only: a step receives a batch of tensors and returns a batch of
tensors. Raw forms, column names, and encodings never appear here — names
resolve to feature-tensor blocks (`TableSchema.feature_slice`) at
composition time, so one step works regardless of how the data was
ingested.

Step invariants:

- Steps never mutate their input; they return a `Batch` (cloning on
  write). Fields a step does not touch pass through unchanged.
- Deterministic steps ignore `rng`; stochastic steps MUST draw from `rng`
  (never the global RNG) so a seeded run reproduces exactly. This makes
  validation-time augmentation a tagging choice, not a code change: tag a
  stochastic step for the `eval` stage and it runs there, reproducibly.
- Steps that change the row count keep `source_indices` and `targets`
  aligned with `features` (label-preserving steps satisfy
  `targets == input_targets[source_indices]`).
"""

from typing import Literal, Protocol

from torch import Generator

from template.data import Batch

Stage = Literal["train", "eval", "predict"]


class ProcessingStep(Protocol):
    """One processing step over batched tensors."""

    def process(self, batch: Batch, rng: Generator) -> Batch: ...


class BlockResolver(Protocol):
    """Resolves a feature column name to its block of the feature tensor.

    Satisfied structurally by `data.TableSchema`, so feature-processing
    config can name columns while this layer keeps its `Batch`-only edge
    into `data` — the same no-import trick as the raw readers.
    """

    def feature_slice(self, column: str) -> slice: ...

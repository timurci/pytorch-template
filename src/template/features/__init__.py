"""The feature processing layer: tensor-side processing of batches.

One contract, one composition: `ProcessingStep` (`Batch` -> `Batch`, tensor
only) and `ProcessingPipeline` (ordered, stage-tagged steps applied per
batch). What used to be split into offline preprocessing over dataframes
and train-time augmentation over tensors is one machinery here: the two
differ only in stage tags and whether a step draws from the `rng`. Raw
forms are handled at ingestion (`template.data` / `template.persistence`);
this layer never sees them.
"""

from template.features.noise import GaussianNoise
from template.features.pipeline import ProcessingPipeline
from template.features.protocol import ProcessingStep, Stage
from template.features.scale import LogScaleByCap, ScaleByCap

__all__ = [
    "GaussianNoise",
    "LogScaleByCap",
    "ProcessingPipeline",
    "ProcessingStep",
    "ScaleByCap",
    "Stage",
]

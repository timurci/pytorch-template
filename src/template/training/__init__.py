"""Training loop and checkpoint strategies.

Knows `nn.Module`, `Optimizer`, an optional epoch `LRScheduler`,
`DataLoader[Batch]`, the tracker and processing protocols, and the
checkpoint read/write helpers in `persistence`.
"""

from template.training.checkpoint import (
    BestCheckpoint,
    CheckpointStrategy,
    RecoveryCheckpoint,
    TrainerState,
    inference_checkpoint,
    load_best_score,
    load_model_weights,
    load_training_state,
    restore_training_state,
)
from template.training.trainer import Trainer

__all__ = [
    "BestCheckpoint",
    "CheckpointStrategy",
    "RecoveryCheckpoint",
    "Trainer",
    "TrainerState",
    "inference_checkpoint",
    "load_best_score",
    "load_model_weights",
    "load_training_state",
    "restore_training_state",
]

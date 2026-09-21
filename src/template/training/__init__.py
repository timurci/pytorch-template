"""Training loop. Knows `nn.Module`, `Optimizer`, `DataLoader[Batch]`, and the tracker and augmenter protocols."""

from template.training.trainer import Trainer

__all__ = ["Trainer"]

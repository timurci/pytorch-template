"""Training loop.

Knows `nn.Module`, `Optimizer`, an optional epoch `LRScheduler`,
`DataLoader[Batch]`, and the tracker and processing protocols.
"""

from template.training.trainer import Trainer

__all__ = ["Trainer"]

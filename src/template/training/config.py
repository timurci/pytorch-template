"""Pydantic schemas for checkpoint strategies.

`kind` picks the strategy in YAML. `build()` returns the plain strategy.
Resuming a run and choosing which file inference loads stay in the CLI.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from template.config_pattern import ConfigModel
from template.training.checkpoint import BestCheckpoint, RecoveryCheckpoint


class RecoveryCheckpointConfig(ConfigModel):
    kind: Literal["recovery"] = "recovery"
    path: Path
    every: int | None = Field(default=None, gt=0)

    def build(self, *, epochs: int) -> RecoveryCheckpoint:
        return RecoveryCheckpoint(self.path, every=self.every, epochs=epochs)


class BestCheckpointConfig(ConfigModel):
    kind: Literal["best"] = "best"
    path: Path

    def build(self, *, score: float | None = None) -> BestCheckpoint:
        return BestCheckpoint(self.path, score=score)


CheckpointConfig = Annotated[
    RecoveryCheckpointConfig | BestCheckpointConfig,
    Field(discriminator="kind"),
]

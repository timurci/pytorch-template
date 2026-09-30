"""Pydantic schemas for `tracking`'s `ExperimentTracker` adapters.

`kind` picks the backend in YAML, `build()` returns the passive tracker.
External lifecycles (the MLflow run) stay in the entrypoints: these
schemas only describe what to log with.
"""

from typing import Annotated, Literal

from pydantic import Field

from template.config_pattern import ConfigModel
from template.tracking.mlflow import MLflowTracker
from template.tracking.null import NullTracker
from template.tracking.protocol import ExperimentTracker
from template.tracking.stdout import StdoutTracker


class NullTrackerConfig(ConfigModel):
    kind: Literal["null"] = "null"

    def build(self) -> ExperimentTracker:
        return NullTracker()


class StdoutTrackerConfig(ConfigModel):
    kind: Literal["stdout"] = "stdout"
    every_n_steps: int = Field(default=1, ge=1)

    def build(self, *, total_steps: int | None = None) -> ExperimentTracker:
        return StdoutTracker(self.every_n_steps, total_steps=total_steps)


class MLflowTrackerConfig(ConfigModel):
    kind: Literal["mlflow"] = "mlflow"
    experiment_name: str = "template"
    run_name: str | None = None
    tracking_uri: str | None = None

    def build(self) -> ExperimentTracker:
        return MLflowTracker()


TrackerConfig = Annotated[
    NullTrackerConfig | StdoutTrackerConfig | MLflowTrackerConfig,
    Field(discriminator="kind"),
]

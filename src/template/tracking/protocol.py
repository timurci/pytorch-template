from collections.abc import Mapping
from typing import Protocol


class ExperimentTracker(Protocol):
    """Observability port: the trainer reports, adapters decide where to.

    Implementations must be cheap and non-failing observers; the entrypoint
    owns any external lifecycle (e.g. opening and closing an MLflow run).
    """

    def log_params(self, params: Mapping[str, object]) -> None: ...

    def log_metrics(
        self, metrics: Mapping[str, float], *, step: int | None = None
    ) -> None: ...

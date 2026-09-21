from collections.abc import Mapping

import mlflow


class MLflowTracker:
    """Log into the active MLflow run; the caller owns the run lifecycle."""

    def log_params(self, params: Mapping[str, object]) -> None:
        mlflow.log_params({name: str(value) for name, value in params.items()})

    def log_metrics(
        self, metrics: Mapping[str, float], *, step: int | None = None
    ) -> None:
        mlflow.log_metrics(dict(metrics), step=step)

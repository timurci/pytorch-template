from collections.abc import Mapping


class NullTracker:
    def log_params(self, params: Mapping[str, object]) -> None:
        return None

    def log_metrics(
        self, metrics: Mapping[str, float], *, step: int | None = None
    ) -> None:
        return None

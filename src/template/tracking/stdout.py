import logging
from collections.abc import Mapping

logger = logging.getLogger(__name__)


class StdoutTracker:
    """Logs to the `"template"` logger, which the CLI points at stdout.

    `every_n_steps` throttles metrics output: only steps that are multiples of
    it are logged, plus the final step when `total_steps` is known.
    """

    def __init__(
        self, every_n_steps: int = 1, *, total_steps: int | None = None
    ) -> None:
        if every_n_steps < 1:
            raise ValueError("every_n_steps must be >= 1")
        self._every_n_steps = every_n_steps
        self._total_steps = total_steps

    def log_params(self, params: Mapping[str, object]) -> None:
        logger.info("params %s", dict(params))

    def log_metrics(
        self, metrics: Mapping[str, float], *, step: int | None = None
    ) -> None:
        if step is not None and not self._should_log(step):
            return
        prefix = f"step={step} " if step is not None else ""
        logger.info("%smetrics %s", prefix, dict(metrics))

    def _should_log(self, step: int) -> bool:
        if step % self._every_n_steps == 0:
            return True
        return self._total_steps is not None and step == self._total_steps - 1

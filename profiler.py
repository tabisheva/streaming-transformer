from contextlib import contextmanager
from collections import defaultdict
import time


metrics = None


class MetricStorage:
    def __init__(self):
        self._sums = defaultdict(float)
        self._counts = defaultdict(float)

    def observe(self, metric_name: str, metric_value: float):
        self._sums[metric_name] += metric_value
        self._counts[metric_name] += 1.0

    def get_avg(self, metric_name: str) -> float:
        if metric_name not in self._sums:
            return 0.0

        return self._sums[metric_name] / self._counts[metric_name]

@contextmanager
def profile() -> MetricStorage:
    global metrics
    if metrics is not None:
        yield
        return

    metrics = MetricStorage()
    yield metrics
    metrics = None


@contextmanager
def record_function(metric_name: str):
    global metrics
    if metrics is None:
        yield
        return

    start = time.perf_counter()
    yield
    end = time.perf_counter()
    delta = end - start
    metrics.observe(metric_name, delta)

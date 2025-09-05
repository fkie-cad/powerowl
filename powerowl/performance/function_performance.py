from typing import Optional, Callable, Union
from functools import wraps
import time

import numpy as np
import tabulate


def measure(func: Callable) -> Callable:
    return FunctionPerformance.measure(func)


class FunctionPerformance:
    enabled: bool = False
    instance: Optional['FunctionPerformance'] = None

    @staticmethod
    def get_instance() -> 'FunctionPerformance':
        if FunctionPerformance.instance is None:
            FunctionPerformance.instance = FunctionPerformance(True)
        return FunctionPerformance.instance

    def __init__(self, enabled: bool = False):
        FunctionPerformance.enabled = enabled
        self._timings = {}

    def _register_timing(self, wrapped: Union[Callable, str], timing: float):
        if isinstance(wrapped, str):
            wrapped_id = wrapped
        else:
            wrapped_id = wrapped.__name__
        self._timings.setdefault(wrapped_id, []).append(timing)

    @staticmethod
    def measure(wrapped: Callable) -> Callable:
        @wraps(wrapped)
        def wrapper(*args, **kwargs):
            if FunctionPerformance.enabled:
                start = time.perf_counter_ns()
                result = wrapped(*args, **kwargs)
                end = time.perf_counter_ns()
                FunctionPerformance.get_instance()._register_timing(wrapped, end - start)
                return result
            else:
                return wrapped(*args, **kwargs)
        return wrapper

    @staticmethod
    def manual_measure(name: str):
        class _PerfContext:
            def __init__(self, fp, _name: str):
                self._fp = fp
                self._name = name
                self._start_time = None

            def __enter__(self):
                self._start_time = time.perf_counter_ns()

            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.perf_counter_ns() - self._start_time
                self._fp._register_timing(self._name, duration)

        return _PerfContext(FunctionPerformance.get_instance(), name)

    def print_evaluation(self):
        metrics = ["function", "count", "mean", "max", "min", "sum"]
        aggregations = []

        def format_nanoseconds(nanoseconds: float) -> str:
            remainder = nanoseconds
            if remainder < 1000:
                return f"{nanoseconds:.2f} ns"
            remainder /= 1000
            if remainder < 1000:
                return f"{remainder:.2f} μs"
            remainder /= 1000
            if remainder < 1000:
                return f"{remainder:.2f} ms"
            remainder /= 1000
            return f"{remainder:.2f} s"

        for name, timings in self._timings.items():
            aggregations.append({
                "function": name,
                "count": len(timings),
                "mean": format_nanoseconds(np.mean(timings)),
                "max": format_nanoseconds(np.max(timings)),
                "min": format_nanoseconds(np.min(timings)),
                "sum": format_nanoseconds(np.sum(timings))
            })
        print(tabulate.tabulate(aggregations, headers="keys"))

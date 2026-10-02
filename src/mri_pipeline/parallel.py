"""Sequential and process-pool runners with timing.

All worker functions live in importable modules (not in ``__main__`` or a
notebook), so these runners work with the "spawn" start method used on
Windows and macOS as well as "fork" on Linux.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

STRATEGIES = ("sequential", "map", "imap_unordered")


def start_method_from_env() -> str | None:
    """Start method forced through the MRI_START_METHOD variable, if set."""
    value = os.environ.get("MRI_START_METHOD", "").strip()
    return value or None


def available_cores() -> int:
    """CPU cores this process may use (affinity-aware where supported)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def get_context(start_method: str | None = None) -> mp.context.BaseContext:
    return mp.get_context(start_method or start_method_from_env())


def auto_chunksize(n_items: int, processes: int) -> int:
    """The chunk size ``Pool.map`` picks when ``chunksize=None``."""
    chunks, extra = divmod(n_items, processes * 4)
    return max(1, chunks + (1 if extra else 0))


class _Indexed:
    """Picklable wrapper: calls ``func(item)`` and returns ``(index, result)``."""

    def __init__(self, func: Callable[[Any], Any]):
        self.func = func

    def __call__(self, pair: tuple[int, Any]) -> tuple[int, Any]:
        index, item = pair
        return index, self.func(item)


def run(
    func: Callable[[Any], Any],
    items: Sequence[Any],
    strategy: str = "map",
    processes: int | None = None,
    chunksize: int | None = None,
    start_method: str | None = None,
) -> list[Any]:
    """Apply ``func`` to every item and return results in input order.

    ``strategy``: "sequential" (plain loop, no pool), "map" (``Pool.map``) or
    "imap_unordered" (results arrive as they finish and are put back in input
    order). ``chunksize=None`` means the ``Pool.map`` default; for
    imap_unordered the same default is computed explicitly (its own default is 1).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    if strategy == "sequential":
        return [func(item) for item in items]
    processes = processes or available_cores()
    ctx = get_context(start_method)
    with ctx.Pool(processes=processes) as pool:
        if strategy == "map":
            return pool.map(func, items, chunksize=chunksize)
        size = chunksize or auto_chunksize(len(items), processes)
        out: list[Any] = [None] * len(items)
        for index, result in pool.imap_unordered(_Indexed(func), enumerate(items), chunksize=size):
            out[index] = result
        return out


@dataclass
class Timed:
    results: list[Any]
    seconds: float


def timed_run(func: Callable[[Any], Any], items: Sequence[Any], **kwargs: Any) -> Timed:
    """``run`` with wall-clock timing (``time.perf_counter``), pool start-up included."""
    start = time.perf_counter()
    results = run(func, items, **kwargs)
    return Timed(results, time.perf_counter() - start)


def speedup_efficiency(t_serial: float, t_parallel: float, processes: int) -> tuple[float, float]:
    speedup = t_serial / t_parallel
    return speedup, speedup / processes

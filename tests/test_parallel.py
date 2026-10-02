from __future__ import annotations

import functools
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from mri_pipeline import io, parallel
from mri_pipeline.experiments import compare_outputs
from mri_pipeline.filters import FILTER_NAMES, apply_filters

START_METHODS = ["spawn"] + (["fork"] if sys.platform.startswith("linux") else [])


@pytest.fixture(scope="module")
def images(synthetic_dir) -> list[np.ndarray]:
    return io.load_images(io.list_images(synthetic_dir))


@pytest.mark.parametrize("start_method", START_METHODS)
@pytest.mark.parametrize(
    ("strategy", "chunksize"),
    [("map", None), ("map", 1), ("imap_unordered", 1), ("imap_unordered", None)],
)
def test_parallel_equals_sequential_every_filter(images, strategy, chunksize, start_method):
    task = functools.partial(apply_filters, mode="fast")
    reference = parallel.run(task, images, strategy="sequential")
    result = parallel.run(
        task, images, strategy=strategy, processes=2, chunksize=chunksize, start_method=start_method
    )
    check = compare_outputs(reference, result)
    assert set(check) == set(FILTER_NAMES)
    assert all(v["array_equal"] and v["allclose"] for v in check.values()), check


def test_imap_unordered_restores_input_order():
    items = list(range(23))
    out = parallel.run(abs, [-i for i in items], strategy="imap_unordered", processes=3)
    assert out == items


def test_unknown_strategy():
    with pytest.raises(ValueError, match="strategy"):
        parallel.run(abs, [1], strategy="threads")


@pytest.mark.parametrize(
    ("n", "p", "expected"), [(40, 4, 3), (40, 1, 10), (5, 4, 1), (16, 4, 1), (17, 4, 2)]
)
def test_auto_chunksize_matches_pool_map_rule(n, p, expected):
    assert parallel.auto_chunksize(n, p) == expected


def test_timed_run_and_speedup():
    timed = parallel.timed_run(abs, [-1, -2], strategy="sequential")
    assert timed.results == [1, 2] and timed.seconds >= 0
    assert parallel.speedup_efficiency(10.0, 4.0, 4) == (2.5, 0.625)


def test_start_method_env(monkeypatch):
    monkeypatch.setenv("MRI_START_METHOD", "spawn")
    assert parallel.start_method_from_env() == "spawn"
    assert parallel.get_context().get_start_method() == "spawn"
    monkeypatch.setenv("MRI_START_METHOD", "")
    assert parallel.start_method_from_env() is None


def test_available_cores_positive():
    assert parallel.available_cores() >= 1


def test_compare_outputs_detects_difference(images):
    ref = [apply_filters(im, "fast") for im in images[:2]]
    changed = [dict(d) for d in ref]
    changed[1] = {**changed[1], "Sobel": changed[1]["Sobel"] + 1e-3}
    check = compare_outputs(ref, changed)
    assert not check["Sobel"]["array_equal"] and not check["Sobel"]["allclose"]
    assert check["Hessian"]["array_equal"]


SPAWN_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    from mri_pipeline.filters import gaussian_filter
    from mri_pipeline.parallel import run

    images = [np.full((8, 8), i, dtype=np.uint8) for i in range(4)]
    print(len(run(gaussian_filter, images, processes=2, start_method="spawn")))
    """
)


def _runs_under_spawn(script: str) -> bool:
    """Run ``script`` like an interactive session (python -c: no importable
    __main__, same as a Jupyter kernel) and report whether it finished."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=90,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired:
        return False  # workers died unpickling the task; Pool.map waits forever
    return proc.returncode == 0 and proc.stdout.strip() == "4"


@pytest.mark.slow
def test_spawn_safe_from_an_interactive_main():
    """Workers live in an importable module, so a pool started from a session with
    no importable __main__ (like a notebook) works under the spawn start method."""
    assert _runs_under_spawn(SPAWN_SCRIPT)

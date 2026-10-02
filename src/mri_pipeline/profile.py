"""Profile where the time goes before deciding what to parallelise.

cProfile wraps the sequential pipeline on a few images; the table reports the
cumulative time spent inside each filter function and inside the GLCM step.
Run it from a plain Python process (``python -m mri_pipeline profile``): inside
a Jupyter kernel I found cProfile's per-function times to be unreliable.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from pathlib import Path

import numpy as np

from mri_pipeline import filters
from mri_pipeline.features import features_from_filtered

STAGES = {
    "Entropy": "entropy_filter",
    "Gaussian": "gaussian_filter",
    "Sobel": "sobel_filter",
    "Gabor": "gabor_filter",
    "Hessian": "hessian_filter",
    "Prewitt": "prewitt_filter",
    "GLCM (7 images x 24 features)": "features_from_filtered",
}


def _pipeline(images: list[np.ndarray], mode: str) -> None:
    for image in images:
        features_from_filtered(filters.apply_filters(image, mode), "minmax")


def profile_pipeline(images: list[np.ndarray], mode: str = "fast") -> list[dict[str, float | str]]:
    """Return one row per stage: cumulative seconds, seconds per image, share."""
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    _pipeline(images, mode)
    profiler.disable()
    # Wall-clock total. cProfile's own cumulative time for ``_pipeline`` is not
    # reliable inside a Jupyter kernel (cells run in a coroutine and the call
    # can be counted twice), which made "Other" negative there.
    total = time.perf_counter() - start
    stats = pstats.Stats(profiler)
    cumulative: dict[str, float] = {}
    for (filename, _line, func_name), row in stats.stats.items():  # type: ignore[attr-defined]
        cumtime = row[3]
        in_package = Path(filename).parent.name == "mri_pipeline"
        for stage, target in STAGES.items():
            if func_name == target and in_package:
                cumulative[stage] = cumulative.get(stage, 0.0) + cumtime
    rows: list[dict[str, float | str]] = []
    for stage in STAGES:
        seconds = cumulative.get(stage, 0.0)
        rows.append(
            {
                "stage": stage,
                "total_s": seconds,
                "per_image_s": seconds / len(images),
                "share_pct": 100.0 * seconds / total if total else 0.0,
            }
        )
    other = total - sum(cumulative.values())
    rows.append(
        {
            "stage": "Other (loop, dict building, profiler overhead)",
            "total_s": other,
            "per_image_s": other / len(images),
            "share_pct": 100.0 * other / total if total else 0.0,
        }
    )
    rows.append(
        {"stage": "Total", "total_s": total, "per_image_s": total / len(images), "share_pct": 100.0}
    )
    return rows

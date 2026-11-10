"""GLCM texture features and the raw-pixel baseline features.

``glcm_features`` follows the ``compute_glcm_features`` helper supplied with
the original exercise. Changed here: each filter output is min-max rescaled to
0-255 before the uint8 cast; the helper's direct cast
``(image * 255).astype(np.uint8)`` is kept as ``rescale_direct_cast`` and used
only in a rescale ablation.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from mri_pipeline.filters import FILTER_NAMES, apply_filters

GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
RESCALE_METHODS = ("minmax", "direct")
RAW_PIXEL_SIZE = 64


def rescale_to_uint8(image: np.ndarray) -> np.ndarray:
    """Min-max rescale any numeric array to 0-255 uint8 without wrap-around.

    A constant array maps to all zeros. Non-finite values are treated as the
    nearest finite extreme (NaN as the minimum).
    """
    arr = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    arr = np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo)
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    span = hi - lo
    if np.isfinite(span):
        scaled = (arr - lo) / span * 255.0
    else:
        # Range wider than float64 can hold (e.g. -1e308 to 1e308): halve first.
        scaled = (arr / 2 - lo / 2) / (hi / 2 - lo / 2) * 255.0
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def rescale_direct_cast(image: np.ndarray) -> np.ndarray:
    """Direct uint8 cast, used only in the rescale ablation.

    For uint8 input, ``x * 255`` overflows and maps v to (256 - v) % 256 (a
    grey-level inversion). Float input above 1 (for example the entropy
    filter, range about 0-4) wraps modulo 256.
    """
    return (image * 255).astype(np.uint8)


def glcm_features(image_u8: np.ndarray, prefix: str) -> dict[str, float]:
    """24 GLCM features (6 properties x 4 angles, distance 1, 256 levels)."""
    if image_u8.dtype != np.uint8:
        raise ValueError("glcm_features expects a uint8 image; rescale first")
    matrix = graycomatrix(image_u8, [1], list(GLCM_ANGLES), levels=256, symmetric=True, normed=True)
    out: dict[str, float] = {}
    for prop in GLCM_PROPS:
        for i, value in enumerate(graycoprops(matrix, prop).ravel()):
            out[f"{prefix}_{prop}_{i + 1}"] = float(value)
    return out


def glcm_feature_names() -> list[str]:
    return [
        f"{name}_{prop}_{i + 1}"
        for name in FILTER_NAMES
        for prop in GLCM_PROPS
        for i in range(len(GLCM_ANGLES))
    ]


def features_from_filtered(filtered: dict[str, np.ndarray], rescale: str = "minmax") -> np.ndarray:
    """168 GLCM features (7 images x 24) as a float64 vector."""
    if rescale not in RESCALE_METHODS:
        raise ValueError(f"rescale must be one of {RESCALE_METHODS}, got {rescale!r}")
    convert = rescale_to_uint8 if rescale == "minmax" else rescale_direct_cast
    values: dict[str, float] = {}
    for name in FILTER_NAMES:
        values.update(glcm_features(convert(filtered[name]), name))
    return np.array([values[k] for k in glcm_feature_names()], dtype=np.float64)


def glcm_vector(image: np.ndarray, mode: str = "fast") -> np.ndarray:
    """Filters + GLCM (min-max rescale) for one image: 168 floats."""
    return features_from_filtered(apply_filters(image, mode), "minmax")


def image_features(image: np.ndarray, mode: str = "fast") -> dict[str, np.ndarray]:
    """Worker for the ML pipeline: filter once, return both rescale variants.

    Returning two small vectors (instead of seven filtered images) keeps the
    data sent back from worker processes small.
    """
    filtered = apply_filters(image, mode)
    return {
        "glcm_minmax": features_from_filtered(filtered, "minmax"),
        "glcm_direct": features_from_filtered(filtered, "direct"),
    }


def raw_pixel_features(image: np.ndarray, size: int = RAW_PIXEL_SIZE) -> np.ndarray:
    """Baseline: the image resized to ``size x size`` and flattened to 0-1."""
    small = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return small.astype(np.float64).ravel() / 255.0

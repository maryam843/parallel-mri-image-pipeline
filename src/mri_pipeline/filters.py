"""The six image filters from the course template, as module-level functions.

Provided by DSAI3202 course staff (UDST). Modified by me: yes (split into one
module-level function per filter so they can be pickled by ``multiprocessing``
under the "spawn" start method; the Hessian sigma range is a parameter with a
documented fast mode; filter parameters are unchanged).

Known quirk kept on purpose: ``skimage.filters.gabor`` on a uint8 image returns
uint8, so negative responses of the imaginary part wrap around. I kept the
course filter settings as they are (see docs/DECISIONS.md).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as nd
from skimage.filters import gabor, hessian, prewitt, sobel
from skimage.filters.rank import entropy
from skimage.morphology import disk

FILTER_NAMES = ("Input", "Entropy", "Gaussian", "Sobel", "Gabor", "Hessian", "Prewitt")

# The template used sigmas=range(1, 100, 1). That single filter takes most of
# the run time, so the default here is a fast mode with sigmas 1-9.
HESSIAN_SIGMAS = {"fast": range(1, 10), "full": range(1, 100)}


def entropy_filter(image: np.ndarray) -> np.ndarray:
    return entropy(image, disk(2))


def gaussian_filter(image: np.ndarray) -> np.ndarray:
    return nd.gaussian_filter(image, sigma=1)


def sobel_filter(image: np.ndarray) -> np.ndarray:
    return sobel(image)


def gabor_filter(image: np.ndarray) -> np.ndarray:
    return gabor(image, frequency=0.9)[1]


def hessian_filter(image: np.ndarray, mode: str = "fast") -> np.ndarray:
    return hessian(image, sigmas=HESSIAN_SIGMAS[mode])


def prewitt_filter(image: np.ndarray) -> np.ndarray:
    return prewitt(image)


def apply_filters(image: np.ndarray, mode: str = "fast") -> dict[str, np.ndarray]:
    """Apply all six filters to one grayscale uint8 image.

    Returns a dict with the unfiltered image under ``'Input'`` followed by
    the six filter outputs.
    """
    if mode not in HESSIAN_SIGMAS:
        raise ValueError(f"mode must be one of {sorted(HESSIAN_SIGMAS)}, got {mode!r}")
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("expected a 2-D uint8 grayscale image")
    return {
        "Input": image,
        "Entropy": entropy_filter(image),
        "Gaussian": gaussian_filter(image),
        "Sobel": sobel_filter(image),
        "Gabor": gabor_filter(image),
        "Hessian": hessian_filter(image, mode),
        "Prewitt": prewitt_filter(image),
    }

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from mri_pipeline import features, filters


@pytest.fixture
def image(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 256, size=(48, 40), dtype=np.uint8)


def test_apply_filters_keys_shapes(image: np.ndarray) -> None:
    out = filters.apply_filters(image, "fast")
    assert tuple(out) == filters.FILTER_NAMES
    assert all(v.shape == image.shape for v in out.values())
    assert out["Input"] is image


def test_apply_filters_validates_input(image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="mode"):
        filters.apply_filters(image, "medium")
    with pytest.raises(ValueError, match="uint8"):
        filters.apply_filters(image.astype(float), "fast")


def test_full_and_fast_hessian_differ(image: np.ndarray) -> None:
    assert list(filters.HESSIAN_SIGMAS["full"]) == list(range(1, 100))
    assert list(filters.HESSIAN_SIGMAS["fast"]) == list(range(1, 10))
    small = image[:24, :24]
    fast = filters.hessian_filter(small, "fast")
    full = filters.hessian_filter(small, "full")
    assert fast.shape == full.shape


def test_filters_match_template_calls(image: np.ndarray) -> None:
    """Same calls as the original filter code, so the refactor did not change results."""
    from scipy import ndimage as nd
    from skimage.filters import gabor, hessian, prewitt, sobel
    from skimage.filters.rank import entropy
    from skimage.morphology import disk

    out = filters.apply_filters(image, "fast")
    np.testing.assert_array_equal(out["Entropy"], entropy(image, disk(2)))
    np.testing.assert_array_equal(out["Gaussian"], nd.gaussian_filter(image, sigma=1))
    np.testing.assert_array_equal(out["Sobel"], sobel(image))
    np.testing.assert_array_equal(out["Gabor"], gabor(image, frequency=0.9)[1])
    np.testing.assert_array_equal(out["Hessian"], hessian(image, sigmas=range(1, 10)))
    np.testing.assert_array_equal(out["Prewitt"], prewitt(image))


# ------------------------------------------------------------------- rescale

numeric_arrays = st.one_of(
    hnp.arrays(np.uint8, hnp.array_shapes(min_dims=2, max_dims=2, max_side=12)),
    hnp.arrays(
        np.float64,
        hnp.array_shapes(min_dims=2, max_dims=2, max_side=12),
        elements=st.floats(-1e300, 1e300, allow_nan=False, allow_infinity=False),
    ),
    hnp.arrays(
        np.float32,
        hnp.array_shapes(min_dims=2, max_dims=2, max_side=12),
        elements=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False, width=32),
    ),
)


@settings(max_examples=300, deadline=None)
@given(numeric_arrays)
def test_property_rescale_in_range_and_order_preserving(arr: np.ndarray) -> None:
    out = features.rescale_to_uint8(arr)
    assert out.dtype == np.uint8 and out.shape == arr.shape
    flat_in = arr.astype(np.float64).ravel()
    flat_out = out.astype(np.int64).ravel()
    order = np.argsort(flat_in, kind="stable")
    # monotonic: a larger input never gets a smaller output (no wrap-around)
    assert np.all(np.diff(flat_out[order]) >= 0)
    if flat_in.max() > flat_in.min():
        assert flat_out[order[0]] == 0 and flat_out[order[-1]] == 255
    else:
        assert np.all(flat_out == 0)


def test_rescale_handles_non_finite() -> None:
    arr = np.array([[np.nan, 0.0], [np.inf, -np.inf]])
    out = features.rescale_to_uint8(arr)
    assert out.tolist() == [[0, 0], [0, 0]]  # finite range is a single value
    arr = np.array([[np.nan, 1.0], [2.0, np.inf]])
    assert features.rescale_to_uint8(arr).tolist() == [[0, 0], [255, 255]]
    assert features.rescale_to_uint8(np.full((2, 2), np.nan)).tolist() == [[0, 0], [0, 0]]


def test_rescale_huge_range_and_subnormal() -> None:
    out = features.rescale_to_uint8(np.array([-1.7e308, 0.0, 1.7e308]))
    assert out.tolist() == [0, 128, 255]
    out = features.rescale_to_uint8(np.array([0.0, 5e-324]))
    assert out.tolist() == [0, 255]


DIRECT_CAST_INPUTS = {
    "uint8 image": np.array([[0, 1, 100, 200, 255]], dtype=np.uint8),
    "entropy float 0-3.7": np.array([[0.0, 0.5, 1.2, 2.5, 3.7]]),
}


def _order_preserved(convert, arr: np.ndarray) -> bool:
    out = convert(arr).astype(np.int64).ravel()
    order = np.argsort(arr.astype(np.float64).ravel(), kind="stable")
    return bool(np.all(np.diff(out[order]) >= 0))


@pytest.mark.parametrize("name", list(DIRECT_CAST_INPUTS))
def test_minmax_rescale_keeps_order_where_direct_cast_wraps(name: str) -> None:
    """The ablation's direct cast wraps around on these inputs; min-max never does."""
    arr = DIRECT_CAST_INPUTS[name]
    assert _order_preserved(features.rescale_to_uint8, arr)
    assert not _order_preserved(features.rescale_direct_cast, arr)


def test_direct_cast_behaviour() -> None:
    """What the direct cast used in the rescale ablation does."""
    u8 = np.array([0, 1, 56, 200, 255], dtype=np.uint8)
    assert features.rescale_direct_cast(u8).tolist() == [0, 255, 200, 56, 1]  # 256 - v
    assert features.rescale_direct_cast(np.array([3.7])).tolist() == [175]  # 943 mod 256


# ---------------------------------------------------------------------- GLCM


def test_glcm_feature_count_and_names(image: np.ndarray) -> None:
    feats = features.glcm_features(image, "Input")
    assert len(feats) == 24
    assert list(feats)[:2] == ["Input_contrast_1", "Input_contrast_2"]
    names = features.glcm_feature_names()
    assert len(names) == 168 and len(set(names)) == 168


def test_glcm_rejects_float(image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="uint8"):
        features.glcm_features(image.astype(float), "x")


def test_constant_image_glcm() -> None:
    feats = features.glcm_features(np.full((10, 10), 7, dtype=np.uint8), "c")
    assert feats["c_contrast_1"] == 0.0
    assert feats["c_energy_1"] == pytest.approx(1.0)


def test_image_features_vectors(image: np.ndarray) -> None:
    out = features.image_features(image, "fast")
    assert set(out) == {"glcm_minmax", "glcm_direct"}
    assert out["glcm_minmax"].shape == (168,)
    assert np.all(np.isfinite(out["glcm_minmax"]))
    # Input and Gaussian are uint8, so the direct cast inverts them:
    # the two variants must differ.
    assert not np.array_equal(out["glcm_minmax"], out["glcm_direct"])


def test_features_from_filtered_validates(image: np.ndarray) -> None:
    filtered = filters.apply_filters(image, "fast")
    with pytest.raises(ValueError, match="rescale"):
        features.features_from_filtered(filtered, "zscore")


def test_raw_pixel_features(image: np.ndarray) -> None:
    vec = features.raw_pixel_features(image)
    assert vec.shape == (64 * 64,)
    assert vec.min() >= 0.0 and vec.max() <= 1.0

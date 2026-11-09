import numpy as np
import pytest
from scipy import ndimage as ndi

from domb_napari.core import preprocess


def test_back_substr_is_signed_and_keeps_input():
    rng = np.random.default_rng(0)
    for dtype in (np.uint16, np.int32, np.float32):
        img = rng.integers(90, 200, (4, 32, 32)).astype(dtype)
        img_copy = img.copy()
        out = preprocess.back_substr(img, percentile=10.0)
        assert np.array_equal(img, img_copy)
        assert out.dtype == np.float32 and out.min() < 0
        for frame_in, frame_out in zip(img, out):
            np.testing.assert_allclose(frame_out, frame_in - np.percentile(frame_in, 10.0), atol=1e-3)


def test_background_noise_mean_is_preserved():
    rng = np.random.default_rng(1)
    img = rng.poisson(100, (3, 200, 200)).astype(np.uint16)
    out = preprocess.back_substr(img, percentile=50.0)
    assert abs(out.mean()) < 1.0      # clipping at zero would give ~4


def test_median_filter_stack():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 4000, (9, 24, 24)).astype(np.uint16)
    out = preprocess.median_filter_stack(img, 3)
    assert out.dtype == img.dtype
    assert np.array_equal(out, ndi.median_filter(img, size=(1, 3, 3)))


@pytest.mark.parametrize('method', ['exp', 'biexp'])
def test_bleaching_fit_with_plateau(method):
    x = np.arange(300)
    profile = 1500 * np.exp(-x / 60) + 800
    fit, r2 = preprocess.bleaching_fit(profile, method)
    np.testing.assert_allclose(fit, profile, rtol=1e-3)
    assert r2 > 0.9999


def test_pb_correction():
    x = np.arange(100)
    profile = 1000 * np.exp(-x / 40) + 300 * np.exp(-x / 400)
    img = profile[:, None, None] * np.ones((1, 8, 8))
    mask = np.ones((8, 8), dtype=int)
    for method in ('biexp', '1st'):
        out, coefs, metric = preprocess.pb_correction(img, mask, method)
        assert out.dtype == np.float32 and isinstance(metric, str)
        np.testing.assert_allclose(out, profile[0], rtol=2e-3)
        np.testing.assert_allclose(coefs, profile / profile[0], rtol=2e-3)
    with pytest.raises(ValueError):
        preprocess.pb_correction(img, np.zeros((8, 8)), 'exp')

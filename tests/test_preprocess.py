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


def _cell_frame(rng, shape=(256, 256), background=100.0, frac=0.3, peak=3000.0):
    rows, cols = np.indices(shape)
    radius = np.sqrt(frac * shape[0] * shape[1] / np.pi)
    cell = np.hypot(rows - shape[0] / 2, cols - shape[1] / 2) < radius
    texture = ndi.gaussian_filter(rng.random(shape), 4)
    texture = (texture - texture.min()) / np.ptp(texture)
    return rng.poisson(np.where(cell, peak * (0.2 + 0.8 * texture), 0.0) + background), cell


@pytest.mark.parametrize('background,peak', [(10.0, 500.0), (100.0, 3000.0), (500.0, 30000.0)])
def test_histogram_mode_of_background(background, peak):
    rng = np.random.default_rng(0)
    frame, _ = _cell_frame(rng, background=background, peak=peak)
    mode = preprocess.histogram_mode(frame)
    assert abs(mode - (background - 0.5)) < max(1.0, 0.02 * background)
    assert preprocess.histogram_mode(frame.astype(np.float32) + rng.normal(0, 2, frame.shape)) == \
        pytest.approx(background - 0.5, abs=1.0)


def test_histogram_mode_edge_cases():
    assert preprocess.histogram_mode(np.full(100, 7.0)) == 7.0
    values = np.r_[np.random.default_rng(1).normal(50, 3, 5000), np.nan, np.inf]
    assert preprocess.histogram_mode(values) == pytest.approx(50, abs=0.5)
    with pytest.raises(ValueError):
        preprocess.histogram_mode(np.array([np.nan, np.nan]))


def test_background_methods():
    rng = np.random.default_rng(2)
    frames = [_cell_frame(rng, background=bg) for bg in (100.0, 120.0, 140.0)]
    stack = np.stack([f for f, _ in frames]).astype(np.uint16)
    cell = frames[0][1]
    # background mask made of several labels and a NaN pixel
    labels = np.zeros(cell.shape, dtype=np.int32)
    labels[:20, :20] = 1
    labels[-20:, -20:] = 4
    stack_f = stack.astype(np.float32)
    stack_f[:, 0, 0] = np.nan
    levels = {m: preprocess.background_levels(stack_f, method=m, mask=labels) for m in preprocess.BACKGROUND_METHODS}
    np.testing.assert_allclose(levels['mask'], [np.nanmean(f[labels > 0]) for f in stack_f])
    np.testing.assert_allclose(levels['mask'], [100, 120, 140], atol=1.0)
    np.testing.assert_allclose(levels['mode'], [99.5, 119.5, 139.5], atol=1.0)
    assert np.all(levels['percentile'] < np.array([100, 120, 140]) - 15)

    out, subtracted = preprocess.back_substr(stack, method='mask', mask=labels, return_levels=True)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, stack - subtracted[:, None, None], atol=1e-3)
    assert abs(out[:, labels > 0].mean()) < 1e-3


def test_background_mask_validation():
    stack = np.ones((2, 8, 8))
    with pytest.raises(ValueError):
        preprocess.background_levels(stack, method='mask')
    with pytest.raises(ValueError):
        preprocess.background_levels(stack, method='mask', mask=np.zeros((8, 8)))
    with pytest.raises(ValueError):
        preprocess.background_levels(stack, method='mask', mask=np.ones((8, 9)))
    with pytest.raises(ValueError):
        preprocess.background_levels(stack, method='median')

import numpy as np
import pytest

from domb_napari.core import series


def test_red_green_windows():
    ramp = np.arange(12, dtype=np.uint16)[:, None, None] * np.ones((1, 4, 4), dtype=np.uint16)
    out = series.red_green_series(ramp, 1, 0, 1)
    assert out.shape == (11, 4, 4) and out.dtype == np.float32
    np.testing.assert_allclose(out, 1.0)
    # mean(img[i+2+3 : i+2+3+4]) - mean(img[i : i+2])
    out = series.red_green_series(ramp, left_frames=2, space_frames=3, right_frames=4)
    assert out.shape == (12 - 9 + 1, 4, 4)
    np.testing.assert_allclose(out, 6.5 - 0.5)


def test_red_green_is_signed_for_unsigned_input():
    img = np.zeros((3, 2, 2), dtype=np.uint16)
    img[0] = 100
    out = series.red_green_series(img, 1, 0, 1)
    np.testing.assert_allclose(out[0], -100.0)


def test_red_green_reference(rng=np.random.default_rng(0)):
    img = rng.normal(100, 10, (20, 8, 8)).astype(np.float32)
    l, s, r = 3, 1, 2
    out = series.red_green_series(img, l, s, r)
    for i in (0, 5, out.shape[0] - 1):
        ref = img[i+l+s:i+l+s+r].mean(axis=0) - img[i:i+l].mean(axis=0)
        np.testing.assert_allclose(out[i], ref, rtol=1e-5, atol=1e-4)
    weighted = series.red_green_series(img, l, s, r, intensity_weighted=True)
    ref_w = ref * 0.5 * (img[i+l+s:i+l+s+r].mean(axis=0) + img[i:i+l].mean(axis=0)) / np.abs(img).max()
    np.testing.assert_allclose(weighted[i], ref_w, rtol=1e-4, atol=1e-4)


def test_red_green_validation():
    img = np.zeros((3, 2, 2))
    with pytest.raises(ValueError):
        series.red_green_series(img, 2, 1, 1)
    with pytest.raises(ValueError):
        series.red_green_series(img, 0, 0, 1)


def test_delta_f():
    rng = np.random.default_rng(1)
    img = rng.integers(100, 4000, (10, 16, 16)).astype(np.uint16)
    f0 = img[:5].mean(axis=0)
    out = series.delta_img(img, mode='dF', win_size=5)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, img - f0, rtol=1e-5, atol=1e-2)
    rel = series.delta_img(img, mode='dF/F0', win_size=5)
    np.testing.assert_allclose(rel, (img - f0) / f0, rtol=1e-4, atol=1e-5)


def test_delta_f_f0_undefined_baseline_is_nan():
    img = np.full((6, 4, 4), 50.0)
    img[:, 0, 0] = 0.0
    img[:, 1, 1] = -3.0
    img[:, 2, 2] = 5.0
    rel = series.delta_img(img, mode='dF/F0', win_size=3)
    assert np.all(np.isnan(rel[:, 0, 0])) and np.all(np.isnan(rel[:, 1, 1]))
    assert np.all(np.isfinite(rel[:, 2, 2]))
    assert np.nanmax(np.abs(rel)) < 1e6
    rel = series.delta_img(img, mode='dF/F0', win_size=3, f0_min=10.0)
    assert np.all(np.isnan(rel[:, 2, 2])) and np.all(np.isfinite(rel[:, 3, 3]))
    assert np.all(np.isfinite(series.delta_img(img, mode='dF', win_size=3)))

""" Tests of domb_napari.core with synthetic data of known ground truth.

"""
import pathlib

import numpy as np
import pytest
import yaml
from scipy import ndimage as ndi
from scipy import stats

from domb_napari.core import fret, masking, preprocess, profiles, registration, series
from domb_napari.core.fret import FRETCoefs, CubesFRET

A, D, G, XI = 0.12, 0.35, 2.5, 0.05


def simulate(donor, acceptor, eff, b=0.0, c=0.0, intact=1.0):
    """ Noise-free 3-cube intensities of a 1:1 complex with efficiency eff. """
    don = donor * (1 - eff * intact)
    acc = acceptor * intact
    fc = G * donor * eff * intact
    return don + b * acc, fc + A * acc + D * don, acc + c * don


def blocks(shape, size=10):
    rows, cols = np.indices(shape)
    return (rows // size) * (shape[1] // size) + cols // size + 1


def test_fret_estimation():
    rng = np.random.default_rng(0)
    # numba kernel equals the numpy reference, exact unmixing with b and c
    imgs = [rng.integers(0, 4000, (3, 16, 16)).astype(np.uint16) for _ in range(3)]
    coefs = FRETCoefs(a=A, d=D, G=G, xi=XI, b=0.02, c=0.01)
    cubes = CubesFRET.from_coefs(*imgs, coefs)
    ref = fret.fret_from_intensities(*imgs, coefs)
    for key in ('Fc', 'E_D', 'E_A', 'D', 'A'):
        img = getattr(cubes, key + '_img')()
        assert img.dtype == np.float32
        np.testing.assert_allclose(img, ref[key], rtol=2e-5, atol=1e-3, equal_nan=True)
    out = fret.fret_from_intensities(*simulate(1000.0, 800.0, 0.3, b=0.05, c=0.03),
                                     FRETCoefs(a=A, d=D, G=G, b=0.05, c=0.03))
    assert out['E_D'] == pytest.approx(0.3, rel=1e-9)

    # no clipping: without FRET the mean Fc stays at zero; invalid pixels are NaN
    shape = (20, 48, 48)
    noisy = [rng.poisson(i + 20, shape) - 20.0 for i in simulate(50.0, 50.0, 0.0)]
    fc = CubesFRET(*noisy, A, D).Fc_img()
    assert fc.min() < 0 and abs(fc.mean()) < 5 * fc.std() / np.sqrt(fc.size)
    zeros = np.zeros((1, 2, 2))
    assert np.all(np.isnan(CubesFRET(zeros, zeros, zeros, A, D, G_val=G, eps_rel_val=XI).E_A_img()))

    # ROI efficiency from mean intensities is unbiased at low signal
    noisy = [rng.poisson(i + 10, (50, 40, 40)) - 10.0 for i in simulate(30.0, 30.0, 0.25)]
    roi = CubesFRET(*noisy, A, D, G_val=G).roi_profiles(np.ones((40, 40), int))
    assert np.mean(roi['E_D']) == pytest.approx(0.25, abs=0.01)

    # Ecorr: acceptor bleaching curve from the mask, pixels without acceptor are ignored
    intact = np.exp(-np.arange(30) / 20.0)
    stack = [np.empty((30, 16, 16)) for _ in range(3)]
    for t in range(30):
        for img, val in zip(stack, simulate(1000.0, 900.0, 0.4, intact=intact[t])):
            img[t] = val
    stack[2][:, :4, :4] = 0.0
    mask = np.ones((16, 16), int)
    mask[:4, :4] = 0
    ecorr = CubesFRET(*stack, A, D, G_val=G).Ecorr_img(mask=mask, f0_frames=1)
    np.testing.assert_allclose(ecorr[:, 8, 8], 0.4, rtol=1e-5)

    # coefficients: validation, YAML round trip, bundled configuration
    with pytest.raises(ValueError):
        FRETCoefs(a=0.1, d=0.2, G=0).require('E_D')
    with pytest.raises(ValueError):
        FRETCoefs(a=0.1, d=0.2).require('E_A')
    path = pathlib.Path(fret.__file__).parent.parent / '_e_fret_coefs.yaml'
    for pair in yaml.safe_load(open(path)):
        FRETCoefs.from_yaml(path, pair)


def test_fret_calibration(tmp_path):
    rng = np.random.default_rng(1)
    # cross-talk: Deming slope is free of regression dilution
    acceptor = rng.uniform(40, 120, (64, 64))
    stack = lambda img: rng.poisson(np.broadcast_to(img, (6, 64, 64))).astype(float)
    _, fit_arr, summary = fret.CrossTalkEstimation(np.ones((64, 64)), stack(np.full((64, 64), 5.0)) - 5.0,
                                                   stack(A * acceptor), stack(acceptor)).estimate_a()
    res = summary.iloc[0]
    assert res['a_val'] == pytest.approx(A, rel=0.03) and res['a_ratio'] == pytest.approx(A, rel=0.02)
    assert 0 <= res['a_r^2'] <= 1 and fit_arr.shape == (2, 64 * 64)

    # G by acceptor photobleaching (Zal)
    donor = rng.uniform(300, 1500, (60, 60))
    high = [rng.poisson(i).astype(float) for i in simulate(donor, donor, 0.35)]
    low = [rng.poisson(i).astype(float) for i in simulate(donor, donor, 0.35, intact=0.2)]
    labels = blocks(donor.shape)
    g_df, fit_arr = fret.GFactorEstimation(labels, None, *high, *low, a_val=A, d_val=D).estimate_g_zal()
    assert g_df['g_val'][0] == pytest.approx(G, rel=0.02) and fit_arr.shape == (2, 36)
    assert abs(g_df['g_val'][0] - G) < 4 * g_df['g_err'][0]

    # G with two constructs (Chen), bootstrap error does not shrink with resamples
    high = [rng.poisson(i).astype(float) for i in simulate(donor, donor, 0.45)]
    low = [rng.poisson(i).astype(float) for i in simulate(donor, donor, 0.10)]
    est = fret.GFactorEstimation(labels, labels, *high, *low, a_val=A, d_val=D)
    res = est.estimate_g_chen(n_boot=500)[0].iloc[0]
    assert res['g_val'] == pytest.approx(G, rel=0.03) and res['g_ci_low'] < res['g_val'] < res['g_ci_high']
    assert est.estimate_g_chen(n_boot=4000)[0]['g_err'][0] == pytest.approx(res['g_err'], rel=0.25)

    # k factor and writing coefficients to a configuration
    imgs = simulate(donor, donor / 1.7, 0.3)
    k_df, _ = fret.KFactorEstimation(labels, *imgs, a_val=A, d_val=D, G_val=G).estimate_k()
    assert k_df['k_val'][0] == pytest.approx(1.7, rel=1e-4)
    config = tmp_path / 'coefs.yaml'
    fret.update_yaml(config, 'X/Y', a=0.1, d=0.2, G=3.0)
    fret.update_yaml(config, 'Q/W', a=0.3, d=0.4)
    fret.update_yaml(config, 'X/Y', G=4.0)
    assert FRETCoefs.from_yaml(config, 'X/Y').G == 4.0 and FRETCoefs.from_yaml(config, 'Q/W').G is None


def test_preprocess():
    rng = np.random.default_rng(2)
    rows, cols = np.indices((256, 256))
    cell = np.hypot(rows - 128, cols - 128) < 80
    frames = np.stack([rng.poisson(np.where(cell, 2000.0, 0.0) + bg) for bg in (100.0, 140.0)])
    labels = np.zeros((256, 256), int)
    labels[:20, :20] = 1
    labels[-20:, -20:] = 4                                   # several labels form one mask
    levels = {m: preprocess.background_levels(frames, method=m, mask=labels)
              for m in preprocess.BACKGROUND_METHODS}
    np.testing.assert_allclose(levels['mask'], [100, 140], atol=1.0)
    np.testing.assert_allclose(levels['mode'], [99.5, 139.5], atol=1.0)
    assert np.all(levels['percentile'] < [85, 125])
    frames_u16 = frames.astype(np.uint16)
    out = preprocess.back_substr(frames_u16, method='mask', mask=labels)
    assert out.dtype == np.float32 and out.min() < 0 and abs(out[:, labels > 0].mean()) < 1e-3
    assert np.array_equal(frames_u16, frames.astype(np.uint16))
    with pytest.raises(ValueError):
        preprocess.background_levels(frames, method='mask', mask=np.zeros((256, 256)))

    stack = rng.integers(0, 4000, (5, 24, 24)).astype(np.uint16)
    assert np.array_equal(preprocess.median_filter_stack(stack, 3), ndi.median_filter(stack, size=(1, 3, 3)))

    x = np.arange(100)
    profile = 1000 * np.exp(-x / 40) + 300 * np.exp(-x / 400)
    corrected, coefs, _ = preprocess.pb_correction(profile[:, None, None] * np.ones((1, 8, 8)),
                                                   np.ones((8, 8)), 'biexp')
    np.testing.assert_allclose(corrected, profile[0], rtol=2e-3)


def test_series_and_profiles():
    # red-green: mean(img[i+l+s : i+l+s+r]) - mean(img[i : i+l])
    ramp = np.arange(12, dtype=np.uint16)[:, None, None] * np.ones((1, 4, 4), dtype=np.uint16)
    out = series.red_green_series(ramp, left_frames=2, space_frames=3, right_frames=4)
    assert out.shape == (4, 4, 4) and out.dtype == np.float32
    np.testing.assert_allclose(out, 6.0)
    assert series.red_green_series(ramp[::-1], 1, 0, 1).min() == -1.0

    # ΔF and ΔF/F0 without weighting, undefined ratio is NaN
    img = np.random.default_rng(3).integers(100, 4000, (10, 16, 16)).astype(np.float64)
    img[:, 0, 0] = 0.0
    f0 = img[:5].mean(axis=0)
    np.testing.assert_allclose(series.delta_img(img, 'dF', 5), img - f0, atol=1e-2)
    rel = series.delta_img(img, 'dF/F0', 5)
    assert np.all(np.isnan(rel[:, 0, 0]))
    np.testing.assert_allclose(rel[:, 1:, 1:], ((img - f0) / f0)[:, 1:, 1:], rtol=1e-4)

    # ROI profiles follow label values, statistics are over ROIs
    labels = np.zeros((16, 16), int)
    labels[1:5, 1:5], labels[8:12, 3:9], labels[13:16, 10:16] = 1, 2, 5
    prof = profiles.labels_to_profiles(labels, img)
    np.testing.assert_allclose(prof, np.array([ndi.mean(f, labels, [1, 2, 5]) for f in img]).T, rtol=1e-6)
    center, sem = profiles.profile_stats(prof, 'se')
    np.testing.assert_allclose(sem, stats.sem(prof, axis=0), rtol=1e-5)
    _, iqr = profiles.profile_stats(prof, 'iqr')
    np.testing.assert_allclose(iqr.sum(axis=0), stats.iqr(prof, axis=0), rtol=1e-5)
    df = profiles.profiles_table(prof, profiles.label_ids(labels), time_scale=0.5,
                                 distances=profiles.stim_distances(labels, (0, 0)))
    assert len(df) == 2 * 3 * 10 and sorted(df['roi'].unique()) == [1, 2, 5]
    assert list(df['index'][:10]) == list(range(10))


def test_masking():
    mask = np.zeros((256, 256), bool)
    mask[40:200, 60:180] = True
    seg = masking.mask_segmentation(mask, 30)
    areas = np.bincount(seg.ravel())[1:]
    assert len(areas) == 30 and areas.max() / areas.min() < 3 and np.array_equal(seg > 0, mask)

    # in-ROIs detection keeps labels of ROIs with overlapping bounding boxes
    img = np.zeros((40, 40))
    img[5:15, 5:15] = img[22:32, 22:32] = 10.0
    rois = np.zeros((40, 40), int)
    rois[:, 20:], rois[35:, :20], rois[2:18, 2:18] = 7, 7, 3
    up = masking.up_mask(img, det_th=0.5, rois_mask=rois, in_rois_method='threshold',
                         in_rois_th_corr=1.0, opening_fp=0, dilation_fp=0)
    assert up[10, 10] == 3 and up[27, 27] == 7 and np.all(rois[up > 0] == up[up > 0])

    dots = np.zeros((3, 64, 64))
    dots[:, [16, 16, 48], [16, 48, 32]] = 100.0
    assert masking.dots_mask(dots, mask_radius=4).max() == 3


def test_registration():
    matrix = np.array([[1.004, 0.003, 2.3], [-0.002, 0.997, -3.1], [0, 0, 1]])
    rng = np.random.default_rng(0)
    img = np.zeros((256, 256))
    img[rng.integers(20, 236, 30), rng.integers(20, 236, 30)] = rng.uniform(500, 3000, 30)
    img = ndi.gaussian_filter(img, 4) * 200 + 100
    moving = rng.poisson(ndi.affine_transform(img, matrix, order=1) * 0.6 + 100).astype(float)
    result = registration.AffineRegistration().optimize(rng.poisson(img).astype(float), moving)
    corners = np.array([[0, 0, 1], [0, 256, 1], [256, 0, 1], [256, 256, 1]]).T
    assert result.info['accepted']
    assert np.abs((result.affine @ corners - np.linalg.inv(matrix) @ corners)[:2]).max() < 0.3

    stack = rng.integers(0, 4000, (3, 40, 40)).astype(np.uint16)
    out = registration.apply_affine_stack(stack, matrix)
    ref = ndi.affine_transform(stack[1].astype(float), matrix, order=1, cval=np.nan)
    np.testing.assert_allclose(out[1], ref, rtol=1e-5, equal_nan=True)

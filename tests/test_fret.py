import numpy as np
import pytest

from domb_napari.core import fret
from domb_napari.core.fret import FRETCoefs, CubesFRET

A, D, G, XI = 0.12, 0.35, 2.5, 0.05


def simulate(donor, acceptor, eff, a=A, d=D, g=G, b=0.0, c=0.0, intact=1.0):
    """ Noise-free 3-cube intensities of a 1:1 complex with efficiency eff. """
    don = donor * (1 - eff * intact)
    acc = acceptor * intact
    fc = g * donor * eff * intact
    dd = don + b * acc
    aa = acc + c * don
    da = fc + a * acc + d * don
    return dd, da, aa


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_kernel_matches_reference(rng):
    dd, da, aa = (rng.integers(0, 4000, (5, 32, 32)).astype(np.uint16) for _ in range(3))
    coefs = FRETCoefs(a=A, d=D, G=G, xi=XI, b=0.02, c=0.01)
    cubes = CubesFRET.from_coefs(dd, da, aa, coefs)
    ref = fret.fret_from_intensities(dd, da, aa, coefs)
    for key, img in (('Fc', cubes.Fc_img()), ('E_D', cubes.E_D_img()),
                     ('E_A', cubes.E_A_img()), ('D', cubes.D_img()), ('A', cubes.A_img())):
        assert img.dtype == np.float32
        np.testing.assert_allclose(img, ref[key], rtol=2e-5, atol=1e-3, equal_nan=True)


def test_exact_unmixing_recovers_efficiency():
    donor, acceptor, eff = 1000.0, 800.0, 0.3
    for b, c in ((0.0, 0.0), (0.05, 0.03)):
        dd, da, aa = simulate(donor, acceptor, eff, b=b, c=c)
        out = fret.fret_from_intensities(dd, da, aa, FRETCoefs(a=A, d=D, G=G, xi=XI, b=b, c=c))
        assert out['E_D'] == pytest.approx(eff, rel=1e-9)
        assert out['D'] == pytest.approx(donor * (1 - eff), rel=1e-9)
        assert out['A'] == pytest.approx(acceptor, rel=1e-9)


def test_fc_is_not_rectified(rng):
    # no FRET: Fc is pure noise around zero, its mean must stay at zero
    dd, da, aa = simulate(50.0, 50.0, 0.0)
    shape = (20, 64, 64)
    dd_n = rng.poisson(dd + 20, shape) - 20.0
    da_n = rng.poisson(da + 20, shape) - 20.0
    aa_n = rng.poisson(aa + 20, shape) - 20.0
    fc = CubesFRET(dd_n, da_n, aa_n, A, D).Fc_img()
    assert fc.min() < 0
    assert abs(fc.mean()) < 5 * fc.std() / np.sqrt(fc.size)


def test_invalid_pixels_are_nan():
    dd = np.zeros((1, 2, 2)); da = np.zeros((1, 2, 2)); aa = np.zeros((1, 2, 2))
    cubes = CubesFRET(dd, da, aa, A, D, G_val=G, eps_rel_val=XI)
    assert np.all(np.isnan(cubes.E_D_img()))
    assert np.all(np.isnan(cubes.E_A_img()))
    assert np.all(cubes.Fc_img() == 0)


def test_roi_efficiency_is_unbiased_at_low_snr(rng):
    eff = 0.25
    dd, da, aa = simulate(30.0, 30.0, eff)
    shape = (50, 40, 40)
    offset = 10
    imgs = [rng.poisson(i + offset, shape) - float(offset) for i in (dd, da, aa)]
    cubes = CubesFRET(*imgs, A, D, G_val=G)
    labels = np.ones(shape[1:], dtype=int)
    roi = cubes.roi_profiles(labels)
    assert roi['E_D'].shape == (1, shape[0])
    assert np.mean(roi['E_D']) == pytest.approx(eff, abs=0.01)


def test_ecorr_compensates_acceptor_bleaching():
    n_t, eff = 30, 0.4
    intact = np.exp(-np.arange(n_t) / 20.0)
    stack = [np.empty((n_t, 16, 16)) for _ in range(3)]
    for t in range(n_t):
        for img, val in zip(stack, simulate(1000.0, 900.0, eff, intact=intact[t])):
            img[t] = val
    stack[2][:, :4, :4] = 0.0     # pixels without acceptor must not break the correction
    cubes = CubesFRET(*stack, A, D, G_val=G)
    mask = np.ones((16, 16), dtype=int); mask[:4, :4] = 0
    np.testing.assert_allclose(cubes.acceptor_bleaching(mask, f0_frames=1), intact, rtol=1e-9)
    ecorr = cubes.Ecorr_img(mask=mask, f0_frames=1)
    assert np.all(np.isfinite(ecorr))
    np.testing.assert_allclose(ecorr[:, 8, 8], eff, rtol=1e-5)
    e_d = cubes.E_D_img()
    assert e_d[-1, 8, 8] < 0.5 * eff


def test_coefs_validation(tmp_path):
    with pytest.raises(ValueError):
        FRETCoefs(a=0.1, d=0.2, G=0).require('E_D')
    with pytest.raises(ValueError):
        FRETCoefs(a=0.1, d=0.2, xi=None).require('E_A')
    with pytest.raises(ValueError):
        FRETCoefs(a=None, d=0.2)
    FRETCoefs(a=0.1, d=0.2).require('Fc')

    path = tmp_path / 'coefs.yaml'
    fret.update_yaml(path, 'X/Y', a=0.1, d=0.2, G=3.0)
    fret.update_yaml(path, 'Q/W', a=0.3, d=0.4)
    fret.update_yaml(path, 'X/Y', G=4.0)
    coefs = FRETCoefs.from_yaml(path, 'X/Y')
    assert (coefs.a, coefs.d, coefs.G, coefs.b) == (0.1, 0.2, 4.0, 0.0)
    assert FRETCoefs.from_yaml(path, 'Q/W').G is None


def test_bundled_config_is_valid():
    import pathlib
    import yaml
    path = pathlib.Path(fret.__file__).parent.parent / '_e_fret_coefs.yaml'
    for pair in yaml.safe_load(open(path)):
        FRETCoefs.from_yaml(path, pair)


def test_linear_fits_errors_in_variables(rng):
    slope, n = 0.2, 40000
    truth = rng.uniform(40, 120, n)
    x = rng.poisson(truth).astype(float)
    y = rng.poisson(slope * truth).astype(float)
    fit = fret.linear_fits(x, y)
    assert fit['ols'] < slope * 0.92                       # regression dilution
    assert fit['deming'] == pytest.approx(slope, rel=0.02)
    assert fit['ratio'] == pytest.approx(slope, rel=0.02)
    assert abs(fit['deming'] - slope) < 4 * fit['deming_err']
    assert 0 <= fit['r2'] <= 1


def test_deming_is_robust_to_offset(rng):
    slope, n = 0.2, 20000
    truth = rng.uniform(50, 400, n)
    x = rng.poisson(truth).astype(float)
    y = rng.poisson(slope * truth).astype(float) + 15.0
    fit = fret.linear_fits(x, y, delta=slope)
    assert fit['deming'] == pytest.approx(slope, rel=0.03)
    assert fit['deming_i'] == pytest.approx(15.0, abs=2.0)
    assert fit['ratio'] > slope * 1.2


def test_cross_talk_estimation(rng):
    n_t, shape = 6, (48, 48)
    acceptor = rng.uniform(100, 1500, shape)
    aa = rng.poisson(np.broadcast_to(acceptor, (n_t,) + shape)).astype(float)
    da = rng.poisson(np.broadcast_to(A * acceptor, (n_t,) + shape)).astype(float)
    dd = rng.poisson(np.full((n_t,) + shape, 5.0)) - 5.0
    mask = np.ones(shape, dtype=int)
    coef_df, fit_arr, summary = fret.CrossTalkEstimation(mask, dd, da, aa).estimate_a()
    assert len(coef_df) == n_t and fit_arr.shape == (2, mask.size)
    res = summary.iloc[0]
    assert res['a_val'] == pytest.approx(A, rel=0.02)
    assert res['b_val'] == pytest.approx(0.0, abs=0.005)
    assert res['a_frames_median'] == pytest.approx(A, rel=0.03)
    assert 0 <= res['a_r^2'] <= 1
    assert isinstance(res['a_p'], float)


def _g_images(rng, eff_h, eff_l, intact_l=1.0, noise=True, shape=(60, 60), acceptor_ratio=1.0):
    donor = rng.uniform(300, 1500, shape)
    acceptor = donor * acceptor_ratio
    high = simulate(donor, acceptor, eff_h)
    low = simulate(donor, acceptor, eff_l if intact_l == 1.0 else eff_h, intact=intact_l)
    if noise:
        high = [rng.poisson(i).astype(float) for i in high]
        low = [rng.poisson(i).astype(float) for i in low]
    return high, low


def _blocks(shape, size=10):
    rows, cols = np.indices(shape)
    return (rows // size) * (shape[1] // size) + cols // size + 1


def test_g_zal(rng):
    high, low = _g_images(rng, 0.35, None, intact_l=0.2)
    labels = _blocks(high[0].shape)
    est = fret.GFactorEstimation(labels, None, *high, *low, a_val=A, d_val=D)
    g_df, fit_arr = est.estimate_g_zal()
    res = g_df.iloc[0]
    assert fit_arr.shape == (2, 36) and res['n_roi'] == 36
    assert res['g_val'] == pytest.approx(G, rel=0.02)
    assert abs(res['g_val'] - G) < 4 * res['g_err']
    assert abs(res['g_deming'] - G) < 4 * res['g_deming_err']
    assert 0 <= res['g_r^2'] <= 1


def test_g_zal_uses_frame_average(rng):
    high, low = _g_images(rng, 0.35, None, intact_l=0.2, noise=False)
    labels = _blocks(high[0].shape)
    high3 = [np.stack([i, i + 1000.0]) for i in high]
    low3 = [np.stack([i, i + 1000.0]) for i in low]
    g_first = fret.GFactorEstimation(labels, None, *high3, *low3, a_val=A, d_val=D,
                                     n_frames=1).estimate_g_zal()[0]['g_val'][0]
    assert g_first == pytest.approx(G, rel=1e-6)


def test_g_chen(rng):
    shape = (60, 60)
    donor_l = rng.uniform(300, 1500, shape)
    low = [rng.poisson(i).astype(float) for i in simulate(donor_l, donor_l, 0.1)]
    high = [rng.poisson(i).astype(float) for i in simulate(donor_l, donor_l, 0.45)]
    labels = _blocks(shape)
    est = fret.GFactorEstimation(labels, labels, *high, *low, a_val=A, d_val=D)
    g_df, fit_arr = est.estimate_g_chen(n_boot=500)
    res = g_df.iloc[0]
    assert res['g_val'] == pytest.approx(G, rel=0.03)
    assert res['g_ci_low'] < res['g_val'] < res['g_ci_high']
    assert len(fit_arr) == 2 and fit_arr[0].shape == (2, 36)
    # bootstrap standard error does not shrink with the number of resamples
    err_more = est.estimate_g_chen(n_boot=4000)[0]['g_err'][0]
    assert err_more == pytest.approx(res['g_err'], rel=0.25)


def test_k_factor(rng):
    donor = rng.uniform(300, 1500, (40, 40))
    imgs = simulate(donor, donor / 1.7, 0.3)
    labels = _blocks(donor.shape)
    k_df, _ = fret.KFactorEstimation(labels, *imgs, a_val=A, d_val=D, G_val=G).estimate_k()
    assert k_df['k_val'][0] == pytest.approx(1.7, rel=1e-4)


def test_legacy_module_alias():
    from domb_napari import _e_fret
    assert _e_fret.CubesFRET is CubesFRET

""" 3³/E-FRET

Stand-alone module for 3-cube E-FRET calibration and estimation, it does not
depend on napari and can be used for batch processing.

Notation
--------
I_DD, I_DA, I_AA: intensities in donor ex./donor em., donor ex./acceptor em.
and acceptor ex./acceptor em. channels, background subtracted (signed,
not clipped).

Cross-talk coefficients are estimated with single-fluorophore samples:
a = I_DA(A) / I_AA(A), b = I_DD(A) / I_AA(A),
c = I_AA(D) / I_DD(D), d = I_DA(D) / I_DD(D).

Spectral unmixing (exact solution of the linear system, it reduces to the
conventional equations if b = c = 0):
D  = (I_DD - b * I_AA) / (1 - b * c)      donor signal
A  = (I_AA - c * I_DD) / (1 - b * c)      directly excited acceptor signal
Fc = I_DA - a * A - d * D                 sensitized emission

E_D   = Fc / (Fc + G * D)                 apparent efficiency, donor-centric
E_A   = xi * Fc / (a * A)                 apparent efficiency, acceptor-centric
Ecorr = E_D * A(0) / A(t)                 corrected for acceptor photobleaching

No value is clipped or replaced: Fc, E_D and E_A are signed, pixels with
a non-positive denominator are NaN. Rectification of the noise (clipping at
zero) biases the mean of any region, so thresholding has to be done by masks.

References
----------
- Erickson et al., 2001. "Preassociation of calmodulin with voltage-gated Ca(2+) channels revealed by FRET in single living cells". doi: 10.1016/s0896-6273(01)00438-x.
- Zal and Gascoigne, 2004. "Photobleaching-corrected FRET efficiency imaging of live cells". doi: 10.1529/biophysj.103.022087
- Chen et al., 2006. "Measurement of FRET efficiency and ratio of donor to acceptor concentration in living cells". doi: 10.1529/biophysj.106.088773
- Butz et al., 2016, "Quantifying macromolecular interactions in living cells using FRET two-hybrid assays". doi:10.1038/nprot.2016.128

"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit, prange

from scipy import stats

from ._parallel import NUMBA_LOCK
from .profiles import labels_to_profiles, label_ids


# --------------------------------------------------------------------------
# coefficients
# --------------------------------------------------------------------------
@dataclass
class FRETCoefs():
    """ Calibration coefficients of a FRET pair.

    Attributes
    ----------
    a, d: float
        acceptor and donor cross-talk to the DA channel
    G: float
        gauge factor, required for E_D and Ecorr
    xi: float
        ratio of acceptor to donor extinction coefficients at the donor
        excitation wavelength, required for E_A
    b, c: float
        acceptor cross-talk to the DD channel and donor cross-talk to the
        AA channel, 0 if negligible

    """
    a: float
    d: float
    G: float = None
    xi: float = None
    b: float = 0.0
    c: float = 0.0

    def __post_init__(self):
        for name in ('a', 'd', 'b', 'c'):
            val = getattr(self, name)
            val = 0.0 if (val is None and name in ('b', 'c')) else val
            if val is None or not np.isfinite(val):
                raise ValueError(f'Coefficient {name} must be a finite number!')
            setattr(self, name, float(val))
        if self.b * self.c >= 1.0:
            raise ValueError('Product of b and c must be less than 1!')

    def require(self, output_type:str):
        """ Raises ValueError if the coefficients are not sufficient for the output.

        """
        if output_type in ('E_D', 'Ecorr') and not (self.G is not None and self.G > 0):
            raise ValueError(f'G factor must be a positive number for {output_type} estimation!')
        if output_type == 'E_A':
            if not (self.xi is not None and self.xi > 0):
                raise ValueError('xi must be a positive number for E_A estimation!')
            if not self.a > 0:
                raise ValueError('Coefficient a must be a positive number for E_A estimation!')

    @classmethod
    def from_yaml(cls, path, fret_pair:str):
        """ Loads the coefficients of a FRET pair from a YAML file.

        """
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if fret_pair not in config:
            raise KeyError(f'FRET pair {fret_pair} is not in {path}!')
        pair = config[fret_pair]
        return cls(a=pair.get('a'), d=pair.get('d'), G=pair.get('G'),
                   xi=pair.get('xi'), b=pair.get('b', 0.0), c=pair.get('c', 0.0))


def update_yaml(path, fret_pair:str, **coefs):
    """ Writes coefficients of a FRET pair to a YAML file, other pairs are kept.

    """
    import os
    import yaml
    config = {}
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    pair = dict(config.get(fret_pair) or {})
    pair.update({k: (None if v is None else float(v)) for k, v in coefs.items()})
    config[fret_pair] = pair
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


# --------------------------------------------------------------------------
# pixel-wise and intensity-wise estimation
# --------------------------------------------------------------------------
_MODES = {'Fc': 0, 'E_D': 1, 'E_A': 2, 'D': 3, 'A': 4}


@njit(parallel=True, cache=True)
def _fret_kernel(dd_img, da_img, aa_img, a, b, c, d, G, xi, corr, mode):
    """ Single-pass pixel-wise kernel, float64 arithmetic, float32 output.

    """
    n_t, n_x, n_y = dd_img.shape
    out = np.empty((n_t, n_x, n_y), dtype=np.float32)
    k = 1.0 / (1.0 - b * c)
    for i in prange(n_t):
        corr_i = corr[i]
        for x in range(n_x):
            for y in range(n_y):
                dd = np.float64(dd_img[i, x, y])
                da = np.float64(da_img[i, x, y])
                aa = np.float64(aa_img[i, x, y])
                don = (dd - b * aa) * k
                acc = (aa - c * dd) * k
                fc = da - a * acc - d * don
                if mode == 0:
                    val = fc
                elif mode == 1:
                    den = fc + G * don
                    val = fc / den * corr_i if den > 0.0 else np.nan
                elif mode == 2:
                    den = a * acc
                    val = xi * fc / den if den > 0.0 else np.nan
                elif mode == 3:
                    val = don
                else:
                    val = acc
                out[i, x, y] = val
    return out


def fret_from_intensities(dd, da, aa, coefs:FRETCoefs):
    """ FRET estimation from intensities of any shape (numpy reference).

    Meant for ROI-averaged intensities: efficiency of a region has to be
    calculated from its mean intensities (ratio of means), the mean of
    pixel-wise ratios is biased at low signal-to-noise ratio.

    Returns
    -------
    output: dict of ndarray
        'D', 'A', 'Fc' and, if the coefficients are provided, 'E_D', 'E_A'

    """
    dd = np.asarray(dd, dtype=np.float64)
    da = np.asarray(da, dtype=np.float64)
    aa = np.asarray(aa, dtype=np.float64)
    k = 1.0 / (1.0 - coefs.b * coefs.c)
    don = (dd - coefs.b * aa) * k
    acc = (aa - coefs.c * dd) * k
    fc = da - coefs.a * acc - coefs.d * don
    output = {'D': don, 'A': acc, 'Fc': fc}
    if coefs.G is not None and coefs.G > 0:
        den = fc + coefs.G * don
        e_d = np.full(fc.shape, np.nan)
        np.divide(fc, den, out=e_d, where=den > 0)
        output['E_D'] = e_d
    if coefs.xi is not None and coefs.xi > 0 and coefs.a > 0:
        den = coefs.a * acc
        e_a = np.full(fc.shape, np.nan)
        np.divide(coefs.xi * fc, den, out=e_a, where=den > 0)
        output['E_A'] = e_a
    return output


def _masked_profile(img:np.ndarray, mask:np.ndarray=None):
    """ Mean intensity of every frame inside of a mask, NaN are ignored.

    """
    if mask is None:
        return np.nanmean(img.reshape(img.shape[0], -1), axis=1, dtype=np.float64)
    mask = np.asarray(mask) != 0
    if mask.shape != img.shape[1:]:
        raise ValueError(f'Mask shape {mask.shape} and frame shape {img.shape[1:]} are different!')
    if not mask.any():
        raise ValueError('Mask is empty!')
    return np.nanmean(img[:, mask], axis=1, dtype=np.float64)


class CubesFRET():
    """ Class for estimating FRET efficiency in image time series

    Parameters/attributes
    ----------
    dd_img: ndarray [t,x,y] or [x,y]
        Time series with donor excitation-donor emission
    da_img: ndarray [t,x,y] or [x,y]
        Time series with donor excitation-acceptor emission
    aa_img: ndarray [t,x,y] or [x,y]
        Time series with acceptor excitation-acceptor emission
    a_val: float
        Acceptor cross-talk coefficient (I_DA(A) / I_AA(A))
    d_val: float
        Donor cross-talk coefficient (I_DA(D) / I_DD(D))
    G_val: float
        Gauge ("G") factor of imaging system
    eps_rel_val: float
        Ratio of acceptor to donor extinction coefficients at donor excitation wavelength
    b_val: float
        Acceptor cross-talk to DD channel (I_DD(A) / I_AA(A))
    c_val: float
        Donor cross-talk to AA channel (I_AA(D) / I_DD(D))

    Methods
    -------
    Fc_img():
        Sensitized emission, signed.
    E_D_img():
        Apparent FRET efficiency (Zal and Gascoigne, 2004).
    E_A_img():
        Acceptor-centric apparent FRET efficiency (Erickson et al., 2001).
    Ecorr_img(mask=None, f0_frames=2):
        E_D corrected for acceptor photobleaching (Zal and Gascoigne, 2004).
    D_img(), A_img():
        Unmixed donor and acceptor signals.
    roi_profiles(labels):
        ROI-wise estimation from mean intensities.

    All image methods return float32 ndarray [t,x,y].

    """
    def __init__(self, dd_img:np.ndarray, da_img:np.ndarray, aa_img:np.ndarray,
                 a_val:float, d_val:float,
                 G_val:float=None, eps_rel_val:float=None,
                 b_val:float=0.0, c_val:float=0.0):
        imgs = [np.asarray(img) for img in (dd_img, da_img, aa_img)]
        if imgs[0].ndim == 2:
            imgs = [img[np.newaxis, ...] for img in imgs]
        if not all(img.ndim == 3 and img.shape == imgs[0].shape for img in imgs):
            raise ValueError('DD, DA and AA images must have the same 2D or 3D shape!')
        self.dd_img, self.da_img, self.aa_img = imgs
        self.coefs = FRETCoefs(a=a_val, d=d_val, G=G_val, xi=eps_rel_val,
                               b=b_val, c=c_val)

    @classmethod
    def from_coefs(cls, dd_img, da_img, aa_img, coefs:FRETCoefs):
        return cls(dd_img, da_img, aa_img, a_val=coefs.a, d_val=coefs.d,
                   G_val=coefs.G, eps_rel_val=coefs.xi, b_val=coefs.b, c_val=coefs.c)

    def _calc(self, mode:str, corr:np.ndarray=None):
        cf = self.coefs
        if corr is None:
            corr = np.ones(self.dd_img.shape[0], dtype=np.float64)
        with NUMBA_LOCK:
            return _fret_kernel(self.dd_img, self.da_img, self.aa_img,
                                cf.a, cf.b, cf.c, cf.d,
                                1.0 if cf.G is None else float(cf.G),
                                1.0 if cf.xi is None else float(cf.xi),
                                np.asarray(corr, dtype=np.float64), _MODES[mode])

    def Fc_img(self):
        return self._calc('Fc')

    def D_img(self):
        return self._calc('D')

    def A_img(self):
        return self._calc('A')

    def E_D_img(self):
        self.coefs.require('E_D')
        return self._calc('E_D')

    def E_A_img(self):
        self.coefs.require('E_A')
        return self._calc('E_A')

    def acceptor_bleaching(self, mask:np.ndarray=None, f0_frames:int=2):
        """ Fraction of the non-bleached acceptor in every frame, A(t) / A(0).

        Estimated from the mean unmixed acceptor signal inside of the mask
        (whole frame if None), so the redistribution of the acceptor between
        pixels does not affect the estimation.

        """
        aa_prof = _masked_profile(self.aa_img, mask)
        dd_prof = _masked_profile(self.dd_img, mask) if self.coefs.c != 0 else 0.0
        acc_prof = (aa_prof - self.coefs.c * dd_prof) / (1.0 - self.coefs.b * self.coefs.c)
        acc_0 = np.mean(acc_prof[:max(1, int(f0_frames))])
        if not (np.all(np.isfinite(acc_prof)) and acc_0 > 0 and np.all(acc_prof > 0)):
            raise ValueError('Acceptor intensity inside of the mask must be positive in every frame!')
        return acc_prof / acc_0

    def Ecorr_img(self, mask:np.ndarray=None, f0_frames:int=2):
        self.coefs.require('Ecorr')
        return self._calc('E_D', corr=1.0 / self.acceptor_bleaching(mask, f0_frames))

    def roi_profiles(self, labels:np.ndarray, bleach_mask:np.ndarray=None,
                     f0_frames:int=2):
        """ ROI-wise FRET estimation from mean intensities of the regions.

        Returns
        -------
        output: dict of ndarray [lab,t]
            'DD', 'DA', 'AA', 'D', 'A', 'Fc' and, if the coefficients are
            provided, 'E_D', 'E_A', 'Ecorr'; rows follow profiles.label_ids

        """
        dd = labels_to_profiles(labels, self.dd_img)
        da = labels_to_profiles(labels, self.da_img)
        aa = labels_to_profiles(labels, self.aa_img)
        output = {'DD': dd.astype(np.float64), 'DA': da.astype(np.float64),
                  'AA': aa.astype(np.float64)}
        output.update(fret_from_intensities(dd, da, aa, self.coefs))
        if 'E_D' in output:
            mask = (np.asarray(labels) != 0) if bleach_mask is None else bleach_mask
            output['Ecorr'] = output['E_D'] / self.acceptor_bleaching(mask, f0_frames)[np.newaxis, :]
        return output


# --------------------------------------------------------------------------
# linear models for calibration
# --------------------------------------------------------------------------
def _group_sums(x, y, w, n_groups:int, seed:int):
    """ Weighted moment sums of random groups of observations [6,groups].

    """
    n = x.shape[0]
    k = int(min(n_groups, n))
    group = np.random.default_rng(seed).permutation(n) % k
    terms = (w, w * x, w * y, w * x * x, w * y * y, w * x * y)
    return np.stack([np.bincount(group, weights=t, minlength=k) for t in terms])


def _ratio_est(s):
    return s[2] / s[1]


def _deming_est(s, delta):
    mx, my = s[1] / s[0], s[2] / s[0]
    vxx = s[3] / s[0] - mx * mx
    vyy = s[4] / s[0] - my * my
    vxy = s[5] / s[0] - mx * my
    with np.errstate(invalid='ignore', divide='ignore'):
        slope = (vyy - delta * vxx + np.sqrt((vyy - delta * vxx) ** 2 + 4.0 * delta * vxy ** 2)) / (2.0 * vxy)
    return np.stack((slope, my - slope * mx))


def _jackknife(sums, estimator):
    """ Estimate and its delete-a-group jackknife standard error.

    """
    total = sums.sum(axis=1)
    theta = estimator(total)
    k = sums.shape[1]
    if k < 2:
        return theta, np.full(np.shape(theta), np.nan)
    loo = estimator(total[:, np.newaxis] - sums)
    err = np.sqrt((k - 1) / k * np.sum((loo - np.mean(loo, axis=-1, keepdims=True)) ** 2, axis=-1))
    return theta, err


def linear_fits(x, y, delta:float=None, w=None, n_groups:int=50, seed:int=0):
    """ Slope of the linear relation y ~ x by three estimators.

    - 'deming': errors-in-variables regression with an intercept. Consistent
      if both variables are noisy, delta is the ratio of the noise variances
      var(err_y) / var(err_x). By default delta = |mean(y) / mean(x)|, which
      holds for the shot noise of two channels detected with the same gain.
    - 'ratio': sum(y) / sum(x), line through the origin. Consistent if both
      variables are noisy and there is no offset.
    - 'ols': ordinary least squares with an intercept. Underestimates the
      slope if x is noisy (regression dilution), kept for diagnostics.

    Standard errors of 'deming' and 'ratio' are delete-a-group jackknife ones.
    Observations are assumed independent, which is not strictly true for
    neighbouring pixels, so the errors are the lower bounds.

    Parameters
    ----------
    x, y: array_like
        observations, non-finite pairs are dropped
    delta: float
        ratio of the noise variances for Deming regression
    w: array_like
        weights of observations for 'deming' and 'ratio' (e.g. ROI areas)

    Returns
    -------
    fit: dict
        deming, deming_i, deming_err, deming_i_err, ratio, ratio_err,
        ols, ols_i, ols_err, ols_i_err, r2, p, delta, n

    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=np.float64).ravel()
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[keep], y[keep], w[keep]
    fit = {key: np.nan for key in ('deming', 'deming_i', 'deming_err', 'deming_i_err',
                                   'ratio', 'ratio_err',
                                   'ols', 'ols_i', 'ols_err', 'ols_i_err', 'r2', 'p', 'delta')}
    fit['n'] = int(x.size)
    if x.size < 3 or np.ptp(x) == 0:
        return fit

    if delta is None:
        mx, my = np.average(x, weights=w), np.average(y, weights=w)
        delta = abs(my / mx) if (mx != 0 and my != 0) else 1.0
    fit['delta'] = float(delta)

    sums = _group_sums(x, y, w, n_groups, seed)
    ratio, ratio_err = _jackknife(sums, _ratio_est)
    deming, deming_err = _jackknife(sums, lambda s: _deming_est(s, delta))
    ols = stats.linregress(x, y)
    fit.update({'deming': float(deming[0]), 'deming_i': float(deming[1]),
                'deming_err': float(deming_err[0]), 'deming_i_err': float(deming_err[1]),
                'ratio': float(ratio), 'ratio_err': float(ratio_err),
                'ols': float(ols.slope), 'ols_i': float(ols.intercept),
                'ols_err': float(ols.stderr), 'ols_i_err': float(ols.intercept_stderr),
                'r2': float(ols.rvalue ** 2), 'p': float(ols.pvalue)})
    return fit


def _coef_row(name:str, fit:dict):
    """ Row of the calibration table for one coefficient.

    """
    return {name + '_val': fit['deming'], name + '_err': fit['deming_err'],
            name + '_i': fit['deming_i'], name + '_i_err': fit['deming_i_err'],
            name + '_ratio': fit['ratio'], name + '_ratio_err': fit['ratio_err'],
            name + '_ols': fit['ols'], name + '_ols_err': fit['ols_err'],
            name + '_ols_i': fit['ols_i'], name + '_ols_i_err': fit['ols_i_err'],
            name + '_r^2': fit['r2'], name + '_p': fit['p']}


def _mean_frames(img:np.ndarray, n_frames:int=None):
    """ 2D float64 image as the mean of the first n_frames of a stack.

    """
    img = np.asarray(img)
    if img.ndim == 2:
        return img.astype(np.float64)
    if img.ndim != 3:
        raise ValueError('Incorrect input image shape, must be 2D image or 3D stack!')
    return np.mean(img[:n_frames], axis=0, dtype=np.float64)


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
class CrossTalkEstimation():
    """ Class for estimating cross-talk coefficients from image time series.

    The coefficient is the slope of the cross-talk channel intensity against
    the reference channel intensity over the pixels of the mask. The primary
    estimate ('_val') is the Deming regression slope, the ratio and OLS
    estimates are reported for diagnostics, see linear_fits. A non-zero
    intercept ('_i') points to an inaccurate background subtraction.

    Parameters/attributes
    ----------
    mask: ndarray [x,y]
        Binary mask for pixels to be used in estimation
    dd_img: ndarray [t,x,y]
        Time series with donor excitation-donor emission
    da_img: ndarray [t,x,y]
        Time series with donor excitation-acceptor emission
    aa_img: ndarray [t,x,y]
        Time series with acceptor excitation-acceptor emission

    Methods
    -------
    estimate_a():
        Acceptor-only sample: 'a' (DA vs AA) and 'b' (DD vs AA)
    estimate_d():
        Donor-only sample: 'd' (DA vs DD) and 'c' (AA vs DD)

    Both return tuple (coef_df, fit_arr, summary_df):
    coef_df - coefficients estimated in every frame,
    fit_arr - [2,px] reference and cross-talk intensities of time-averaged images,
    summary_df - one row, final coefficients estimated with time-averaged
    images (the lowest noise), median and MAD of the frame-wise estimates.

    """
    def __init__(self, mask,
                 dd_img=None, da_img=None, aa_img=None):
        if any(img is None for img in [dd_img, da_img, aa_img]):
            raise ValueError("All spectral channels must be provided!")
        self.dd_img, self.da_img, self.aa_img = [np.asarray(img) if np.ndim(img) == 3
                                                 else np.asarray(img)[np.newaxis, ...]
                                                 for img in (dd_img, da_img, aa_img)]
        self.mask = np.asarray(mask) != 0
        if not self.mask.any():
            raise ValueError('Mask is empty!')

    @staticmethod
    def _coef_calc(img_ref, img_prm, img_off, img_mask,
                   c_prm_name, c_off_name):
        rows = []
        for i in range(len(img_ref)):
            arr_ref = img_ref[i][img_mask]
            row = {'frame_n': i}
            row.update(_coef_row(c_prm_name, linear_fits(arr_ref, img_prm[i][img_mask])))
            row.update(_coef_row(c_off_name, linear_fits(arr_ref, img_off[i][img_mask])))
            rows.append(row)
        coef_df = pd.DataFrame(rows)

        # final estimation with time-averaged images
        ref_mean = _mean_frames(img_ref)[img_mask]
        prm_mean = _mean_frames(img_prm)[img_mask]
        off_mean = _mean_frames(img_off)[img_mask]
        summary = {'n_frames': len(img_ref), 'n_px': int(np.sum(img_mask))}
        for name, arr in ((c_prm_name, prm_mean), (c_off_name, off_mean)):
            summary.update(_coef_row(name, linear_fits(ref_mean, arr)))
            frame_vals = coef_df[name + '_val'].to_numpy(dtype=np.float64)
            summary[name + '_frames_median'] = np.nanmedian(frame_vals)
            summary[name + '_frames_mad'] = stats.median_abs_deviation(frame_vals, nan_policy='omit')
        return (coef_df, np.array([ref_mean, prm_mean]), pd.DataFrame([summary]))

    def estimate_a(self):
        """ Estimate acceptor cross-talk coefficients 'a' and 'b'

        """
        return self._coef_calc(img_ref=self.aa_img,
                               img_prm=self.da_img,
                               img_off=self.dd_img,
                               img_mask=self.mask,
                               c_prm_name='a',
                               c_off_name='b')

    def estimate_d(self):
        """ Estimate donor cross-talk coefficients 'd' and 'c'

        """
        return self._coef_calc(img_ref=self.dd_img,
                               img_prm=self.da_img,
                               img_off=self.aa_img,
                               img_mask=self.mask,
                               c_prm_name='d',
                               c_off_name='c')


def _roi_intensities(labels, dd_img, da_img, aa_img, coefs:FRETCoefs):
    """ Unmixed ROI-mean signals and ROI areas of 2D images.

    """
    labels = np.asarray(labels)
    if labels.dtype == bool:
        labels = labels.astype(np.uint8)
    ids = label_ids(labels)
    if ids.size == 0:
        raise ValueError('Mask is empty!')
    area = np.bincount(np.searchsorted(ids, labels[labels != 0])).astype(np.float64)
    dd = labels_to_profiles(labels, dd_img)[:, 0]
    da = labels_to_profiles(labels, da_img)[:, 0]
    aa = labels_to_profiles(labels, aa_img)[:, 0]
    signals = fret_from_intensities(dd, da, aa, coefs)
    signals.update({'DD': dd.astype(np.float64), 'DA': da.astype(np.float64),
                    'AA': aa.astype(np.float64), 'area': area})
    return signals


class GFactorEstimation():
    """ Class for estimating G factor

    Parameters/attributes
    ----------
    mask: ndarray [x,y]
        labels of ROIs to be used in estimation (higher FRET images)
    l_mask: ndarray [x,y]
        labels of ROIs of lower FRET images, used in Chen method, mask if None
    h_dd_img, h_da_img, h_aa_img: ndarray [x,y] or [t,x,y]
        DD, DA and AA images with higher expected FRET
        (before acceptor photobleaching or high FRET construct)
    l_dd_img, l_da_img, l_aa_img: ndarray [x,y] or [t,x,y]
        DD, DA and AA images with lower expected FRET
        (after acceptor photobleaching or low FRET construct)
    a_val, d_val: float
        cross-talk coefficients
    b_val, c_val: float
        minor cross-talk coefficients, 0 if negligible
    n_frames: int
        number of the first frames of 3D inputs to average, all if None

    Methods
    -------
    estimate_g_zal():
        Acceptor photobleaching method (Zal and Gascoigne, 2004)
    estimate_g_chen(n_boot=2000, seed=0):
        Two constructs with different FRET efficiency (Chen et al., 2006)

    """
    def __init__(self, mask, l_mask=None,
                 h_dd_img=None, h_da_img=None, h_aa_img=None,
                 l_dd_img=None, l_da_img=None, l_aa_img=None,
                 a_val=None, d_val=None, b_val=0.0, c_val=0.0,
                 n_frames:int=None):
        if any(img is None for img in [h_dd_img, h_da_img, h_aa_img,
                                       l_dd_img, l_da_img, l_aa_img]):
            raise ValueError("All input images must be provided!")
        self.h_dd_img, self.h_da_img, self.h_aa_img = [_mean_frames(img, n_frames) for img
                                                       in (h_dd_img, h_da_img, h_aa_img)]
        self.l_dd_img, self.l_da_img, self.l_aa_img = [_mean_frames(img, n_frames) for img
                                                       in (l_dd_img, l_da_img, l_aa_img)]
        self.coefs = FRETCoefs(a=a_val, d=d_val, b=b_val, c=c_val)
        self.mask = mask
        self.l_mask = l_mask if l_mask is not None else mask

    def _signals(self):
        sig_h = _roi_intensities(self.mask, self.h_dd_img, self.h_da_img,
                                 self.h_aa_img, self.coefs)
        sig_l = _roi_intensities(self.l_mask, self.l_dd_img, self.l_da_img,
                                 self.l_aa_img, self.coefs)
        return sig_h, sig_l

    def estimate_g_zal(self):
        """ Estimate G factor by acceptor photobleaching

        G = (Fc - Fc_post) / (D_post - D) by definition, i.e. the line
        dFc = G * dD goes through the origin. Both differences are noisy,
        so the primary estimate ('g_val') is the area-weighted ratio of sums
        over ROIs, which equals the ratio of the whole mask means. Deming
        and OLS slopes with free intercept are reported for diagnostics:
        an intercept far from zero points to wrong cross-talk coefficients,
        misregistration of pre- and post-bleaching images or donor bleaching.

        ROIs of the same cell are not independent, so 'g_err' (delete-one-ROI
        jackknife) is the within-sample error only.

        Returns
        -------
        g_df: pd.DataFrame
            one row: g_val, g_err, g_deming, g_deming_err, g_ols, g_ols_err,
            g_i, g_i_err, g_r^2, g_p, n_roi
        fit_arr: np.ndarray [2,roi]
            (dD, dFc) of ROIs

        """
        sig_h, sig_l = self._signals()
        if sig_h['area'].shape != sig_l['area'].shape:
            raise ValueError('The same ROIs must be used for both images in Zal method!')
        fc_delta = sig_h['Fc'] - sig_l['Fc']
        don_delta = sig_l['D'] - sig_h['D']

        fit = linear_fits(don_delta, fc_delta, w=sig_h['area'], n_groups=don_delta.size)

        # Deming regression needs independent errors of both variables, but
        # the noise of DD enters dFc with the factor d. So the regression is
        # done for dFc - d * dD = d(DA - a * A), which does not contain DD,
        # and d is added back to the slope. The ratio of the shot noise
        # variances is estimated from the mean intensities.
        cf = self.coefs
        var_y = sum(np.average(np.abs(s['DA']) + cf.a ** 2 * np.abs(s['AA']), weights=s['area'])
                    for s in (sig_h, sig_l))
        var_x = sum(np.average(np.abs(s['DD']), weights=s['area']) for s in (sig_h, sig_l))
        fit_dem = linear_fits(don_delta, fc_delta - cf.d * don_delta,
                              delta=var_y / var_x if var_x > 0 else 1.0,
                              w=sig_h['area'], n_groups=don_delta.size)
        fit['deming'], fit['deming_err'] = fit_dem['deming'] + cf.d, fit_dem['deming_err']

        g_df = pd.DataFrame([{'g_val': fit['ratio'], 'g_err': fit['ratio_err'],
                              'g_deming': fit['deming'], 'g_deming_err': fit['deming_err'],
                              'g_ols': fit['ols'], 'g_ols_err': fit['ols_err'],
                              'g_i': fit['ols_i'], 'g_i_err': fit['ols_i_err'],
                              'g_r^2': fit['r2'], 'g_p': fit['p'],
                              'n_roi': int(don_delta.size)}])
        return (g_df, np.array([don_delta, fc_delta]))

    def estimate_g_chen(self, n_boot:int=2000, seed:int=0):
        """ Estimate G factor with two constructs of different FRET efficiency

        G = (Fc_h/A_h - Fc_l/A_l) / (D_l/A_l - D_h/A_h), where every term is
        the ratio of area-weighted sums over ROIs of one construct.
        The standard error and the 95% interval are bootstrap ones (ROIs are
        resampled independently for both constructs).

        Returns
        -------
        g_df: pd.DataFrame
            one row: g_val, g_err, g_ci_low, g_ci_high, d_a_h, fc_a_h, d_a_l,
            fc_a_l (ratios of sums, D/A and Fc/A of both constructs), n_roi_h, n_roi_l
        fit_arr: tuple of np.ndarray [2,roi]
            (D/A, Fc/A) of ROIs of the high and of the low FRET constructs

        """
        sig_h, sig_l = self._signals()

        def _sums(sig):
            return np.stack([sig['area'] * sig[key] for key in ('Fc', 'D', 'A')])

        def _g(s_h, s_l):
            with np.errstate(invalid='ignore', divide='ignore'):
                return (s_h[0] / s_h[2] - s_l[0] / s_l[2]) / (s_l[1] / s_l[2] - s_h[1] / s_h[2])

        val_h, val_l = _sums(sig_h), _sums(sig_l)
        g_val = float(_g(val_h.sum(axis=1), val_l.sum(axis=1)))

        g_err, g_low, g_high = np.nan, np.nan, np.nan
        n_h, n_l = val_h.shape[1], val_l.shape[1]
        if n_boot > 0 and n_h > 1 and n_l > 1:
            rng = np.random.default_rng(seed)
            boot_h = val_h[:, rng.integers(0, n_h, (n_boot, n_h))].sum(axis=2)
            boot_l = val_l[:, rng.integers(0, n_l, (n_boot, n_l))].sum(axis=2)
            g_boots = _g(boot_h, boot_l)
            g_boots = g_boots[np.isfinite(g_boots)]
            if g_boots.size > 1:
                g_err = float(np.std(g_boots, ddof=1))
                g_low, g_high = (float(q) for q in np.percentile(g_boots, [2.5, 97.5]))

        tot_h, tot_l = val_h.sum(axis=1), val_l.sum(axis=1)
        g_df = pd.DataFrame([{'g_val': g_val, 'g_err': g_err,
                              'g_ci_low': g_low, 'g_ci_high': g_high,
                              'd_a_h': tot_h[1] / tot_h[2], 'fc_a_h': tot_h[0] / tot_h[2],
                              'd_a_l': tot_l[1] / tot_l[2], 'fc_a_l': tot_l[0] / tot_l[2],
                              'n_roi_h': n_h, 'n_roi_l': n_l}])
        with np.errstate(invalid='ignore', divide='ignore'):
            fit_arr = (np.array([sig_h['D'] / sig_h['A'], sig_h['Fc'] / sig_h['A']]),
                       np.array([sig_l['D'] / sig_l['A'], sig_l['Fc'] / sig_l['A']]))
        return (g_df, fit_arr)


class KFactorEstimation():
    """ Class for estimating k factor for donor/acceptor ratio estimation

    Based on Chen et al., 2006. For a construct with 1:1 stoichiometry
    k = (D + Fc / G) / A, estimated as the area-weighted ratio of sums
    over ROIs, Deming and OLS slopes are reported for diagnostics.

    Parameters/attributes
    ----------
    mask: ndarray [x,y]
        labels of ROIs to be used in estimation
    dd_img, da_img, aa_img: ndarray [x,y] or [t,x,y]
        DD, DA and AA images
    a_val, d_val, G_val: float
        cross-talk coefficients and G factor
    b_val, c_val: float
        minor cross-talk coefficients, 0 if negligible
    n_frames: int
        number of the first frames of 3D inputs to average, all if None

    Methods
    -------
    estimate_k():
        returns tuple (k_df, fit_arr) with one-row dataframe and
        [2,roi] array (A, D + Fc / G)

    """
    def __init__(self, mask,
                 dd_img=None, da_img=None, aa_img=None,
                 a_val=None, d_val=None, G_val=None, b_val=0.0, c_val=0.0,
                 n_frames:int=None):
        if any(img is None for img in [dd_img, da_img, aa_img]):
            raise ValueError("All spectral channels must be provided!")
        self.dd_img, self.da_img, self.aa_img = [_mean_frames(img, n_frames) for img
                                                 in (dd_img, da_img, aa_img)]
        self.mask = mask
        self.coefs = FRETCoefs(a=a_val, d=d_val, G=G_val, b=b_val, c=c_val)
        self.coefs.require('E_D')

    def estimate_k(self):
        sig = _roi_intensities(self.mask, self.dd_img, self.da_img,
                               self.aa_img, self.coefs)
        don_tot = sig['D'] + sig['Fc'] / self.coefs.G
        fit = linear_fits(sig['A'], don_tot, w=sig['area'], n_groups=don_tot.size)
        k_df = pd.DataFrame([{'k_val': fit['ratio'], 'k_err': fit['ratio_err'],
                              'k_deming': fit['deming'], 'k_deming_err': fit['deming_err'],
                              'k_ols': fit['ols'], 'k_ols_err': fit['ols_err'],
                              'k_i': fit['ols_i'], 'k_i_err': fit['ols_i_err'],
                              'k_r^2': fit['r2'], 'k_p': fit['p'],
                              'n_roi': int(don_tot.size)}])
        return (k_df, np.array([sig['A'], don_tot]))

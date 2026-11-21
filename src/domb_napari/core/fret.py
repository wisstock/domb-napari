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
        for name in COEF_KEYS:
            val = getattr(self, name)
            if val is None:
                if name in ('a', 'd'):
                    raise ValueError(f'Coefficient {name} is required!')
                val = 0.0 if name in ('b', 'c') else None
            else:
                val = _coef_value(name, val)
            setattr(self, name, val)
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
        """ Loads the coefficients of a FRET pair from a YAML file, see read_config.

        """
        config = read_config(path)
        if fret_pair not in config:
            raise KeyError(f'FRET pair {fret_pair} is not in {path}!')
        pair = config[fret_pair]
        if not isinstance(pair, dict):
            raise ValueError(f'{fret_pair}: coefficients must be a mapping, e.g. "a: 0.1"!')
        unknown = sorted(str(key) for key in pair if key not in COEF_KEYS)
        if unknown:
            raise ValueError(f'{fret_pair}: unknown coefficients {unknown}, allowed are {list(COEF_KEYS)} (case-sensitive)!')
        try:
            return cls(**{key: pair.get(key) for key in COEF_KEYS})
        except ValueError as err:
            raise ValueError(f'{fret_pair}: {err}') from err


COEF_KEYS = ('a', 'd', 'b', 'c', 'G', 'xi')


def _coef_value(name:str, raw):
    """ Coefficient as a finite float.

    YAML 1.1 reads exponents without a dot (1e-2) as strings, so strings are
    converted; booleans (YAML reads yes/no/on/off as booleans) are rejected.

    """
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError(f'Coefficient {name} = {raw!r} is a boolean, a number is required!')
    try:
        val = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f'Coefficient {name} = {raw!r} is not a number!') from None
    if not np.isfinite(val):
        raise ValueError(f'Coefficient {name} must be a finite number!')
    return val


def read_config(path):
    """ Reads a YAML configuration of FRET pairs.

    The file has to be a mapping of pair names to mappings of coefficients.
    Duplicate keys raise ValueError (plain YAML loaders silently keep the
    last value). An empty file is an empty configuration.

    Returns
    -------
    config: dict

    """
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    def _mapping(loader, node, deep=False):
        keys = [loader.construct_object(key, deep=deep) for key, _ in node.value]
        duplicates = sorted({str(key) for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f'Duplicate keys {duplicates} in {path}!')
        return loader.construct_mapping(node, deep=deep)

    _Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=_Loader)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f'{path} must contain a mapping of FRET pairs!')
    return config


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


def frame_noise_variance(stack:np.ndarray):
    """ Noise variance of single frames from successive differences.

    var = mean((I[t+1] - I[t])^2) / 2 along the time axis, which is
    insensitive to slow intensity drift (e.g. photobleaching) of a static
    sample. NaN of the first axis-0 positions are ignored.

    Parameters
    ----------
    stack: ndarray [t,...]
        at least 2 frames

    Returns
    -------
    var: ndarray [...]

    """
    stack = np.asarray(stack, dtype=np.float64)
    with np.errstate(invalid='ignore'):
        return np.nanmean(np.diff(stack, axis=0) ** 2, axis=0) / 2.0


def _coef_row(name:str, fit:dict):
    """ Row of the calibration table for one coefficient.

    """
    return {name + '_val': fit['deming'], name + '_err': fit['deming_err'],
            name + '_i': fit['deming_i'], name + '_i_err': fit['deming_i_err'],
            name + '_ratio': fit['ratio'], name + '_ratio_err': fit['ratio_err'],
            name + '_ols': fit['ols'], name + '_ols_err': fit['ols_err'],
            name + '_ols_i': fit['ols_i'], name + '_ols_i_err': fit['ols_i_err'],
            name + '_r^2': fit['r2'], name + '_p': fit['p'],
            name + '_delta': fit['delta']}


def _select_frames(img:np.ndarray, frames:tuple=None):
    """ Frames start:stop of a stack (stop is exclusive), all frames if None.

    """
    img = np.asarray(img)
    if img.ndim != 3:
        if frames is not None:
            raise ValueError('Frame range can be selected for 3D stacks only!')
        return img
    if frames is None:
        return img
    if len(frames) != 2:
        raise ValueError('Frame range should have 2 elements: start and stop!')
    start, stop = (int(i) for i in frames)
    if not 0 <= start < stop <= img.shape[0]:
        raise ValueError(f'Frame range [{start}, {stop}] is incorrect for a stack of {img.shape[0]} frames, '
                         f'0 <= start < stop <= {img.shape[0]} is required!')
    return img[start:stop]


def _mean_frames(img:np.ndarray, frames:tuple=None):
    """ 2D float64 image as the mean of frames start:stop of a stack, see _select_frames.

    """
    img = _select_frames(img, frames)
    if img.ndim == 2:
        return img.astype(np.float64)
    if img.ndim != 3:
        raise ValueError('Incorrect input image shape, must be 2D image or 3D stack!')
    return np.mean(img, axis=0, dtype=np.float64)


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

    The ratio of noise variances of the Deming regression ('_delta') is
    measured from successive frames, so it includes shot, background and
    read noise ('noise' = 'temporal'). For a single frame the shot noise
    approximation mean(y)/mean(x) is used ('noise' = 'poisson approximation').

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
    def _noise_ratios(img_ref, imgs, img_mask):
        """ Ratios of noise variances var(y) / var(x) of the masked pixels.

        Measured from successive frames if there are 2+ frames, the same
        ratio holds for single frames and for time-averaged images. Otherwise
        None (Deming regression falls back to the shot noise approximation).

        """
        if len(img_ref) < 2:
            return [None] * len(imgs), 'poisson approximation'
        var_x = np.nanmean(frame_noise_variance(img_ref[:, img_mask]))
        ratios = [np.nanmean(frame_noise_variance(img[:, img_mask])) / var_x for img in imgs]
        if not (var_x > 0 and all(np.isfinite(r) and r > 0 for r in ratios)):
            return [None] * len(imgs), 'poisson approximation'
        return ratios, 'temporal'

    @staticmethod
    def _coef_calc(img_ref, img_prm, img_off, img_mask,
                   c_prm_name, c_off_name):
        (delta_prm, delta_off), noise = CrossTalkEstimation._noise_ratios(img_ref, (img_prm, img_off), img_mask)
        rows = []
        for i in range(len(img_ref)):
            arr_ref = img_ref[i][img_mask]
            row = {'frame_n': i}
            row.update(_coef_row(c_prm_name, linear_fits(arr_ref, img_prm[i][img_mask], delta=delta_prm)))
            row.update(_coef_row(c_off_name, linear_fits(arr_ref, img_off[i][img_mask], delta=delta_off)))
            rows.append(row)
        coef_df = pd.DataFrame(rows)

        # final estimation with time-averaged images
        ref_mean = _mean_frames(img_ref)[img_mask]
        prm_mean = _mean_frames(img_prm)[img_mask]
        off_mean = _mean_frames(img_off)[img_mask]
        summary = {'n_frames': len(img_ref), 'n_px': int(np.sum(img_mask)), 'noise': noise}
        for name, arr, delta in ((c_prm_name, prm_mean, delta_prm), (c_off_name, off_mean, delta_off)):
            summary.update(_coef_row(name, linear_fits(ref_mean, arr, delta=delta)))
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
    h_frames, l_frames: tuple
        (start, stop) frame ranges of the high and the low FRET stacks to
        average, stop is exclusive; all frames if None. The noise of the
        averaged images is measured from the same frames.

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
                 h_frames:tuple=None, l_frames:tuple=None):
        if any(img is None for img in [h_dd_img, h_da_img, h_aa_img,
                                       l_dd_img, l_da_img, l_aa_img]):
            raise ValueError("All input images must be provided!")
        raw_h = [_select_frames(img, h_frames) for img in (h_dd_img, h_da_img, h_aa_img)]
        raw_l = [_select_frames(img, l_frames) for img in (l_dd_img, l_da_img, l_aa_img)]
        self.h_dd_img, self.h_da_img, self.h_aa_img = [_mean_frames(img) for img in raw_h]
        self.l_dd_img, self.l_da_img, self.l_aa_img = [_mean_frames(img) for img in raw_l]
        self.n_frames = {state: (raw[0].shape[0] if raw[0].ndim == 3 else 1)
                         for state, raw in (('h', raw_h), ('l', raw_l))}
        # noise of the averaged images is measured from the selected frames, see estimate_g_zal
        self._raw = {}
        for state, raw in (('h', raw_h), ('l', raw_l)):
            if all(img.ndim == 3 and img.shape[0] >= 2 for img in raw):
                self._raw[state] = (raw, raw[0].shape[0])
        self.coefs = FRETCoefs(a=a_val, d=d_val, b=b_val, c=c_val)
        as_labels = lambda m: np.asarray(m).astype(np.uint8) if np.asarray(m).dtype == bool else np.asarray(m)
        self.mask = as_labels(mask)
        self.l_mask = as_labels(l_mask) if l_mask is not None else self.mask

    def _signals(self):
        sig_h = _roi_intensities(self.mask, self.h_dd_img, self.h_da_img,
                                 self.h_aa_img, self.coefs)
        sig_l = _roi_intensities(self.l_mask, self.l_dd_img, self.l_da_img,
                                 self.l_aa_img, self.coefs)
        return sig_h, sig_l

    def _roi_noise(self, state:str, labels, sig):
        """ Noise variances of the ROI means of DD, DA and AA [3,roi].

        Measured from successive selected frames of the ROI mean profiles if
        there are 2+ of them, divided by the number of averaged frames.
        Otherwise the shot noise approximation |mean| / area, which ignores
        the background and read noise.

        """
        if state in self._raw:
            stacks, n_avg = self._raw[state]
            var = np.array([frame_noise_variance(labels_to_profiles(labels, st).T) for st in stacks]) / n_avg
            if np.all(np.isfinite(var)) and np.all(var.sum(axis=1) > 0):
                return var, 'temporal'
        return np.array([np.abs(sig[key]) / sig['area'] for key in ('DD', 'DA', 'AA')]), 'poisson approximation'

    def _zal_errors(self, sig_h, sig_l):
        """ Variance of dD, covariance of dD and dFc, variance of dFc (pooled over ROIs).

        Channel noise is independent, D = k(DD - b AA) and
        Fc = DA + k(ac - d) DD + k(db - a) AA with k = 1 / (1 - bc).

        """
        cf = self.coefs
        k = 1.0 / (1.0 - cf.b * cf.c)
        c_d = np.array([k, 0.0, -k * cf.b])
        c_fc = np.array([k * (cf.a * cf.c - cf.d), 1.0, k * (cf.d * cf.b - cf.a)])
        var_x = var_fc = cov = 0.0
        sources = set()
        for state, sig, labels in (('h', sig_h, self.mask), ('l', sig_l, self.l_mask)):
            var, source = self._roi_noise(state, labels, sig)
            sources.add(source)
            var = var.sum(axis=1)                    # pooled over ROIs
            var_x += np.sum(c_d ** 2 * var)
            var_fc += np.sum(c_fc ** 2 * var)
            cov -= np.sum(c_d * c_fc * var)          # dD = D_l - D_h, dFc = Fc_h - Fc_l
        source = 'temporal' if sources == {'temporal'} else 'poisson approximation'
        return var_x, cov, var_fc, source

    def estimate_g_zal(self):
        """ Estimate G factor by acceptor photobleaching

        G is the slope of dFc against dD over ROIs, where
        dFc = Fc - Fc_post and dD = D_post - D.

        The primary estimate ('g_val') is the slope of Deming regression with
        a free intercept ('g_i'), which is consistent when both differences
        are noisy. Its errors are correlated (DD enters both dD and dFc), so
        the regression is done for y = dFc - s * dD with s = cov / var(dD),
        whose errors are uncorrelated with dD, and s is added back to the
        slope. The noise variances are measured from successive frames
        ('g_noise' = 'temporal') or, for single frames, approximated by the
        shot noise ('g_noise' = 'poisson approximation'). ROIs are weighted
        by area.

        Diagnostics: 'g_ratio' (area-weighted ratio of sums, line through the
        origin) and OLS. An intercept far from zero points to wrong cross-talk
        coefficients, misregistration of pre- and post-bleaching images or
        donor bleaching. ROIs of the same cell are not independent, so the
        errors (delete-one-ROI jackknife) are the within-sample errors only.

        Returns
        -------
        g_df: pd.DataFrame
            one row: g_val, g_err, g_i, g_i_err, g_ratio, g_ratio_err,
            g_ols, g_ols_err, g_ols_i, g_ols_i_err, g_r^2, g_p, g_delta,
            g_shift, g_noise, n_roi, n_frames_h, n_frames_l
        fit_arr: np.ndarray [2,roi]
            (dD, dFc) of ROIs
        roi_df: pd.DataFrame
            ROI-mean DD, DA, AA, D, A, Fc of both images, dD, dFc

        """
        sig_h, sig_l = self._signals()
        if sig_h['area'].shape != sig_l['area'].shape:
            raise ValueError('The same ROIs must be used for both images in Zal method!')
        fc_delta = sig_h['Fc'] - sig_l['Fc']
        don_delta = sig_l['D'] - sig_h['D']
        area = sig_h['area']

        fit = linear_fits(don_delta, fc_delta, w=area, n_groups=don_delta.size)
        var_x, cov, var_fc, source = self._zal_errors(sig_h, sig_l)
        shift = cov / var_x
        delta = max((var_fc - cov ** 2 / var_x) / var_x, np.finfo(float).eps)
        fit_dem = linear_fits(don_delta, fc_delta - shift * don_delta, delta=delta,
                              w=area, n_groups=don_delta.size)
        fit.update({'deming': fit_dem['deming'] + shift, 'deming_err': fit_dem['deming_err'],
                    'deming_i': fit_dem['deming_i'], 'deming_i_err': fit_dem['deming_i_err'],
                    'delta': delta})
        row = _coef_row('g', fit)
        row.update({'g_shift': shift, 'g_noise': source, 'n_roi': int(don_delta.size),
                    'n_frames_h': self.n_frames['h'], 'n_frames_l': self.n_frames['l']})

        roi_df = pd.DataFrame({'roi': label_ids(self.mask), 'area': area,
                               **{f'{key}_high': sig_h[key] for key in ('DD', 'DA', 'AA', 'D', 'A', 'Fc')},
                               **{f'{key}_low': sig_l[key] for key in ('DD', 'DA', 'AA', 'D', 'A', 'Fc')},
                               'dD': don_delta, 'dFc': fc_delta})
        return (pd.DataFrame([row]), np.array([don_delta, fc_delta]), roi_df)

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
            fc_a_l (ratios of sums, D/A and Fc/A of both constructs), n_roi_h, n_roi_l,
            n_frames_h, n_frames_l
        fit_arr: tuple of np.ndarray [2,roi]
            (D/A, Fc/A) of ROIs of the high and of the low FRET constructs
        roi_df: pd.DataFrame
            ROI-mean DD, DA, AA, D, A, Fc, D/A, Fc/A of both constructs

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
                              'n_roi_h': n_h, 'n_roi_l': n_l,
                              'n_frames_h': self.n_frames['h'], 'n_frames_l': self.n_frames['l']}])
        tables = []
        with np.errstate(invalid='ignore', divide='ignore'):
            fit_arr = (np.array([sig_h['D'] / sig_h['A'], sig_h['Fc'] / sig_h['A']]),
                       np.array([sig_l['D'] / sig_l['A'], sig_l['Fc'] / sig_l['A']]))
            for name, sig, labels in (('high', sig_h, self.mask), ('low', sig_l, self.l_mask)):
                tables.append(pd.DataFrame({'construct': name, 'roi': label_ids(labels), 'area': sig['area'],
                                            **{key: sig[key] for key in ('DD', 'DA', 'AA', 'D', 'A', 'Fc')},
                                            'D/A': sig['D'] / sig['A'], 'Fc/A': sig['Fc'] / sig['A']}))
        return (g_df, fit_arr, pd.concat(tables, ignore_index=True))


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
    frames: tuple
        (start, stop) frame range of 3D inputs to average, stop is
        exclusive; all frames if None

    Methods
    -------
    estimate_k():
        returns tuple (k_df, fit_arr) with one-row dataframe and
        [2,roi] array (A, D + Fc / G)

    """
    def __init__(self, mask,
                 dd_img=None, da_img=None, aa_img=None,
                 a_val=None, d_val=None, G_val=None, b_val=0.0, c_val=0.0,
                 frames:tuple=None):
        if any(img is None for img in [dd_img, da_img, aa_img]):
            raise ValueError("All spectral channels must be provided!")
        self.dd_img, self.da_img, self.aa_img = [_mean_frames(img, frames) for img
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

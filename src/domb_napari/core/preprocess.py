""" Preprocessing of image series.

"""
import numpy as np

from scipy import ndimage as ndi
from scipy import optimize

from ._parallel import map_frames


def back_substr(input_img:np.ndarray, percentile:float=1.0):
    """ Background subtraction by percentile value of every frame.

    The result is signed and is not clipped at zero: clipping rectifies the
    noise of dim pixels and biases mean intensities of regions, which then
    propagates to every ratio (FRET efficiency, dF/F0).

    Parameters
    ----------
    input_img: ndarray [t,x,y]
        input image series
    percentile: float
        percentile of frame intensity used as background level

    Returns
    -------
    output_img: ndarray [t,x,y]
        background subtracted image series, float32

    """
    input_img = np.asarray(input_img)
    background = np.nanpercentile(input_img.reshape(input_img.shape[0], -1),
                                  percentile, axis=1)
    return (input_img - background[:, np.newaxis, np.newaxis]).astype(np.float32)


def median_filter_stack(input_img:np.ndarray, kernel:int=3, workers:int=None):
    """ Frame-by-frame median filter, frames are processed in parallel.

    An odd kernel size keeps the filter centered, an even one shifts the
    image by half a pixel.

    """
    return map_frames(lambda frame: ndi.median_filter(frame, size=kernel),
                      input_img, dtype=input_img.dtype, workers=workers)


def _exp(x, a, b, c):
    return a * np.exp(-b * x) + c


def _bi_exp(x, a, b, c, d, e):
    return a * np.exp(-b * x) + c * np.exp(-d * x) + e


def bleaching_fit(bleach_profile:np.ndarray, method:str='exp'):
    """ Fit of a photobleaching profile.

    Models: 'exp' - a*exp(-b*x) + c, 'biexp' - a*exp(-b*x) + c*exp(-d*x) + e.
    All parameters are non-negative, the constant term describes the
    non-bleaching fraction.

    Returns
    -------
    fit: ndarray [t]
        fitted profile
    r2: float
        coefficient of determination

    """
    y = np.asarray(bleach_profile, dtype=np.float64)
    n = y.shape[0]
    x = np.arange(n, dtype=np.float64)
    amp = max(y[0] - y[-1], np.finfo(float).eps)
    floor = max(min(y[-1], y[0]), 0.0)
    if method == 'exp':
        func, p0 = _exp, (amp, 3.0 / n, floor)
    elif method == 'biexp':
        func, p0 = _bi_exp, (amp / 2, 10.0 / n, amp / 2, 1.0 / n, floor)
    else:
        raise ValueError('Incorrect correction method!')
    popt, _ = optimize.curve_fit(func, x, y, p0=p0, bounds=(0.0, np.inf), maxfev=20000)
    fit = func(x, *popt)
    ss_res = np.sum((y - fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return fit, r2


def pb_correction(input_img:np.ndarray, mask:np.ndarray, method:str='exp'):
    """ Image series photobleaching correction by exponential fit or normalization to 1st frame.

    The bleaching profile is the mean intensity of the masked area, every
    frame is divided by the profile normalized to its first point.

    The correction rescales every channel with its own curve, so corrected
    stacks are not valid inputs for E-FRET estimation.

    Parameters
    ----------
    input_img: ndarray [t,x,y]
        input image series, background subtracted
    mask: ndarray [x,y]
        area of interest
    method: str
        'exp', 'biexp' (see bleaching_fit) or '1st' (raw profile)

    Returns
    -------
    corrected_img: ndarray [t,x,y]
        float32
    bleach_coefs: ndarray [t]
        fraction of the initial intensity
    correction_metric: str

    """
    mask = np.asarray(mask) != 0
    if not mask.any():
        raise ValueError('Photobleaching correction mask is empty!')
    bleach_profile = np.nanmean(input_img[:, mask], axis=1, dtype=np.float64)

    if method in ('exp', 'biexp'):
        bleach_fit, r2 = bleaching_fit(bleach_profile, method)
        bleach_coefs = bleach_fit / bleach_fit[0]
        correction_metric = f'R²={r2:.4f}, remaining intensity {bleach_coefs[-1]*100:.0f}%'
    elif method == '1st':
        bleach_coefs = bleach_profile / bleach_profile[0]
        correction_metric = f'remaining intensity {bleach_coefs[-1]*100:.0f}%'
    else:
        raise ValueError('Incorrect correction method!')
    if not np.all(bleach_coefs > 0):
        raise ValueError('Photobleaching profile is not positive, check the mask and the background subtraction!')
    corrected_img = (input_img / bleach_coefs[:, np.newaxis, np.newaxis]).astype(np.float32)
    return corrected_img, bleach_coefs, correction_metric

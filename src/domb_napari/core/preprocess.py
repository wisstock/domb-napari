""" Preprocessing of image series.

"""
import numpy as np

from scipy import ndimage as ndi
from scipy import optimize

from ._parallel import map_frames


BACKGROUND_METHODS = ('percentile', 'mode', 'mask')


def _log_parabola_peak(centers:np.ndarray, counts:np.ndarray, k:int):
    """ Sub-bin position of a histogram peak.

    A parabola is fitted to the logarithm of the counts of the contiguous
    bins above the half maximum around the bin k, which is exact for a
    Gaussian peak. The bin center is returned if the fit is not concave or
    its vertex leaves the fitted range.

    """
    half = counts[k] / 2.0
    left, right = k, k
    while left > 0 and counts[left - 1] > half:
        left -= 1
    while right < counts.size - 1 and counts[right + 1] > half:
        right += 1
    left, right = max(0, min(left, k - 1)), min(counts.size - 1, max(right, k + 1))
    sel = slice(left, right + 1)
    if right - left < 2 or np.any(counts[sel] <= 0):
        return float(centers[k])
    c2, c1, _ = np.polyfit(centers[sel], np.log(counts[sel]), 2)
    if c2 >= 0:
        return float(centers[k])
    vertex = -c1 / (2.0 * c2)
    if not centers[left] <= vertex <= centers[right]:
        return float(centers[k])
    return float(vertex)


def histogram_mode(values:np.ndarray):
    """ Position of the highest peak of the intensity histogram.

    Two passes:
    1. histogram of the values between the 0.1 and 99.9 percentiles with
       the Freedman-Diaconis bin width, smoothed with a [1, 2, 1] kernel,
       gives the position of the peak;
    2. histogram with a 4 times finer bin inside of +-4 coarse bins around
       the peak, the peak position is refined by a log-parabola fit of the
       bins above the half maximum.
    Bin widths of integer-valued data are integer.

    The mode is the background level when background pixels are the most
    frequent intensity of the frame. For Poisson noise the mode is about
    0.5 count below the mean.

    Parameters
    ----------
    values: ndarray
        intensities, non-finite values are ignored

    Returns
    -------
    mode: float

    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError('There are no finite values for the histogram mode estimation!')
    lo, hi = np.percentile(v, [0.1, 99.9])
    v = v[(v >= lo) & (v <= hi)]
    if hi <= lo or v.size < 10:
        return float(np.median(v)) if v.size else float(lo)
    integer = bool(np.all(v == np.round(v)))

    q25, q75 = np.percentile(v, [25, 75])
    width = 2.0 * (q75 - q25) * v.size ** (-1.0 / 3.0)
    if width <= 0:
        width = (hi - lo) / 256.0
    if integer:
        width = max(1.0, float(np.round(width)))
    edges = np.arange(lo - width / 2.0, hi + width, width)
    counts, edges = np.histogram(v, bins=edges)
    smooth = np.convolve(counts, [0.25, 0.5, 0.25], mode='same')
    peak = 0.5 * (edges[np.argmax(smooth)] + edges[np.argmax(smooth) + 1])

    # refinement: the window and the bin shrink to the width of the peak
    for _ in range(4):
        fine = width / 4.0
        if integer:
            fine = max(1.0, float(np.round(fine)))
        window = v[np.abs(v - peak) <= 4.0 * width]
        edges = np.arange(peak - 4.0 * width - fine / 2.0, peak + 4.0 * width + fine, fine)
        counts, edges = np.histogram(window, bins=edges)
        counts = np.convolve(counts, [0.25, 0.5, 0.25], mode='same')
        centers = 0.5 * (edges[:-1] + edges[1:])
        k = int(np.argmax(counts))
        peak = _log_parabola_peak(centers, counts, k)
        above = np.flatnonzero(counts > counts[k] / 2.0)
        sigma = (above[-1] - above[0] + 1) * fine / 2.355      # FWHM of a Gaussian peak
        if integer:
            sigma = max(sigma, 1.0)
        if sigma >= 0.7 * width or fine <= (1.0 if integer else 0.0):
            break
        width = sigma
    return peak


def background_levels(input_img:np.ndarray, method:str='percentile',
                      percentile:float=1.0, mask:np.ndarray=None):
    """ Background level of every frame of a stack.

    Parameters
    ----------
    input_img: ndarray [t,x,y]
        input image series
    method: str
        'percentile' - the percentile of frame intensity; the 1st percentile
        lies about 2.3 standard deviations of the background noise below its
        mean, so a positive offset remains after the subtraction;
        'mode' - the highest peak of the intensity histogram, see histogram_mode;
        'mask' - the mean intensity of the pixels of a background mask
    percentile: float
        percentile for the 'percentile' method
    mask: ndarray [x,y]
        background mask for the 'mask' method, all non-zero pixels are
        used as one region regardless of their label values

    Returns
    -------
    levels: ndarray [t]
        float64

    """
    input_img = np.asarray(input_img)
    if input_img.ndim != 3:
        raise ValueError('Input image for background estimation must be a 3D stack [t,x,y]!')
    flat = input_img.reshape(input_img.shape[0], -1)
    if method == 'percentile':
        return np.nanpercentile(flat, percentile, axis=1).astype(np.float64)
    elif method == 'mode':
        return np.array([histogram_mode(frame) for frame in flat])
    elif method == 'mask':
        if mask is None:
            raise ValueError('Background mask is required for the mask method!')
        mask = np.asarray(mask)
        if mask.shape != input_img.shape[1:]:
            raise ValueError(f'Background mask shape {mask.shape} and frame shape {input_img.shape[1:]} are different!')
        mask = mask != 0
        if not mask.any():
            raise ValueError('Background mask is empty!')
        with np.errstate(invalid='ignore'):
            levels = np.nanmean(input_img[:, mask], axis=1, dtype=np.float64)
        if not np.all(np.isfinite(levels)):
            raise ValueError('Background mask contains no finite pixels in some frames!')
        return levels
    raise ValueError(f"Unknown background method {method}! Use one of {BACKGROUND_METHODS}.")


def back_substr(input_img:np.ndarray, percentile:float=1.0, method:str='percentile',
                mask:np.ndarray=None, return_levels:bool=False):
    """ Frame-wise background subtraction.

    The result is signed and is not clipped at zero: clipping rectifies the
    noise of dim pixels and biases mean intensities of regions, which then
    propagates to every ratio (FRET efficiency, dF/F0).

    Parameters
    ----------
    input_img: ndarray [t,x,y]
        input image series
    percentile: float
        percentile of frame intensity for the 'percentile' method
    method: str
        'percentile', 'mode' or 'mask', see background_levels
    mask: ndarray [x,y]
        background mask for the 'mask' method
    return_levels: bool
        also return the subtracted level of every frame

    Returns
    -------
    output_img: ndarray [t,x,y]
        background subtracted image series, float32
    levels: ndarray [t]
        if return_levels

    """
    input_img = np.asarray(input_img)
    levels = background_levels(input_img, method=method, percentile=percentile, mask=mask)
    output_img = (input_img - levels[:, np.newaxis, np.newaxis]).astype(np.float32)
    if return_levels:
        return output_img, levels
    return output_img


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

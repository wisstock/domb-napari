""" Differential and relative image series.

All functions return float32 stacks, signed and not clipped. NaN of the
input propagate to the output.

"""
import numpy as np
from numba import njit, prange

from ._parallel import NUMBA_LOCK


@njit(parallel=True, cache=True)
def _red_green_kernel(img, left, space, right, weight_norm):
    n_t, n_x, n_y = img.shape
    n_out = n_t - (left + space + right) + 1
    out = np.empty((n_out, n_x, n_y), dtype=np.float32)
    for i in prange(n_out):
        stim_start = i + left + space
        for x in range(n_x):
            for y in range(n_y):
                base = 0.0
                for t in range(i, i + left):
                    base += img[t, x, y]
                base /= left
                stim = 0.0
                for t in range(stim_start, stim_start + right):
                    stim += img[t, x, y]
                stim /= right
                val = stim - base
                if weight_norm > 0.0:
                    val *= 0.5 * (stim + base) / weight_norm
                out[i, x, y] = val
    return out


def red_green_series(input_img:np.ndarray, left_frames:int=1, space_frames:int=0,
                     right_frames:int=1, intensity_weighted:bool=False):
    """ Differential ("red-green") image series.

    out[i] = mean(img[i+left+space : i+left+space+right]) - mean(img[i : i+left])

    The output frame i corresponds to the transition between the input frames
    i+left-1 and i+left+space, the output has t-(left+space+right)+1 frames.

    Parameters
    ----------
    input_img: ndarray [t,x,y]
        image series
    left_frames: int
        number of frames of the baseline window
    space_frames: int
        number of skipped frames between the windows
    right_frames: int
        number of frames of the response window
    intensity_weighted: bool
        multiply the difference by the mean intensity of both windows divided
        by the maximal intensity of the stack; suppresses the noise of dim
        areas for visualization and masking, the values are not quantitative

    Returns
    -------
    output_img: ndarray [t-(left+space+right)+1,x,y]

    """
    input_img = np.asarray(input_img)
    if input_img.ndim != 3:
        raise ValueError('The input image should have 3 dimensions!')
    if left_frames < 1 or right_frames < 1 or space_frames < 0:
        raise ValueError('Left and right windows must have at least 1 frame, space can not be negative!')
    if input_img.shape[0] < left_frames + space_frames + right_frames:
        raise ValueError('The input image has not enough frames for the selected windows!')

    weight_norm = 0.0
    if intensity_weighted:
        weight_norm = float(np.nanmax(np.abs(input_img)))
        if not weight_norm > 0:
            raise ValueError('The input image is empty, intensity weighting is impossible!')
    with NUMBA_LOCK:
        return _red_green_kernel(input_img, int(left_frames), int(space_frames),
                                 int(right_frames), weight_norm)


@njit(parallel=True, cache=True)
def _delta_kernel(img, base_img, weight_img, relative, f0_min):
    n_t, n_x, n_y = img.shape
    out = np.empty((n_t, n_x, n_y), dtype=np.float32)
    for i in prange(n_t):
        for x in range(n_x):
            for y in range(n_y):
                f0 = base_img[x, y]
                val = img[i, x, y] - f0
                if relative:
                    val = val / f0 if f0 > f0_min else np.nan
                out[i, x, y] = val * weight_img[x, y]
    return out


def baseline_img(input_img:np.ndarray, win_size:int=5):
    """ Pixel-wise baseline F0 as the mean of the first win_size frames.

    """
    return np.mean(input_img[:max(1, int(win_size))], axis=0, dtype=np.float64)


def delta_img(input_img:np.ndarray, mode:str='dF', win_size:int=5,
              f0_min:float=0.0, intensity_weighted:bool=False):
    """ Pixel-wise intensity changes relative to the baseline.

    Parameters
    ----------
    input_img: ndarray [t,x,y]
        image series
    mode: str
        'dF' - F(t) - F0, 'dF/F0' - (F(t) - F0) / F0
    win_size: int
        number of the first frames for the baseline F0 estimation
    f0_min: float
        for 'dF/F0' mode pixels with F0 <= f0_min are NaN, the ratio is
        undefined for a non-positive baseline
    intensity_weighted: bool
        multiply the result by the maximal intensity projection normalized to
        the range 0-1; suppresses the noise of dim areas for visualization
        and masking, the values are not quantitative

    Returns
    -------
    output_img: ndarray [t,x,y]

    """
    input_img = np.asarray(input_img)
    if input_img.ndim != 3:
        raise ValueError('The input image should have 3 dimensions!')
    if mode not in ('dF', 'dF/F0'):
        raise ValueError("Unknown mode! Use 'dF' or 'dF/F0'.")
    if f0_min < 0:
        raise ValueError('f0_min can not be negative!')

    base = baseline_img(input_img, win_size)
    weight = np.ones(base.shape, dtype=np.float64)
    if intensity_weighted:
        max_img = np.nanmax(input_img, axis=0).astype(np.float64)
        low, high = np.nanmin(max_img), np.nanmax(max_img)
        if high > low:
            weight = (max_img - low) / (high - low)
    with NUMBA_LOCK:
        return _delta_kernel(input_img, base, weight, mode == 'dF/F0', float(f0_min))

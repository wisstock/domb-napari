""" ROI intensity profiles, baselines, statistics and data frames.

"""
import numpy as np
import pandas as pd
from numba import njit, prange

from scipy import ndimage as ndi
from scipy import stats

from ._parallel import NUMBA_LOCK


def label_ids(input_label:np.ndarray):
    """ Sorted non-zero label values of a labels image.

    """
    ids = np.unique(input_label)
    return ids[ids != 0]


@njit(parallel=True, cache=True)
def _profiles_kernel(flat_img, px_idx, px_lab, lab_count):
    n_frames = flat_img.shape[0]
    n_lab = lab_count.shape[0]
    out = np.empty((n_lab, n_frames), dtype=np.float32)
    for t in prange(n_frames):
        acc = np.zeros(n_lab, dtype=np.float64)
        for j in range(px_idx.shape[0]):
            acc[px_lab[j]] += flat_img[t, px_idx[j]]
        for l in range(n_lab):
            out[l, t] = acc[l] / lab_count[l]
    return out


def labels_to_profiles(input_label:np.ndarray, input_img:np.ndarray):
    """ Mean intensity of every label region in every frame.

    Rows follow the sorted non-zero label values, see label_ids.

    Parameters
    ----------
    input_label: ndarray [x,y]
        labels image, 0 is the background
    input_img: ndarray [t,x,y] or [x,y]
        image series

    Returns
    -------
    prof_arr: ndarray [lab,t]
        mean intensity profiles, float32

    """
    input_label = np.asarray(input_label)
    input_img = np.asarray(input_img)
    if input_img.ndim == 2:
        input_img = input_img[np.newaxis, ...]
    if input_img.shape[1:] != input_label.shape:
        raise ValueError(f'Frame shape {input_img.shape[1:]} and labels shape {input_label.shape} are different!')

    flat_lab = input_label.ravel()
    px_idx = np.flatnonzero(flat_lab)
    if px_idx.size == 0:
        return np.empty((0, input_img.shape[0]), dtype=np.float32)
    _, px_lab = np.unique(flat_lab[px_idx], return_inverse=True)
    lab_count = np.bincount(px_lab).astype(np.float64)

    flat_img = np.ascontiguousarray(input_img).reshape(input_img.shape[0], -1)
    with NUMBA_LOCK:
        return _profiles_kernel(flat_img, px_idx.astype(np.int64),
                                px_lab.astype(np.int64), lab_count)


def _relative(delta:np.ndarray, base:np.ndarray):
    """ delta / base, NaN where the baseline is not positive.

    """
    base = np.broadcast_to(base, delta.shape)
    out = np.full(delta.shape, np.nan, dtype=np.float64)
    np.divide(delta, base, out=out, where=base > 0)
    return out


def simple_baseline(prof_arr:np.ndarray, win_size:int=5):
    """ Baseline as the mean of the first win_size points of every profile.

    Returns
    -------
    base_arr: ndarray [lab,1]

    """
    prof_arr = np.asarray(prof_arr, dtype=np.float64)
    return np.mean(prof_arr[:, :win_size], axis=1, keepdims=True)


def dietrich_baseline(prof_arr:np.ndarray, win_size:int=4, stds:float=1.5):
    """ Baseline of every profile by the Dietrich's method (pybaselines).

    Parameters
    ----------
    prof_arr: ndarray [lab,t]
        intensity profiles
    win_size: int
        half window of the moving average smoothing
    stds: float
        number of standard deviations for thresholding

    Returns
    -------
    base_arr: ndarray [lab,t]

    """
    from pybaselines import Baseline

    prof_arr = np.asarray(prof_arr, dtype=np.float64)
    fitter = Baseline(x_data=np.arange(prof_arr.shape[1]))
    base_arr = np.empty_like(prof_arr)
    for i, prof in enumerate(prof_arr):
        base_arr[i], _ = fitter.dietrich(prof, smooth_half_window=win_size,
                                         num_std=stds)
    return base_arr


def apply_baseline(prof_arr:np.ndarray, base_arr:np.ndarray, mode:str='dF'):
    """ Converts absolute profiles with a known baseline.

    Parameters
    ----------
    mode: str
        'dF' (F - F0), 'dF/F0' ((F - F0) / F0, NaN where F0 <= 0)
        or 'abs' (F - F0 + initial baseline level)

    """
    prof_arr = np.asarray(prof_arr, dtype=np.float64)
    delta = prof_arr - base_arr
    if mode == 'dF':
        return delta
    elif mode == 'dF/F0':
        return _relative(delta, base_arr)
    elif mode == 'abs':
        if base_arr.shape[1] == 1:
            return prof_arr.copy()
        return delta + np.mean(base_arr[:, 0:2], axis=1, keepdims=True)
    raise ValueError("Unknown mode! Use 'dF', 'dF/F0' or 'abs'.")


def delta_prof_simple(prof_arr:np.ndarray, win_size:int=5, mode:str='dF', **kwargs):
    """ Profiles relative to the simple baseline, see simple_baseline.

    """
    return apply_baseline(prof_arr, simple_baseline(prof_arr, win_size), mode)


def delta_prof_pybase(prof_arr:np.ndarray, win_size:int=4, stds:float=1.5,
                      mode:str='dF', **kwargs):
    """ Profiles relative to the Dietrich baseline, see dietrich_baseline.

    """
    return apply_baseline(prof_arr, dietrich_baseline(prof_arr, win_size, stds), mode)


def profile_stats(prof_arr:np.ndarray, method:str='se', alpha:float=0.05):
    """ Central tendency and dispersion across ROIs for every time point.

    NaN values are ignored.

    Parameters
    ----------
    prof_arr: ndarray [lab,t]
        profiles, the statistics are taken over the first axis
    method: str
        'se' - mean, standard error of the mean
        'ci' - mean, half width of the (1 - alpha) t-confidence interval
        'iqr' - median, distances to the 1st and the 3rd quartiles

    Returns
    -------
    center: ndarray [t]
    spread: ndarray [t] or [2,t]
        symmetric error or (lower, upper) errors, the matplotlib yerr format

    """
    prof_arr = np.asarray(prof_arr, dtype=np.float64)
    n = np.sum(np.isfinite(prof_arr), axis=0)
    if method in ('se', 'ci'):
        center = np.nanmean(prof_arr, axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            sem = np.where(n > 1, np.nanstd(prof_arr, axis=0, ddof=1) / np.sqrt(n), np.nan)
        if method == 'se':
            return center, sem
        return center, stats.t.ppf(1 - alpha / 2, df=np.maximum(n - 1, 1)) * sem
    elif method == 'iqr':
        q1, center, q3 = np.nanpercentile(prof_arr, [25, 50, 75], axis=0)
        return center, np.stack((center - q1, q3 - center))
    raise ValueError("Unknown method! Use 'se', 'ci' or 'iqr'.")


def stim_distances(input_label:np.ndarray, position:tuple=None):
    """ Mean distance from every label region to a point.

    Parameters
    ----------
    input_label: ndarray [x,y]
        labels image
    position: tuple
        (row, column) of the point, the image center if None

    Returns
    -------
    dist_arr: ndarray [lab]
        mean distance in pixels, rows follow label_ids

    """
    if position is None:
        position = (input_label.shape[0] // 2, input_label.shape[1] // 2)
    point_img = np.ones(input_label.shape, dtype=bool)
    point_img[int(position[0]), int(position[1])] = False
    dist_img = ndi.distance_transform_edt(point_img)
    return np.asarray(ndi.mean(dist_img, labels=input_label,
                               index=label_ids(input_label)))


def profiles_table(prof_abs:np.ndarray, roi_ids:np.ndarray, time_scale:float=1.0,
                   img_name:str='', labels_name:str='',
                   dF_win:int=4, dietrich_win:int=4, dietrich_std:float=1.25,
                   distances:np.ndarray=None):
    """ Long-format data frame of ROI profiles with two baseline estimators.

    Columns: id, lab_id, roi (label value), dist (optional), index, time,
    abs_int, dF_int, dF/F0_int, base ('simple' or 'dietrich').

    """
    prof_abs = np.asarray(prof_abs, dtype=np.float64)
    n_roi, n_frames = prof_abs.shape
    roi_col = np.repeat(np.asarray(roi_ids), n_frames)
    index_col = np.tile(np.arange(n_frames), n_roi)
    time_col = index_col * float(time_scale)

    base_simple = simple_baseline(prof_abs, dF_win)
    base_dietrich = dietrich_baseline(prof_abs, dietrich_win, dietrich_std)

    blocks = []
    for base_name, base_arr in (('simple', base_simple), ('dietrich', base_dietrich)):
        block = {'id': img_name,
                 'lab_id': labels_name,
                 'roi': roi_col}
        if distances is not None:
            block['dist'] = np.repeat(np.rint(distances).astype(int), n_frames)
        block.update({'index': index_col,
                      'time': time_col,
                      'abs_int': np.round(apply_baseline(prof_abs, base_arr, 'abs'), 4).ravel(),
                      'dF_int': np.round(apply_baseline(prof_abs, base_arr, 'dF'), 4).ravel(),
                      'dF/F0_int': np.round(apply_baseline(prof_abs, base_arr, 'dF/F0'), 4).ravel(),
                      'base': base_name})
        blocks.append(pd.DataFrame(block))
    return pd.concat(blocks, ignore_index=True)

""" Utilities for domb-napari plugin.

"""
import numba
from numba import njit

import numpy as np
import pandas as pd
from skimage import segmentation

from scipy import ndimage as ndi
from scipy import stats
from scipy import stats
from scipy import optimize

import vispy.color
from pybaselines import Baseline


def red_green_cmap():
    """ Red-green colormap for visualizing fluorescence changes.

    """
    return vispy.color.Colormap([[0.0, 1.0, 0.0],
                                 [0.0, 0.9, 0.0],
                                 [0.0, 0.85, 0.0],
                                 [0.0, 0.0, 0.0],
                                 [0.85, 0.0, 0.0],
                                 [0.9, 0.0, 0.0],
                                 [1.0, 0.0, 0.0]])


def magenta_blue_cmap():
    """ Magenta-blue colormap for visualizing fluorescence changes.

    """
    return vispy.color.Colormap([[0.0, 1.0, 1.0],
                                 [0.0, 0.9, 0.9],
                                 [0.0, 0.85, 0.85],
                                 [0.0, 0.0, 0.0],
                                 [0.85, 0.0, 0.85],
                                 [0.9, 0.0, 0.9],
                                 [1.0, 0.0, 1.0]])


def delta_cmap():
    """ Symmetric colormap for visualizing fluorescence changes.

    """
    return vispy.color.Colormap([[0.3, 0.0, 1.0],
                                 [0.2, 0.1, 1.0],
                                 [0.1, 0.2, 1.0],
                                 [0.0, 0.3, 1.0],
                                 [0.0, 0.5, 1.0],
                                 [0.0, 0.6, 0.9],
                                 [0.0, 0.4, 0.7],
                                 [0.0, 0.2, 0.4],
                                 [0.0, 0.0, 0.0],
                                 [0.4, 0.2, 0.0],
                                 [0.7, 0.4, 0.0],
                                 [0.9, 0.6, 0.0],
                                 [1.0, 0.7, 0.0],
                                 [1.0, 0.5, 0.0],
                                 [1.0, 0.3, 0.0],
                                 [1.0, 0.1, 0.0],
                                 [1.0, 0.0, 0.0]])


def delta_smooth_cmap():
    """ Smooth symmetric colormap for visualizing fluorescence changes.

    """
    return vispy.color.Colormap([[0.0, 0.2, 0.6],
                                 [0.0, 0.3, 0.7],
                                 [0.0, 0.4, 0.8],
                                 [0.0, 0.5, 0.9],
                                 [0.2, 0.6, 1.0],
                                 [0.4, 0.7, 1.0],
                                 [0.6, 0.8, 1.0],
                                 [0.0, 0.0, 0.0],
                                 [0.4, 0.2, 0.0],
                                 [0.6, 0.3, 0.0],
                                 [0.8, 0.4, 0.0],
                                 [1.0, 0.5, 0.0],
                                 [1.0, 0.6, 0.2],
                                 [1.0, 0.7, 0.4],
                                 [1.0, 0.8, 0.6]])


def pb_exp_correction(input_img:np.ndarray, mask:np.ndarray, method:str='exp'):
    """ Image series photobleaching correction by exponential fit. Correction proceeds by masked area of interest, not the whole frame to prevent autofluorescence influence.

    Parameters
    ----------
    input_img: ndarray [t,x,y]
        input image series
    mask: ndarray [x,y]
        mask of region of interest, must be same size with image frames
    method: str
        method for correction, exponential (`exp`) or bi-exponential (`bi_exp`)

    Returns
    -------
    corrected_img: ndarray [t,x,y]
        corrected image series
    bleach_coefs: ndarray [t]
        array of correction coeficients for each frame
    r_val: float
        R-squared value of exponential fit

    """
    exp = lambda x,a,b: a * np.exp(-b * x)
    bi_exp = lambda x,a,b,c,d: (a * np.exp(-b * x)) + (c * np.exp(-d * x))

    if method == 'exp':
        func = exp
    elif method == 'bi_exp':
        func = bi_exp
    else:
        raise ValueError('Incorrect method!')

    bleach_profile = np.mean(input_img, axis=(1,2), where=mask)
    x_profile = np.linspace(0, bleach_profile.shape[0], bleach_profile.shape[0])

    popt,_ = optimize.curve_fit(func, x_profile, bleach_profile)
    bleach_fit = np.vectorize(func)(x_profile, *popt)
    bleach_coefs =  bleach_fit / bleach_fit.max()
    bleach_coefs_arr = bleach_coefs.reshape(-1, 1, 1)
    corrected_image = input_img/bleach_coefs_arr

    _,_,r_val,_,_ = stats.linregress(bleach_profile, bleach_fit)

    return corrected_image, bleach_coefs, r_val


def back_substr(input_img:np.ndarray, percentile:float=1.0):
    """ Background substraction by percentile value.

    Parameters
    ----------
    input_img: ndarray [t,x,y]
        input image series
    percentile: float
        percentile value for background substraction

    Returns
    -------
    output_img: ndarray [t,x,y]
        background substracted image series

    """
    input_type = input_img.dtype
    if input_type in [float, np.float32]:
        raise TypeError(f'Float image type is not recommended: {input_type}. Prefered raw data types are: uint8, uint16, int16, int32.')
    if input_img.dtype != np.int32:  # double precision for uint16 images
        input_img = input_img.astype(np.int32)
    corrected_img = np.empty_like(input_img, dtype=np.int32)
    for i in range(input_img.shape[0]):
        f = input_img[i]
        background = np.percentile(f, percentile)
        background = np.array(background, dtype=np.int32)
        f -= background
        f = np.clip(f, 0, None)
        corrected_img[i] = f
    
    return corrected_img.astype(input_type)


# no JIT
def get_bright_channel(input_img:np.ndarray):
    """ Get index of the brightest channel from a multi-channel image series.

    Parameters
    ----------
    input_img: ndarray [t,c,x,y]
        input multi-channel image series

    Returns
    -------parallel=True, cache=True
    return bright_idx: int
        index of the brightest channel

    """
    if input_img.ndim != 4:
        raise ValueError('Input image must be 4D array with TCXY  order')
    bright_idx = np.argmax(np.sum(input_img, axis=(0,2,3)))
    return bright_idx


# no JIT
def mask_segmentation(input_mask:np.ndarray, fragment_num:int=30):
    """ Segmentation of the input binary mask by distance transform and watershed algorithm.

    Parameters
    ----------
    input_mask: ndarray [x,y]
        input binary mask
    fragment_num: int
        number of fragments to segment the mask

    Returns
    -------
    output_mask: ndarray [x,y]
        segmented mask with unique integer labels for each region

    """
    if input_mask.ndim != 2:
        raise ValueError('Input mask must be 2D array!')

    mask_coords = np.argwhere(input_mask)
    np.random.seed(42)
    rand_idx = np.random.choice(len(mask_coords), size=fragment_num, replace=False)
    rand_coords = mask_coords[rand_idx]

    markers_mask = np.zeros(input_mask.shape, dtype=bool)
    markers_mask[tuple(rand_coords.T)] = True
    markers = ndi.label(markers_mask)[0]

    distance = ndi.distance_transform_edt(input_mask)
    rois = segmentation.watershed(distance, markers, mask=input_mask, compactness=0.1)

    return rois


@numba.njit(parallel=True, cache=True)
def _delta_df_kernel(input_img, base_img, norm_img, output_img):
    """Numba function for 'dF' mode calculation."""
    for i in numba.prange(input_img.shape[0]):
        output_img[i] = (input_img[i] - base_img) * norm_img

@numba.njit(parallel=True, cache=True)   # NORM NOT WORKING!
def _delta_df_f0_kernel(input_img, base_img, norm_img, output_img):
    """Numba function for 'dF/F0' mode calculation."""
    epsilon = 1e-9
    for i in numba.prange(input_img.shape[0]):
        output_img[i] = ((input_img[i] - base_img) * norm_img) / ((base_img * norm_img) + epsilon)

def delta_img(input_img: np.ndarray, mode:str='dF', win_size:int=5):
    """
    Compute pixel-wise delta image from the input image series.
    
    Parameters
    ----------
    input_img : np.ndarray
        A 3D array where each slice corresponds to a frame.
    mode : str, optional
        Mode of calculation: 'dF/F0' or 'dF'.
    win_size : int, optional
        The number of initial frames for baseline estimation.
        
    Returns
    -------
    np.ndarray
        A 3D array with the calculated delta values.
    """
    base_img = np.mean(input_img[:win_size], axis=0).astype(np.float32)
    output_img = np.empty_like(input_img, dtype=np.float32)
    norm_img = np.max(input_img, axis=0)
    norm_range = np.max(norm_img) - np.min(base_img)
    if norm_range == 0:
        norm_range = 1.0            
    norm_img = ((norm_img - np.min(base_img)) / norm_range).clip(0, 1)

    if mode == 'dF':        
        _delta_df_kernel(input_img, base_img, norm_img, output_img)
    elif mode == 'dF/F0':
        _delta_df_f0_kernel(input_img, base_img, norm_img, output_img)
    else:
        raise ValueError("Unknown mode! Use 'dF' or 'dF/F0'.")

    return output_img


@njit(parallel=True, cache=True)
def labels_to_profiles(input_label:np.ndarray, input_img:np.ndarray):
    """ Averages the pixel values of each label region across all frames in the input image.
    
    Parameters
    ----------
    input_label : np.ndarray
        A 2D array where each pixel is labeled with a unique integer representing the region.
    input_img : np.ndarray
        A 3D array where each slice corresponds to a frame of the image, and each pixel contains intensity values.
    
    Returns
    -------
    np.ndarray
        A 2D array where each row corresponds to a label region and each column corresponds to a frame,
        containing the average pixel values for that region across all frames [lab, frame, time].

    """
    input_img = input_img.astype(np.float32)
    prof_arr = []
    for label_num in np.unique(input_label)[1:]:
        region_idxs = np.where(input_label == label_num)
        region_prof = []
        for frame in input_img:
            val = np.float64(0)
            for i, j in zip(region_idxs[0], region_idxs[1]):
                val += frame[i, j]
            region_prof.append(val / len(region_idxs[0]))
        prof_arr.append(region_prof)  
    return np.asarray(prof_arr, dtype=np.float32)


# no JIT
def delta_prof_pybase(prof_arr: np.ndarray, win_size:int=4, stds:float=1.5,
                      mode:str='ΔF', **kwargs):
    """ Computes the baseline of each profile in the input array using the Dietrich's method.
    pybaselines docs: https://pybaselines.readthedocs.io/en/latest/generated/api/pybaselines.Baseline.dietrich.html#pybaselines.Baseline.dietrich

    Parameters
    ----------
    prof_arr : np.ndarray
        A 2D array where each row corresponds to a profile (e.g., fluorescence intensity over time).
    win_size : int, optional
        The half window to use for smoothing the input data with a moving average.
    stds : float, optional
        The number of standard deviations to include when thresholding.
    mode : str, optional
        The mode of output profile calculation. Options are 'dF/F0' (relative intensity changes), or 'dF' (absolute intensity changes).
   
     Returns
    -------
    np.ndarray
        A 2D array where each row corresponds to a profile with the baseline subtracted or normalized according to the specified mode.
    
    Notes
    -----
    The function uses the `pybaselines` library to compute the baseline using a moving median method.
    
    """
    output_arr = []
    for prof in prof_arr:
        baseline_fit = Baseline(x_data = range(len(prof)))
        prof_baseline,_ = baseline_fit.dietrich(prof,
                                                smooth_half_window=win_size,
                                                num_std=stds)
        if mode == 'ΔF/F0':
            output_prof = (prof - prof_baseline) / prof_baseline
        elif mode == 'ΔF':
            output_prof = prof - prof_baseline
        elif mode == 'abs':
            output_prof = prof
        output_arr.append(output_prof)
    return np.asarray(output_arr)


# no JIT
def delta_prof_simple(prof_arr: np.ndarray, win_size:int=5, mode:str='ΔF', **kwargs):
    """ Computes the baseline of each profile in the input array using a baseline estimation by the begingng of the profiles.
    Parameters
    ----------
    prof_arr : np.ndarray
        A 2D array where each row corresponds to a profile (e.g., fluorescence intensity over time).
    win_size : int, optional
        The size of the moving window used for baseline estimation.
    mode : str, optional
        The mode of output profile calculation. Options are 'dF/F0' (relative intensity changes), or 'dF' (absolute intensity changes).
    ---------
    Returns
    -------
    np.ndarray
        A 2D array where each row corresponds to a profile with the baseline subtracted or normalized according to the specified mode.
    -------
    Notes
    -----
    The function uses a simple moving average method for baseline estimation.

    """
    output_arr = []
    for prof in prof_arr:
        F0 = np.mean(prof[:win_size])
        if mode == 'ΔF/F0':
            output_prof = (prof - F0) / F0
        elif mode == 'ΔF':
            output_prof = prof - F0
        output_arr.append(output_prof)
    return np.asarray(output_arr)


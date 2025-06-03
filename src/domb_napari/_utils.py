""" Utilities for domb-napari plugin.

"""

import pathlib
import os
import time

import numpy as np
from numpy import ma
import pandas as pd

from scipy import ndimage as ndi
from scipy import stats

from numba import jit, njit
from dipy.align.transforms import AffineTransform2D
from dipy.align.imaffine import AffineRegistration
from pybaselines import Baseline


@njit()
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
    prof_arr = []
    for label_num in np.unique(input_label)[1:]:
        region_idxs = np.where(input_label == label_num)
        region_prof = []
        for frame in input_img:
            val = 0
            for i, j in zip(region_idxs[0], region_idxs[1]):
                val += frame[i, j]
            region_prof.append(val / len(region_idxs[0]))
        prof_arr.append(region_prof)  
    return np.asarray(prof_arr)


def delta_prof_pybase(prof_arr: np.ndarray, win_size:int=10, mode:str='dF'):
    """ Computes the baseline of each profile in the input array using a moving median estimation.
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
    The function uses the `pybaselines` library to compute the baseline using a moving median method.
    
    """
    output_arr = []
    for prof in prof_arr:
        baseline_fit = Baseline(x_data = range(len(prof)))
        prof_baseline,_ = baseline_fit.noise_median(prof, half_window=win_size)
        if mode == 'dF/F0':
            output_prof = (prof - prof_baseline) / prof_baseline
        elif mode == 'dF':
            output_prof = prof - prof_baseline
        output_arr.append(output_prof)
    return np.asarray(output_arr)


def delta_prof_simple(prof_arr: np.ndarray, win_size:int=5, mode:str='dF'):
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
        if mode == 'dF/F0':
            output_prof = (prof - F0) / F0
        elif mode == 'dF':
            output_prof = prof - F0
        output_arr.append(output_prof)
    return np.asarray(output_arr)
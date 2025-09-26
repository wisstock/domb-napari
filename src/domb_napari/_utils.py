""" Utilities for domb-napari plugin.

"""

import numpy as np
from numpy import ma
from numba import jit, njit
import pandas as pd

from scipy import ndimage as ndi
from scipy import stats
from scipy import signal
from scipy import stats
from scipy import optimize

import vispy.color
from pybaselines import Baseline

from dipy.align.transforms import AffineTransform2D
from dipy.align.imaffine import AffineRegistration



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
    """ Symmetric colormap for visualizing fluorescence changes.

    """
    return vispy.color.Colormap([[0.0, 0.2, 0.6],      # темно-синій
                                [0.0, 0.3, 0.7],      # синій
                                [0.0, 0.4, 0.8],      # світло-синій
                                [0.0, 0.5, 0.9],      # яскравий синій
                                [0.2, 0.6, 1.0],      # світлий синій
                                [0.4, 0.7, 1.0],      # дуже світлий синій
                                [0.6, 0.8, 1.0],      # блідо-синій
                                [0.0, 0.0, 0.0],      # чорний (центр)
                                [0.4, 0.2, 0.0],      # темно-коричневий
                                [0.6, 0.3, 0.0],      # коричневий
                                [0.8, 0.4, 0.0],      # темно-оранжевий
                                [1.0, 0.5, 0.0],      # оранжевий
                                [1.0, 0.6, 0.2],      # світло-оранжевий
                                [1.0, 0.7, 0.4],      # персиковий
                                [1.0, 0.8, 0.6]])     # світло-персиковий


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
    if input_type in [float, np.float32, np.float64]:
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


def get_bright_channel(input_img:np.ndarray):
    """ Get index of the brightest channel from a multi-channel image series.

    Parameters
    ----------
    input_img: ndarray [t,c,x,y]
        input multi-channel image series

    Returns
    -------
    return bright_idx: int
        index of the brightest channel

    """
    if input_img.ndim != 4:
        raise ValueError('Input image must be 4D array with TCXY  order')
    bright_idx = np.argmax(np.sum(input_img, axis=(0,2,3)))
    return bright_idx


def delta_img(input_img: np.ndarray, mode:str='dF', win_size:int=5):
    """ Compute pixel-wise delta image from the input image series.
    
    Parameters
    ----------
    input_img : np.ndarray
        A 3D array where each slice corresponds to a frame of the image, and each pixel contains intensity values.
    mode : str, optional
        The mode of output image calculation. Options are 'dF/F0' (relative intensity changes), or 'dF' (absolute intensity changes normalized to initial intensity).
    win_size : int, optional
        The number of the initial frames used for baseline estimation. Default is 5.
    Returns
    -------
    np.ndarray
        A 3D array same shape as input_img, where each pixel value is the delta value calculated according to the specified mode.
    """
    if input_img.dtype not in [np.int16, np.int32, np.int64]:
        input_img = input_img.astype(np.int64)

    base_img = np.mean(input_img[:win_size], axis=0)
    norm_img = np.max(input_img, axis=0)
    norm_img = ((norm_img - np.min(base_img)) / (np.max(norm_img) - np.min(base_img))).clip(0, 1)

    output_img = np.empty_like(input_img, dtype=np.float64)
    for i in range(input_img.shape[0]):
        if mode == 'dF':
            output_img[i] = (input_img[i] - base_img)  * norm_img
        elif mode == 'dF/F0':
            output_img[i] = ((input_img[i] - base_img) / base_img)
        else:
            raise ValueError("Unknown mode! Use 'dF' or 'dF/F0'.")

    del base_img, norm_img  # Free memory
    return output_img


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


# @magic_factory(call_button='Align stack')
# def dw_registration_old(viewer: Viewer, offset_img:Image, reference_img:Image,
#                     use_reference_img:bool=False,
#                     ch_ref:int=3,
#                     ch_offset:int=0,
#                     input_crop:int=30, output_crop:int=20):
#     if input is not None:
#         if offset_img.data.ndim == 4:

#             def _save_aligned(img):
#                 xform_name = offset_img.name+'_xform'
#                 try: 
#                     viewer.layers[xform_name].data = img
#                     viewer.layers[xform_name].colormap = 'turbo'
#                 except KeyError:
#                     viewer.add_image(img, name=xform_name, colormap='turbo')

#             @thread_worker(connect={'yielded':_save_aligned})
#             def _dw_registration():
#                 offset_series = offset_img.data
#                 master_img = reference_img.data

#                 if input_crop != 0:
#                     y, x = offset_series.shape[-2:]
#                     offset_series = offset_series[:,:,input_crop:y-input_crop,input_crop:x-input_crop]
#                     master_img = master_img[:,input_crop:y-input_crop,input_crop:x-input_crop]

#                 if use_reference_img:
#                     master_img_ref, master_img_offset = master_img[1], master_img[0]
#                 else:
#                     master_img_ref = np.mean(offset_series[:,ch_ref,:,:], axis=0)
#                     master_img_offset = np.mean(offset_series[:,ch_offset,:,:], axis=0)

#                 affreg = AffineRegistration()
#                 transform = AffineTransform2D()
#                 affine = affreg.optimize(master_img_ref, master_img_offset,
#                                         transform, params0=None)

#                 ch0_xform = np.asarray([affine.transform(frame) for frame in offset_series[:,0,:,:]])
#                 ch2_xform = np.asarray([affine.transform(frame) for frame in offset_series[:,2,:,:]])
#                 xform_series = np.stack((ch0_xform,
#                                          offset_series[:,1,:,:],
#                                          ch2_xform,
#                                          offset_series[:,3,:,:]),
#                                         axis=1)
#                 if output_crop != 0:
#                     yo, xo = xform_series.shape[-2:]
#                     xform_series = xform_series[:,:,output_crop:yo-output_crop,output_crop:xo-output_crop]
#                 yield xform_series.astype(offset_series.dtype)
                    
#             _dw_registration()
#         else:
#             raise ValueError('Incorrect input image shape!')
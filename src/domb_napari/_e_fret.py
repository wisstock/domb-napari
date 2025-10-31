""" E-FRET calculations for napari plugin.

"""

import pathlib
import os
import time

from numba import njit, prange

import pandas as pd
import numpy as np
from numpy import ma
from scipy import ndimage as ndi
from scipy import stats

from skimage import filters
from skimage import morphology
from skimage import measure
from skimage import restoration
from skimage import feature
from skimage import segmentation


@njit(parallel=True, cache=True)
def _Fc_calc(dd_img, da_img, aa_img, a_val, d_val):
    """ Sensitized fluorescence calculation """
    Fc_img = da_img - aa_img * a_val - dd_img * d_val
    return np.clip(Fc_img, a_min=0, a_max=None)

@njit(parallel=True, cache=True)
def _Eapp_calc(dd_img, da_img, aa_img, a_val, d_val, G_val):
    """ Apparent FRET efficiency calculation """
    fc_img = _Fc_calc(dd_img, da_img, aa_img, a_val, d_val)
    E_app_img = np.zeros_like(fc_img)

    epsilon = 1e-12

    for i in prange(fc_img.shape[0]):
        DD_frame = dd_img[i]

        Fc_frame = fc_img[i]
        R_frame = Fc_frame / (DD_frame + epsilon)
        RG_frame = R_frame + G_val

        E_app_frame = R_frame / (RG_frame + epsilon)
                            
        E_app_img[i] = np.clip(E_app_frame, a_min=0, a_max=None)
    return E_app_img

@njit(parallel=True, cache=True)
def _Ecor_calc(dd_img, da_img, aa_img, a_val, d_val, G_val, corr_img):
    """ Corrected FRET efficiency calculation """
    E_app_img = _Eapp_calc(dd_img, da_img, aa_img, a_val, d_val, G_val)
    E_cor_img = E_app_img * corr_img
    return E_cor_img


class E_FRET():
    """ Class for estimating FRET efficiency in image time series.

    Parameters
    ----------
    dd_img: ndarray [t,x,y]
        image time series with donor excitation-donor emission
    da_img: ndarray [t,x,y]
        image time series with donor excitation-acceptor emission
    aa_img: ndarray [t,x,y]
        image time series with acceptor excitation-acceptor emission
    a_val: float
        acceptor bleedthrough coefficient
    d_val: float
        donor bleedthrough coefficient
    G_val: float
        gauge ("G") parameter of imaging system

    Attributes
    ----------
    DD_img: ndarray [t,x,y]
        image time series with donor excitation-donor emission
        (e.g. 435 nm - CFP ch.)
    DA_img: ndarray [t,x,y]
        image time series with donor excitation-acceptor emission
        (e.g. 435 nm - YFP ch.)
    AD_img: ndarray [t,x,y]
        image time series with acceptor excitation-donor emission
        (e.g. 505 nm - CFP ch.)
    AA_img: ndarray [t,x,y]
        image time series with acceptor excitation-acceptor emission
        (e.g. 505 nm - YFP ch.)
    a: float
        acceptor bleedthrough coefficient (I_DA(A) / I_AA(A))
    b: float
        acceptor bleedthrough coefficient (I_DD(A) / I_AA(A))
    c: float
        donor bleedthrough coefficient (I_AA(D) / I_DD(D))
    d: float
        donor bleedthrough coefficient (I_DA(D) / I_DD(D))
    G: float
        gauge ("G") parameter of imaging system
    Fc_img: ndarray [t,x,y]
        image time series of sensitized fluorescence
    R_img: ndarray [t,x,y]
        image time series of sensitized fluorescence to donor emission ratio
        (F_c / I_DD)
    Eapp_img: ndarray [t,x,y]
        image time series of E-FRET
    Ecorr_img: ndarray [t,x,y]
        image time series of E-FRET corrected for photobleaching

    """
    def __init__(self, dd_img, da_img, aa_img, a_val, d_val, G_val):
        self.dd_img = dd_img
        self.da_img = da_img
        self.aa_img = aa_img
        self.a_val = a_val
        self.d_val = d_val
        self.G_val = G_val

    def Fc_img(self):
        return _Fc_calc(self.dd_img, self.da_img, self.aa_img,
                        self.a_val, self.d_val)

    def Eapp_img(self):
        return _Eapp_calc(self.dd_img, self.da_img, self.aa_img,
                          self.a_val, self.d_val, self.G_val)
    
    def Ecorr_img(self, f0_frames:int=3):
        aa_f0_img = np.mean(self.aa_img[:f0_frames], axis=0, dtype=np.float32)
        corr_img = aa_f0_img / self.aa_img
        return _Ecor_calc(self.dd_img, self.da_img, self.aa_img,
                          self.a_val, self.d_val, self.G_val,
                          corr_img)
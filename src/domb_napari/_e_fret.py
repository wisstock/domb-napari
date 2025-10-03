""" E-FRET calculations for napari plugin.

"""

import pathlib
import os
import time

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


def Fc_img(dd_img, da_img, aa_img, a, d):
    dd_img = dd_img.astype(np.float32)
    da_img = da_img.astype(np.float32)
    aa_img = aa_img.astype(np.float32)
    Fc_img = da_img - aa_img*a - dd_img*d
    return Fc_img.clip(min=0.0)

class Eapp():
    def __init__(self, dd_img:np.ndarray, da_img:np.ndarray, aa_img:np.ndarray,  # ad_img:np.ndarray,
                 abcd_list:list, G_val:float,
                 **kwargs):
        """ Class for estimating FRET efficiency in image time series.
    
        Parameters
        ----------
        dd_img: ndarray [t,x,y]
            image time series with donor excitation-donor emission
        da_img: ndarray [t,x,y]
            image time series with donor excitation-acceptor emission
        ad_img: ndarray [t,x,y]
            image time series with acceptor excitation-donor emission
        aa_img: ndarray [t,x,y]
            image time series with acceptor excitation-acceptor emission
        abcd_list: list
            list of crosstalk coefficients
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
        self.DD_img = dd_img  # 435-CFP  DD
        self.DA_img = da_img  # 435-YFP  DA
        # self.AD_img = ad_img  # 505-CFP  AD
        self.AA_img = aa_img  # 505-YFP  AA

        self.a = abcd_list[0]
        self.b = abcd_list[1]
        self.c = abcd_list[2]
        self.d = abcd_list[3]

        self.G = G_val

        self.Fc_img = self.Fc_calc(dd_img=self.DD_img,
                                   da_img=self.DA_img,
                                   aa_img=self.AA_img,
                                   a=self.a, b=self.b, c=self.c, d=self.d)

        self.R_img, self.Eapp_img = self.E_app_calc(fc_img=self.Fc_img,
                                                    dd_img=self.DD_img,
                                                    G=self.G)
        
        self.Ecorr_img = self.E_cor_calc(e_app_img=self.Eapp_img,
                                         aa_img=self.AA_img,
                                         dd_img=self.DD_img,
                                         c=self.c,
                                         **kwargs)


    @staticmethod
    def Fc_calc(dd_img:np.ndarray, da_img:np.ndarray, aa_img:np.ndarray,
                a:float, b:float, c:float, d:float):
        Fc_img = []
        for frame_num in range(dd_img.shape[0]):
            DD_frame = dd_img[frame_num]
            DA_frame = da_img[frame_num]
            AA_frame = aa_img[frame_num]
            
            if b==c==0:
                Fc_frame = DA_frame - a*AA_frame - d*DD_frame
            else:
                Fc_frame = DA_frame - a*(AA_frame - c*DD_frame) - d*(DD_frame - b*AA_frame)
            Fc_img.append(Fc_frame.clip(min=0))

        return np.asarray(Fc_img)


    @staticmethod
    def E_app_calc(fc_img:np.ndarray, dd_img:np.ndarray, G:float):
        R_img = []
        E_app_img = []
        for frame_num in range(fc_img.shape[0]):
            Fc_frame = fc_img[frame_num]
            DD_frame = dd_img[frame_num]

            R_frame = np.divide(Fc_frame, DD_frame, out=np.zeros_like(Fc_frame), where=DD_frame!=0)
            R_G_frame = R_frame + G
            E_app_frame = np.divide(R_frame, R_G_frame,
                                    out=np.zeros_like(R_frame), where=R_G_frame!=0)
            E_app_frame[E_app_frame < 0] = 0

            R_img.append(R_frame)
            E_app_img.append(E_app_frame)

        return np.asarray(R_img), np.asarray(E_app_img)
    

    @staticmethod
    def E_app_calc_alt(dd_img:np.ndarray, da_img:np.ndarray, aa_img:np.ndarray,
                       a:float, d:float, G:float):
        E_app_img = []
        for frame_num in range(dd_img.shape[0]):
            DD_frame = dd_img[frame_num]
            DA_frame = da_img[frame_num]
            AA_frame = aa_img[frame_num]

            Fc_frame = DA_frame - a*AA_frame - d*DD_frame
            R_G_frame = DA_frame - a*AA_frame - (G-d)*DD_frame

            E_app_frame = np.divide(Fc_frame, R_G_frame,
                                    out=np.zeros_like(Fc_frame), where=R_G_frame!=0)
            E_app_frame[E_app_frame < 0] = 0
            E_app_img.append(E_app_frame)

        return np.asarray(E_app_img)


    @staticmethod
    def E_cor_calc(e_app_img:np.ndarray, aa_img:np.ndarray, dd_img:np.ndarray,
                   c:float, mask:np.ndarray, corr_by_mask=False):
        aa_0 = np.mean(aa_img[:2], axis=0)
        dd_0 = np.mean(dd_img[:2], axis=0)

        if corr_by_mask:
            aa_masked = ma.masked_where(~mask, aa_0)
            dd_c_masked = ma.masked_where(~mask, (dd_0*c))
            I_0_val = np.mean(aa_masked - (dd_c_masked * c))
            I_t_series = np.asarray([np.mean(ma.masked_where(~mask, aa_img[i] - (dd_img[i]*c)).compressed()) \
                                     for i in range(aa_img.shape[0])])
        else:
            I_0_val = np.mean(aa_0 - (dd_0 * c))
            I_t_series = np.asarray([np.mean(aa_img[i] - (dd_img[i]*c)) \
                                     for i in range(aa_img.shape[0])])

        E_corr = [e_app_img[i] * (I_0_val / I_t_series[i]) \
                  for i in range(e_app_img.shape[0])]

        return np.asarray(E_corr)
    

# @jit
# def Fc_calc_jit(dd_img:np.ndarray, da_img:np.ndarray, aa_img:np.ndarray,  # a=0.122, d=0.794
#             a:float=0.122, b:float=0, c:float=0, d:float=0.794):
#     Fc_img = np.empty_like(dd_img, dtype=np.float32)
#     for frame_num in range(dd_img.shape[0]):
#         DD_frame = dd_img[frame_num]
#         DA_frame = da_img[frame_num]
#         AA_frame = aa_img[frame_num]
        
#         if b==c==0:
#             Fc_frame = DA_frame - a*AA_frame - d*DD_frame
#         else:
#             Fc_frame = DA_frame - a*(AA_frame - c*DD_frame) - d*(DD_frame - b*AA_frame)
#         Fc_img[frame_num] = Fc_frame

#     return Fc_img

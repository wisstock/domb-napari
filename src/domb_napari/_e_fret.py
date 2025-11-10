""" E-FRET calculations for napari plugin.

It's standalone module for domb-napari plugin and it could be used independently for E-FRET calibration and estimation.

References
----------
Zal and Gascoigne, 2004. "Photobleaching-corrected FRET efficiency imaging of live cells". doi: 10.1529/biophysj.103.022087

"""
import numpy as np
from numpy import ma
from numba import njit, prange

from scipy import ndimage as ndi
from scipy import stats

import pandas as pd

from skimage import morphology
from skimage import measure
from skimage import feature


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
    dd_img: ndarray [t,x,y]
        image time series with donor excitation-donor emission
    da_img: ndarray [t,x,y]
        image time series with donor excitation-acceptor emission
    aa_img: ndarray [t,x,y]
        image time series with acceptor excitation-acceptor emission
    a: float
        acceptor bleedthrough coefficient (I_DA(A) / I_AA(A))
    d: float
        donor bleedthrough coefficient (I_DA(D) / I_DD(D))
    G: float
        gauge ("G") parameter of imaging system

    Methods
    -------
    Fc_img():
        Calculate sensitized fluorescence image time series.
        returns ndarray [t,x,y]
    Eapp_img():
        Calculate apparent FRET efficiency image time series.
        returns ndarray [t,x,y]
    Ecorr_img(f0_frames:int=3):
        Calculate corrected FRET efficiency image time series.
        returns ndarray [t,x,y]

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
    

class cross_talk_estimation():
    """ Class for estimating cross-talk coefficients from image time series.

    Parameters/attributes
    ----------
    dd_img: ndarray [t,x,y]
        image time series with donor excitation-donor emission
    da_img: ndarray [t,x,y]
        image time series with donor excitation-acceptor emission
    aa_img: ndarray [t,x,y]
        image time series with acceptor excitation-acceptor emission
    mask: ndarray [x,y]
        binary mask for pixels to be used in estimation
        
    Methods
    -------
    estimate_a():
        Estimate acceptor bleedthrough coefficient 'a'
        returns tupple (coef_df, fit_arr) with dataframe of coefficients estimated for all frames and array of fit data
    estimate_d():
        Estimate donor bleedthrough coefficient 'd'
        returns tupple (coef_df, fit_arr) with dataframe of coefficients estimated for all frames and array of fit data
        
    """
    def __init__(self, mask,
                 dd_img=None, da_img=None, aa_img=None):
        if any(img is None for img in [dd_img, da_img, aa_img]):
            raise ValueError("All spectral channels must be provided!")

        self.dd_img = dd_img
        self.da_img = da_img
        self.aa_img = aa_img
        self.mask = mask

    @staticmethod
    def _coef_calc(img_ref, img_prm, img_off, img_mask,
                   c_prm_name, c_off_name):
        """ Generalized function for cross-talk coefficient calculation with linear fit
        
        Returns
        -------
        coef_df: pd.DataFrame
            DataFrame with estimated coefficients for all frames
        fit_arr: np.ndarray
            Array with fit data for all frames [(arr_ref, arr_prm), ...]

        """
        col_list = ['frame_n',
                    c_prm_name+'_val', c_prm_name+'_p', c_prm_name+'_err', c_prm_name+'_i', c_prm_name+'_i_err', c_prm_name+'_r^2',
                    c_off_name+'_val', c_off_name+'_p', c_off_name+'_err', c_off_name+'_i', c_off_name+'_i_err', c_off_name+'_r^2']
        c_df = pd.DataFrame(columns=col_list)

        fit_arr = []
        for i in range(len(img_ref)):
            # pixels intensities for primary channel and cross-talk channels
            arr_ref = ma.masked_array(img_ref[i], mask=~img_mask).compressed()
            arr_prm = ma.masked_array(img_prm[i], mask=~img_mask).compressed()

            # pixels intensities for b/c cross-talk channel, sholuld be uncorrelated
            arr_off = ma.masked_array(img_off[i], mask=~img_mask).compressed()

            fit_arr.append((arr_ref, arr_prm))

            c_prm_fit = stats.linregress(arr_ref, arr_prm, alternative='greater')
            c_off_fit = stats.linregress(arr_ref, arr_off, alternative='greater')

            row_dict =  {'frame_n': i,
                         c_prm_name+'_val': c_prm_fit.slope,
                         c_prm_name+'_p': "{:.5f}".format(c_prm_fit.pvalue),
                         c_prm_name+'_err': c_prm_fit.stderr,
                         c_prm_name+'_i': c_prm_fit.intercept,
                         c_prm_name+'_i_err': c_prm_fit.intercept_stderr,
                         c_prm_name+'_r^2': c_prm_fit.rvalue,
                         c_off_name+'_val': c_off_fit.slope,
                         c_off_name+'_p': "{:.5f}".format(c_off_fit.pvalue),
                         c_off_name+'_err': c_off_fit.stderr,
                         c_off_name+'_i': c_off_fit.intercept,
                         c_off_name+'_i_err': c_off_fit.intercept_stderr,
                         c_off_name+'_r^2': c_off_fit.rvalue}      
            row_df = pd.DataFrame(row_dict, index=[0])
            c_df = pd.concat([c_df.astype(row_df.dtypes),
                                row_df.astype(c_df.dtypes)],
                                ignore_index=True)
        return (c_df, np.array(fit_arr))

    def estimate_a(self):
        """ Estimate acceptor cross-talk coefficient 'a'
        
        """
        coefs = self._coef_calc(img_ref = self.aa_img,
                                img_prm = self.da_img,
                                img_off = self.dd_img,
                                img_mask = self.mask,
                                c_prm_name = 'a',
                                c_off_name = 'b')
        return coefs
    
    def estimate_d(self):
        """ Estimate donor cross-talk coefficient 'd'

        """
        coefs = self._coef_calc(img_ref = self.dd_img,
                                img_prm = self.da_img,
                                img_off = self.aa_img,
                                img_mask = self.mask,
                                c_prm_name = 'd',
                                c_off_name = 'c')
        return coefs


class G_factor_estimation():
    """ Class for estimating G factor from image time series.

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

    Attributes
    ----------
    DD_img: ndarray [t,x,y]
        image time series with donor excitation-donor emission
        (e.g. 435 nm - CFP ch.)
    DA_img: ndarray [t,x,y]
        image time series with donor excitation-acceptor emission
        (e.g. 435 nm - YFP ch.)

    """

    def __init__(self, dd_img, da_img, aa_img, a_val, d_val):
        self.dd_img = dd_img
        self.da_img = da_img
        self.aa_img = aa_img
        self.a_val = a_val
        self.d_val = d_val
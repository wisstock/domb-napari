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

import ._utils as utils


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

    Parameters/attributes
    ----------
    dd_img: ndarray [t,x,y]
        image time series with donor excitation-donor emission
    da_img: ndarray [t,x,y]
        image time series with donor excitation-acceptor emission
    aa_img: ndarray [t,x,y]
        image time series with acceptor excitation-acceptor emission
    a_val: float
        acceptor cross-talk coefficient (I_DA(A) / I_AA(A))
    d_val: float
        donor cross-talk coefficient (I_DA(D) / I_DD(D))
    G_val: float
        gauge ("G") parameter of imaging system

    Methods
    -------
    Fc_img():
        Calculate sensitized fluorescence image time series.
        returns ndarray [t,x,y]
    Eapp_img():
        Calculate apparent FRET efficiency image time series.
        returns ndarray [t,x,y]
    Ecorr_img(f0_frames:int=2):
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
    
    def Ecorr_img(self, f0_frames:int=2):
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

    Parameters/attributes
    ----------
    mask: ndarray [x,y]
        binary mask for pixels to be used in estimation
    pre_dd_img: ndarray [x,y]
        image time series with donor excitation-donor emission
        with larger expected FRET
    pre_da_img: ndarray [x,y]
        image time series with donor excitation-acceptor emission
        with larger expected FRET
    pre_aa_img: ndarray [x,y]
        image time series with acceptor excitation-acceptor emission
        with larger expected FRET
    post_dd_img: ndarray [t,x,y]
        image time series with donor excitation-donor emission
        with smaller expected FRET
    post_da_img: ndarray [t,x,y]
        image time series with donor excitation-acceptor emission
        with smaller expected FRET
    post_aa_img: ndarray [t,x,y]
        image time series with acceptor excitation-acceptor emission
        with smaller expected FRET
    a_val: float
        acceptor cross-talk coefficient
    d_val: float
        donor cross-talk coefficient

    Methods
    -------
    estimate_g():
        Estimate G factor of imaging system

    """
    def __init__(self, mask,
                 pre_dd_img=None, pre_da_img=None, pre_aa_img=None,
                 post_dd_img=None, post_da_img=None, post_aa_img=None,
                 a_val=None, d_val=None):
        if any(img is None for img in [pre_dd_img, pre_da_img, pre_aa_img,
                                       post_dd_img, post_da_img, post_aa_img]):
            raise ValueError("All spectral channels must be provided!")
        if not np.all([pre_dd_img.ndim == 2, pre_da_img.ndim == 2, pre_aa_img.ndim == 2]):
            raise ValueError('Incorrect input pre-image shape, should be 2D frame!')
        if not np.all([post_dd_img.ndim == 3, post_da_img.ndim == 3, post_aa_img.ndim == 3]):
            raise ValueError('Incorrect input post-image shape, should be 3D time series!')

        self.pre_dd_img = pre_dd_img
        self.pre_da_img = pre_da_img
        self.pre_aa_img = pre_aa_img
        self.post_dd_img = post_dd_img
        self.post_da_img = post_da_img
        self.post_aa_img = post_aa_img
        self.a_val = a_val
        self.d_val = d_val
        self.mask = mask

    def estimate_g(self):
        """ Estimate G factor of imaging system

        Returns
        -------
        g_val: float
            estimated G factor

        """
        g_df = pd.DataFrame(columns=['frame_n', 'g_val', 'g_p', 'g_err', 'g_i', 'g_i_err', 'g_r^2'])

        Fc_img_pre = _Fc_calc(dd_img=self.pre_dd_img,
                              da_img=self.pre_da_img,
                              aa_img=self.pre_aa_img,
                              a_val=self.a_val, d_val=self.d_val)
        Fc_img_post = _Fc_calc(dd_img=self.post_dd_img,
                               da_img=self.post_da_img,
                               aa_img=self.post_aa_img,
                               a_val=self.a_val, d_val=self.d_val)
        
        Fc_arr_pre = utils.labels_to_profiles(self.mask, Fc_img_pre)
        Fc_arr_post = utils.labels_to_profiles(self.mask, Fc_img_post)
        Fc_arr_delta = Fc_arr_pre - Fc_arr_post
        Fc_arr_delta = Fc_arr_delta.T

        DD_arr_pre = utils.labels_to_profiles(self.mask, self.pre_dd_img[pre_f_start:pre_frame_end])
        DD_arr_post = utils.labels_to_profiles(self.mask, self.post_dd_img)
        DD_arr_delta = DD_arr_post - DD_arr_pre
        DD_arr_delta = DD_arr_delta.T

        i = 0
        for fc, dd in zip(Fc_arr_delta, DD_arr_delta):
            g_fit = stats.linregress(dd, fc)
            row_dict =  {'frame_n': i,
                            'g_val': g_fit.slope,
                            'g_p': "{:.5f}".format(g_fit.pvalue),
                            'g_err': g_fit.stderr,
                            'g_i': g_fit.intercept,
                            'g_i_err': g_fit.intercept_stderr,
                            'g_r^2': g_fit.rvalue}      
            row_df = pd.DataFrame(row_dict, index=[0])
            g_df = pd.concat([g_df.astype(row_df.dtypes),
                                row_df.astype(g_df.dtypes)],
                                ignore_index=True)
            i += 1
        show_info(f'{output_name} G-factor estimated in {end - start:.2f} s')
        return (g_df, roi_mask)
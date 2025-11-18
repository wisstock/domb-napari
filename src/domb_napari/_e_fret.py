""" 3³/E-FRET

Stand-alone module for domb-napari plugin and it could be used independently for 3³/E-FRET calibration and estimation.

References
----------
- Erickson et al., 2001. "Preassociation of calmodulin with voltage-gated Ca(2+) channels revealed by FRET in single living cells ". doi: 10.1016/s0896-6273(01)00438-x.
- Zal and Gascoigne, 2004. "Photobleaching-corrected FRET efficiency imaging of live cells". doi: 10.1529/biophysj.103.022087
- Chen et al., 2006. "Measurement of FRET efficiency and ratio of donor to acceptor concentration in living cells". doi: 10.1529/biophysj.106.088773
- Butz et al., 2016, "Quantifying macromolecular interactions in living cells using FRET two-hybrid assays". doi:10.1038/nprot.2016.128

"""
import numpy as np
from numpy import ma
from numba import njit, prange

from scipy import stats
import pandas as pd

import domb_napari._utils as utils


@njit(parallel=True, cache=True)
def _Fc_calc(dd_img, da_img, aa_img, a_val, d_val):
    """ Sensitized fluorescence calculation
    
    """
    Fc_img = da_img - aa_img * a_val - dd_img * d_val
    return np.clip(Fc_img, a_min=0, a_max=None)

@njit(parallel=True, cache=True)
def _Eapp_calc(dd_img, da_img, aa_img, a_val, d_val, G_val):
    """ Apparent FRET efficiency calculation
    
    """
    fc_img = _Fc_calc(dd_img, da_img, aa_img, a_val, d_val)
    E_app_img = np.zeros_like(fc_img)

    epsilon = 1e-13

    for i in prange(fc_img.shape[0]):
        DD_frame = dd_img[i]
        Fc_frame = fc_img[i]

        Fc_G_frame = Fc_frame / G_val
        E_app_frame = Fc_G_frame / (DD_frame + Fc_G_frame + epsilon)
                            
        E_app_img[i] = np.clip(E_app_frame, a_min=0, a_max=None)
    return E_app_img

@njit(parallel=True, cache=True)
def _Ecor_calc(dd_img, da_img, aa_img, a_val, d_val, G_val, corr_img):
    """ Corrected FRET efficiency calculation
    
    """
    E_app_img = _Eapp_calc(dd_img, da_img, aa_img, a_val, d_val, G_val)
    E_cor_img = E_app_img * corr_img
    return E_cor_img

class E_FRET():
    """ Class for estimating FRET efficiency in image time series

    Parameters/attributes
    ----------
    dd_img: ndarray [t,x,y]
        Time series with donor excitation-donor emission
    da_img: ndarray [t,x,y]
        Time series with donor excitation-acceptor emission
    aa_img: ndarray [t,x,y]
        Time series with acceptor excitation-acceptor emission
    a_val: float
        Acceptor cross-talk coefficient (I_DA(A) / I_AA(A))
    d_val: float
        Donor cross-talk coefficient (I_DA(D) / I_DD(D))
    G_val: float
        Gauge ("G") factor of imaging system

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
    def __init__(self, dd_img, da_img, aa_img,
                 a_val, d_val, G_val):
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
    

class CrossTalkEstimation():
    """ Class for estimating cross-talk coefficients from image time series.

    Parameters/attributes
    ----------
    dd_img: ndarray [t,x,y]
        Time series with donor excitation-donor emission
    da_img: ndarray [t,x,y]
        Time series with donor excitation-acceptor emission
    aa_img: ndarray [t,x,y]
        Time series with acceptor excitation-acceptor emission
    mask: ndarray [x,y]
        Binary mask for pixels to be used in estimation
        
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
        """ Generalized function for cross-talk coefficient calculation
        with linear fit
        
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


class GFactorEstimation():
    """ Class for estimating G factor from image time series

    Parameters/attributes
    ----------
    mask: ndarray [x,y]
        binary mask for pixels to be used in estimation
    l_mask: ndarray [x,y]
        binary mask for pixels to be used in estimation for lower FRET images, used in Chen method
    h_dd_img: ndarray [x,y]
        One frame obtained with donor excitation-donor emission
        with higher expected FRET (before acceptor photobleaching or after acceptor recovery)
    h_da_img: ndarray [x,y]
        One frame obtained with donor excitation-acceptor emission
        with higher expected FRET (before acceptor photobleaching or after acceptor recovery)
    h_aa_img: ndarray [x,y]
        One frame obtained with acceptor excitation-acceptor emission
        with higher expected FRET (before acceptor photobleaching or after acceptor recovery)
    l_dd_img: ndarray [t,x,y]
        Time series with donor excitation-donor emission
        with lower expected FRET (after acceptor photobleaching or before acceptor recovery)
    l_da_img: ndarray [t,x,y]
        Time series with donor excitation-acceptor emission
        with lower expected FRET (after acceptor photobleaching or before acceptor recovery)
    l_aa_img: ndarray [t,x,y]
        Time series with acceptor excitation-acceptor emission
        with lower expected FRET (after acceptor photobleaching or before acceptor recovery)
    a_val: float
        acceptor cross-talk coefficient
    d_val: float
        donor cross-talk coefficient

    Methods
    -------
    estimate_g():
        Estimate G factor of imaging system
        returns tupple (g_df, fit_arr) with dataframe of G factors estimated for all post frames
        and array of fit data

    """
    def __init__(self, mask, l_mask=None,
                 h_dd_img=None, h_da_img=None, h_aa_img=None,
                 l_dd_img=None, l_da_img=None, l_aa_img=None,
                 a_val=None, d_val=None):
        if any(img is None for img in [h_dd_img, h_da_img, h_aa_img,
                                       l_dd_img, l_da_img, l_aa_img]):
            raise ValueError("All input images must be provided!")
        if not np.all([h_dd_img.ndim == 2, h_da_img.ndim == 2, h_aa_img.ndim == 2,
                       l_dd_img.ndim == 2, l_da_img.ndim == 2, l_aa_img.ndim == 2]):
            raise ValueError('Incorrect input image shape, must be 2D image!')
            
        # add new axis for time dimension consistency
        self.h_dd_img = h_dd_img[np.newaxis, ...]
        self.h_da_img = h_da_img[np.newaxis, ...]
        self.h_aa_img = h_aa_img[np.newaxis, ...]
        self.l_dd_img = l_dd_img[np.newaxis, ...]
        self.l_da_img = l_da_img[np.newaxis, ...]
        self.l_aa_img = l_aa_img[np.newaxis, ...]

        self.a_val = a_val
        self.d_val = d_val
        self.mask = mask
        self.l_mask = l_mask if l_mask is not None else mask

    def estimate_g_zal(self):
        """ Estimate G factor of imaging system using Zal and Gascoigne method
        based on acceptor photobleaching

        Returns
        -------
        g_df: pd.DataFrame
            DataFrame with estimated G factors for all post frames
        fit_arr: np.ndarray
            Array with fit data (DD_arr_delta, Fc_arr_delta)

        """
        # sensitized fluorescence images
        Fc_img_h = _Fc_calc(dd_img=self.h_dd_img,
                              da_img=self.h_da_img,
                              aa_img=self.h_aa_img,
                              a_val=self.a_val, d_val=self.d_val)
        Fc_img_l = _Fc_calc(dd_img=self.l_dd_img,
                               da_img=self.l_da_img,
                               aa_img=self.l_aa_img,
                               a_val=self.a_val, d_val=self.d_val)
        
        # profiles extraction
        Fc_arr_h = utils.labels_to_profiles(self.mask, Fc_img_h)
        Fc_arr_l = utils.labels_to_profiles(self.mask, Fc_img_l)
        Fc_arr_delta = Fc_arr_h - Fc_arr_l

        DD_arr_h = utils.labels_to_profiles(self.mask, self.h_dd_img)
        DD_arr_l = utils.labels_to_profiles(self.mask, self.l_dd_img)
        DD_arr_delta = DD_arr_l - DD_arr_h

        g_fit = stats.linregress(DD_arr_delta[:,0],
                                 Fc_arr_delta[:,0])
        
        g_df = pd.DataFrame(columns=['g_val', 'g_p', 'g_err',
                                     'g_i', 'g_i_err', 'g_r^2'])
        row_dict =  {'g_val': g_fit.slope,
                     'g_p': "{:.5f}".format(g_fit.pvalue),
                     'g_err': g_fit.stderr,
                     'g_i': g_fit.intercept,
                     'g_i_err': g_fit.intercept_stderr,
                     'g_r^2': g_fit.rvalue}      
        row_df = pd.DataFrame(row_dict, index=[0])
        g_df = pd.concat([g_df.astype(row_df.dtypes),
                            row_df.astype(g_df.dtypes)],
                            ignore_index=True)
        
        return (g_df, np.array([DD_arr_delta, Fc_arr_delta]))
    
    def estimate_g_chen(self, estimate_error:bool=True, n_boot:int=1000):
        """ Estimate G factor of imaging system using Chen et al. method
        based on different FRET levels

        Parameters
        ----------
        estimate_error: bool
            Whether to estimate error using bootstrap resampling
        n_boot: int
            Number of bootstrap resampling iterations

        Returns
        -------
        g_df: pd.DataFrame
            DataFrame with estimated G factors for all post frames
        fit_arr: np.ndarray
            Array with fit data for all post frames [(dd_prm, da_prm), ...]

        """
        # sensitized fluorescence images
        Fc_img_h = _Fc_calc(dd_img=self.h_dd_img,
                            da_img=self.h_da_img,
                            aa_img=self.h_aa_img,
                            a_val=self.a_val, d_val=self.d_val)
        Fc_img_l = _Fc_calc(dd_img=self.l_dd_img,
                            da_img=self.l_da_img,
                            aa_img=self.l_aa_img,
                            a_val=self.a_val, d_val=self.d_val)
        
        # profiles for high and low FRET images
        Fc_arr_h = utils.labels_to_profiles(self.mask, Fc_img_h)[:,0]
        AA_arr_h = utils.labels_to_profiles(self.mask, self.h_aa_img)[:,0]
        DD_arr_h = utils.labels_to_profiles(self.mask, self.h_dd_img)[:,0]
        Fc_AA_arr_h = Fc_arr_h / AA_arr_h
        DD_AA_arr_h = DD_arr_h / AA_arr_h

        Fc_arr_l = utils.labels_to_profiles(self.l_mask, Fc_img_l)[:,0]
        AA_arr_l = utils.labels_to_profiles(self.l_mask, self.l_aa_img)[:,0]
        DD_arr_l = utils.labels_to_profiles(self.l_mask, self.l_dd_img)[:,0]
        Fc_AA_arr_l = Fc_arr_l / AA_arr_l
        DD_AA_arr_l = DD_arr_l / AA_arr_l

        # intersect-based G factor estimation
        Fc_AA_delta = np.mean(Fc_AA_arr_h) - np.mean(Fc_AA_arr_l)
        DD_AA_delta = np.mean(DD_AA_arr_l) - np.mean(DD_AA_arr_h)
        g_val = Fc_AA_delta / DD_AA_delta
    
        # bootstrap resampling for error estimation
        g_boots = np.empty(n_boot)
        if estimate_error:
            for i in range(n_boot):  
                idx_h = np.random.randint(0, Fc_AA_arr_h.shape[0], Fc_AA_arr_h.shape[0])
                idx_l = np.random.randint(0, Fc_AA_arr_l.shape[0], Fc_AA_arr_l.shape[0])
                
                Fc_AA_delta_bs = np.mean(Fc_AA_arr_h[idx_h]) - np.mean(Fc_AA_arr_l[idx_l])
                DD_AA_delta_bs = np.mean(DD_AA_arr_l[idx_l]) - np.mean(DD_AA_arr_h[idx_h])
                g_val_bs = Fc_AA_delta_bs / DD_AA_delta_bs
                g_boots[i] = g_val_bs
            g_err = np.std(g_boots) / np.sqrt(n_boot)

            g_df = pd.DataFrame(columns=['g_val', 'g_err'])
            row_dict =  {'g_val': g_val,
                         'g_err': g_err}      
            row_df = pd.DataFrame(row_dict, index=[0])
            g_df = pd.concat([g_df.astype(row_df.dtypes),
                                row_df.astype(g_df.dtypes)],
                                ignore_index=True)
        else:
            g_df = pd.DataFrame({'g_val': [g_val]})
        return (g_df, (np.array([DD_AA_arr_h, Fc_AA_arr_h]), np.array([DD_AA_arr_l, Fc_AA_arr_l])))
    
    # def estimate_g_butz(self):
    #     """ Estimate G factor of imaging system using Butz et al. method
    #     based on different FRET levels

    #     Returns
    #     -------
    #     g_df: pd.DataFrame
    #         DataFrame with estimated G factors for all post frames
    #     fit_arr: np.ndarray
    #         Array with fit data for all post frames [(aa_prm, da_prm), ...]

    #     """
    #     # sensitized fluorescence images
    #     Fc_img_h = _Fc_calc(dd_img=self.h_dd_img,
    #                         da_img=self.h_da_img,
    #                         aa_img=self.h_aa_img,
    #                         a_val=self.a_val, d_val=self.d_val)
    #     Fc_img_l = _Fc_calc(dd_img=self.l_dd_img,
    #                         da_img=self.l_da_img,
    #                         aa_img=self.l_aa_img,
    #                         a_val=self.a_val, d_val=self.d_val)
        
    #     # profiles extraction
    #     Fc_arr_h = utils.labels_to_profiles(self.mask, Fc_img_h)
    #     Fc_arr_l = utils.labels_to_profiles(self.mask, Fc_img_l)
    #     Fc_arr = np.concatenate((Fc_arr_h, Fc_arr_l), axis=0)
        
    #     AA_arr_h = utils.labels_to_profiles(self.mask, self.h_aa_img)
    #     AA_arr_l = utils.labels_to_profiles(self.mask, self.l_aa_img)
    #     AA_arr = np.concatenate((AA_arr_h, AA_arr_l), axis=0)

    #     DD_arr_h = utils.labels_to_profiles(self.mask, self.h_dd_img)
    #     DD_arr_l = utils.labels_to_profiles(self.mask, self.l_dd_img)
    #     DD_arr = np.concatenate((DD_arr_h, DD_arr_l), axis=0)

    #     print(Fc_arr.shape, AA_arr.shape, DD_arr.shape)

    #     Fc_AA_arr = (Fc_arr / AA_arr) * (1 / self.a_val)
    #     DD_AA_arr = (DD_arr / AA_arr) * (self.d_val / self.a_val)

    #     # linear fit for G factor estimation
    #     g_fit = stats.linregress(DD_AA_arr[:,0],
    #                              Fc_AA_arr[:,0])
        
    #     g_df = pd.DataFrame(columns=['g_val', 'g_p', 'g_err',
    #                                  'r_val', 'r_err', 'g_r^2'])
    #     row_dict =  {'g_val': g_fit.slope,
    #                  'g_p': "{:.5f}".format(g_fit.pvalue),
    #                  'g_err': g_fit.stderr,
    #                  'r_val': g_fit.intercept,
    #                  'r_err': g_fit.intercept_stderr,
    #                  'g_r^2': g_fit.rvalue}      
    #     row_df = pd.DataFrame(row_dict, index=[0])
    #     g_df = pd.concat([g_df.astype(row_df.dtypes),
    #                         row_df.astype(g_df.dtypes)],
    #                         ignore_index=True)
        
    #     return (g_df, np.array([DD_AA_arr, Fc_AA_arr]))


class KFactorEstimation():
    """ Class for estimating k factor for donor/acceptor ratio estimation

    Based on Chen et al., 2006

    Parameters/attributes
    ----------
    dd_img: ndarray [t,x,y]
        Time series with donor excitation-donor emission
    da_img: ndarray [t,x,y]
        Time series with donor excitation-acceptor emission
    aa_img: ndarray [t,x,y]
        Time series with acceptor excitation-acceptor emission
    mask: ndarray [x,y]
        Binary mask for pixels to be used in estimation
        
    Methods
    -------
    estimate_k():
        Estimate k factor of imaging system
        returns tupple (k_df, fit_arr) with dataframe of k factors estimated for all frames and array of fit data

    """
    def __init__(self, mask,
                 dd_img=None, da_img=None, aa_img=None,
                 a_val=None, d_val=None, G_val=None):
        if any(img is None for img in [dd_img, da_img, aa_img]):
            raise ValueError("All spectral channels must be provided!")
        if not np.all([dd_img.ndim == 2, da_img.ndim == 2, aa_img.ndim == 2]):
            raise ValueError('Incorrect input image shape, must be 2D image!')

        self.dd_img = dd_img[np.newaxis, ...]
        self.da_img = da_img[np.newaxis, ...]
        self.aa_img = aa_img[np.newaxis, ...]
        self.mask = mask
        self.a_val = a_val
        self.d_val = d_val
        self.G_val = G_val

    def estimate_k(self):
        """ Estimate k factor of imaging system
        
        """
        # sensitized fluorescence images
        Fc_img = _Fc_calc(dd_img=self.dd_img,
                          da_img=self.da_img,
                          aa_img=self.aa_img,
                          a_val=self.a_val, d_val=self.d_val)
        
        # profiles extraction
        Fc_arr = utils.labels_to_profiles(self.mask, Fc_img)
        DD_arr = utils.labels_to_profiles(self.mask, self.dd_img)
        D_tot_arr = DD_arr + (Fc_arr / self.G_val)

        AA_arr = utils.labels_to_profiles(self.mask, self.aa_img)

        # linear fit for k factor estimation
        k_fit = stats.linregress(AA_arr[:,0],
                                 D_tot_arr[:,0])

        k_df = pd.DataFrame(columns=['k_val', 'k_p', 'k_err',
                                     'k_i', 'k_i_err', 'k_r^2'])
        row_dict =  {'k_val': k_fit.slope,
                     'k_p': "{:.5f}".format(k_fit.pvalue),
                     'k_err': k_fit.stderr,
                     'k_i': k_fit.intercept,
                     'k_i_err': k_fit.intercept_stderr,
                     'k_r^2': k_fit.rvalue}      
        row_df = pd.DataFrame(row_dict, index=[0])
        k_df = pd.concat([k_df.astype(row_df.dtypes),
                            row_df.astype(k_df.dtypes)],
                            ignore_index=True)
        return (k_df, np.array([AA_arr, D_tot_arr]))
from magicgui import magic_factory

from napari import Viewer
from napari.layers import Image, Labels, Points
from napari.utils.notifications import show_info, show_warning
from napari.qt.threading import thread_worker

import pathlib
import os
import time

import pandas as pd
import numpy as np
from scipy import ndimage as ndi
from scipy import stats

from skimage import filters
from skimage import morphology
from skimage import measure
from skimage import feature
from skimage import segmentation

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas

from dipy.align.transforms import AffineTransform2D
from dipy.align.imaffine import AffineRegistration

import domb_napari._utils as utils
import domb_napari._e_fret as e_fret


@magic_factory(call_button='Preprocess stack',
               stack_order={"choices": ['TCXY', 'CTXY']},
               correction_method={"choices": ['exp', 'bi_exp']},)
def split_channels(viewer: Viewer, img:Image,
                   stack_order:str='TCXY',
                   median_filter:bool=False, median_kernel:int=2,
                   background_substraction:bool=True,
                   photobleaching_correction:bool=False,
                   use_correction_mask:bool=False,
                   correction_mask:Labels=None,
                   correction_method:str='exp',
                   drop_frames:bool=False,
                   frames_range:list=[0,10],
                   frames_crop:int=0):
    if input is not None:
        def _save_ch(params):
            img = params[0]
            img_name = params[1]
            try: 
                viewer.layers[img_name].data = img
            except KeyError:
                viewer.add_image(img, name=img_name, colormap='turbo')

        @thread_worker(connect={'yielded':_save_ch})
        def _split_channels():
            def _preprocessing(ch_img, ch_suffix):
                start = time.perf_counter()
                if drop_frames:
                    if len(frames_range) == 2:
                        ch_img = ch_img[frames_range[0]:frames_range[-1],:,:]
                        ch_suffix = f'_{frames_range[0]}-{frames_range[-1]}'+ch_suffix
                    else:
                        raise ValueError('List of indexes should has 2 elements!')
                if median_filter:
                    median_axis = lambda x,k: np.array([ndi.median_filter(f, size=k) for f in x], dtype=x.dtype)
                    ch_img = median_axis(ch_img, median_kernel)
                if background_substraction:
                    ch_img = utils.back_substr(ch_img, percentile=1.0)
                if photobleaching_correction:
                    if correction_mask is not None and use_correction_mask:
                        pb_mask = correction_mask.data
                        show_info(f'{ch_suffix} photobleaching correction with mask {correction_mask.name}')
                    else:
                        pb_mask = ch_img[:,:,0] > filters.threshold_otsu(ch_img[:,:,0])
                        show_info(f'{ch_suffix} photobleaching correction with Otsu mask')
                    ch_img,_,r_corr = utils.pb_exp_correction(input_img=ch_img,
                                                              mask=pb_mask,
                                                              method=correction_method)
                    show_info(f'{correction_method} photobleaching correction, r^2={r_corr}')
                if frames_crop != 0:
                    yo, xo = ch_img.shape[1:]
                    ch_img = ch_img[:,frames_crop:yo-frames_crop,frames_crop:xo-frames_crop]
                end = time.perf_counter()
                show_info(f'{ch_suffix} preprocessing time: {end - start:.2f} s, shape {ch_img.shape}, data type {ch_img.dtype}')
                return (ch_img, img.name+ch_suffix)

            show_info(f'{img.name}: preprocessing started, data type {img.data.dtype}, shape {img.data.shape}')
            if img.data.ndim == 4:
                show_info(f'{img.name}: Ch. split and preprocessing mode, shape {img.data.shape}')
                if stack_order == 'TCXY':    # for LA data
                    input_img = img.data   
                elif stack_order == 'CTXY':  # for confocal data
                    input_img = np.moveaxis(img.data,0,1)
                for i in range(0,img.data.shape[1]):
                    show_info(f'{img.name}: Ch. {i} preprocessing started')
                    yield _preprocessing(ch_img=input_img[:,i,:,:], ch_suffix=f'_ch{i}')
            elif img.data.ndim == 3:
                show_info(f'{img.name}: image already has 3 dimensions, preprocessing only mode')
                yield _preprocessing(ch_img=img.data, ch_suffix='')
            else:
                raise ValueError('Input image has to have 3 or 4 dimensions!')       
        
        _split_channels()


@magic_factory(call_button='Align stack',
               align_method={"choices": ['internal', 'load matrix']},
               load_matrix={'mode': 'r', 'filter': 'Text files (*.txt)'},
               saving_path={'mode': 'd'},)
def dw_registration(viewer: Viewer, offset_img:Image,
                    input_crop:int=30, output_crop:int=30,
                    align_method:str='internal',  # reference_img:Image=None,
                    manual_channels:bool=False,  # ch0:int=0, ch1:int=1,
                    ref_off_ch:list=[0,1],  # for manual_channels=True
                    load_matrix:pathlib.Path = None,
                    save_matrix:bool=False,
                    saving_path:pathlib.Path = os.getcwd()):
    if input is not None:
        if offset_img.data.ndim == 4:
            def _save_aligned(img):
                xform_name = offset_img.name+'_algn'
                try: 
                    viewer.layers[xform_name].data = img
                    viewer.layers[xform_name].colormap = 'turbo'
                except KeyError:
                    viewer.add_image(img, name=xform_name, colormap='turbo')

            @thread_worker(connect={'yielded':_save_aligned})
            def _dw_registration():
                input_data = offset_img.data
                
                start = time.perf_counter()
                if input_crop != 0:  # optional input crop for better registration and border artefacts removal
                    y, x = input_data.shape[-2:]
                    input_data = input_data[:,:,input_crop:y-input_crop,input_crop:x-input_crop]

                if align_method == 'internal':
                    show_info(f'{offset_img.name}: internal alignment mode')
                    affreg = AffineRegistration()
                    transform = AffineTransform2D()

                    if manual_channels:  # manual channel selection
                        show_info(f'{offset_img.name}: move ch. {ref_off_ch[1]} to ch.{ref_off_ch[0]}')
                        ref_frame = np.mean(input_data[:,ref_off_ch[0],:,:], axis=0)
                        move_frame = np.mean(input_data[:,ref_off_ch[1],:,:], axis=0)
                    elif input_data.shape[1] == 2:  # 1 ext with DV
                        show_info(f'{offset_img.name}: 2 spectral ch.')
                        ref_frame = np.mean(input_data[:,1,:,:], axis=0)
                        move_frame = np.mean(input_data[:,0,:,:], axis=0)
                    elif input_data.shape[1] == 4:  # 2 ext with DV
                        show_info(f'{offset_img.name}: 4 spectral ch.')
                        ref_frame = np.mean(input_data[:,3,:,:], axis=0)
                        move_frame = np.mean(input_data[:,0,:,:], axis=0)
                    else:
                        raise ValueError(f'The input image have {input_data.shape[1]} spectral ch., but 2 or 4 ch. are required!')

                    affine_params = affreg.optimize(ref_frame, move_frame,
                                                    transform, params0=None)
                    
                    if input_data.shape[1] == 2:  # 1 ext with DV
                        reg_channel = np.array([affine_params.transform(frame) for frame in input_data[:,0,:,:]],
                                               dtype=input_data.dtype)
                        output_data = np.stack((reg_channel,
                                                input_data[:,1,:,:]),
                                               axis=1)
                    if input_data.shape[1] == 4:  # 2 ext with DV
                        reg_channel_0 = np.array([affine_params.transform(frame) for frame in input_data[:,0,:,:]],
                                                 dtype=input_data.dtype)
                        reg_channel_2 = np.array([affine_params.transform(frame) for frame in input_data[:,2,:,:]],
                                                 dtype=input_data.dtype)
                        output_data = np.stack((reg_channel_0,
                                                input_data[:,1,:,:],
                                                reg_channel_2,
                                                input_data[:,3,:,:]),
                                               axis=1)
                    if save_matrix:
                        matrix_name = f"{offset_img.name}_affine_matrix.txt"
                        np.savetxt(os.path.join(saving_path, matrix_name), affine_params.affine)
                        show_info(f'{offset_img.name}: affine matrix saved to {os.path.join(saving_path, matrix_name)}')
                elif align_method == 'load matrix':
                    affine_matrix = np.loadtxt(load_matrix)
                    show_info(f'{offset_img.name}: alignment with matrix mode, loaded from {load_matrix}')
                    if input_data.shape[1] == 2:  # 1 ext with DV
                        show_info(f'{offset_img.name}: 2 spectral ch.')
                        reg_channel = np.array([ndi.affine_transform(frame, affine_matrix, output_shape=frame.shape) for frame in input_data[:,0,:,:]],
                                                dtype=input_data.dtype)
                        output_data = np.stack((reg_channel,
                                                input_data[:,1,:,:]),
                                                axis=1)
                    elif input_data.shape[1] == 4:  # 2 ext with DV
                        show_info(f'{offset_img.name}: 4 spectral ch.')
                        reg_channel_0 = np.array([ndi.affine_transform(frame, affine_matrix, output_shape=frame.shape) for frame in input_data[:,0,:,:]],
                                                 dtype=input_data.dtype)
                        reg_channel_2 = np.array([ndi.affine_transform(frame, affine_matrix, output_shape=frame.shape) for frame in input_data[:,2,:,:]],
                                                 dtype=input_data.dtype)
                        output_data = np.stack((reg_channel_0,
                                                input_data[:,1,:,:],
                                                reg_channel_2,
                                                input_data[:,3,:,:]),
                                               axis=1)                   
                    else:
                        raise ValueError(f'The input image have {input_data.shape[1]} spectral ch., but 2 or 4 ch. are required!')
                elif align_method == 'reference':
                    show_warning('Sorry, this mode is under development!')

                if output_crop != 0:  # optional output crop for border artefacts of the registration removal
                    yo, xo = output_data.shape[-2:]
                    output_data = output_data[:,:,input_crop:yo-input_crop,input_crop:xo-input_crop]
                
                end = time.perf_counter()
                show_info(f'{offset_img.name}: stack aligned in {end-start:.2f} s')
                yield output_data.astype(input_data.dtype)
                    
            _dw_registration()
        else:
            raise ValueError('Incorrect dimensions of the input image!')


@magic_factory(call_button='Estimate cross-talk',
               presented_fluorophore={"choices": ['A', 'D']},
               saving_path={'mode': 'd'})
def cross_calc(viewer: Viewer, DD_img:Image, DA_img:Image, AA_img:Image,
               mask: Labels,
               presented_fluorophore:str='A',
               saving_path:pathlib.Path = os.getcwd()):
    if input is not None:
        if not np.all([DD_img.data.ndim == 3, DA_img.data.ndim == 3, AA_img.data.ndim == 3]):
            raise ValueError('Incorrect input image shape!')

        def _save_cross_data(output):
            output_name = AA_img.name.replace('_ch3','')
            
            # data frame saving
            output_df = output[0]
            output_df.insert(0, 'id', output_name)
            df_name = f"{output_name}_{presented_fluorophore}.csv"
            output_df.to_csv(os.path.join(saving_path, df_name))
            show_info(f'{df_name}: {presented_fluorophore} coeficients saved')
                
            # regression line for last frame
            x_arr, y_arr = output[1][-1,0], output[1][-1,1]
            last_slope = output_df.iloc[-1,2]
            last_intercept = output_df.iloc[-1,5]
            last_r2 = output_df.iloc[-1,7]
            x_line = np.linspace(np.min(x_arr), np.max(x_arr), 100)
            y_line = last_slope*x_line + last_intercept
            show_info(f'Last frame slope {last_slope:.3f}, intercept {last_intercept:.3f}, r² {last_r2:.2f}')

            # last frame plot
            axis_lab_dict = {'A':['AA, a.u.','DA, a.u.', 'a'],
                             'D':['DD, a.u.','DA, a.u.', 'd']}

            mpl_fig = plt.figure()
            ax = mpl_fig.add_subplot(111)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.scatter(x_arr, y_arr, s=5, alpha=0.15, color='blue')
            ax.plot(x_line, y_line, color='red', linewidth=2)
            ax.grid(color='grey', linewidth=.25)
            ax.set_xlabel(axis_lab_dict[presented_fluorophore][0])
            ax.set_ylabel(axis_lab_dict[presented_fluorophore][1])
            plt.title(f'{output_name}, esimation of {axis_lab_dict[presented_fluorophore][2]}')
            viewer.window.add_dock_widget(FigureCanvas(mpl_fig), name='Cross-talk estimation')

        @thread_worker(connect={'returned':_save_cross_data})
        def _cross_calc():
            input_mask = mask.data != 0
            cross_estimator = e_fret.CrossTalkEstimation(mask=input_mask,
                                                         dd_img=DD_img.data,
                                                         da_img=DA_img.data,
                                                         aa_img=AA_img.data)
            if presented_fluorophore == 'A':
                coefs = cross_estimator.estimate_a()
            if presented_fluorophore == 'D':
                coefs = cross_estimator.estimate_d()
            return coefs

        _cross_calc()


@magic_factory(call_button='Estimate G-factor',
               estimation_method={"choices": ['Zal', 'Chen']},
               saving_path={'mode': 'd'})
def g_calc(viewer: Viewer,
           DD_img_high_FRET:Image, DA_img_high_FRET:Image, AA_img_high_FRET:Image,
           DD_img_low_FRET:Image, DA_img_low_FRET:Image, AA_img_low_FRET:Image,
           mask: Labels,
           estimation_method:str='Zal',
           segment_mask:bool=True,
           a:float=0.0136, d:float=0.2646,  # a & d for TagBFP+mBaoJin
           saving_path:pathlib.Path = os.getcwd()):
    if input is not None:
        if not np.all([DD_img_low_FRET.data.ndim == 3, DA_img_low_FRET.data.ndim == 3, AA_img_low_FRET.data.ndim == 3]):
            raise ValueError('Incorrect input low FRET image shape!')
        if not np.all([DD_img_high_FRET.data.ndim == 3, DA_img_high_FRET.data.ndim == 3, AA_img_high_FRET.data.ndim == 3]):
            raise ValueError('Incorrect input high FRET image shape!')

        def _save_g_data(output):
            output_name = AA_img_low_FRET.name.replace('_ch3','')
            # data frame saving
            output_df = output[0]
            output_df.insert(0, 'id', output_name)
            df_name = f"{output_name}_G.csv"
            output_df.to_csv(os.path.join(saving_path, df_name))
            show_info(f'{df_name}: G-factor saved')

            # segmented mask saving
            if segment_mask:
                lab_name = f"{output_name}_seg_labels"
                try:
                    viewer.layers[lab_name].data = output[-1]
                except KeyError:
                    new_labels = viewer.add_labels(output[-1], name=lab_name, opacity=0.5)
                    new_labels.contour = 0

            # regression line for 1st frame
            x_arr, y_arr = output[1][0], output[1][1]
            fit_slope = output_df.iloc[0,1]
            fit_intercept = output_df.iloc[0,4]
            fit_r2 = output_df.iloc[0,6]
            x_line = np.linspace(np.min(x_arr), np.max(x_arr), 100)
            y_line = fit_slope*x_line + fit_intercept
            show_info(f'G-factor fit slope {fit_slope:.3f}, intercept {fit_intercept:.3f}, r² {fit_r2:.4f}')

            # plotting
            mpl_fig = plt.figure()
            
            # G-factor fit
            ax0 = mpl_fig.add_subplot(121)
            ax0.spines['top'].set_visible(False)
            ax0.spines['right'].set_visible(False)
            ax0.scatter(x_arr, y_arr, s=15, alpha=0.25, color='blue')
            ax0.plot(x_line, y_line, color='red', linewidth=2)
            ax0.grid(color='grey', linewidth=.25)
            ax0.set_xlabel('ΔDD, a.u.')
            ax0.set_ylabel('ΔFc, a.u.')
            ax0.set_title(f'Linear fit, G={fit_slope:.3f}', fontsize=10)

            # ΔFc/ΔDD in ROIs
            ax1 = mpl_fig.add_subplot(122)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.boxplot(y_arr/x_arr, positions=[1], widths=0.5)
            ax1.scatter(np.full(len(y_arr), 1), y_arr/x_arr, s=35, alpha=0.25, color='blue')
            ax1.errorbar([1], output_df['g_val'],
                    yerr = 1.96 * output_df['g_err'],
                    fmt ='-o', capsize=5,
                    alpha=0.75, color='blue')
            ax1.grid(color='grey', linewidth=.25)
            ax1.set_xticklabels([''])
            ax1.set_xlabel('')
            ax1.set_ylabel('ΔFc/ΔDD')
            ax1.set_title(f'ROIs ΔFc/ΔDD and 95% CI for G', fontsize=10)
            plt.suptitle(f'{output_name} G-factor estimation', fontsize=10)
            viewer.window.add_dock_widget(FigureCanvas(mpl_fig), name='G-factor estimation')

        @thread_worker(connect={'returned':_save_g_data})
        def _g_calc():
            start = time.perf_counter()
            show_info(f'G-factor estimation started, using a={a}, d={d}')

            if segment_mask:
                img_mask = mask.data != 0
                img_mask_area = np.sum(img_mask)
                fragment_area = img_mask_area // 30
                if fragment_area < 300:
                    raise ValueError('The mask area is too small for segmentation, please select larger area!')
                roi_mask = utils.mask_segmentation(img_mask, fragment_num=30)
            else:
                roi_mask = mask.data
                show_info(f'G-factor estimation with provided mask, there are {np.max(roi_mask)} ROIs')

            g_estimator = e_fret.GFactorEstimation(mask=roi_mask,
                                                     h_dd_img=DD_img_high_FRET.data[0],
                                                     h_da_img=DA_img_high_FRET.data[0],
                                                     h_aa_img=AA_img_high_FRET.data[0],
                                                     l_dd_img=DD_img_low_FRET.data[0],
                                                     l_da_img=DA_img_low_FRET.data[0],
                                                     l_aa_img=AA_img_low_FRET.data[0],
                                                     a_val=a,
                                                     d_val=d)
            # coef = g_estimator.estimate_g_zal()
            if estimation_method == 'Zal':
                coef = g_estimator.estimate_g_zal()
            elif estimation_method == 'Chen':
                coef = g_estimator.estimate_g_chen()
            end = time.perf_counter()
            show_info(f'G-factor estimated in {end - start:.2f}s')
            if segment_mask:
                return (*coef, roi_mask)
            else:
                return coef

        _g_calc()


@magic_factory(call_button='Estimate E-FRET',
               output_type={"choices": ['Fc', 'Eapp', 'Ecorr']},)
def e_app_calc(viewer: Viewer, DD_img:Image, DA_img:Image, AA_img:Image,
               a:float=0.0136, d:float=0.2646, G:float=2.99,  # CFP+YFP: a=0.122, d=0.794, G=3.6 | TagBFP+mBaoJin: a=0.0136, d=0.2646, G=2.992
               output_type:str='Fc',
               save_normalized:bool=True):
    if input is not None:
        if not np.all([DD_img.data.ndim == 3, DA_img.data.ndim == 3, AA_img.data.ndim == 3]):
            raise ValueError('Incorrect input image shape!')

        def _save_e_app(params):
            img = params[0]
            img_name = params[1]
            try: 
                viewer.layers[img_name].data = img
            except KeyError:
                viewer.add_image(img, name=img_name, colormap='turbo')

        @thread_worker(connect={'yielded':_save_e_app})
        def _e_app_calc():
            start = time.perf_counter()
            e_fret_img = e_fret.E_FRET(dd_img=DD_img.data,
                                       da_img=DA_img.data,
                                       aa_img=AA_img.data,
                                       a_val=a,
                                       d_val=d,
                                       G_val=G)
            output_name = AA_img.name.replace('_ch3', '')
            if output_type == 'Ecorr':
                output_fret_img = e_fret_img.Ecorr_img()
                output_suffix = '_Ecorr'
            elif output_type == 'Eapp':
                output_fret_img = e_fret_img.Eapp_img()
                output_suffix = '_Eapp'
            elif output_type == 'Fc':
                output_fret_img = e_fret_img.Fc_img()
                output_suffix = '_Fc'
            yield (output_fret_img, output_name + output_suffix)
            if save_normalized:
                epsilon = 1e-12
                img_norm = np.mean(AA_img.data, axis=0)
                img_norm = (img_norm-np.min(img_norm)) / (np.max(img_norm)-np.min(img_norm) + epsilon)
                output_norm = output_fret_img*img_norm
                yield (output_norm, output_name + output_suffix + '_norm')
            end = time.perf_counter()
            show_info(f'{output_type} img calculated in {end-start:.2f}s')

        _e_app_calc()


@magic_factory(call_button='Calc Red-Green')
def der_series(viewer: Viewer, img:Image,
               left_frames:int=1, space_frames:int=0, right_frames:int=1,
               normalize_by_int:bool=True,
               save_MIP:bool=False):
    if input is not None:
        if img.data.ndim != 3:
            raise ValueError('The input image should have 3 dimensions!')

        def _save_rg_img(params):
            img = params[0]
            img_name = params[1]
            try: 
                viewer.layers[img_name].data = img
            except KeyError:
                if normalize_by_int:
                    c_lim = np.max(np.abs(img)) * 0.3
                else:
                    c_lim = np.max(np.abs(img)) * 0.75
                new_image = viewer.add_image(img, name=img_name, contrast_limits=[-c_lim, c_lim])
                new_image.colormap = 'red-green', utils.red_green_cmap()

        @thread_worker(connect={'yielded':_save_rg_img})
        def _der_series():
            ref_img = img.data

            der_img = []
            for i in range(ref_img.shape[0]-(left_frames+right_frames+space_frames)):
                img_base = np.mean(ref_img[i:i+left_frames+1], axis=0)
                img_stim = np.mean(ref_img[i+left_frames+right_frames:i+left_frames+right_frames+space_frames+1], axis=0)
                
                img_diff = img_stim-img_base

                if normalize_by_int:
                    img_norm = np.mean(np.stack((img_base,img_diff), axis=0), axis=0)
                    img_norm = (img_norm-np.min(img_norm)) / (np.max(img_norm)-np.min(img_norm))
                    img_diff = img_diff * img_norm

                der_img.append(img_diff)

            der_img = np.asarray(der_img, dtype=float)
            yield (der_img, img.name + '_red-green')

            if save_MIP:
                der_mip = np.max(der_img, axis=0)
                yield (der_mip, img.name + '_red-green-MIP')

        _der_series()


@magic_factory(call_button='Calc relative intensity',
               values_mode={"choices": ['ΔF', 'ΔF/F0']},)
def rel_series(viewer: Viewer, img:Image, values_mode:str='ΔF', F0_win:int=5):
    if input is not None:
        if img.data.ndim != 3:
            raise ValueError('The input image should have 3 dimensions!')

        def _save_rel_img(params):
            img = params[0]
            img_name = params[1]
            try:
                viewer.layers[img_name].data = img
            except KeyError:
                c_lim = np.max(np.abs(img)) * 0.8
                new_image = viewer.add_image(img, name=img_name, contrast_limits=[-c_lim, c_lim])
                new_image.colormap = 'delta', utils.delta_smooth_cmap()

        @thread_worker(connect={'yielded':_save_rel_img})
        def _rel_series():
            start = time.perf_counter()

            input_img = img.data
            mode_dict = {'ΔF': 'dF', 'ΔF/F0': 'dF/F0'}
            output_img = utils.delta_img(input_img, mode=mode_dict[values_mode], win_size=F0_win)
        
            end = time.perf_counter()
            show_info(f'{img.name}: {values_mode} img calculated in {end-start:.2f}s')

            yield (output_img, img.name + f'_{values_mode}')

        _rel_series()


@magic_factory(call_button='Build Dots Mask',
               background_level={"widget_type": "FloatSlider", 'min':50.0, 'max': 99.0, 'step':1.0},
               detection_level={"widget_type": "FloatSlider",'min':1.0, 'max': 100.0, 'step':1.0},)
def dot_mask_calc(viewer: Viewer, img:Image, background_level:float=75.0, detection_level:float=25.0,
                  minimal_distance:int=2, mask_diamets:int=5):
    if input is not None:
        if img.data.ndim != 3:
            raise ValueError('The input image should have 3 dimensions!')
        labels_name = img.name + '_dots-labels'

        def _save_dot_labels(params):
            lab = params[0]
            name = params[1]
            try:
                viewer.layers[name].data = lab
            except KeyError:
                new_labels = viewer.add_labels(lab, name=name, opacity=1)
                new_labels.contour = 1

        @thread_worker(connect={'yielded':_save_dot_labels})
        def _dot_mask_calc():
            prc_filt = lambda x, p: np.array(x - np.percentile(x, p)).clip(min=0).astype(dtype=x.dtype)

            input_img = img.data
            input_mip = np.max(input_img, axis=0)
            detection_mip = prc_filt(x=input_mip, p=background_level)

            peaks_coord = feature.peak_local_max(detection_mip,
                                                 min_distance=2,
                                                 threshold_rel=detection_level/100.0)
            peaks_img = np.zeros_like(input_mip, dtype=bool)
            peaks_img[tuple(peaks_coord.T)] = True
            peaks_mask = morphology.dilation(peaks_img, footprint=morphology.disk(mask_diamets))

            mask_dist_img = ndi.distance_transform_edt(peaks_mask)
            mask_centers_coord = feature.peak_local_max(mask_dist_img,
                                                        min_distance=minimal_distance)
            mask_centers = np.zeros_like(input_mip, dtype=bool)
            mask_centers[tuple(mask_centers_coord.T)] = True

            peaks_labels = segmentation.watershed(-mask_dist_img,
                                                  markers=morphology.label(mask_centers),
                                                  mask=peaks_mask,
                                                  compactness=10)
            show_info(f'{img.name}: detected {np.max(peaks_labels)} dots labels')
            yield (peaks_labels, labels_name)

        _dot_mask_calc()


@magic_factory(call_button='Build Up Mask',
               det_th={"widget_type": "FloatSlider", 'max': 1},
               in_ROIs_det_method={"choices": ['otsu', 'threshold']},)  # insertions_threshold={'widget_type': 'FloatSlider', 'max': 1}
def up_mask_calc(viewer: Viewer, img:Image, ROIs_mask:Labels,
                 det_frame_index:int=2,
                 det_th:float=0.25,
                 in_ROIs_det:bool=False,
                 in_ROIs_det_method:str='otsu',
                 in_ROIs_det_th_corr:float=0.1,
                 final_opening_fp:int=1,
                 final_dilation_fp:int=0,
                 save_total_up_mask:bool=False):
    if input is not None:
        if img.data.ndim != 3:
            raise ValueError('The input image should have 3 dimensions!')

        def _save_up_labels(params):
            lab = params[0]
            name = params[1]
            try:
                viewer.layers[name].data = lab
            except KeyError:
                new_labels = viewer.add_labels(lab, name=name, opacity=1)
                new_labels.contour = 1

        @thread_worker(connect={'yielded':_save_up_labels})
        def _up_mask_calc():
            input_img = img.data
            detection_img = input_img[det_frame_index]

            def up_detection(img, method, th, div, op_f, d_f):
                if method == 'threshold':
                    up_m = img > np.max(np.abs(img)) * (th*div)
                    up_m = morphology.erosion(up_m, footprint=morphology.disk(2))
                    up_m = morphology.dilation(up_m, footprint=morphology.disk(1))
                    up_m = ndi.binary_fill_holes(up_m)
                    up_m = up_m.astype(int)
                elif method == 'otsu':
                    up_m = img > filters.threshold_otsu(img)
                if op_f != 0:
                    up_m = morphology.opening(up_m, footprint=morphology.disk(op_f))
                up_m = morphology.dilation(up_m, footprint=morphology.disk(d_f))
                return up_m.astype(bool)

            if in_ROIs_det:
                rois_mask = ROIs_mask.data
                up_labels = np.zeros_like(rois_mask)
                for roi_region in measure.regionprops(rois_mask):
                    one_roi_box = roi_region.bbox
                    one_roi_img = detection_img[one_roi_box[0]:one_roi_box[2],one_roi_box[1]:one_roi_box[3]]
                    one_roi_input_mask = rois_mask[one_roi_box[0]:one_roi_box[2],one_roi_box[1]:one_roi_box[3]] == 0

                    one_roi_mask = up_detection(img=one_roi_img,
                                                method=in_ROIs_det_method,
                                                th=det_th,
                                                div=in_ROIs_det_th_corr,
                                                op_f=final_opening_fp,
                                                d_f=final_dilation_fp)
                    one_roi_mask[one_roi_input_mask] = 0
                    one_roi_mask = one_roi_mask * roi_region.label
                    up_labels[one_roi_box[0]:one_roi_box[2],one_roi_box[1]:one_roi_box[3]] = one_roi_mask
                    up_mask = up_labels > 0
            else:
                up_mask = up_detection(img=detection_img,
                                       method='threshold',
                                       th=det_th,
                                       div=0.1,
                                       op_f=final_opening_fp,
                                       d_f=final_dilation_fp)
                up_labels = measure.label(up_mask)

            show_info(f'{img.name}: detected {np.max(measure.label(up_mask))} labels')

            labels_name = img.name + '_up-labels'
            yield (up_labels, labels_name)
            if save_total_up_mask:
                mask_name = img.name + '_up-mask'
                yield (up_mask, mask_name)

        _up_mask_calc()                


@magic_factory(call_button='Build Mask',
               masking_mode={"choices": ['up', 'down']},)
def mask_calc(viewer: Viewer, img:Image, det_frame_index:int=2,
              masking_mode:str='up',
              up_threshold:float=0.2,
              down_threshold:float=-0.9,
              opening_footprint:int=0):
    if input is not None:
        if img.data.ndim != 3:
            raise ValueError('The input image should have 3 dimensions!')

        if masking_mode == 'up':
            labels_name = img.name + '_up-labels'
        elif masking_mode == 'down':        
            labels_name = img.name + '_down-labels'

        def _save_rg_labels(params):
            lab = params[0]
            name = params[1]
            try:
                viewer.layers[name].data = lab
            except KeyError:
                new_labels = viewer.add_labels(lab, name=labels_name, opacity=1)
                new_labels.contour = 1

        @thread_worker(connect={'yielded':_save_rg_labels})
        def _mask_calc():
            input_img = img.data
            detection_img = input_img[det_frame_index]

            if masking_mode == 'up':
                mask = detection_img >= np.max(np.abs(detection_img)) * up_threshold
            elif masking_mode == 'down':        
                mask = detection_img <= np.max(np.abs(detection_img)) * down_threshold

            mask = morphology.erosion(mask, footprint=morphology.disk(2))
            mask = morphology.dilation(mask, footprint=morphology.disk(1))
            mask = ndi.binary_fill_holes(mask)
            mask = mask.astype(int)

            if opening_footprint != 0:
                mask = morphology.opening(mask, footprint=morphology.disk(opening_footprint))
                mask = morphology.dilation(mask, footprint=morphology.disk(1))

            labels = measure.label(mask)
            show_info(f'{img.name}: detected {np.max(labels)} "{masking_mode}" labels')

            yield (labels, labels_name)

        _mask_calc()
            

@magic_factory(call_button='Build Profiles',
               values_mode={"choices": ['ΔF', 'ΔF/F0', 'abs.']},)
def labels_profile_line(viewer: Viewer, img:Image, labels:Labels,
                        time_scale:float=1.0,
                        values_mode='ΔF',
                        use_simple_baseline:bool=True,
                        ΔF_win:int=4,
                        Dietrich_std:float=1.25):
    if input is not None:
        input_img = img.data
        input_labels = labels.data
        time_line = np.linspace(0, (input_img.shape[0]-1)*time_scale, \
                                num=input_img.shape[0])

        baseline_params = {'win_size': ΔF_win, 
                           'mode': values_mode,
                           'stds': Dietrich_std}  # for pybaselines

        if use_simple_baseline:
            fun_delta = utils.delta_prof_simple
            show_info(f'{img.name}: simple baseline estimation, win size={baseline_params["win_size"]}')
        else:
            fun_delta = utils.delta_prof_pybase
            show_info(f'{img.name}: Dietrich baseline estimation, std={baseline_params["stds"]}, win size={baseline_params["win_size"]}')

        y_lab_dict = {'abs.': 'a.u.',
                      'ΔF': 'ΔF',
                      'ΔF/F0': 'ΔF/F0'}
        ylab = y_lab_dict[values_mode]

        start = time.perf_counter()
        profile_abs = utils.labels_to_profiles(input_label=input_labels,
                                               input_img=input_img)

        profile_to_plot = []
        if all([values_mode == 'abs.', use_simple_baseline]):
            profile_to_plot = np.round(profile_abs, decimals=4)
        else:
            profile_to_plot = fun_delta(profile_abs,
                                        **baseline_params)
        end = time.perf_counter()
        show_info(f'{img.name}: profiles calculated in {end-start:.2f} s')

        lab_colors = labels.get_color([prop['label'] for prop in measure.regionprops(label_image=input_labels)])

        mpl_fig = plt.figure()
        ax = mpl_fig.add_subplot(111)
        if values_mode == 'ΔF/F0' or values_mode == 'ΔF':
            ax.axhline(y=0.0, color='k', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for num_ROI, color in enumerate(lab_colors):
            profile_ROI = profile_to_plot[num_ROI]
            ax.plot(time_line, profile_ROI,
                        alpha=0.45, marker='o', color=color)
        ax.grid(color='grey', linewidth=.25)
        ax.set_xlabel('Time, s')
        ax.set_ylabel(ylab)
        plt.title(f'{img.name} ROIs profiles, {values_mode} mode, labels: {labels.name}')
        viewer.window.add_dock_widget(FigureCanvas(mpl_fig), name='ROIs Prof.')


@magic_factory(call_button='Build Profiles',
               profiles_num={"choices": ['1', '2', '3']},
               values_mode={"choices": ['abs.', 'ΔF', 'ΔF/F0']},
               stat_method={"choices": ['se', 'iqr', 'ci']},)
def labels_multi_profile_stat(viewer: Viewer, img_0:Image, img_1:Image, img_2:Image,
                              lab:Labels,
                              profiles_num:str='1',
                              time_scale:float=1.0,
                              values_mode:str='ΔF/F0',
                              use_simple_baseline:bool=True,
                              ΔF_win:int=4,
                              Dietrich_std:float=1.25,
                              stat_method:str='se'):
    if input is not None:
        # mean, se
        arr_se_stat = lambda x: (np.mean(x, axis=0), \
                                 np.std(x, axis=0)/np.sqrt(x.shape[1]))
        # meadian, IQR
        arr_iqr_stat = lambda x: (np.median(x, axis=0), \
                                  stats.iqr(x, axis=0))
        # mean, CI
        arr_ci_stat = lambda x, alpha=0.05: (np.mean(x, axis=0), \
                                             stats.t.ppf(1-alpha/2, df=x.shape[1]-1) \
                                                         *np.std(x, axis=0, ddof=1)/np.sqrt(x.shape[1]))
        stat_dict = {'se':arr_se_stat,
                     'iqr':arr_iqr_stat,
                     'ci':arr_ci_stat}

        baseline_params = {'win_size': ΔF_win, 
                           'mode': values_mode,
                           'stds': Dietrich_std}  # for pybaselines

        if use_simple_baseline:
            fun_delta = utils.delta_prof_simple
            show_info(f'{lab.name}: simple baseline estimation, win size={baseline_params["win_size"]}')
        else:
            fun_delta = utils.delta_prof_pybase
            show_info(f'{lab.name}: Dietrich baseline estimation, std={baseline_params["stds"]}, win size={baseline_params["win_size"]}')

        y_lab_dict = {'abs.': 'a.u.',
                      'ΔF': 'ΔF',
                      'ΔF/F0': 'ΔF/F0'}
        ylab = y_lab_dict[values_mode]

        # processing
        input_labels = lab.data

        start = time.perf_counter()
        # img 0
        input_img_0 = img_0.data
        time_line_0 = np.linspace(0, (input_img_0.shape[0]-1)*time_scale, \
                                  num=input_img_0.shape[0])
        profile_abs_0 = utils.labels_to_profiles(input_label=input_labels,
                                                 input_img=input_img_0)
        if values_mode == 'abs.':
            selected_profile_0 = np.round(profile_abs_0, decimals=4)
        else:
            selected_profile_0 = fun_delta(profile_abs_0,
                                           **baseline_params)
        arr_val_0, arr_var_0 = stat_dict[stat_method](selected_profile_0)

        # img 1
        if profiles_num == '2' or profiles_num == '3':
            input_img_1 = img_1.data
            time_line_1 = np.linspace(0, (input_img_1.shape[0]-1)*time_scale, \
                                    num=input_img_1.shape[0])
            profile_abs_1 = utils.labels_to_profiles(input_label=input_labels,
                                                     input_img=input_img_1)
            if values_mode == 'abs.':
                selected_profile_1 = np.round(profile_abs_1, decimals=4)
            else:
                selected_profile_1 = fun_delta(profile_abs_1,
                                                **baseline_params)
            arr_val_1, arr_var_1 = stat_dict[stat_method](selected_profile_1)

        # img 2
        if profiles_num == '3':
            input_img_2 = img_2.data
            time_line_2 = np.linspace(0, (input_img_2.shape[0]-1)*time_scale, \
                                    num=input_img_2.shape[0])
            profile_abs_2 = utils.labels_to_profiles(input_label=input_labels,
                                                     input_img=input_img_2)
            if values_mode == 'abs.':
                selected_profile_2 = np.round(profile_abs_2, decimals=4)
            else:
                selected_profile_2 = fun_delta(profile_abs_2,
                                                **baseline_params)
            arr_val_2, arr_var_2 = stat_dict[stat_method](selected_profile_2)
        end = time.perf_counter()
        show_info(f'Profiles calculated in {end-start:.2f} s')

        # plotting
        mpl_fig = plt.figure()
        ax = mpl_fig.add_subplot(111)
        if values_mode == 'ΔF/F0' or values_mode == 'ΔF':
            ax.axhline(y=0.0, color='k', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(color='grey', linewidth=.25)
        ax.set_xlabel('Time, s')
        ax.set_ylabel(ylab)
        ax.errorbar(time_line_0, arr_val_0,
                    yerr = arr_var_0,
                    fmt ='-o', capsize=2, label=img_0.name,
                    alpha=0.75, color='black')
        if profiles_num == '2' or profiles_num == '3':
            ax.errorbar(time_line_1, arr_val_1,
                        yerr = arr_var_1,
                        fmt ='-o', capsize=2, label=img_1.name,
                        alpha=0.75, color='red')
        if profiles_num == '3':
            ax.errorbar(time_line_2, arr_val_2,
                        yerr = arr_var_2,
                        fmt ='-o', capsize=2, label=img_2.name,
                        alpha=0.75, color='blue')
        plt.legend()
        plt.title(f'{lab.name}, method {stat_method}')
        viewer.window.add_dock_widget(FigureCanvas(mpl_fig), name='Multiple Img Stat Prof.')


@magic_factory(call_button='Build Profiles',
               labels_num={"choices": ['1', '2', '3']},
               values_mode={"choices": ['abs.', 'ΔF', 'ΔF/F0']},
               stat_method={"choices": ['se', 'iqr', 'ci']},)
def multi_labels_profile_stat(viewer: Viewer, img:Image,
                        lab_0:Labels, lab_1:Labels, lab_2:Labels,
                        time_scale:float=1.0,
                        labels_num:str='1',
                        values_mode:str='ΔF/F0',
                        use_simple_baseline:bool=True,
                        ΔF_win:int=4,
                        Dietrich_std:float=1.25,
                        stat_method:str='se'):
    if input is not None:
        # mean, se
        arr_se_stat = lambda x: (np.mean(x, axis=0), \
                                 np.std(x, axis=0)/np.sqrt(x.shape[1]))
        # meadian, IQR
        arr_iqr_stat = lambda x: (np.median(x, axis=0), \
                                  stats.iqr(x, axis=0))
        # mean, CI
        arr_ci_stat = lambda x, alpha=0.05: (np.mean(x, axis=0), \
                                             stats.t.ppf(1-alpha/2, df=x.shape[1]-1) \
                                                         *np.std(x, axis=0, ddof=1)/np.sqrt(x.shape[1]))
        stat_dict = {'se':arr_se_stat,
                     'iqr':arr_iqr_stat,
                     'ci':arr_ci_stat}

        baseline_params = {'win_size': ΔF_win, 
                           'mode': values_mode,
                           'stds': Dietrich_std}  # for pybaselines

        if use_simple_baseline:
            fun_delta = utils.delta_prof_simple
            show_info(f'{img.name}: simple baseline estimation, win size={baseline_params["win_size"]}')
        else:
            fun_delta = utils.delta_prof_pybase
            show_info(f'{img.name}: Dietrich baseline estimation, std={baseline_params["stds"]}, win size={baseline_params["win_size"]}')

        y_lab_dict = {'abs.': 'a.u.',
                      'ΔF': 'ΔF',
                      'ΔF/F0': 'ΔF/F0'}
        ylab = y_lab_dict[values_mode]

        # processing
        input_img = img.data
        time_line = np.linspace(0, (input_img.shape[0]-1)*time_scale, \
                                num=input_img.shape[0])

        start = time.perf_counter()
        # lab 0
        input_lab_0 = lab_0.data
        profile_abs_0 = utils.labels_to_profiles(input_label=input_lab_0,
                                                 input_img=input_img)
        if values_mode == 'abs.':
            selected_profile_0 = np.round(profile_abs_0, decimals=4)
        else:
            selected_profile_0 = fun_delta(profile_abs_0,
                                           **baseline_params)
        arr_val_0, arr_var_0 = stat_dict[stat_method](selected_profile_0)

        # lab 1
        if labels_num == '2' or labels_num == '3':
            input_lab_1 = lab_1.data
            profile_abs_1 = utils.labels_to_profiles(input_label=input_lab_1,
                                            input_img=input_img)
            if values_mode == 'abs.':
                selected_profile_1 = np.round(profile_abs_1, decimals=4)
            else:
                selected_profile_1 = fun_delta(profile_abs_1,
                                               **baseline_params)
            arr_val_1, arr_var_1 = stat_dict[stat_method](selected_profile_1)

        # lab 2
        if labels_num == '3':
            input_lab_2 = lab_2.data
            profile_abs_2 = utils.labels_to_profiles(input_label=input_lab_2,
                                            input_img=input_img)
            if values_mode == 'abs.':
                selected_profile_2 = np.round(profile_abs_2, decimals=4)
            else:
                selected_profile_2 = fun_delta(profile_abs_2,
                                               **baseline_params)
            arr_val_2, arr_var_2 = stat_dict[stat_method](selected_profile_2)
        end = time.perf_counter()
        show_info(f'Profiles calculated in {end-start:.2f} s')

        # plotting        
        mpl_fig = plt.figure()
        ax = mpl_fig.add_subplot(111)
        if values_mode == 'ΔF/F0' or values_mode == 'ΔF':
            ax.axhline(y=0.0, color='k', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(color='grey', linewidth=.25)
        ax.set_xlabel('Time, s')
        ax.set_ylabel(ylab)
        ax.errorbar(time_line, arr_val_0,
                    yerr = arr_var_0,
                    fmt ='-o', capsize=2, label=lab_0.name,
                    alpha=0.75, color='black')
        if labels_num == '2' or labels_num == '3':
            ax.errorbar(time_line, arr_val_1,
                        yerr = arr_var_1,
                        fmt ='-o', capsize=2, label=lab_1.name,
                        alpha=0.75, color='red')
        if labels_num == '3':
            ax.errorbar(time_line, arr_val_2,
                        yerr = arr_var_2,
                        fmt ='-o', capsize=2, label=lab_2.name,
                        alpha=0.75, color='blue')
        plt.legend()
        plt.title(f'{img.name}, method {stat_method}')
        viewer.window.add_dock_widget(FigureCanvas(mpl_fig), name='Multiple Lab Stat Prof.')


@magic_factory(call_button='Build Profiles',
               saving_path={'mode': 'd'})
def save_df(img:Image, labels:Labels,
            time_scale:float=1.0,
            ΔF_win:int=4,
            Dietrich_win:int=4,
            Dietrich_std:float=1.25,
            save_ROIs_distances:bool=False,
            custom_stim_position:bool=False,
            stim_position:Points=None,
            saving_path:pathlib.Path = os.getcwd()):
    if input is not None:
        if img.data[0].shape != labels.data.shape:
            raise ValueError(f'Frames of the input image and labels should have the same size! Input shape {input_img.shape}, labels shape {input_labels.shape}.')
        input_img = img.data
        input_labels = labels.data
        df_name = img.name + '_' + labels.name
        df_name = df_name.replace('_xform','')
        time_line = np.linspace(0, (input_img.shape[0]-1)*time_scale, \
                                num=input_img.shape[0])
        show_info(f'Input shape {input_img.shape}, labels shape {input_labels.shape}')

        if save_ROIs_distances:
            col_list = ['id', 'lab_id', 'roi', 'dist', 'index', 'time', 'abs_int', 'dF_int', 'dF/F0_int', 'base']
            tip_position_img = np.ones_like(input_img[0], dtype=bool)
            if custom_stim_position:
                try:
                    tip_x, tip_y = int(stim_position.data[0][-2]), int(stim_position.data[0][-1])  # for time series
                    show_info(f'{img.name}: custom stim position {tip_x, tip_y}')
                    tip_position_img[tip_x,tip_y] = False
                except AttributeError:
                    show_warning(f"{img.name}: no stim position, using img center position!")
                    tip_x, tip_y = tip_position_img.shape[0]//2, tip_position_img.shape[1]//2
                    show_info(f'{img.name}: center stim position {tip_x, tip_y}')
                    tip_position_img[tip_x,tip_y] = False
            else:
                tip_x, tip_y = tip_position_img.shape[0]//2, tip_position_img.shape[1]//2
                show_info(f'{img.name}: center stim position {tip_x, tip_y}')
                tip_position_img[tip_x,tip_y] = False
            tip_distance_img = ndi.distance_transform_edt(tip_position_img)
            distance_list = []
            for label_num in np.unique(input_labels)[1:]:
                region_mask = input_labels == label_num
                distance_list.append(round(np.mean(tip_distance_img, where=region_mask)))
        else:
            col_list = ['id', 'lab_id', 'roi', 'index', 'time', 'abs_int', 'dF_int', 'dF/F0_int', 'base']

        start = time.perf_counter()
        # simple baseline calc
        profile_abs = utils.labels_to_profiles(input_label=input_labels,
                                               input_img=input_img)
        profile_dF = utils.delta_prof_simple(profile_abs, mode='ΔF',
                                             win_size=ΔF_win)
        profile_dF_F0 = utils.delta_prof_simple(profile_abs, mode='ΔF/F0',
                                                win_size=ΔF_win)
        # Dietrich baseline calc
        profile_abs_base = utils.delta_prof_pybase(profile_abs, mode='abs',
                                                   win_size=Dietrich_win, stds=Dietrich_std)
        profile_dF_base = utils.delta_prof_pybase(profile_abs, mode='ΔF',
                                                  win_size=Dietrich_win, stds=Dietrich_std)
        profile_dF_F0_base = utils.delta_prof_pybase(profile_abs, mode='ΔF/F0',
                                                     win_size=Dietrich_win, stds=Dietrich_std)
        end = time.perf_counter()
        show_info(f'{img.name}: profiles calculated in {end-start:.2f} s')

        output_df = pd.DataFrame(columns=col_list)
        # simple baseline saving
        for num_ROI in range(profile_abs.shape[0]):
            ROI_abs = np.round(profile_abs[num_ROI], decimals=4)
            ROI_dF = np.round(profile_dF[num_ROI], decimals=4)
            ROI_dF_F0 = np.round(profile_dF_F0[num_ROI], decimals=4)
            dict_ROI = {'id':img.name,
                        'lab_id':labels.name,
                        'roi':num_ROI+1,
                        'index': np.linspace(0, input_img.shape[0], num=input_img.shape[0], dtype=int),
                        'time':time_line,
                        'abs_int':ROI_abs,
                        'dF_int':ROI_dF,
                        'dF/F0_int':ROI_dF_F0,
                        'base':'simple'}
            if save_ROIs_distances:
                dict_ROI['dist'] = distance_list[num_ROI]
            df_ROI = pd.DataFrame(dict_ROI)
            output_df = pd.concat([output_df.astype(df_ROI.dtypes),
                                    df_ROI.astype(output_df.dtypes)],
                                    ignore_index=True)
        # Dietrich baseline saving
        for num_ROI in range(profile_abs.shape[0]):
            ROI_abs = np.round(profile_abs_base[num_ROI], decimals=4)
            ROI_dF = np.round(profile_dF_base[num_ROI], decimals=4)
            ROI_dF_F0 = np.round(profile_dF_F0_base[num_ROI], decimals=4)
            dict_ROI = {'id':img.name,
                        'lab_id':labels.name,
                        'roi':num_ROI+1,
                        'index': np.linspace(0, input_img.shape[0], num=input_img.shape[0], dtype=int),
                        'time':time_line,
                        'abs_int':ROI_abs,
                        'dF_int':ROI_dF,
                        'dF/F0_int':ROI_dF_F0,
                        'base':'dietrich'}
            if save_ROIs_distances:
                dict_ROI['dist'] = distance_list[num_ROI]
            df_ROI = pd.DataFrame(dict_ROI)
            output_df = pd.concat([output_df.astype(df_ROI.dtypes),
                                    df_ROI.astype(output_df.dtypes)],
                                    ignore_index=True)
        output_df.to_csv(os.path.join(saving_path, df_name+'.csv'))


if __name__ == '__main__':
    import napari
    viewer = napari.Viewer()
    viewer = Viewer()

    split_channels_widget = split_channels()
    viewer.window.add_dock_widget(split_channels_widget, name = 'Preprocessing',
                                  area='right')
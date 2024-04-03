from magicgui import magic_factory

import napari
from napari import Viewer
from napari.layers import Image, Labels
from napari.utils.notifications import show_info
from napari.qt.threading import thread_worker

import pathlib
import os

import numpy as np
from scipy import ndimage as ndi
from scipy import stats
from scipy import signal

from skimage import filters
from skimage import morphology
from skimage import measure
from skimage import restoration

import vispy.color

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas

from dipy.align.transforms import AffineTransform2D
from dipy.align.imaffine import AffineRegistration

from domb.utils import masking
from domb.fret.e_fret import e_app


def _red_green():
     """ Red-green colormap

     """
     return vispy.color.Colormap([[0.0, 1.0, 0.0],
                                  [0.0, 0.9, 0.0],
                                  [0.0, 0.85, 0.0],
                                  [0.0, 0.0, 0.0],
                                  [0.85, 0.0, 0.0],
                                  [0.9, 0.0, 0.0],
                                  [1.0, 0.0, 0.0]])


@magic_factory(call_button='Preprocess stack',
               stack_order={"choices": ['TCXY', 'CTXY']},
               correction_method={"choices": ['exp', 'bi_exp']},)
def split_channels(viewer: Viewer, img:Image,
                   stack_order:str='TCXY',
                   median_filter:bool=True, median_kernel:int=3,  #gaussian_blur:bool=True, gaussian_sigma=0.75,
                   background_substraction:bool=True,
                   photobleaching_correction:bool=False,
                   correction_method:str='exp',
                   drop_frames:bool=False,
                   frames_range:list=[0,10]):
    if input is not None:
        def _save_ch(params):
            img = params[0]
            img_name = params[1]
            try: 
                viewer.layers[img_name].data = img
            except KeyError:
                new_image = viewer.add_image(img, name=img_name, colormap='turbo')

        @thread_worker(connect={'yielded':_save_ch})
        def _split_channels():
            def _preprocessing(ch_img):
                if drop_frames:
                    if len(frames_range) == 2:
                        ch_img = ch_img[frames_range[0]:frames_range[-1],:,:]
                    else:
                        raise ValueError('List of indexes should has 2 elements!')
                if median_filter:
                    median_axis = lambda x,k: np.array([ndi.median_filter(f, size=k) for f in x], dtype=x.dtype)
                    ch_img = median_axis(ch_img, median_kernel)
                # if gaussian_blur:
                #     ch_img = filters.gaussian(ch_img, sigma=gaussian_sigma, channel_axis=0)
                #     show_info(f'Img series blured with sigma {gaussian_sigma}')
                if background_substraction:
                    bc_p = lambda x: np.array([f - np.percentile(f, 0.5) for f in x]).clip(min=0).astype(dtype=x.dtype)
                    ch_img = bc_p(ch_img)
                if photobleaching_correction:
                    pb_mask = masking.proc_mask(np.mean(ch_img, axis=0))
                    ch_img,_,r_corr = masking.pb_exp_correction(input_img=ch_img,
                                                                mask=pb_mask,
                                                                method=correction_method)
                    show_info(f'{correction_method} photobleaching correction, r^2={r_corr}')
                return ch_img

            if img.data.ndim == 4:
                show_info(f'{img.name}: Ch. split and preprocessing mode, shape {img.data.shape}')
                if stack_order == 'TCXY':
                    input_img = img.data
                elif stack_order == 'CTXY':
                    input_img = np.moveaxis(img.data,0,1)
                for i in range(0,img.data.shape[1]):
                    show_info(f'{img.name}: Ch. {i} preprocessing')
                    yield (_preprocessing(ch_img=input_img[:,i,:,:]), img.name + f'_ch{i}')
            elif img.data.ndim == 3:
                show_info(f'{img.name}: Image already has 3 dimensions, preprocessing only mode')
                yield (_preprocessing(ch_img=img.data), img.name + '_ch0')
            else:
                raise ValueError('Input image should have 3 or 4 dimensions!')       
        
        _split_channels()


@magic_factory(call_button='Align stack')
def dw_registration(viewer: Viewer, offset_img:Image, reference_img:Image,
                    input_crop:int=25, output_crop:int=10):
    if input is not None:
        if (offset_img.data.ndim == 4) and (reference_img.data.ndim == 3):

            def _save_aligned(img):
                xform_name = offset_img.name+'_xform'
                try: 
                    viewer.layers[xform_name].data = img
                    viewer.layers[xform_name].colormap = 'turbo'
                except KeyError:
                    viewer.add_image(img, name=xform_name, colormap='turbo')

            @thread_worker(connect={'yielded':_save_aligned})
            def _dw_registration():
                offset_series = offset_img.data
                master_img = reference_img.data

                if input_crop != 0:
                    y, x = offset_series.shape[-2:]
                    offset_series = offset_series[:,:,input_crop:y-input_crop,input_crop:x-input_crop]
                    master_img = master_img[:,input_crop:y-input_crop,input_crop:x-input_crop]

                master_img_ref, master_img_offset = master_img[1], master_img[0]
                affreg = AffineRegistration()
                transform = AffineTransform2D()
                affine = affreg.optimize(master_img_ref, master_img_offset,
                                        transform, params0=None)
                master_img_xform = affine.transform(master_img_offset)

                masking.misalign_estimate(master_img_ref, master_img_offset,
                                          title='Master raw', show_img=False, rough_estimate=True)
                masking.misalign_estimate(master_img_ref, master_img_xform,
                                          title='Master xform', show_img=False, rough_estimate=True)
                # masking.misalign_estimate(np.mean(offset_series[:,0,:,:], axis=0),
                #                           np.mean(offset_series[:,-1,:,:], axis=0),
                #                           title='Raw', show_img=False, rough_estimate=True)

                ch0_xform = np.asarray([affine.transform(frame) for frame in offset_series[:,0,:,:]])
                ch2_xform = np.asarray([affine.transform(frame) for frame in offset_series[:,2,:,:]])
                xform_series = np.stack((ch0_xform,
                                         offset_series[:,1,:,:],
                                         ch2_xform,
                                         offset_series[:,3,:,:]),
                                        axis=1)
                if output_crop != 0:
                    yo, xo = xform_series.shape[-2:]
                    xform_series = xform_series[:,:,output_crop:yo-output_crop,output_crop:xo-output_crop]

                # masking.misalign_estimate(np.mean(xform_series[:,0,:,:], axis=0),
                #                           np.mean(xform_series[:,-1,:,:], axis=0),
                #                           title='Xform', show_img=False, rough_estimate=True)
                
                yield xform_series.astype(offset_series.dtype)
                    
            _dw_registration()
        else:
            raise ValueError('Incorrect input image shape!')


@magic_factory(call_button='Split SEP',
               pH_1st_frame={"choices": ['7.3', '6.0']},)
def split_sep(viewer: Viewer, img:Image,
              pH_1st_frame:str='7.3',
              calc_surface_img:bool=False,
              calc_projections:bool=False):
    if input is not None:
        if img.data.ndim == 3:

            def _save_sep(params):
                img = params[0]
                img_name = params[1]
                cmap_rg = False
                if len(params) == 3:
                    cmap_rg = params[2]
                try: 
                    viewer.layers[img_name].data = img
                except KeyError:
                    new_image = viewer.add_image(img, name=img_name, colormap='turbo')
                    if cmap_rg:
                        new_image.colormap = 'red-green', _red_green()
                    else:
                        new_image.colormap = 'turbo'

            @thread_worker(connect={'yielded':_save_sep})
            def _split_sep():
                sep_img = img.data.astype(float)

                if pH_1st_frame == '7.3':
                    total_start_i, intra_start_i = 0, 1
                elif pH_1st_frame == '6.0':
                    total_start_i, intra_start_i = 1, 0

                total_img = sep_img[total_start_i::2,:,:]  # 0
                intra_img = sep_img[intra_start_i::2,:,:]  # 1

                total_name = img.name + '_total'
                intra_name = img.name + '_intra'

                yield (total_img, total_name)
                yield (intra_img, intra_name)

                if calc_projections:
                    projections_diff = lambda x: np.max(x, axis=0) - np.mean(x, axis=0)
                    yield (projections_diff(total_img),
                           img.name + '_total-projection',
                           True)
                    yield (projections_diff(intra_img),
                           img.name + '_intra-projection',
                           True)
                    yield (np.max(intra_img, axis=0),
                           img.name + '_intra-mip')

                if calc_surface_img:
                    surface_img = total_img - intra_img
                    yield (surface_img,
                           img.name + '_surface')
                    if calc_projections:
                        yield (projections_diff(surface_img),
                               img.name + '_surface-projection',
                               True)
            
            _split_sep()
        else:
            raise ValueError('The input image should have 3 dimensions!')


@magic_factory(call_button='Calc E-FRET')
def e_app_calc(viewer: Viewer, DD_img:Image, DA_img:Image, AA_img:Image,
          a:float=0.122, d:float=0.794, G:float=3.6,
          Eapp_correction:bool=False):
    if input is not None:
        if (DD_img.data.ndim == 3) and (DA_img.data.ndim == 3) and (AA_img.data.ndim == 3):

            def _save_e_app(params):
                img = params[0]
                img_name = params[1]
                try: 
                    viewer.layers[img_name].data = img
                except KeyError:
                    viewer.add_image(img, name=img_name, colormap='turbo')

            @thread_worker(connect={'yielded':_save_e_app})
            def _e_app_calc():
                e_fret_img = e_app.Eapp(dd_img=DD_img.data, da_img=DA_img.data, aa_img=AA_img.data,
                                        abcd_list=[a,0,0,d], G_val=G,
                                        mask=masking.proc_mask(np.mean(AA_img.data, axis=0)))
                if Eapp_correction:
                    output_fret_img = e_fret_img.Ecorr_img
                    output_suffix = '_Ecorr'
                else:
                    output_fret_img = e_fret_img.Eapp_img
                    output_suffix = '_Eapp'
                yield (output_fret_img, AA_img.name + output_suffix)

            _e_app_calc()
        else:
            raise ValueError('Incorrect input image shape!')


@magic_factory(call_button='Calc Red-Green')
def der_series(viewer: Viewer, img:Image,
               left_frames:int=2, space_frames:int=2, right_frames:int=2,
               normalize_by_int:bool=False):
    if input is not None:
        if img.data.ndim != 3:
            raise ValueError('The input image should have 3 dimensions!')
        img_name = img.name + '_red-green'

        def _save_rg_img(img):
            try: 
                viewer.layers[img_name].data = img
            except KeyError:
                c_lim = np.max(np.abs(img)) * 0.75
                new_image = viewer.add_image(img, name=img_name, contrast_limits=[-c_lim, c_lim])
                new_image.colormap = 'red-green', _red_green()

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
            yield der_img

        _der_series()


@magic_factory(call_button='Build Up Mask',
               insertion_threshold={"widget_type": "FloatSlider", 'max': 5},)  # insertions_threshold={'widget_type': 'FloatSlider', 'max': 1}
def up_mask_calc(viewer: Viewer, img:Image, detection_img_index:int=2,
                 insertion_threshold:float=0.2,
                 opening_footprint:int=0,
                 save_mask:bool=False):
    if input is not None:
        if img.data.ndim != 3:
            raise ValueError('The input image should have 3 dimensions!')
        labels_name = img.name + '_up-labels'
        mask_name = img.name + '_up-mask'

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
            detection_img = input_img[detection_img_index]
            
            up_mask = detection_img >= np.max(np.abs(detection_img)) * (insertion_threshold/100.0)
            up_mask = morphology.erosion(up_mask, footprint=morphology.disk(2))
            up_mask = morphology.dilation(up_mask, footprint=morphology.disk(1))
            up_mask = ndi.binary_fill_holes(up_mask)
            up_mask = up_mask.astype(int)

            if opening_footprint != 0:
                up_mask = morphology.opening(up_mask, footprint=morphology.disk(opening_footprint))
                up_mask = morphology.dilation(up_mask, footprint=morphology.disk(1))

            up_labels = measure.label(up_mask)
            show_info(f'{img.name}: detected {np.max(up_labels)} labels')

            yield (up_labels, labels_name)
            if save_mask:
                yield (up_mask, mask_name)

        _up_mask_calc()                


@magic_factory(call_button='Build Mask',
               masking_mode={"choices": ['up', 'down']},)
def mask_calc(viewer: Viewer, img:Image, detection_frame_index:int=2,
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
            detection_img = input_img[detection_frame_index]

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
               saving_path={'mode': 'd'})
def labels_profile_line(viewer: Viewer, img:Image, labels:Labels,
                        time_scale:float=5.0,
                        absolute_intensity:bool=True,
                        ΔF_win:int=5,
                        ΔF_aplitude_lim:list=[10.0, 10.0],
                        profiles_crop:bool=False,
                        profiles_range:list=[0,10],
                        save_data_frame:bool=False,
                        saving_path:pathlib.Path = os.getcwd()):
    if input is not None:
        input_img = img.data
        input_labels = labels.data
        df_name = img.name + '_lab_prof'

        profile_dF, profile_raw = masking.label_prof_arr(input_label=input_labels,
                                                         input_img_series=input_img,
                                                         f0_win=ΔF_win)
        time_line = np.linspace(0, input_img.shape[0]*time_scale, \
                                num=input_img.shape[0])

        if absolute_intensity:
            profile_to_plot = profile_raw
            ylab = 'Intensity, a.u.'
            df_name = df_name + '_absolute'
        else:
            profile_to_plot = profile_dF
            ylab = 'ΔF/F0'
            df_name = df_name + '_ΔF'

        if save_data_frame:
            import pandas as pd
            output_df = pd.DataFrame(columns=['id','roi','int', 'index', 'time'])
            for num_ROI in range(profile_to_plot.shape[0]):
                profile_ROI = profile_to_plot[num_ROI]
                df_ROI = pd.DataFrame({'id':np.full(profile_ROI.shape[0], img.name),
                                       'roi':np.full(profile_ROI.shape[0], num_ROI+1),
                                       'int':profile_ROI,
                                       'index': np.linspace(0, input_img.shape[0], num=input_img.shape[0], dtype=int),
                                       'time':time_line})
                output_df = pd.concat([output_df.astype(df_ROI.dtypes),
                                       df_ROI.astype(output_df.dtypes)],
                                      ignore_index=True)
            output_df.to_csv(os.path.join(saving_path, df_name+'.csv'))

        # plotting
        if profiles_crop:
            profile_to_plot = profile_to_plot[:,profiles_range[0]:profiles_range[1]]
            time_line = time_line[profiles_range[0]:profiles_range[1]]

        lab_colors = labels.get_color([prop['label'] for prop in measure.regionprops(label_image=input_labels)])

        print(profile_to_plot.shape, time_line.shape, lab_colors.shape)

        mpl_fig = plt.figure()
        ax = mpl_fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for num_ROI, color in enumerate(lab_colors):
            profile_ROI = profile_to_plot[num_ROI]
            print(profile_ROI.shape, time_line.shape, color.shape)
            if absolute_intensity:
                ax.plot(time_line, profile_ROI,
                         alpha=0.45, marker='o', color=color)
                plt_title = f'{img.name} absolute intensity profiles, labels {labels.name}'
            elif (profile_ROI.min() > -ΔF_aplitude_lim[0]) | (profile_ROI.max() < ΔF_aplitude_lim[1]):
                ax.plot(time_line, profile_ROI,
                         alpha=0.45, marker='o', color=color)
                plt_title = f'{img.name} ΔF/F0 profiles (lim -{ΔF_aplitude_lim[0]}, {ΔF_aplitude_lim[1]}), labels {labels.name}'
            else:
                continue
        ax.grid(color='grey', linewidth=.25)
        ax.set_xlabel('Time, s')
        ax.set_ylabel(ylab)
        plt.title(plt_title)
        viewer.window.add_dock_widget(FigureCanvas(mpl_fig), name=f'{img.name} Profile')


@magic_factory(call_button='Build Profile',
               stat_method={"choices": ['se', 'iqr', 'ci']},)
def labels_profile_stat(viewer: Viewer, img_0:Image, img_1:Image, labels:Labels,
                        raw_intensity:bool=False,
                        two_profiles:bool=False, 
                        time_scale:float=5.0,
                        ΔF_win:int=5,
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

        # processing
        input_img_0 = img_0.data
        input_labels = labels.data
        
        profile_dF_0, profile_raw_0 = masking.label_prof_arr(input_label=input_labels,
                                                             input_img_series=input_img_0,
                                                             f0_win=ΔF_win)
        if raw_intensity:
            selected_profile_0  = profile_raw_0
        else:
            selected_profile_0  = profile_dF_0
        arr_val_0, arr_var_0 = stat_dict[stat_method](selected_profile_0)

        if two_profiles:
            input_img_1 = img_1.data
            profile_dF_1, profile_raw_1 = masking.label_prof_arr(input_label=input_labels,
                                                                 input_img_series=input_img_1,
                                                                 f0_win=ΔF_win)
            if raw_intensity:
                selected_profile_1  = profile_raw_1
            else:
                selected_profile_1  = profile_dF_1
            arr_val_1, arr_var_1 = stat_dict[stat_method](selected_profile_1)

        # plotting
        time_line = np.linspace(0, input_img_0.shape[0]*time_scale, \
                                num=input_img_0.shape[0])
        
        mpl_fig = plt.figure()
        ax = mpl_fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)        
        if two_profiles:
            ax.errorbar(time_line, arr_val_0,
                        yerr = arr_var_0,
                        fmt ='-o', capsize=2, label=img_0.name, alpha=0.75)
            ax.errorbar(time_line, arr_val_1,
                        yerr = arr_var_1,
                        fmt ='-o', capsize=2, label=img_1.name, alpha=0.75)
            ax.grid(color='grey', linewidth=.25)
            ax.set_xlabel('Time, s')
            ax.set_ylabel('ΔF/F0')
            plt.legend()
            plt.title(f'Two labels profiles (method {stat_method})')
            viewer.window.add_dock_widget(FigureCanvas(mpl_fig), name='Two Profiles')
        else:
            ax.errorbar(time_line, arr_val_0,
                        yerr = arr_var_0,
                        fmt ='-o', capsize=2)
            ax.grid(color='grey', linewidth=.25)
            ax.set_xlabel('Time, s')
            ax.set_ylabel('ΔF/F0')
            plt.title(f'{img_0.name} labels profile (method {stat_method})')
            viewer.window.add_dock_widget(FigureCanvas(mpl_fig), name=f'{img_0.name} Profile')


if __name__ == '__main__':

    import napari
    viewer = napari.Viewer()
    viewer = Viewer()

    split_channels_widget = split_channels()
    viewer.window.add_dock_widget(split_channels_widget, name = 'Preprocessing',
                                area='right')
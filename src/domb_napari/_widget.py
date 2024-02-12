from magicgui import magic_factory

import napari
from napari import Viewer
from napari.layers import Image, Labels
from napari.utils.notifications import show_info

import pathlib
import os

import numpy as np
from scipy import ndimage as ndi
from scipy import stats

from skimage import filters
from skimage import morphology
from skimage import measure

import vispy.color

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas

from domb.utils import masking


def _save_img(viewer: Viewer, img:np.ndarray, img_name:str):
    try: 
        viewer.layers[img_name].data = img
    except KeyError:
        viewer.add_image(img, name=img_name, colormap='turbo')


@magic_factory(call_button='Preprocess Image',
               correction_method={"choices": ['exp', 'bi_exp']},)
def split_channels(viewer: Viewer, img:Image,
                   gaussian_blur:bool=False, gaussian_sigma=0.5,
                   photobleaching_correction:bool=False,
                   correction_method:str='exp'):
    if input is not None:
        series_dim = img.data.ndim
        if series_dim == 4:
            preprocessing_info = f'{img.name}: Ch. split and preprocessing mode'
            print(preprocessing_info)
            show_info(preprocessing_info)

            for i in range(img.data.shape[1]):
                ch_name = img.name + f'_ch{i}'
                ch_img = img.data[:,i,:,:].astype(float)
                corr_img = np.mean(ch_img, axis=0)
                corr_mask = corr_img > filters.threshold_otsu(corr_img)
                corr_mask = morphology.dilation(corr_mask, footprint=morphology.disk(10))
                if gaussian_blur:
                    ch_img = filters.gaussian(ch_img, sigma=gaussian_sigma, channel_axis=0)
                    print(f'{ch_name}: Img series blured with sigma {gaussian_sigma}')
                if photobleaching_correction:
                    ch_img,_,r_corr = masking.pb_exp_correction(input_img=ch_img,
                                                                mask=corr_mask,
                                                                method=correction_method)
                    print(f'{ch_name}: Photobleaching correction, r^2={r_corr}')

                _save_img(viewer=viewer, img=ch_img, img_name=ch_name)
        elif series_dim == 3:
            preprocessing_info = f'{img.name}: Image already has 3 dimensions, preprocessing only mode'
            print(preprocessing_info)
            show_info(preprocessing_info)
            
            ch_name = img.name + '_ch0'
            ch_img = img.data
            if gaussian_blur:
                ch_img = filters.gaussian(ch_img, sigma=gaussian_sigma, channel_axis=0)
                print(f'{ch_name}: Img series blured with sigma {gaussian_sigma}')
            if photobleaching_correction:
                corr_img = np.mean(ch_img, axis=0)
                corr_mask = corr_img > filters.threshold_otsu(corr_img)
                ch_img,_,r_corr = masking.pb_exp_correction(input_img=ch_img, mask=corr_mask)
                print(f'{ch_name}: Photobleaching correction, r^2={r_corr}')
            _save_img(viewer=viewer, img=ch_img, img_name=ch_name)
        else:
            raise ValueError('The input image should have 4 dimensions!')
        

@magic_factory(call_button='Split SEP')
def split_sep(viewer: Viewer, img:Image,
              calc_surface_img:bool=False):
    if input is not None:
        series_dim = img.data.ndim
        if series_dim == 3:
            sep_img = img.data.astype(float)
            total_img = sep_img[0::2,:,:]
            intra_img = sep_img[1::2,:,:]

            total_name = img.name + '_total'
            intra_name = img.name + '_intra'

            _save_img(viewer=viewer, img=total_img, img_name=total_name)
            _save_img(viewer=viewer, img=intra_img, img_name=intra_name)

            if calc_surface_img:
                surface_img = total_img - intra_img
                surface_name = img.name + '_surface'
                _save_img(viewer=viewer, img=surface_img, img_name=surface_name)
        else:
            raise ValueError('The input image should have 3 dimensions!')


@magic_factory(call_button='Calc Red-Green',
               insertion_threshold={"widget_type": "FloatSlider", 'max': 1},)
def der_series(viewer: Viewer, img:Image,
               left_frames:int=2, space_frames:int=2, right_frames:int=2,
               save_mask_series:bool=False,
               insertion_threshold:float=0.2):
    if input is not None:
        if img.data.ndim != 3:
            raise ValueError('The input image should have 3 dimensions!')
        ref_img = img.data

        der_img = []
        mask_img = []
        for i in range(ref_img.shape[0]-(left_frames+right_frames+space_frames)):
            img_base = np.mean(ref_img[i:i+left_frames+1], axis=0)
            img_stim = np.mean(ref_img[i+left_frames+right_frames:i+left_frames+right_frames+space_frames+1], axis=0)
            
            img_diff = img_stim-img_base
            img_mask = img_diff >= np.max(np.abs(img_diff)) * insertion_threshold

            der_img.append(img_diff)
            mask_img.append(img_mask)

        der_img = np.asarray(der_img, dtype=float)
        mask_img = np.asarray(mask_img, dtype=float)
        der_contrast_lim = np.max(np.abs(der_img)) * 0.75
        print(f'Derivate series shape: {der_img.shape}')

        red_green_cmap = vispy.color.Colormap([[0.0, 1.0, 0.0],
                                               [0.0, 0.9, 0.0],
                                               [0.0, 0.5, 0.0],
                                               [0.0, 0.0, 0.0],
                                               [0.5, 0.0, 0.0],
                                               [0.9, 0.0, 0.0],
                                               [1.0, 0.0, 0.0]])

        img_name = img.name + '_red-green'
        try:
            viewer.layers[img_name].data = der_img
        except KeyError:
            der_layer = viewer.add_image(der_img, name=img_name,
                                         contrast_limits=[-der_contrast_lim, der_contrast_lim])
            der_layer.colormap = 'red-green', red_green_cmap

        if save_mask_series:
            mask_name = img.name + '_red-mask'
            _save_img(viewer=viewer, img=mask_img, img_name=mask_name)


@magic_factory(call_button='Build Up Mask',
               insertion_threshold={"widget_type": "FloatSlider", 'max': 1},)  # insertions_threshold={'widget_type': 'FloatSlider', 'max': 1}
def up_mask_calc(viewer: Viewer, img:Image, detection_img_index:int=2,
                 insertion_threshold:float=0.2,
                 save_mask:bool=False):
    if input is not None:
        if img.data.ndim != 3:
            raise ValueError('The input image should have 2 dimensions!')
        input_img = img.data
        detection_img = input_img[detection_img_index]
        
        up_mask = detection_img >= np.max(np.abs(detection_img)) * insertion_threshold
        up_mask = morphology.opening(up_mask, footprint=morphology.disk(1))
        up_mask = ndi.binary_fill_holes(up_mask)
        up_mask = up_mask.astype(int)
        up_labels = measure.label(up_mask)
        print(f'Up mask shape: {up_mask.shape}, detected {np.max(up_labels)} labels')
            
        labels_name = img.name + '_up-labels'
        try:
            viewer.layers[labels_name].data = up_labels
        except KeyError:
            viewer.add_labels(up_labels, name=labels_name, opacity=0.6)

        if save_mask:
            mask_name = img.name + '_up-mask'
            try:
                viewer.layers[mask_name].data = up_mask
            except KeyError:
                viewer.add_labels(up_mask, name=mask_name,
                                num_colors=1, color={1:(255,0,0,255)},
                                opacity=0.6)


@magic_factory(call_button='Build Mask',
               masking_mode={"choices": ['up', 'down']},)
def mask_calc(viewer: Viewer, img:Image, detection_frame_index:int=2,
              masking_mode:str='up',
              up_threshold:float=0.2,
              down_threshold:float=-0.1):
    if input is not None:
        if img.data.ndim != 3:
            raise ValueError('The input image should have 3 dimensions!')
        input_img = img.data
        detection_img = input_img[detection_frame_index]

        if masking_mode == 'up':
            mask = detection_img >= np.max(np.abs(detection_img)) * up_threshold
            labels_name = img.name + '_up-labels'
            mask_name = img.name + '_up-mask'
        elif masking_mode == 'down':        
            mask = detection_img <= np.max(np.abs(detection_img)) * down_threshold
            labels_name = img.name + '_down-labels'
            mask_name = img.name + '_down-mask'

        mask = morphology.opening(mask, footprint=morphology.disk(3))
        mask = ndi.binary_fill_holes(mask)
        mask = mask.astype(int)
        labels = measure.label(mask)

        mask_info = f'{img.name}: detected {np.max(labels)} labels'
        print(mask_info)
        show_info(mask_info)
            
        try:
            viewer.layers[labels_name].data = labels
        except KeyError:
            viewer.add_labels(labels, name=labels_name, opacity=0.6)

        if save_mask:
            try:
                viewer.layers[mask_name].data = mask
            except KeyError:
                viewer.add_labels(mask, name=mask_name,
                                num_colors=1, color={1:(255,0,0,255)},
                                opacity=0.6)


@magic_factory(call_button='Build Profiles',
               saving_path={'mode': 'd'})
def labels_profile_line(viewer: Viewer, img:Image, labels:Labels,
                        time_scale:float=2.0,
                        raw_intensity:bool=True,
                        ΔF_win:int=5,
                        min_amplitude:float=0.0,
                        max_amplitude:float=5.0,
                        frame_crop:bool=False,
                        start_frame:int=0,
                        stop_frame:int=10,
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

        if frame_crop:
            profile_dF = profile_dF[start_frame:stop_frame] 
            profile_raw = profile_raw[start_frame:stop_frame]
            time_line = time_line[start_frame:stop_frame]

        if raw_intensity:
            profile_to_plot = profile_raw
            ylab = 'Intensity, a.u.'
            df_name = df_name + '_raw'
        else:
            profile_to_plot = profile_dF
            ylab = 'ΔF/F0'
            df_name = df_name + '_dF'

        if save_data_frame:
            import pandas as pd
            output_df = pd.DataFrame(columns=['id','roi','int', 'time'])
            for num_ROI in range(profile_to_plot.shape[0]):
                profile_ROI = profile_to_plot[num_ROI]
                df_ROI = pd.DataFrame({'id':np.full(profile_ROI.shape[0], img.name),
                                       'roi':np.full(profile_ROI.shape[0], num_ROI+1),
                                       'int':profile_ROI,
                                       'time':time_line})
                output_df = pd.concat([output_df.astype(df_ROI.dtypes),
                                       df_ROI.astype(output_df.dtypes)],
                                      ignore_index=True)
            output_df.to_csv(os.path.join(saving_path, df_name+'.csv'))

        # plotting        
        mpl_fig = plt.figure()
        ax = mpl_fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for num_ROI in range(profile_to_plot.shape[0]):
            profile_ROI = profile_to_plot[num_ROI]
            if raw_intensity:
                ax.plot(time_line, profile_ROI,
                         alpha=0.35, marker='o')
                plt_title = f'{img.name} individual labels raw profiles'
            elif (profile_ROI.max() > min_amplitude) | (profile_ROI.max() < max_amplitude):
                ax.plot(time_line, profile_ROI,
                         alpha=0.35, marker='o')
                plt_title = f'{img.name} individual labels profiles (min={min_amplitude}, max={max_amplitude})'
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
                        two_profiles:bool=False, 
                        time_scale:float=2,
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
        arr_val_0, arr_var_0 = stat_dict[stat_method](profile_dF_0)

        if two_profiles:
            input_img_1 = img_1.data
            profile_dF_1, profile_raw_1 = masking.label_prof_arr(input_label=input_labels,
                                                                 input_img_series=input_img_1,
                                                                 f0_win=ΔF_win)
            arr_val_1, arr_var_1 = stat_dict[stat_method](profile_dF_1)

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
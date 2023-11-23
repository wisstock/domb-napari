from magicgui import magicgui
from magicgui import magic_factory
from magicgui.widgets import FunctionGui

import napari
from napari import Viewer
from napari.types import LabelsData, ImageData
from napari.layers import Image, Points, Labels

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


@magic_factory(call_button='Preprocess Image',
               correction_method={"choices": ['exp', 'bi_exp']},)
def split_channels(viewer: Viewer, img:Image,
                   gaussian_blur:bool=False, gaussian_sigma=0.75,
                   photobleaching_correction:bool=False,
                   correction_method:str='exp'):
    if input is not None:
        series_dim = img.data.ndim
        if series_dim == 4:
            print(f'{img.name}: Split and preprocessing mode')
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
                try: 
                    # if the layer exists, update the data
                    viewer.layers[ch_name].data = ch_img
                except KeyError:
                    # otherwise add it to the viewer
                    viewer.add_image(ch_img, name=ch_name, colormap='turbo')
        elif series_dim == 3:
            print(f'{img.name}: Image already has 3 dimensions, preprocessing only mode')
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
            try: 
                viewer.layers[ch_name].data = ch_img
            except KeyError:
                viewer.add_image(ch_img, name=ch_name, colormap='turbo')
        else:
            raise ValueError('The input image should have 4 dimensions!')
        

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
            img_base = np.mean(ref_img[i:i+left_frames], axis=0)
            img_stim = np.mean(ref_img[i+left_frames+right_frames:i+left_frames+right_frames+space_frames], axis=0)
            
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
            try:
                viewer.layers[mask_name].data = mask_img
            except KeyError:
                viewer.add_image(mask_img, name=mask_name, colormap='red')


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
                

# @magic_factory(call_button='Build Down Mask',
#                insertion_threshold={"widget_type": "FloatSlider", 'max': 0, 'min': -0.75},)  # insertions_threshold={'widget_type': 'FloatSlider', 'max': 1}
# def down_mask_calc(viewer: Viewer, img:Image, detection_img_index:int=2,
#                    insertion_threshold:float=-0.2,
#                    save_mask:bool=False):
#     if input is not None:
#         if img.data.ndim != 3:
#             raise ValueError('The input image should have 2 dimensions!')
#         input_img = img.data
#         detection_img = input_img[detection_img_index]
        
#         up_mask = detection_img >= np.max(np.abs(detection_img)) * insertion_threshold
#         up_mask = morphology.opening(up_mask, footprint=morphology.disk(1))
#         up_mask = ndi.binary_fill_holes(up_mask)
#         up_mask = up_mask.astype(int)
#         up_labels = measure.label(up_mask)
#         print(f'Up mask shape: {up_mask.shape}, detected {np.max(up_labels)} labels')
            
#         labels_name = img.name + '_down-labels'
#         try:
#             viewer.layers[labels_name].data = up_labels
#         except KeyError:
#             viewer.add_labels(up_labels, name=labels_name, opacity=0.6)

#         if save_mask:
#             mask_name = img.name + '_down-mask'
#             try:
#                 viewer.layers[mask_name].data = up_mask
#             except KeyError:
#                 viewer.add_labels(up_mask, name=mask_name,
#                                 num_colors=1, color={1:(255,0,0,255)},
#                                 opacity=0.6)


@magic_factory(call_button='Build Set Labels Profiles',
               saving_path={'mode': 'd'})
def labels_profile_line(viewer: Viewer, img:Image, labels:Labels,
                        time_scale:float=2.0,
                        raw_intensity:bool=False,
                        ΔF_win:int=5,
                        min_amplitude:float=0.0,
                        max_amplitude:float=5.0,
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


@magic_factory(call_button='Build Labels Profile',
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
                                area='right')  # add_vertical_stretch=True

    der_series_widget = der_series()
    viewer.window.add_dock_widget(der_series_widget, name = 'Red-Green Series',
                                area='right')

    up_mask_calc_widget = up_mask_calc()
    viewer.window.add_dock_widget(up_mask_calc_widget, name='Up Mask',
                                area='right')

    labels_profile_widget = labels_profile_stat()
    viewer.window.add_dock_widget(labels_profile_widget, name='Labels Profile',
                                  area='right')
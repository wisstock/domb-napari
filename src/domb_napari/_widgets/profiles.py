""" Widgets: ROI profiles plotting and data frame saving.

"""
import os
import pathlib
import time

import numpy as np
from magicgui import magic_factory

from napari import Viewer
from napari.layers import Image, Labels, Points
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning

from ..core import profiles
from ._common import start_worker, new_figure, style_axis, show_figure


_MODES = {'ΔF': 'dF', 'ΔF/F0': 'dF/F0', 'abs.': 'abs'}
_YLAB = {'abs.': 'a.u.', 'ΔF': 'ΔF', 'ΔF/F0': 'ΔF/F0'}
_COLORS = ['black', 'red', 'blue']
_VALUES_OPTION = {"choices": ['ΔF', 'ΔF/F0', 'abs.']}
_STAT_OPTION = {"choices": ['se', 'iqr', 'ci']}


def _check_pair(img_data, labels_data):
    if img_data.ndim != 3:
        raise ValueError('The input image should have 3 dimensions!')
    if img_data.shape[1:] != labels_data.shape:
        raise ValueError(f'Frames of the input image and labels should have the same size! Input shape {img_data.shape}, labels shape {labels_data.shape}.')


def _profiles_for(img_data, labels_data, values_mode, use_simple_baseline, win, stds):
    """ Profiles of ROIs in the selected mode [lab,t].

    """
    prof_abs = profiles.labels_to_profiles(labels_data, img_data)
    if prof_abs.shape[0] == 0:
        raise ValueError('Labels are empty!')
    if use_simple_baseline:
        return profiles.delta_prof_simple(prof_abs, win_size=win, mode=_MODES[values_mode])
    return profiles.delta_prof_pybase(prof_abs, win_size=win, stds=stds, mode=_MODES[values_mode])


def _baseline_info(name, use_simple_baseline, win, stds):
    if use_simple_baseline:
        show_info(f'{name}: simple baseline estimation, win size={win}')
    else:
        show_info(f'{name}: Dietrich baseline estimation, std={stds}, win size={win}')


def _new_axis(values_mode):
    mpl_fig = new_figure()
    ax = mpl_fig.add_subplot(111)
    if values_mode in ('ΔF/F0', 'ΔF'):
        ax.axhline(y=0.0, color='k', linestyle='--')
    style_axis(ax, 'Time, s', _YLAB[values_mode])
    return mpl_fig, ax


@magic_factory(call_button='Build Profiles',
               values_mode=_VALUES_OPTION,)
def labels_profile_line(viewer: Viewer, img:Image, labels:Labels,
                        time_scale:float=1.0,
                        values_mode:str='ΔF',
                        use_simple_baseline:bool=True,
                        ΔF_win:int=4,
                        Dietrich_std:float=1.25):
    img_name, img_data = img.name, img.data
    labels_name, labels_data = labels.name, np.asarray(labels.data)
    _check_pair(img_data, labels_data)
    _baseline_info(img_name, use_simple_baseline, ΔF_win, Dietrich_std)
    lab_colors = labels.get_color(list(profiles.label_ids(labels_data)))

    def _plot(profile_to_plot):
        time_line = np.arange(profile_to_plot.shape[1]) * time_scale
        mpl_fig, ax = _new_axis(values_mode)
        for profile_ROI, color in zip(profile_to_plot, lab_colors):
            ax.plot(time_line, profile_ROI, alpha=0.45, marker='o', color=color)
        ax.set_title(f'{img_name} ROIs profiles, {values_mode} mode, labels: {labels_name}, simple baseline: {use_simple_baseline}')
        show_figure(viewer, mpl_fig, 'ROIs Prof.')

    @thread_worker
    def _calc():
        start = time.perf_counter()
        output = _profiles_for(img_data, labels_data, values_mode,
                               use_simple_baseline, ΔF_win, Dietrich_std)
        show_info(f'{img_name}: profiles calculated in {time.perf_counter()-start:.2f} s')
        return output

    start_worker('ROIs profiles', _calc(), on_return=_plot)


def _stat_profiles(viewer, key, title, dock_name, sets, time_scale, values_mode,
                   use_simple_baseline, win, stds, stat_method):
    """ Shared implementation of the statistical profiles widgets.

    sets: list of (legend, image data, labels data)

    """
    for _, img_data, labels_data in sets:
        _check_pair(img_data, labels_data)

    def _plot(stat_list):
        mpl_fig, ax = _new_axis(values_mode)
        for (legend, _, _), (center, spread), color in zip(sets, stat_list, _COLORS):
            time_line = np.arange(center.shape[0]) * time_scale
            ax.errorbar(time_line, center, yerr=spread, fmt='-o', capsize=2,
                        label=legend, alpha=0.75, color=color)
        ax.legend()
        ax.set_title(f'{title}, method {stat_method}, simple baseline: {use_simple_baseline}')
        show_figure(viewer, mpl_fig, dock_name)

    @thread_worker
    def _calc():
        start = time.perf_counter()
        stat_list = [profiles.profile_stats(_profiles_for(img_data, labels_data, values_mode,
                                                          use_simple_baseline, win, stds),
                                            method=stat_method)
                     for _, img_data, labels_data in sets]
        show_info(f'Profiles calculated in {time.perf_counter()-start:.2f} s')
        return stat_list

    start_worker(key, _calc(), on_return=_plot)


@magic_factory(call_button='Build Profiles',
               profiles_num={"choices": ['1', '2', '3']},
               values_mode=_VALUES_OPTION,
               stat_method=_STAT_OPTION,)
def labels_multi_profile_stat(viewer: Viewer, img_0:Image, img_1:Image=None, img_2:Image=None,
                              lab:Labels=None,
                              profiles_num:str='1',
                              time_scale:float=1.0,
                              values_mode:str='ΔF',
                              use_simple_baseline:bool=True,
                              ΔF_win:int=4,
                              Dietrich_std:float=1.25,
                              stat_method:str='se'):
    imgs = [img_0, img_1, img_2][:int(profiles_num)]
    if lab is None or any(img is None for img in imgs):
        raise ValueError(f'Labels and {profiles_num} image(s) are required!')
    _baseline_info(lab.name, use_simple_baseline, ΔF_win, Dietrich_std)
    lab_data = np.asarray(lab.data)
    _stat_profiles(viewer, 'Multiple img stat profiles', lab.name, 'Multiple Img Stat Prof.',
                   [(img.name, img.data, lab_data) for img in imgs],
                   time_scale, values_mode, use_simple_baseline, ΔF_win, Dietrich_std, stat_method)


@magic_factory(call_button='Build Profiles',
               labels_num={"choices": ['1', '2', '3']},
               values_mode=_VALUES_OPTION,
               stat_method=_STAT_OPTION,)
def multi_labels_profile_stat(viewer: Viewer, img:Image,
                              lab_0:Labels, lab_1:Labels=None, lab_2:Labels=None,
                              time_scale:float=1.0,
                              labels_num:str='1',
                              values_mode:str='ΔF',
                              use_simple_baseline:bool=True,
                              ΔF_win:int=4,
                              Dietrich_std:float=1.25,
                              stat_method:str='se'):
    labs = [lab_0, lab_1, lab_2][:int(labels_num)]
    if img is None or any(lab is None for lab in labs):
        raise ValueError(f'Image and {labels_num} labels layer(s) are required!')
    _baseline_info(img.name, use_simple_baseline, ΔF_win, Dietrich_std)
    _stat_profiles(viewer, 'Multiple labels stat profiles', img.name, 'Multiple Lab Stat Prof.',
                   [(lab.name, img.data, np.asarray(lab.data)) for lab in labs],
                   time_scale, values_mode, use_simple_baseline, ΔF_win, Dietrich_std, stat_method)


@magic_factory(call_button='Save Data',
               saving_path={'mode': 'd'})
def save_df(img:Image, labels:Labels,
            time_scale:float=1.0,
            ΔF_win:int=4,
            Dietrich_win:int=4,
            Dietrich_std:float=1.25,
            save_ROIs_distances:bool=False,
            custom_stim_position:bool=False,
            stim_position:Points=None,
            saving_path:pathlib.Path = pathlib.Path.home()):
    img_name, img_data = img.name, img.data
    labels_name, labels_data = labels.name, np.asarray(labels.data)
    _check_pair(img_data, labels_data)
    df_name = (img_name + '_' + labels_name).replace('_algn','')
    show_info(f'Input shape {img_data.shape}, labels shape {labels_data.shape}')

    position = None
    if save_ROIs_distances:
        if custom_stim_position and stim_position is not None and len(stim_position.data) > 0:
            position = (int(stim_position.data[0][-2]), int(stim_position.data[0][-1]))  # for time series
            show_info(f'{img_name}: custom stim position {position}')
        else:
            if custom_stim_position:
                show_warning(f"{img_name}: no stim position, using img center position!")
            position = (labels_data.shape[0]//2, labels_data.shape[1]//2)
            show_info(f'{img_name}: center stim position {position}')

    def _saved(df_path):
        show_info(f'{img_name}: data frame saved to {df_path}')

    @thread_worker
    def _save_df():
        start = time.perf_counter()
        profile_abs = profiles.labels_to_profiles(labels_data, img_data)
        distances = profiles.stim_distances(labels_data, position) if save_ROIs_distances else None
        output_df = profiles.profiles_table(profile_abs, profiles.label_ids(labels_data),
                                            time_scale=time_scale,
                                            img_name=img_name, labels_name=labels_name,
                                            dF_win=ΔF_win,
                                            dietrich_win=Dietrich_win, dietrich_std=Dietrich_std,
                                            distances=distances)
        show_info(f'{img_name}: profiles calculated in {time.perf_counter()-start:.2f}s')
        df_path = os.path.join(saving_path, df_name+'.csv')
        output_df.to_csv(df_path)
        return df_path

    start_worker('Save Data Frame', _save_df(), on_return=_saved)

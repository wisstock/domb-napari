""" Widgets: red-green and ΔF series.

"""
import time

import numpy as np
from magicgui import magic_factory

from napari import Viewer
from napari.layers import Image
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info

from ..core import series
from .. import _cmaps
from ._common import start_worker, robust_limit


def _add_or_update_diverging(viewer, img, img_name, cmap):
    try:
        viewer.layers[img_name].data = img
    except KeyError:
        c_lim = robust_limit(img)
        new_image = viewer.add_image(img, name=img_name, contrast_limits=[-c_lim, c_lim])
        new_image.colormap = cmap


@magic_factory(call_button='Calc Red-Green')
def der_series(viewer: Viewer, img:Image,
               left_frames:int=1, space_frames:int=0, right_frames:int=1,
               normalize_by_int:bool=False,
               save_MIP:bool=False):
    if img.data.ndim != 3:
        raise ValueError('The input image should have 3 dimensions!')
    img_name, img_data = img.name, img.data

    def _save_rg_img(params):
        _add_or_update_diverging(viewer, params[0], params[1],
                                 ('red-green', _cmaps.red_green_cmap()))

    @thread_worker
    def _der_series():
        start = time.perf_counter()
        der_img = series.red_green_series(img_data,
                                          left_frames=left_frames,
                                          space_frames=space_frames,
                                          right_frames=right_frames,
                                          intensity_weighted=normalize_by_int)
        show_info(f'{img_name}: red-green series calculated in {time.perf_counter()-start:.2f}s, '
                  f'output frame i is the transition between input frames i+{left_frames-1} and i+{left_frames+space_frames}')
        yield (der_img, img_name + '_red-green')
        if save_MIP:
            yield (np.nanmax(der_img, axis=0), img_name + '_red-green-MIP')

    start_worker('Red-green series', _der_series(), on_yield=_save_rg_img)


@magic_factory(call_button='Calc relative intensity',
               values_mode={"choices": ['ΔF', 'ΔF/F0']},
               F0_min={'min': 0.0, 'max': 1e9},)
def rel_series(viewer: Viewer, img:Image, values_mode:str='ΔF', F0_win:int=5,
               F0_min:float=0.0,
               intensity_weighted:bool=False):
    if img.data.ndim != 3:
        raise ValueError('The input image should have 3 dimensions!')
    img_name, img_data = img.name, img.data

    def _save_rel_img(params):
        _add_or_update_diverging(viewer, params[0], params[1],
                                 ('delta', _cmaps.delta_smooth_cmap()))

    @thread_worker
    def _rel_series():
        start = time.perf_counter()
        mode_dict = {'ΔF': 'dF', 'ΔF/F0': 'dF/F0'}
        output_img = series.delta_img(img_data, mode=mode_dict[values_mode],
                                      win_size=F0_win, f0_min=F0_min,
                                      intensity_weighted=intensity_weighted)
        show_info(f'{img_name}: {values_mode} img calculated in {time.perf_counter()-start:.2f}s')
        yield (output_img, img_name + f'_{values_mode}')

    start_worker('ΔF series', _rel_series(), on_yield=_save_rel_img)

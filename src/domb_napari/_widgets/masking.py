""" Widgets: dots, up and intensity masking.

"""
import numpy as np
from magicgui import magic_factory

from napari import Viewer
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info

from ..core import masking
from ..core.profiles import label_ids
from ._common import start_worker, add_or_update_labels


def _check_frame(img, det_frame_index):
    if img.data.ndim != 3:
        raise ValueError('The input image should have 3 dimensions!')
    if not -img.data.shape[0] <= det_frame_index < img.data.shape[0]:
        raise ValueError(f'det_frame_index should be less than {img.data.shape[0]}!')


@magic_factory(call_button='Build Dots Mask',
               background_level={"widget_type": "FloatSlider", 'min':50.0, 'max': 99.0, 'step':1.0},
               detection_level={"widget_type": "FloatSlider",'min':1.0, 'max': 100.0, 'step':1.0},)
def dot_mask_calc(viewer: Viewer, img:Image, background_level:float=75.0, detection_level:float=25.0,
                  minimal_distance:int=2, mask_radius:int=5):
    if img.data.ndim != 3:
        raise ValueError('The input image should have 3 dimensions!')
    img_name, img_data = img.name, img.data

    def _save_dot_labels(params):
        add_or_update_labels(viewer, params[0], params[1])

    @thread_worker
    def _dot_mask_calc():
        peaks_labels = masking.dots_mask(img_data,
                                         background_level=background_level,
                                         detection_level=detection_level,
                                         minimal_distance=minimal_distance,
                                         mask_radius=mask_radius)
        show_info(f'{img_name}: detected {np.max(peaks_labels)} dots labels')
        yield (peaks_labels, img_name + '_dots-labels')

    start_worker('Dot-patterns masking', _dot_mask_calc(), on_yield=_save_dot_labels)


@magic_factory(call_button='Build Up Mask',
               det_th={"widget_type": "FloatSlider", 'max': 1},
               in_ROIs_det_method={"choices": ['otsu', 'threshold']},)
def up_mask_calc(viewer: Viewer, img:Image, ROIs_mask:Labels=None,
                 det_frame_index:int=2,
                 det_th:float=0.25,
                 in_ROIs_det:bool=False,
                 in_ROIs_det_method:str='otsu',
                 in_ROIs_det_th_corr:float=0.1,
                 final_opening_fp:int=1,
                 final_dilation_fp:int=0,
                 save_total_up_mask:bool=False):
    _check_frame(img, det_frame_index)
    if in_ROIs_det and ROIs_mask is None:
        raise ValueError('ROIs mask is required for in-ROIs detection!')
    img_name = img.name
    detection_img = np.asarray(img.data[det_frame_index])
    rois_mask = np.asarray(ROIs_mask.data) if in_ROIs_det else None

    def _save_up_labels(params):
        add_or_update_labels(viewer, params[0], params[1])

    @thread_worker
    def _up_mask_calc():
        up_labels = masking.up_mask(detection_img, det_th=det_th,
                                    rois_mask=rois_mask,
                                    in_rois_method=in_ROIs_det_method,
                                    in_rois_th_corr=in_ROIs_det_th_corr,
                                    opening_fp=final_opening_fp,
                                    dilation_fp=final_dilation_fp)
        show_info(f'{img_name}: detected {label_ids(up_labels).size} labels')
        yield (up_labels, img_name + '_up-labels')
        if save_total_up_mask:
            yield (up_labels > 0, img_name + '_up-mask')

    start_worker('Up masking', _up_mask_calc(), on_yield=_save_up_labels)


@magic_factory(call_button='Build Mask',
               masking_mode={"choices": ['up', 'down']},)
def mask_calc(viewer: Viewer, img:Image, det_frame_index:int=2,
              masking_mode:str='up',
              up_threshold:float=0.2,
              down_threshold:float=-0.9,
              opening_footprint:int=0):
    _check_frame(img, det_frame_index)
    img_name = img.name
    detection_img = np.asarray(img.data[det_frame_index])

    def _save_rg_labels(params):
        add_or_update_labels(viewer, params[0], params[1])

    @thread_worker
    def _mask_calc():
        labels = masking.intensity_mask(detection_img,
                                        masking_mode=masking_mode,
                                        up_threshold=up_threshold,
                                        down_threshold=down_threshold,
                                        opening_footprint=opening_footprint)
        show_info(f'{img_name}: detected {np.max(labels)} "{masking_mode}" labels')
        yield (labels, img_name + f'_{masking_mode}-labels')

    start_worker('Intensity masking', _mask_calc(), on_yield=_save_rg_labels)

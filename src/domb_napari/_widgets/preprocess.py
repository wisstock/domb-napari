""" Widgets: multichannel stack preprocessing and dual-view registration.

"""
import os
import pathlib
import time

import numpy as np
from magicgui import magic_factory

from napari import Viewer
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning

from skimage import filters

from ..core import preprocess
from ..core.registration import AffineRegistration, apply_affine_stack
from ._common import start_worker, add_or_update_image


# expected limit of scaling and shear between the sides of a dual-view splitter
MAX_LINEAR_DEVIATION = 0.05


@magic_factory(call_button='Preprocess stack',
               stack_order={"choices": ['TCXY', 'CTXY']},
               correction_method={"choices": ['exp', 'biexp', '1st']},
               background_percentile={'min': 0.0, 'max': 100.0},)
def split_channels(viewer: Viewer, img:Image,
                   stack_order:str='TCXY',
                   median_filter:bool=False, median_kernel:int=3,
                   background_substraction:bool=True,
                   background_percentile:float=1.0,
                   photobleaching_correction:bool=False,
                   correction_mask:Labels=None,
                   correction_method:str='exp',
                   drop_frames:bool=False,
                   frames_range:list=[0,10],
                   frames_crop:int=0):
    if img.data.ndim not in (3, 4):
        raise ValueError('Input image has to have 3 or 4 dimensions!')
    if drop_frames and len(frames_range) != 2:
        raise ValueError('List of indexes should has 2 elements!')
    pb_mask_input = None if correction_mask is None else np.asarray(correction_mask.data)
    pb_mask_name = None if correction_mask is None else correction_mask.name
    img_name, img_data = img.name, img.data

    def _save_ch(params):
        add_or_update_image(viewer, params[0], params[1])

    @thread_worker
    def _split_channels():
        def _preprocessing(ch_img, ch_suffix):
            start = time.perf_counter()
            if drop_frames:
                ch_img = ch_img[frames_range[0]:frames_range[-1],:,:]
                ch_suffix = f'_{frames_range[0]}-{frames_range[-1]}'+ch_suffix
            if median_filter:
                ch_img = preprocess.median_filter_stack(ch_img, median_kernel)
            if background_substraction:
                ch_img = preprocess.back_substr(ch_img, percentile=background_percentile)
            if photobleaching_correction:
                if pb_mask_input is not None:
                    pb_mask = pb_mask_input
                    show_info(f'Photobleaching correction with {pb_mask_name} mask')
                else:
                    first_frame = np.nan_to_num(np.asarray(ch_img[0], dtype=np.float64))
                    pb_mask = first_frame > filters.threshold_otsu(first_frame)
                    show_info('Photobleaching correction with 1st frame Otsu mask')
                ch_img,_,corr_metric = preprocess.pb_correction(input_img=ch_img,
                                                                mask=pb_mask,
                                                                method=correction_method)
                ch_suffix = ch_suffix + '_' + correction_method
                show_info(f'{correction_method} photobleaching correction, ' + corr_metric)
            if frames_crop != 0:
                yo, xo = ch_img.shape[1:]
                ch_img = ch_img[:,frames_crop:yo-frames_crop,frames_crop:xo-frames_crop]
            end = time.perf_counter()
            show_info(f'{img_name+ch_suffix} preprocessing time: {end - start:.2f} s, shape {ch_img.shape}, data type {ch_img.dtype}')
            return (ch_img, img_name+ch_suffix)

        show_info(f'{img_name}: preprocessing started, data type {img_data.dtype}, shape {img_data.shape}')
        if photobleaching_correction:
            show_warning('Photobleaching-corrected stacks are not valid inputs for E-FRET estimation!')
        if img_data.ndim == 4:
            if stack_order == 'TCXY':    # for LA data
                input_img = img_data
            elif stack_order == 'CTXY':  # for confocal data
                input_img = np.moveaxis(img_data,0,1)
            for i in range(input_img.shape[1]):
                show_info(f'{img_name}: Ch. {i} preprocessing started')
                yield _preprocessing(ch_img=input_img[:,i,:,:], ch_suffix=f'_ch{i}')
        else:
            show_info(f'{img_name}: image already has 3 dimensions, preprocessing only mode')
            yield _preprocessing(ch_img=img_data, ch_suffix='')

    start_worker('Multichannel stack preprocessing', _split_channels(), on_yield=_save_ch)


@magic_factory(call_button='Align stack',
               align_method={"choices": ['internal', 'load matrix']},
               load_matrix={'mode': 'r', 'filter': 'Text files (*.txt)'},
               saving_path={'mode': 'd'},)
def dw_registration(viewer: Viewer, offset_img:Image,
                    input_crop:int=30, output_crop:int=30,
                    align_method:str='internal',
                    manual_channels:bool=False,
                    ref_off_ch:list=[0,1],        # for manual_channels=True
                    load_matrix:pathlib.Path = None,
                    save_matrix:bool=False,
                    saving_path:pathlib.Path = pathlib.Path.home()):
    if offset_img.data.ndim != 4:
        raise ValueError('Incorrect dimensions of the input image!')
    n_ch = offset_img.data.shape[1]
    if manual_channels:
        if len(ref_off_ch) != 2 or not all(0 <= ch < n_ch for ch in ref_off_ch) or ref_off_ch[0] == ref_off_ch[1]:
            raise ValueError(f'ref_off_ch should be two different channel indexes less than {n_ch}!')
        ref_ch, off_ch = int(ref_off_ch[0]), int(ref_off_ch[1])
    elif n_ch in (2, 4):  # 1 or 2 excitations with dual-view
        ref_ch, off_ch = n_ch - 1, 0
    else:
        raise ValueError(f'The input image have {n_ch} spectral ch., but 2 or 4 ch. are required, or select channels manually!')
    if align_method == 'load matrix' and (load_matrix is None or not pathlib.Path(load_matrix).is_file()):
        raise ValueError('Select a file with the affine matrix!')
    # emission sides alternate along the channel axis, so every channel
    # of the offset side has the same parity of the index
    moving_chs = list(range(off_ch % 2, n_ch, 2))
    img_name, img_data = offset_img.name, offset_img.data

    def _save_aligned(output_img):
        add_or_update_image(viewer, output_img, img_name+'_algn')

    @thread_worker
    def _dw_registration():
        start = time.perf_counter()
        input_data = img_data
        if input_crop != 0:  # optional input crop for better registration and border artefacts removal
            y, x = input_data.shape[-2:]
            input_data = input_data[:,:,input_crop:y-input_crop,input_crop:x-input_crop]

        if align_method == 'internal':
            show_info(f'{img_name}: internal alignment mode, move ch. {moving_chs} to ch. {ref_ch}')
            ref_frame = np.mean(input_data[:,ref_ch,:,:], axis=0, dtype=np.float64)
            move_frame = np.mean(input_data[:,off_ch,:,:], axis=0, dtype=np.float64)
            affine_map = AffineRegistration().optimize(ref_frame, move_frame)
            affine_matrix = affine_map.affine
            info = affine_map.info
            show_info(f'{img_name}: mutual information {info["mi_identity"]:.4f} -> {info["mi"]:.4f}, affine matrix\n{np.round(affine_matrix, 5)}')
            if not info['accepted']:
                show_warning(f'{img_name}: registration failed to improve the alignment, the stack is not transformed!')
            elif np.abs(affine_matrix[:2,:2] - np.eye(2)).max() > MAX_LINEAR_DEVIATION:
                show_warning(f'{img_name}: the estimated scaling/shear is unusually large for a dual-view splitter, '
                             'check the result! Mutual information is not reliable for images without fine structure.')
            if save_matrix:
                matrix_path = os.path.join(saving_path, f"{img_name}_affine_matrix.txt")
                np.savetxt(matrix_path, affine_matrix,
                           header=f'affine matrix (output px -> input px), input_crop={input_crop}, frame shape={input_data.shape[-2:]}')
                show_info(f'{img_name}: affine matrix saved to {matrix_path}')
        else:
            affine_matrix = np.loadtxt(load_matrix)
            if affine_matrix.shape != (3, 3):
                raise ValueError('The affine matrix should have 3x3 shape!')
            show_info(f'{img_name}: alignment with matrix mode, loaded from {load_matrix}, move ch. {moving_chs}')

        output_data = np.empty(input_data.shape, dtype=np.float32)
        for ch in range(n_ch):
            if ch in moving_chs:
                output_data[:,ch] = apply_affine_stack(input_data[:,ch], affine_matrix, order=1)
            else:
                output_data[:,ch] = input_data[:,ch]

        if output_crop != 0:  # optional output crop for border artefacts of the registration removal
            yo, xo = output_data.shape[-2:]
            output_data = output_data[:,:,output_crop:yo-output_crop,output_crop:xo-output_crop]

        end = time.perf_counter()
        show_info(f'{img_name}: stack aligned in {end-start:.2f} s')
        yield output_data

    start_worker('Dual-view stack registration', _dw_registration(), on_yield=_save_aligned)

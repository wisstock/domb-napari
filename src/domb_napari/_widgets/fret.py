""" Widgets: E-FRET calibration and estimation.

"""
import os
import pathlib
import time

import numpy as np
import pandas as pd
import yaml
from magicgui import magic_factory

from napari import Viewer
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning

from ..core import fret as e_fret
from ..core.masking import mask_segmentation
from ..core.profiles import label_ids
from ._common import (start_worker, add_or_update_image, add_or_update_labels,
                      new_figure, style_axis, show_figure, subsample, robust_limit)


DEFAULT_FRET_CONFIG_PATH = pathlib.Path(__file__).parent.parent / '_e_fret_coefs.yaml'
MINOR_CROSSTALK_WARNING = 0.01


def _check_3d(*layers):
    if not all(layer is not None and layer.data.ndim == 3 for layer in layers):
        raise ValueError('Incorrect input image shape, 3D stacks [t,x,y] are required!')


def _base_name(aa_layer):
    return aa_layer.name.replace('_ch3', '')


# --------------------------------------------------------------------------
# cross-talk
# --------------------------------------------------------------------------
@magic_factory(call_button='Estimate cross-talk',
               presented_fluorophore={"choices": ['A', 'D']},
               saving_path={'mode': 'd'})
def cross_calc(viewer: Viewer, DD_img:Image, DA_img:Image, AA_img:Image,
               mask: Labels,
               presented_fluorophore:str='A',
               saving_path:pathlib.Path = pathlib.Path.home()):
    _check_3d(DD_img, DA_img, AA_img)
    if mask is None:
        raise ValueError('Mask is required!')
    output_name = _base_name(AA_img)
    prm_name, off_name = ('a', 'b') if presented_fluorophore == 'A' else ('d', 'c')
    ref_lab = 'AA, a.u.' if presented_fluorophore == 'A' else 'DD, a.u.'
    dd_data, da_data, aa_data, mask_data = DD_img.data, DA_img.data, AA_img.data, mask.data

    def _save_cross_data(output):
        coef_df, fit_arr, summary_df = output
        for df, suffix in ((coef_df, ''), (summary_df, '_summary')):
            df.insert(0, 'id', output_name)
            df.to_csv(os.path.join(saving_path, f"{output_name}_{presented_fluorophore}{suffix}.csv"))
        res = summary_df.iloc[0]
        show_info(f'{output_name}: {prm_name}={res[prm_name+"_val"]:.4f}±{res[prm_name+"_err"]:.4f} (Deming), '
                  f'ratio {res[prm_name+"_ratio"]:.4f}, OLS {res[prm_name+"_ols"]:.4f}, '
                  f'intercept {res[prm_name+"_i"]:.2f}, r² {res[prm_name+"_r^2"]:.3f}; '
                  f'{off_name}={res[off_name+"_val"]:.4f}±{res[off_name+"_err"]:.4f}')
        if abs(res[off_name+'_val']) > MINOR_CROSSTALK_WARNING:
            show_warning(f'Coefficient {off_name}={res[off_name+"_val"]:.4f} is not negligible, add it to the FRET pair configuration!')

        x_arr, y_arr = fit_arr
        idx = subsample(x_arr.size)
        x_line = np.linspace(np.min(x_arr), np.max(x_arr), 100)
        mpl_fig = new_figure()
        ax = mpl_fig.add_subplot(111)
        ax.scatter(x_arr[idx], y_arr[idx], s=5, alpha=0.15, color='blue')
        ax.plot(x_line, res[prm_name+'_val']*x_line + res[prm_name+'_i'],
                color='red', linewidth=2, label='Deming')
        ax.plot(x_line, res[prm_name+'_ols']*x_line + res[prm_name+'_ols_i'],
                color='black', linewidth=1, linestyle='--', label='OLS')
        ax.legend()
        style_axis(ax, ref_lab, 'DA, a.u.',
                   f'{output_name}, time-averaged, {prm_name}={res[prm_name+"_val"]:.4f}±{res[prm_name+"_err"]:.4f}, r² {res[prm_name+"_r^2"]:.3f}')
        show_figure(viewer, mpl_fig, 'Cross-talk estimation')

    @thread_worker
    def _cross_calc():
        cross_estimator = e_fret.CrossTalkEstimation(mask=mask_data,
                                                     dd_img=dd_data,
                                                     da_img=da_data,
                                                     aa_img=aa_data)
        if presented_fluorophore == 'A':
            return cross_estimator.estimate_a()
        return cross_estimator.estimate_d()

    start_worker('E-FRET crosstalk estimation', _cross_calc(), on_return=_save_cross_data)


# --------------------------------------------------------------------------
# G-factor
# --------------------------------------------------------------------------
def _g_calc_init(widget):
    """ G-factor estimation widget initialization function for dynamic interface update

    """
    options_map = {'Zal': ['mask', 'segment_mask', 'fragments_num'],
                   'Chen': ['mask_high', 'mask_low']}
    all_dynamic_widgets = [name for params in options_map.values() for name in params]

    def update_interface(method_name):
        for name in all_dynamic_widgets:
            getattr(widget, name).visible = name in options_map.get(method_name, [])

    def update_config(write):
        widget.config_path.visible = write
        widget.fret_pair.visible = write

    widget.estimation_method.changed.connect(update_interface)
    widget.write_to_config.changed.connect(update_config)
    update_interface(widget.estimation_method.value)
    update_config(widget.write_to_config.value)


@magic_factory(widget_init=_g_calc_init,
               call_button='Estimate',
               estimation_method={"choices": ["Zal", "Chen"], "label": "Estimation Method"},
               config_path={'mode': 'w', 'filter': '*.yaml;*.yml'},
               saving_path={'mode': 'd'},)
def g_calc(viewer: Viewer, estimation_method:str='Zal',
           DD_img_high_FRET:Image=None, DA_img_high_FRET:Image=None, AA_img_high_FRET:Image=None,
           DD_img_low_FRET:Image=None, DA_img_low_FRET:Image=None, AA_img_low_FRET:Image=None,
           # options for Zal
           mask: Labels=None,
           segment_mask:bool=True,
           fragments_num:int=30,
           # options for Chen
           mask_high: Labels=None,
           mask_low: Labels=None,
           # common options
           frames_to_average:int=1,
           a:float=0.0136, d:float=0.2646,  # a & d for TagBFP+mBaoJin
           b:float=0.0, c:float=0.0,
           write_to_config:bool=False,
           config_path:pathlib.Path=None,
           fret_pair:str='',
           saving_path:pathlib.Path = pathlib.Path.home()):
    _check_3d(DD_img_low_FRET, DA_img_low_FRET, AA_img_low_FRET,
              DD_img_high_FRET, DA_img_high_FRET, AA_img_high_FRET)
    zal = estimation_method == 'Zal'
    if zal and mask is None:
        raise ValueError('Mask is required for Zal G-factor estimation!')
    if not zal and (mask_high is None or mask_low is None):
        raise ValueError('Both high and low FRET masks are required for Chen G-factor estimation!')
    if write_to_config and (config_path is None or not fret_pair):
        raise ValueError('Config path and FRET pair name are required to write the coefficients!')
    do_segmentation = zal and segment_mask
    output_name = _base_name(AA_img_low_FRET)
    n_frames = None if frames_to_average <= 0 else frames_to_average
    imgs = dict(h_dd_img=DD_img_high_FRET.data, h_da_img=DA_img_high_FRET.data,
                h_aa_img=AA_img_high_FRET.data, l_dd_img=DD_img_low_FRET.data,
                l_da_img=DA_img_low_FRET.data, l_aa_img=AA_img_low_FRET.data)
    if zal:
        masks = (np.asarray(mask.data), None)
    else:
        masks = (np.asarray(mask_high.data), np.asarray(mask_low.data))

    def _save_g_data(output):
        g_df, fit_arr, roi_mask = output
        g_df.insert(0, 'id', output_name)
        g_df.insert(1, 'method', estimation_method)
        g_df.to_csv(os.path.join(saving_path, f"{output_name}_G.csv"))
        res = g_df.iloc[0]
        g_val, g_err = res['g_val'], res['g_err']
        if do_segmentation:
            add_or_update_labels(viewer, roi_mask, f"{output_name}_seg_labels",
                                 opacity=0.5, contour=0)
        if write_to_config:
            e_fret.update_yaml(config_path, fret_pair, a=a, d=d, b=b, c=c, G=g_val)
            show_info(f'{fret_pair}: a, d, b, c, G written to {config_path}')

        mpl_fig = new_figure()
        ax0 = mpl_fig.add_subplot(121)
        ax1 = mpl_fig.add_subplot(122)
        if zal:
            x_arr, y_arr = fit_arr
            show_info(f'G={g_val:.3f}±{g_err:.3f} (ratio), Deming {res["g_deming"]:.3f}, '
                      f'OLS {res["g_ols"]:.3f} with intercept {res["g_i"]:.2f}±{res["g_i_err"]:.2f}, r² {res["g_r^2"]:.3f}')
            x_line = np.linspace(min(0.0, np.min(x_arr)), np.max(x_arr), 100)
            ax0.scatter(x_arr, y_arr, s=15, alpha=0.25, color='blue')
            ax0.plot(x_line, g_val*x_line, color='red', linewidth=2, label='ratio (through origin)')
            ax0.plot(x_line, res['g_ols']*x_line + res['g_i'], color='black',
                     linewidth=1, linestyle='--', label='OLS')
            ax0.legend()
            style_axis(ax0, 'ΔD, a.u.', 'ΔFc, a.u.', f'G={g_val:.3f}±{g_err:.3f}')

            with np.errstate(invalid='ignore', divide='ignore'):
                roi_ratio = y_arr / x_arr
            roi_ratio = roi_ratio[np.isfinite(roi_ratio)]
            ax1.boxplot(roi_ratio, positions=[1], widths=0.5)
            ax1.scatter(np.full(len(roi_ratio), 1), roi_ratio, s=35, alpha=0.25, color='blue')
            ax1.errorbar([1.4], [g_val], yerr=[1.96 * g_err], fmt='-o', capsize=5,
                         alpha=0.75, color='red')
            ax1.set_xticks([1, 1.4])
            ax1.set_xticklabels(['ROIs', 'G, 95% CI'])
            style_axis(ax1, '', 'ΔFc/ΔD', 'ROIs ΔFc/ΔD and G')
            mpl_fig.suptitle(f'{output_name}, Zal and Gascoigne G estimation', fontsize=10)
        else:
            arr_h, arr_l = fit_arr
            show_info(f'G={g_val:.3f}±{g_err:.3f}, 95% CI [{res["g_ci_low"]:.3f}, {res["g_ci_high"]:.3f}]')
            ax0.scatter(arr_h[0], arr_h[1], s=15, alpha=0.25, color='red', label='High FRET ROIs')
            ax0.scatter(arr_l[0], arr_l[1], s=15, alpha=0.25, color='blue', label='Low FRET ROIs')
            ax0.plot([res['d_a_h'], res['d_a_l']], [res['fc_a_h'], res['fc_a_l']],
                     '-o', color='black', linewidth=2, label='slope = -G')
            ax0.legend()
            style_axis(ax0, 'D/A', 'Fc/A', f'G={g_val:.3f}±{g_err:.3f}')
            ax1.boxplot([arr[np.isfinite(arr)] for arr in (arr_h[0], arr_h[1], arr_l[0], arr_l[1])],
                        tick_labels=['D/A H', 'Fc/A H', 'D/A L', 'Fc/A L'])
            style_axis(ax1, '', 'I/A', 'ROIs I/A')
            mpl_fig.suptitle(f'{output_name}, Chen et al. G estimation', fontsize=10)
        show_figure(viewer, mpl_fig, 'G-factor estimation')

    @thread_worker
    def _g_calc():
        start = time.perf_counter()
        show_info(f'G-factor estimation started, method {estimation_method}, a={a}, d={d}, b={b}, c={c}')
        roi_mask, l_mask = masks
        if do_segmentation:
            if np.sum(roi_mask != 0) // fragments_num < 300:
                raise ValueError('The mask area is too small for segmentation, please select larger area!')
            roi_mask = mask_segmentation(roi_mask != 0, fragment_num=fragments_num)
        show_info(f'G-factor estimation with {label_ids(roi_mask).size} ROIs')
        g_estimator = e_fret.GFactorEstimation(mask=roi_mask, l_mask=l_mask,
                                               a_val=a, d_val=d, b_val=b, c_val=c,
                                               n_frames=n_frames, **imgs)
        coef = g_estimator.estimate_g_zal() if zal else g_estimator.estimate_g_chen()
        show_info(f'G-factor estimated in {time.perf_counter() - start:.2f}s')
        return (*coef, roi_mask)

    start_worker('E-FRET G-factor estimation', _g_calc(), on_return=_save_g_data)


# --------------------------------------------------------------------------
# FRET estimation
# --------------------------------------------------------------------------
def _read_pairs(path):
    if path is None or not pathlib.Path(path).is_file():
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return list((yaml.safe_load(f) or {}).keys())
    except Exception as e:
        show_warning(f'Error loading {pathlib.Path(path).name}: {e}')
        return []


def _config_path(config_mode, config_path):
    return DEFAULT_FRET_CONFIG_PATH if config_mode == 'Default' else config_path


def _config_init(widget):
    """ Keeps the list of FRET pairs in sync with the selected configuration file.

    Choices are provided by a callable, so they survive the reset of choices
    done by napari on every change of the layer list.

    """
    widget._pairs = []
    widget.fret_pair.choices = lambda _: list(widget._pairs)

    def _reload(*_):
        default = widget.config_mode.value == 'Default'
        widget.config_path.visible = not default
        selected = widget.fret_pair.value
        widget._pairs = _read_pairs(_config_path(widget.config_mode.value, widget.config_path.value))
        widget.fret_pair.reset_choices()
        if selected in widget._pairs:
            widget.fret_pair.value = selected

    widget.config_mode.changed.connect(_reload)
    widget.config_path.changed.connect(_reload)
    _reload()


def _load_coefs(config_mode, config_path, fret_pair, output_type):
    load_path = _config_path(config_mode, config_path)
    if not load_path or not pathlib.Path(load_path).is_file():
        raise FileNotFoundError('Configuration file not found!')
    if not fret_pair:
        raise ValueError('FRET pair is not selected!')
    coefs = e_fret.FRETCoefs.from_yaml(load_path, fret_pair)
    coefs.require(output_type)
    show_info(f'Selected FRET pair {fret_pair}: a={coefs.a}, d={coefs.d}, b={coefs.b}, c={coefs.c}, G={coefs.G}, xi={coefs.xi}')
    return coefs


_CONFIG_OPTIONS = dict(config_mode={'label': 'Config Mode', 'widget_type': 'RadioButtons',
                                    'choices': ['Load', 'Default'],
                                    'orientation': 'horizontal'},
                       config_path={'label': 'Config Path', 'widget_type': 'FileEdit',
                                    'mode': 'r', 'filter': '*.yaml;*.yml'},
                       fret_pair={'label': 'FRET Pair', 'choices': [],
                                  'widget_type': 'ComboBox'},
                       output_type={"choices": ['Fc', 'E_D', 'E_A', 'Ecorr'],
                                    'label': 'Estimation Method'})


@magic_factory(widget_init=_config_init,
               call_button='Estimate FRET',
               **_CONFIG_OPTIONS)
def e_app_calc(viewer: Viewer,
               config_mode:str='Default', config_path: pathlib.Path=None,
               fret_pair:str=None,
               output_type:str='Fc',
               DD_img:Image=None, DA_img:Image=None, AA_img:Image=None,
               bleach_mask:Labels=None,
               save_normalized:bool=True):
    _check_3d(DD_img, DA_img, AA_img)
    coefs = _load_coefs(config_mode, config_path, fret_pair, output_type)
    output_name = _base_name(AA_img) + '_' + output_type
    dd_data, da_data, aa_data = DD_img.data, DA_img.data, AA_img.data
    bleach_mask_data = None if bleach_mask is None else np.asarray(bleach_mask.data)

    def _save_e_app(params):
        fret_img, img_name = params
        if output_type == 'Fc':
            c_lim = robust_limit(fret_img)
            limits = [-c_lim, c_lim] if np.nanmin(fret_img[0]) < 0 else [0, c_lim]
        else:
            limits = [0, 1]
        add_or_update_image(viewer, fret_img, img_name, contrast_limits=limits)

    @thread_worker
    def _e_app_calc():
        start = time.perf_counter()
        cubes = e_fret.CubesFRET.from_coefs(dd_data, da_data, aa_data, coefs)
        if output_type == 'Ecorr':
            bleaching = cubes.acceptor_bleaching(bleach_mask_data)
            show_info(f'Acceptor remaining at the last frame: {bleaching[-1]*100:.1f}%')
            output_fret_img = cubes.Ecorr_img(mask=bleach_mask_data)
        elif output_type == 'E_D':
            output_fret_img = cubes.E_D_img()
        elif output_type == 'E_A':
            output_fret_img = cubes.E_A_img()
        else:
            output_fret_img = cubes.Fc_img()
        yield (output_fret_img, output_name)
        if save_normalized:
            # intensity-weighted image for visual inspection and masking only
            img_norm = np.mean(aa_data, axis=0, dtype=np.float64)
            img_norm = np.clip(img_norm / np.nanmax(img_norm), 0, None).astype(np.float32)
            yield (np.nan_to_num(output_fret_img) * img_norm, output_name + '_norm')
        show_info(f'{output_type} calculated in {time.perf_counter()-start:.2f}s')

    start_worker('E-FRET estimation', _e_app_calc(), on_yield=_save_e_app)


@magic_factory(widget_init=_config_init,
               call_button='Estimate ROIs FRET',
               saving_path={'mode': 'd'},
               **_CONFIG_OPTIONS)
def e_fret_rois(viewer: Viewer,
                config_mode:str='Default', config_path: pathlib.Path=None,
                fret_pair:str=None,
                output_type:str='E_D',
                DD_img:Image=None, DA_img:Image=None, AA_img:Image=None,
                labels:Labels=None,
                time_scale:float=1.0,
                save_data_frame:bool=False,
                saving_path:pathlib.Path = pathlib.Path.home()):
    _check_3d(DD_img, DA_img, AA_img)
    if labels is None:
        raise ValueError('Labels are required!')
    coefs = _load_coefs(config_mode, config_path, fret_pair, output_type)
    output_name = _base_name(AA_img)
    dd_data, da_data, aa_data = DD_img.data, DA_img.data, AA_img.data
    labels_data, labels_name = np.asarray(labels.data), labels.name
    roi_ids = label_ids(labels_data)
    lab_colors = labels.get_color(list(roi_ids))

    def _plot_rois(profiles):
        n_roi, n_frames = profiles['Fc'].shape
        time_line = np.arange(n_frames) * time_scale
        mpl_fig = new_figure()
        ax = mpl_fig.add_subplot(111)
        if output_type == 'Fc':
            ax.axhline(y=0.0, color='k', linestyle='--')
        for prof, color in zip(profiles[output_type], lab_colors):
            ax.plot(time_line, prof, alpha=0.45, marker='o', color=color)
        style_axis(ax, 'Time, s', output_type)
        ax.set_title(f'{output_name} ROIs {output_type} from mean intensities, labels: {labels_name}')
        show_figure(viewer, mpl_fig, 'ROIs FRET Prof.')

        if save_data_frame:
            table = {'id': output_name,
                     'lab_id': labels_name,
                     'roi': np.repeat(roi_ids, n_frames),
                     'index': np.tile(np.arange(n_frames), n_roi),
                     'time': np.tile(time_line, n_roi)}
            table.update({key: np.round(val, 6).ravel() for key, val in profiles.items()})
            df_path = os.path.join(saving_path, f'{output_name}_{labels_name}_FRET.csv')
            pd.DataFrame(table).to_csv(df_path)
            show_info(f'ROIs FRET data frame saved to {df_path}')

    @thread_worker
    def _e_fret_rois():
        cubes = e_fret.CubesFRET.from_coefs(dd_data, da_data, aa_data, coefs)
        return cubes.roi_profiles(labels_data)

    start_worker('E-FRET ROIs profiles', _e_fret_rois(), on_return=_plot_rois)

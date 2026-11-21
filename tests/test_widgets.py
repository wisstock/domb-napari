""" Package structure and a headless run of every widget.

"""
import importlib
import inspect
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest
import yaml

napari = pytest.importorskip('napari')

ROOT = pathlib.Path(__file__).parent.parent
MANIFEST = yaml.safe_load(open(ROOT / 'src' / 'domb_napari' / 'napari.yaml'))['contributions']


def _factories():
    out = {}
    for command in MANIFEST['commands']:
        module, name = command['python_name'].split(':')
        out[name] = getattr(importlib.import_module(module), name)
    return out


def test_package():
    # core does not import GUI libraries
    code = ("import sys, domb_napari, domb_napari.core.fret, domb_napari.core.registration, "
            "domb_napari.core.series, domb_napari.core.masking, domb_napari.core.preprocess; "
            "print([m for m in ('napari', 'qtpy', 'magicgui', 'matplotlib', 'vispy') if m in sys.modules])")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == '[]'

    # manifest, package exports and README describe the same widgets and parameters
    import domb_napari
    factories = _factories()
    ids = {c['id'] for c in MANIFEST['commands']}
    assert {w['command'] for w in MANIFEST['widgets']} == ids
    assert {i['command'] for items in MANIFEST['menus'].values() for i in items if 'command' in i} == ids
    assert set(domb_napari.__all__) == set(factories)
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    missing = [f'{name}.{param}' for name, factory in factories.items()
               for param in inspect.signature(factory.keywords['function']).parameters
               if param != 'viewer' and f'`{param}`' not in readme]
    assert not missing, missing


def _run(widget, **kwargs):
    """ Calls a widget and waits for its worker. """
    from qtpy.QtWidgets import QApplication
    from domb_napari._widgets import _common
    widget(**kwargs)
    deadline = time.time() + 120
    while _common._RUNNING and time.time() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()
    assert not _common._RUNNING


def _stack(rng, n_t=8, shape=(96, 96)):
    """ TCXY stack of a 1:1 FRET construct, channels: DD, DA, AD, AA. """
    from scipy import ndimage as ndi
    dots = np.zeros(shape)
    dots[rng.integers(10, shape[0] - 10, 25), rng.integers(10, shape[1] - 10, 25)] = rng.uniform(0.3, 1, 25)
    cell = ndi.gaussian_filter(dots, 3) * 150000 + 300
    a, d, g, eff = 0.014, 0.265, 2.99, 0.3
    dd, aa = cell * (1 - eff), cell
    channels = [dd, g * cell * eff + a * aa + d * dd, 0.02 * cell, aa]
    return np.stack([rng.poisson(np.stack(channels) + 100) for _ in range(n_t)]).astype(np.uint16)


def test_pipeline(tmp_path):
    # napari canvas needs OpenGL, which is not available on the offscreen Qt platform
    if os.environ.get('QT_QPA_PLATFORM') == 'offscreen' or \
       (sys.platform.startswith('linux') and not os.environ.get('DISPLAY')):
        pytest.skip('OpenGL context is not available')
    v = napari.Viewer(show=False)
    w = {name: factory() for name, factory in _factories().items()}
    raw = v.add_image(_stack(np.random.default_rng(0)), name='cell')

    # registration: estimated and loaded matrices give the same stack
    _run(w['dw_registration'], viewer=v, offset_img=raw, input_crop=4, output_crop=4,
         save_matrix=True, saving_path=tmp_path)
    algn = v.layers['cell_algn']
    assert algn.data.shape == (8, 4, 80, 80)
    first = algn.data.copy()
    _run(w['dw_registration'], viewer=v, offset_img=raw, input_crop=4, output_crop=4,
         align_method='load matrix', load_matrix=tmp_path / 'cell_affine_matrix.txt')
    np.testing.assert_allclose(v.layers['cell_algn'].data, first, rtol=1e-5, equal_nan=True)

    # preprocessing: background from a labels layer, all labels are one mask
    split = w['split_channels']
    shown = lambda widget: not widget.native.isHidden()
    split.background_method.value = 'labels layer'
    assert shown(split.background_mask) and not shown(split.background_percentile)
    bg = np.zeros((80, 80), int)
    bg[:6, :6], bg[-6:, -6:] = 1, 3
    bg_layer = v.add_labels(bg, name='bg')
    _run(split, viewer=v, img=algn, background_method='labels layer', background_mask=bg_layer)
    dd, da, aa = (v.layers[f'cell_algn_ch{ch}'] for ch in (0, 1, 3))
    assert abs(np.nanmean(aa.data[:, bg > 0])) < 1e-3
    with pytest.raises(ValueError):
        split(viewer=v, img=raw, background_method='labels layer', background_mask=bg_layer)
    ctxy = v.add_image(np.moveaxis(raw.data, 0, 1), name='ctxy')
    _run(split, viewer=v, img=ctxy, stack_order='CTXY', background_method='histogram mode',
         median_filter=True, photobleaching_correction=True, drop_frames=True,
         frames_range=[0, 6], frames_crop=2)
    assert v.layers['ctxy_0-6_ch3_exp'].data.shape == (6, 92, 92)

    # FRET
    mask = v.add_labels((np.mean(aa.data, axis=0) > 600).astype(int), name='mask')
    pair = dict(viewer=v, config_mode='Default', fret_pair='TagBFP/mBaoJin', DD_img=dd, DA_img=da, AA_img=aa)
    _run(w['e_app_calc'], output_type='Ecorr', bleach_mask=mask, **pair)
    _run(w['e_app_calc'], output_type='E_D', **pair)
    assert np.nanmedian(v.layers['cell_algn_E_D'].data[:, mask.data > 0]) == pytest.approx(0.3, abs=0.05)
    w['e_app_calc'].reset_choices()
    assert 'TagBFP/mBaoJin' in w['e_app_calc'].fret_pair.choices
    _run(w['e_fret_rois'], output_type='E_D', labels=mask, save_data_frame=True, saving_path=tmp_path, **pair)
    assert pd.read_csv(tmp_path / 'cell_algn_mask_FRET.csv')['E_D'].mean() == pytest.approx(0.3, abs=0.02)
    _run(w['cross_calc'], viewer=v, DD_img=dd, DA_img=da, AA_img=aa, mask=mask, saving_path=tmp_path)
    assert 'a_val' in pd.read_csv(tmp_path / 'cell_algn_A_summary.csv').columns
    g_args = dict(viewer=v, DD_img_high_FRET=dd, DA_img_high_FRET=da, AA_img_high_FRET=aa,
                  DD_img_low_FRET=dd, DA_img_low_FRET=da, AA_img_low_FRET=aa, saving_path=tmp_path,
                  config_mode='Default', fret_pair='TagBFP/mBaoJin')
    assert 'TagBFP/mBaoJin' in w['g_calc'].fret_pair.choices
    _run(w['g_calc'], estimation_method='Zal', mask=mask, fragments_num=2,
         select_frames=True, frames_high_FRET=[0, 4], frames_low_FRET=[4, 8], **g_args)
    assert pd.read_csv(tmp_path / 'cell_algn_G.csv')['n_frames_h'][0] == 4
    with pytest.raises(ValueError):
        w['g_calc'](estimation_method='Zal', mask=mask, select_frames=True, frames_high_FRET=[2, 9], **g_args)
    _run(w['g_calc'], estimation_method='Chen', mask_high=v.layers['cell_algn_seg_labels'], mask_low=mask, **g_args)
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from domb_napari._widgets import _common
    plot = _common._DOCKS[(id(v), 'G-factor estimation')].widget()
    assert plot.findChild(NavigationToolbar2QT) is not None       # export of the figure
    assert len(plot.findChild(FigureCanvasQTAgg).figure.axes) == 3
    rois = pd.read_csv(tmp_path / 'cell_algn_G_rois.csv')
    assert set(rois['construct']) == {'high', 'low'} and rois['fret_pair'][0] == 'TagBFP/mBaoJin'

    # series, masks, profiles, data frame
    _run(w['rel_series'], viewer=v, img=aa, values_mode='ΔF/F0', F0_win=3)
    _run(w['der_series'], viewer=v, img=aa, save_MIP=True)
    assert v.layers['cell_algn_ch3_red-green'].data.shape == (7, 80, 80)
    _run(w['dot_mask_calc'], viewer=v, img=aa)
    dots = v.layers['cell_algn_ch3_dots-labels']
    _run(w['up_mask_calc'], viewer=v, img=aa, ROIs_mask=mask, det_frame_index=0, in_ROIs_det=True)
    _run(w['mask_calc'], viewer=v, img=aa, det_frame_index=0)
    _run(w['labels_profile_line'], viewer=v, img=aa, labels=dots, use_simple_baseline=False)
    _run(w['labels_multi_profile_stat'], viewer=v, img_0=aa, img_1=dd, lab=dots, profiles_num='2', stat_method='iqr')
    _run(w['multi_labels_profile_stat'], viewer=v, img=aa, lab_0=dots, lab_1=mask, labels_num='2')
    _run(w['save_df'], img=aa, labels=dots, save_ROIs_distances=True, saving_path=tmp_path)
    df = pd.read_csv(tmp_path / 'cell_ch3_cell_ch3_dots-labels.csv')
    assert len(df) == 2 * dots.data.max() * 8 and 'dist' in df.columns
    v.close()


def test_docked_widgets_are_tabbed():
    if os.environ.get('QT_QPA_PLATFORM') == 'offscreen' or \
       (sys.platform.startswith('linux') and not os.environ.get('DISPLAY')):
        pytest.skip('OpenGL context is not available')
    from qtpy.QtWidgets import QApplication, QScrollArea
    v = napari.Viewer(show=False)
    docks, widgets = [], []
    for name in ('split_channels', 'g_calc', 'e_app_calc'):
        widgets.append(_factories()[name]())
        docks.append(v.window.add_dock_widget(widgets[-1], name=name))
        for _ in range(3):
            QApplication.processEvents()
    assert set(v.window._qt_window.tabifiedDockWidgets(docks[0])) == set(docks[1:])
    for dock, widget in zip(docks, widgets):
        assert isinstance(dock.widget(), QScrollArea) and dock.inner_widget() is widget
    v.close()

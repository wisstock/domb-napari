""" Smoke tests of the widgets with a headless napari viewer.

"""
import importlib
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

SRC = pathlib.Path(__file__).parent.parent / 'src' / 'domb_napari'


def _manifest():
    return yaml.safe_load(open(SRC / 'napari.yaml'))


def _factories():
    out = {}
    for command in _manifest()['contributions']['commands']:
        module, name = command['python_name'].split(':')
        out[name] = getattr(importlib.import_module(module), name)
    return out


def test_core_does_not_import_gui():
    code = ("import sys; import domb_napari; import domb_napari.core.fret, domb_napari.core.registration, "
            "domb_napari.core.series, domb_napari.core.masking, domb_napari.core.preprocess, domb_napari.core.profiles; "
            "print([m for m in ('napari', 'qtpy', 'magicgui', 'matplotlib', 'vispy') if m in sys.modules])")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == '[]'


def test_manifest_is_consistent():
    manifest = _manifest()['contributions']
    ids = {c['id'] for c in manifest['commands']}
    assert len(_factories()) == len(ids) == 15
    assert {w['command'] for w in manifest['widgets']} == ids
    menu_ids = {item['command'] for items in manifest['menus'].values() for item in items if 'command' in item}
    assert menu_ids == ids
    import domb_napari
    assert set(domb_napari.__all__) == set(_factories())
    assert domb_napari.split_channels is _factories()['split_channels']


@pytest.fixture(scope='module')
def viewer():
    # napari canvas needs OpenGL, which is not available on the offscreen
    # Qt platform and on Linux without a display (use xvfb in CI)
    if os.environ.get('QT_QPA_PLATFORM') == 'offscreen' or \
       (sys.platform.startswith('linux') and not os.environ.get('DISPLAY')):
        pytest.skip('OpenGL context is not available')
    v = napari.Viewer(show=False)
    yield v
    v.close()


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
    dd = cell * (1 - eff)
    aa = cell
    da = g * cell * eff + a * aa + d * dd
    channels = [dd, da, 0.02 * cell, aa]
    return np.stack([rng.poisson(np.stack(channels) + 100) for _ in range(n_t)]).astype(np.uint16)


def test_pipeline(viewer, tmp_path):
    rng = np.random.default_rng(0)
    widgets = {name: factory() for name, factory in _factories().items()}
    v = viewer
    raw = v.add_image(_stack(rng), name='cell')

    _run(widgets['dw_registration'], viewer=v, offset_img=raw, input_crop=4, output_crop=4,
         save_matrix=True, saving_path=tmp_path)
    algn = v.layers['cell_algn']
    assert algn.data.shape == (8, 4, 80, 80) and algn.data.dtype == np.float32
    matrix = np.loadtxt(tmp_path / 'cell_affine_matrix.txt')
    assert matrix.shape == (3, 3) and np.abs(matrix - np.eye(3)).max() < 0.3

    _run(widgets['dw_registration'], viewer=v, offset_img=raw, input_crop=4, output_crop=4,
         align_method='load matrix', load_matrix=tmp_path / 'cell_affine_matrix.txt')
    np.testing.assert_allclose(v.layers['cell_algn'].data, algn.data, rtol=1e-5, equal_nan=True)

    _run(widgets['split_channels'], viewer=v, img=algn, median_filter=True)
    for ch in range(4):
        assert v.layers[f'cell_algn_ch{ch}'].data.shape == (8, 80, 80)
    dd, da, aa = (v.layers[f'cell_algn_ch{ch}'] for ch in (0, 1, 3))
    assert dd.data.dtype == np.float32 and dd.data.min() < 0

    ctxy = v.add_image(np.moveaxis(raw.data, 0, 1), name='ctxy')
    _run(widgets['split_channels'], viewer=v, img=ctxy, stack_order='CTXY',
         photobleaching_correction=True, drop_frames=True, frames_range=[0, 6], frames_crop=2)
    assert v.layers['ctxy_0-6_ch3_exp'].data.shape == (6, 92, 92)

    mask = v.add_labels((np.mean(aa.data, axis=0) > 600).astype(int), name='mask')
    assert mask.data.sum() > 600

    fret_widget = widgets['e_app_calc']
    assert 'TagBFP/mBaoJin' in fret_widget.fret_pair.choices
    for output_type in ('Fc', 'E_D', 'Ecorr'):
        _run(fret_widget, viewer=v, config_mode='Default', fret_pair='TagBFP/mBaoJin',
             output_type=output_type, DD_img=dd, DA_img=da, AA_img=aa, bleach_mask=mask)
    e_d = v.layers['cell_algn_E_D'].data
    assert e_d.dtype == np.float32 and 'cell_algn_E_D_norm' in v.layers
    assert np.nanmedian(e_d[:, mask.data > 0]) == pytest.approx(0.3, abs=0.05)
    with pytest.raises(ValueError):
        fret_widget(viewer=v, config_mode='Default', fret_pair='ECFP/EYFP',
                    output_type='E_A', DD_img=dd, DA_img=da, AA_img=aa)
    # choices survive the reset done by napari on layer changes
    fret_widget.reset_choices()
    assert 'TagBFP/mBaoJin' in fret_widget.fret_pair.choices

    _run(widgets['e_fret_rois'], viewer=v, config_mode='Default', fret_pair='TagBFP/mBaoJin',
         output_type='E_D', DD_img=dd, DA_img=da, AA_img=aa, labels=mask,
         save_data_frame=True, saving_path=tmp_path)
    roi_df = pd.read_csv(tmp_path / 'cell_algn_mask_FRET.csv')
    assert roi_df['E_D'].mean() == pytest.approx(0.3, abs=0.02)

    _run(widgets['cross_calc'], viewer=v, DD_img=dd, DA_img=da, AA_img=aa, mask=mask,
         presented_fluorophore='A', saving_path=tmp_path)
    assert len(pd.read_csv(tmp_path / 'cell_algn_A.csv')) == 8
    assert 'a_val' in pd.read_csv(tmp_path / 'cell_algn_A_summary.csv').columns

    config = tmp_path / 'user.yaml'
    g_args = dict(viewer=v, DD_img_high_FRET=dd, DA_img_high_FRET=da, AA_img_high_FRET=aa,
                  DD_img_low_FRET=dd, DA_img_low_FRET=da, AA_img_low_FRET=aa,
                  a=0.014, d=0.265, saving_path=tmp_path)
    _run(widgets['g_calc'], estimation_method='Zal', mask=mask, segment_mask=True, fragments_num=2,
         frames_to_average=0, **g_args)
    assert 'cell_algn_seg_labels' in v.layers
    _run(widgets['g_calc'], estimation_method='Chen', mask_high=v.layers['cell_algn_seg_labels'],
         mask_low=mask, write_to_config=True, config_path=config, fret_pair='test/pair', **g_args)
    g_df = pd.read_csv(tmp_path / 'cell_algn_G.csv')
    assert g_df['method'][0] == 'Chen'
    assert set(yaml.safe_load(open(config))['test/pair']) == {'a', 'd', 'b', 'c', 'G'}

    _run(widgets['rel_series'], viewer=v, img=aa, values_mode='ΔF/F0', F0_win=3)
    _run(widgets['rel_series'], viewer=v, img=aa, values_mode='ΔF', F0_win=3)
    _run(widgets['der_series'], viewer=v, img=aa, save_MIP=True)
    assert v.layers['cell_algn_ch3_red-green'].data.shape == (7, 80, 80)
    assert v.layers['cell_algn_ch3_red-green-MIP'].data.shape == (80, 80)

    _run(widgets['dot_mask_calc'], viewer=v, img=aa)
    dots = v.layers['cell_algn_ch3_dots-labels']
    assert dots.data.max() >= 1
    _run(widgets['up_mask_calc'], viewer=v, img=aa, det_frame_index=0, det_th=0.9,
         save_total_up_mask=True)
    _run(widgets['up_mask_calc'], viewer=v, img=aa, ROIs_mask=mask, det_frame_index=0, in_ROIs_det=True)
    assert set(np.unique(v.layers['cell_algn_ch3_up-labels'].data)) <= {0, 1}
    _run(widgets['mask_calc'], viewer=v, img=aa, det_frame_index=0, up_threshold=0.2)
    assert 'cell_algn_ch3_up-labels' in v.layers

    for mode in ('ΔF', 'ΔF/F0', 'abs.'):
        _run(widgets['labels_profile_line'], viewer=v, img=aa, labels=dots, values_mode=mode)
    _run(widgets['labels_profile_line'], viewer=v, img=aa, labels=dots, use_simple_baseline=False)
    for stat in ('se', 'iqr', 'ci'):
        _run(widgets['labels_multi_profile_stat'], viewer=v, img_0=aa, img_1=dd, lab=dots,
             profiles_num='2', stat_method=stat)
    _run(widgets['multi_labels_profile_stat'], viewer=v, img=aa, lab_0=dots, lab_1=mask, labels_num='2')

    _run(widgets['save_df'], img=aa, labels=dots, time_scale=2.0, save_ROIs_distances=True,
         saving_path=tmp_path)
    df = pd.read_csv(tmp_path / 'cell_ch3_cell_ch3_dots-labels.csv')
    assert len(df) == 2 * dots.data.max() * 8 and list(df['index'][:8]) == list(range(8))
    assert 'dist' in df.columns

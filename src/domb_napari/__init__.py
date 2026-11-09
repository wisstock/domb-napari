""" domb-napari: napari toolkit for analysis of fluorescence redistribution and FRET.

The computational core lives in domb_napari.core and does not import napari.
Widgets are loaded lazily, so the core can be used in scripts without Qt.

"""
import importlib

__version__ = '0.7.0'

_WIDGETS = {'split_channels': 'preprocess',
            'dw_registration': 'preprocess',
            'cross_calc': 'fret',
            'g_calc': 'fret',
            'e_app_calc': 'fret',
            'e_fret_rois': 'fret',
            'der_series': 'series',
            'rel_series': 'series',
            'mask_calc': 'masking',
            'up_mask_calc': 'masking',
            'dot_mask_calc': 'masking',
            'labels_multi_profile_stat': 'profiles',
            'multi_labels_profile_stat': 'profiles',
            'labels_profile_line': 'profiles',
            'save_df': 'profiles'}

__all__ = list(_WIDGETS)


def __getattr__(name):
    if name in _WIDGETS:
        module = importlib.import_module(f'._widgets.{_WIDGETS[name]}', __name__)
        return getattr(module, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

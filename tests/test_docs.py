""" README has to mention every parameter of every widget.

"""
import importlib
import inspect
import pathlib

import pytest
import yaml

pytest.importorskip('napari')

ROOT = pathlib.Path(__file__).parent.parent


def test_readme_lists_all_widget_parameters():
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    manifest = yaml.safe_load(open(ROOT / 'src' / 'domb_napari' / 'napari.yaml'))
    missing = []
    for command in manifest['contributions']['commands']:
        assert command['title'].lower() in readme.lower(), command['title']
        module, name = command['python_name'].split(':')
        factory = getattr(importlib.import_module(module), name)
        for param in inspect.signature(factory.keywords['function']).parameters:
            if param != 'viewer' and f'`{param}`' not in readme:
                missing.append(f'{name}.{param}')
    assert not missing, missing

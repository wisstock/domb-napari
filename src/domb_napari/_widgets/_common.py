""" Shared helpers of the widgets: layers, figures and workers.

"""
import numpy as np

from napari.utils.notifications import show_warning


_RUNNING = set()     # keys of the widgets with a running worker
_DOCKS = {}          # (viewer id, name) -> dock widget with a figure


def start_worker(key:str, worker, on_yield=None, on_return=None):
    """ Starts a napari worker unless the same widget is already running.

    Callbacks are executed in the main thread, so they can safely modify
    layers and create figures.

    """
    if key in _RUNNING:
        show_warning(f'{key}: calculation is already running, wait until it is finished!')
        return None
    _RUNNING.add(key)
    worker.finished.connect(lambda: _RUNNING.discard(key))
    if on_yield is not None:
        worker.yielded.connect(on_yield)
    if on_return is not None:
        worker.returned.connect(on_return)
    worker.start()
    return worker


def robust_limit(img:np.ndarray, percentile:float=99.9, max_px:int=1000000):
    """ Symmetric contrast limit as a percentile of absolute finite values.

    """
    flat = np.asarray(img).ravel()
    if flat.size > max_px:
        flat = flat[::flat.size // max_px]
    flat = np.abs(flat[np.isfinite(flat)])
    if flat.size == 0:
        return 1.0
    limit = float(np.percentile(flat, percentile))
    return limit if limit > 0 else 1.0


def add_or_update_image(viewer, data, name:str, **kwargs):
    """ Updates data of the image layer with the given name or adds a new one.

    """
    try:
        layer = viewer.layers[name]
        layer.data = data
    except KeyError:
        kwargs.setdefault('colormap', 'turbo')
        layer = viewer.add_image(data, name=name, **kwargs)
    return layer


def add_or_update_labels(viewer, data, name:str, opacity:float=1.0, contour:int=1):
    """ Updates data of the labels layer with the given name or adds a new one.

    """
    try:
        layer = viewer.layers[name]
        layer.data = data
    except KeyError:
        layer = viewer.add_labels(data, name=name, opacity=opacity)
        layer.contour = contour
    return layer


def new_figure(**kwargs):
    """ Matplotlib figure which is not registered in the pyplot state.

    """
    from matplotlib.figure import Figure
    return Figure(**kwargs)


def style_axis(ax, xlabel:str='', ylabel:str='', title:str=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='grey', linewidth=.25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize=10)


def show_figure(viewer, fig, name:str):
    """ Shows a figure in a dock widget, the previous one with this name is replaced.

    """
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    key = (id(viewer), name)
    old_dock = _DOCKS.pop(key, None)
    if old_dock is not None:
        try:
            viewer.window.remove_dock_widget(old_dock)
        except Exception:
            pass
    _DOCKS[key] = viewer.window.add_dock_widget(FigureCanvasQTAgg(fig), name=name)
    return _DOCKS[key]


def subsample(n:int, max_n:int=20000, seed:int=0):
    """ Indices of a random subset for plotting.

    """
    if n <= max_n:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, size=max_n, replace=False)

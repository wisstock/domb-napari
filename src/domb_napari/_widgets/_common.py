""" Shared helpers of the widgets: layers, figures and workers.

"""
import numpy as np
from magicgui import magic_factory

from napari.utils.notifications import show_warning


_RUNNING = set()     # keys of the widgets with a running worker
_DOCKS = {}          # (viewer id, name) -> dock widget with a figure


def plugin_widget(widget_init=None, **kwargs):
    """ magic_factory of a plugin widget: scrollable, docked as a new tab.

    napari stacks plugin dock widgets of one area under each other, so
    many open widgets do not fit into the window. After napari docks a
    widget made by this factory, the widget is put into a scroll area and
    its dock is moved into a tab next to the other dock widgets of its area.

    """
    def _init(widget):
        if widget_init is not None:
            widget_init(widget)
        dock_as_tab(widget)

    return magic_factory(widget_init=_init, **kwargs)


def dock_as_tab(widget):
    """ Makes the dock of a widget scrollable and tabifies it with the open
    docks of the same area.

    Runs on the next event loop iteration, when the widget is already
    docked; does nothing for a widget that is not docked.

    """
    try:
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import QApplication
    except Exception:
        return
    if QApplication.instance() is None:
        return
    QTimer.singleShot(0, lambda: _tabify(widget))


def _make_scrollable(dock):
    """ Puts the content of a dock widget into a vertical scroll area.

    napari docks magicgui widgets by their inner QWidget, so the scrollable
    option of magicgui is lost; the scroll area is added to the dock instead.
    The magicgui widget is kept as _magic_widget, which napari uses to find
    the widget of a dock.

    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QFrame, QScrollArea
    inner = dock.widget()
    if inner is None or isinstance(inner, QScrollArea):
        return
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    if hasattr(inner, '_magic_widget'):
        scroll._magic_widget = inner._magic_widget
    dock.setWidget(scroll)
    scroll.setWidget(inner)
    scroll.setMinimumWidth(inner.minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width())


def _tabify(widget):
    from qtpy.QtWidgets import QDockWidget, QMainWindow
    try:
        dock = widget.native
        while dock is not None and not isinstance(dock, QDockWidget):
            dock = dock.parentWidget()
        window = dock.parentWidget() if dock is not None else None
        while window is not None and not isinstance(window, QMainWindow):
            window = window.parentWidget()
        if window is None:
            return
        _make_scrollable(dock)
        if dock.isFloating():
            return
        area = window.dockWidgetArea(dock)
        # a tabbed dock in the background is not visible, so the open state
        # is taken from its toggle action (unchecked while the window is hidden)
        others = [dw for dw in window.findChildren(QDockWidget)
                  if dw is not dock and not dw.isFloating()
                  and (dw.toggleViewAction().isChecked() or not dw.isHidden())
                  and window.dockWidgetArea(dw) == area]
        if others:
            window.tabifyDockWidget(others[-1], dock)
            dock.show()
            dock.raise_()
    except RuntimeError:     # the widget was closed before the timer fired
        pass


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


def figure_widget(fig):
    """ Canvas of a figure with the matplotlib toolbar.

    The toolbar exports the figure (PNG, SVG, PDF, TIFF and other formats
    of matplotlib) and provides zoom and pan.

    """
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from qtpy.QtWidgets import QVBoxLayout, QWidget

    canvas = FigureCanvasQTAgg(fig)
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(NavigationToolbar2QT(canvas, container))
    layout.addWidget(canvas)
    return container


def show_figure(viewer, fig, name:str):
    """ Shows a figure with an export toolbar in a dock widget, the previous
    one with this name is replaced.

    """

    key = (id(viewer), name)
    old_dock = _DOCKS.pop(key, None)
    if old_dock is not None:
        try:
            viewer.window.remove_dock_widget(old_dock)
        except Exception:
            pass
    _DOCKS[key] = viewer.window.add_dock_widget(figure_widget(fig), name=name, tabify=True)
    return _DOCKS[key]


def subsample(n:int, max_n:int=20000, seed:int=0):
    """ Indices of a random subset for plotting.

    """
    if n <= max_n:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, size=max_n, replace=False)

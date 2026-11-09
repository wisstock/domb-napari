""" Colormaps for visualizing fluorescence changes.

"""
import vispy.color


def red_green_cmap():
    """ Red-green colormap for differential images.

    """
    return vispy.color.Colormap([[0.0, 1.0, 0.0],
                                 [0.0, 0.9, 0.0],
                                 [0.0, 0.85, 0.0],
                                 [0.0, 0.0, 0.0],
                                 [0.85, 0.0, 0.0],
                                 [0.9, 0.0, 0.0],
                                 [1.0, 0.0, 0.0]])


def delta_smooth_cmap():
    """ Smooth symmetric colormap for relative intensity changes.

    """
    return vispy.color.Colormap([[0.0, 0.2, 0.6],
                                 [0.0, 0.3, 0.7],
                                 [0.0, 0.4, 0.8],
                                 [0.0, 0.5, 0.9],
                                 [0.2, 0.6, 1.0],
                                 [0.4, 0.7, 1.0],
                                 [0.6, 0.8, 1.0],
                                 [0.0, 0.0, 0.0],
                                 [0.4, 0.2, 0.0],
                                 [0.6, 0.3, 0.0],
                                 [0.8, 0.4, 0.0],
                                 [1.0, 0.5, 0.0],
                                 [1.0, 0.6, 0.2],
                                 [1.0, 0.7, 0.4],
                                 [1.0, 0.8, 0.6]])

""" Helpers for multi-threaded frame-wise processing.

"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np


# numba 'workqueue' threading layer is not thread-safe, so every call of
# a parallel numba kernel is serialized with this lock
NUMBA_LOCK = threading.Lock()


def n_workers(workers:int=None):
    """ Number of threads for frame-wise processing.

    """
    if workers is not None:
        return max(1, int(workers))
    return max(1, min(8, os.cpu_count() or 1))


def map_frames(func, stack:np.ndarray, dtype=None, workers:int=None):
    """ Applies a function to every frame of a stack in a thread pool.

    The function has to release the GIL to benefit from the threads,
    scipy.ndimage filters and interpolators do.

    Parameters
    ----------
    func: callable
        function of a single 2D frame, returns an array
    stack: ndarray [t,x,y]
        input image series
    dtype: dtype
        data type of the output, the type returned by func if None
    workers: int
        number of threads

    Returns
    -------
    output_img: ndarray [t,...]
        stacked results

    """
    first = np.asarray(func(stack[0]))
    out = np.empty((stack.shape[0],) + first.shape,
                   dtype=first.dtype if dtype is None else dtype)
    out[0] = first

    def _job(i):
        out[i] = func(stack[i])

    if stack.shape[0] > 1:
        with ThreadPoolExecutor(n_workers(workers)) as executor:
            list(executor.map(_job, range(1, stack.shape[0])))
    return out

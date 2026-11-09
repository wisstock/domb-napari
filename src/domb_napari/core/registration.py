""" Affine image registration by mutual information.

2D affine registration of two images of the same shape with different
intensity distributions (e.g. two spectral channels of a dual-view splitter).

The similarity measure is the mutual information of the joint intensity
distribution, estimated with Parzen windowing by a cubic B-spline kernel over
a 32-bin histogram (Mattes et al., 2003). The registration is
multi-resolution: at every level of a Gaussian scale space the transform is
optimized by L-BFGS-B with the analytic gradient of the mutual information.

Implementation notes:
- the transform is parametrized in coordinates centered at the image center
  and scaled by the half of the image size, so gradients w.r.t. the linear
  and the translation terms have the same order of magnitude;
- the coarsest level starts with a translation-only optimization;
- on fine levels the metric is estimated with a fixed random subset of pixels;
- the result is accepted only if it has higher mutual information than the
  identity transform at full resolution.

The design of the metric follows dipy.align.imaffine (https://dipy.org).

References
----------
- Parzen, 1962. "On estimation of a probability density function and mode". doi: 10.1214/aoms/1177704472
- Mattes et al., 2003. "PET-CT image registration in the chest using free-form deformations". doi: 10.1109/TMI.2003.809072

"""
import numpy as np

from scipy import ndimage as ndi
from scipy.optimize import minimize

from ._parallel import map_frames


_EPS = np.finfo(np.float64).eps
_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
_TRANSLATION = np.array([2, 5])


def _param_to_matrix(theta:np.ndarray):
    matrix = np.eye(3)
    matrix[0, :] = theta[0:3]
    matrix[1, :] = theta[3:6]
    return matrix


def _apply(matrix:np.ndarray, p0:np.ndarray, p1:np.ndarray):
    return (matrix[0, 0] * p0 + matrix[0, 1] * p1 + matrix[0, 2],
            matrix[1, 0] * p0 + matrix[1, 1] * p1 + matrix[1, 2])


def _interpolate(image:np.ndarray, q0:np.ndarray, q1:np.ndarray):
    """ Bilinear interpolation at arbitrary positions, zero outside of the image.

    Returns the values and the mask of positions with all four neighbours
    inside of the image.

    """
    values = ndi.map_coordinates(image, np.stack((q0, q1)), order=1,
                                 mode='grid-constant', cval=0.0)
    inside = ((q0 >= 0) & (q0 < image.shape[0] - 1) & (q1 >= 0) & (q1 < image.shape[1] - 1))
    return values, inside


def _gradient(image:np.ndarray, q0:np.ndarray, q1:np.ndarray):
    """ Central differences of the interpolated image with a half-pixel step.

    """
    grad = []
    for shift0, shift1 in ((0.5, 0.0), (0.0, 0.5)):
        v_plus, in_plus = _interpolate(image, q0 + shift0, q1 + shift1)
        v_minus, in_minus = _interpolate(image, q0 - shift0, q1 - shift1)
        grad.append(np.where(in_plus & in_minus, v_plus - v_minus, 0.0))
    return grad


def _cubic_spline(x:np.ndarray):
    """ Cubic B-spline kernel of the Parzen window.

    """
    absx = np.abs(x)
    sqrx = x * x
    inner = (4.0 - 6.0 * sqrx + 3.0 * sqrx * absx) / 6.0
    outer = (8.0 - 12.0 * absx + 6.0 * sqrx - sqrx * absx) / 6.0
    return np.where(absx < 1.0, inner, np.where(absx < 2.0, outer, 0.0))


def _cubic_spline_derivative(x:np.ndarray):
    """ Derivative of the cubic B-spline kernel of the Parzen window.

    """
    absx = np.abs(x)
    sqrx = x * x
    inner = np.where(x >= 0.0, -2.0 * x + 1.5 * sqrx, -2.0 * x - 1.5 * sqrx)
    outer = np.where(x >= 0.0, -2.0 + 2.0 * x - 0.5 * sqrx,
                     2.0 + 2.0 * x + 0.5 * sqrx)
    return np.where(absx < 1.0, inner, np.where(absx < 2.0, outer, 0.0))


class _ParzenMI():
    """ Mutual information of static samples and moving values, and its gradient.

    The histogram is padded by two bins at each side, because the cubic
    B-spline kernel has a radius of two bins. Static intensities are binned
    once, the moving ones are spread over five bins by the kernel.

    """
    def __init__(self, static_vals:np.ndarray, moving_img:np.ndarray, nbins:int=32):
        self.nbins = nbins
        self.padding = 2
        self.n = static_vals.size
        inner_bins = nbins - 2 * self.padding

        smin, smax = float(np.min(static_vals)), float(np.max(static_vals))
        mmin, mmax = min(float(np.min(moving_img)), 0.0), float(np.max(moving_img))
        sdelta = (smax - smin) / inner_bins
        self.mdelta = (mmax - mmin) / inner_bins
        if sdelta <= 0 or self.mdelta <= 0:
            raise ValueError('Images for registration must not be constant!')
        self.mmin = mmin / self.mdelta - self.padding

        r = self._bin(static_vals / sdelta - (smin / sdelta - self.padding))
        self.row_offset = r * nbins
        self.smarginal = np.bincount(r, minlength=nbins) / self.n

    def _bin(self, normalized:np.ndarray):
        return np.clip(np.floor(normalized), self.padding,
                       self.nbins - 1 - self.padding).astype(np.intp)

    def __call__(self, moving_vals:np.ndarray, moving_grad:list=None):
        """ Mutual information and, if moving_grad is given, its gradient.

        moving_grad: list of derivatives of moving values w.r.t. the parameters

        """
        nbins = self.nbins
        cn = moving_vals / self.mdelta - self.mmin
        c = self._bin(cn)
        spline_arg = (c - 2) - cn    # kernel argument of the leftmost bin

        joint = np.zeros(nbins * nbins)
        joint_grad = None
        if moving_grad is not None:
            joint_grad = np.zeros((nbins * nbins, len(moving_grad)))
        for offset in range(-2, 3):
            idx = self.row_offset + (c + offset)
            joint += np.bincount(idx, weights=_cubic_spline(spline_arg + (offset + 2)),
                                 minlength=nbins * nbins)
            if moving_grad is not None:
                deriv = _cubic_spline_derivative(spline_arg + (offset + 2))
                for k, dm in enumerate(moving_grad):
                    joint_grad[:, k] -= np.bincount(idx, weights=deriv * dm,
                                                    minlength=nbins * nbins)
        joint = joint.reshape(nbins, nbins) / self.n
        mmarginal = joint.sum(axis=0)

        valid = (joint >= _EPS) & (mmarginal[np.newaxis, :] >= _EPS)
        factor = np.zeros_like(joint)
        ratio = np.divide(joint, mmarginal[np.newaxis, :], out=np.ones_like(joint), where=valid)
        factor[valid] = np.log(ratio[valid])

        keep = valid & (self.smarginal[:, np.newaxis] > _EPS)
        log_smarginal = np.zeros(nbins)
        positive = self.smarginal > _EPS
        log_smarginal[positive] = np.log(self.smarginal[positive])
        value = float(np.sum(joint[keep] * (factor - log_smarginal[:, np.newaxis])[keep]))

        if moving_grad is None:
            return value, None
        joint_grad = joint_grad.reshape(nbins, nbins, -1) / (self.n * self.mdelta)
        return value, np.einsum('ij,ijk->k', factor, joint_grad)


class _LevelMetric():
    """ Negative mutual information of one scale space level as a function
    of the parameters of the transform in normalized coordinates.

    """
    def __init__(self, static_img, moving_img, factor, center, scale,
                 start_matrix, nbins, max_samples, rng):
        self.moving = moving_img
        self.center = center
        self.scale = scale

        p0, p1 = np.meshgrid(np.arange(0, static_img.shape[0], factor),
                             np.arange(0, static_img.shape[1], factor), indexing='ij')
        p0, p1 = p0.ravel(), p1.ravel()
        if max_samples is not None and p0.size > max_samples:
            subset = rng.choice(p0.size, size=int(max_samples), replace=False)
            p0, p1 = p0[subset], p1[subset]
        self.hist = _ParzenMI(static_img[p0, p1].astype(np.float64), moving_img, nbins)

        # pre-aligned sampling points in normalized coordinates
        s0, s1 = _apply(start_matrix, p0.astype(np.float64), p1.astype(np.float64))
        self.u0 = (s0 - center[0]) / scale
        self.u1 = (s1 - center[1]) / scale

    def _moved(self, theta):
        t = _param_to_matrix(theta)
        v0, v1 = _apply(t, self.u0, self.u1)
        return v0 * self.scale + self.center[0], v1 * self.scale + self.center[1]

    def value(self, theta):
        q0, q1 = self._moved(theta)
        return -self.hist(_interpolate(self.moving, q0, q1)[0])[0]

    def value_and_grad(self, theta):
        if not np.all(np.isfinite(theta)):
            return np.inf, np.zeros(6)
        q0, q1 = self._moved(theta)
        moving_vals = _interpolate(self.moving, q0, q1)[0]
        g0, g1 = _gradient(self.moving, q0, q1)
        g0, g1 = g0 * self.scale, g1 * self.scale
        moving_grad = [self.u0 * g0, self.u1 * g0, g0,
                       self.u0 * g1, self.u1 * g1, g1]
        value, grad = self.hist(moving_vals, moving_grad)
        return -value, -grad

    def minimize(self, free, max_fun):
        """ Optimizes the selected parameters starting from the identity.

        """
        free = np.asarray(free)

        def _fun(x):
            theta = _IDENTITY.copy()
            theta[free] = x
            value, grad = self.value_and_grad(theta)
            return value, grad[free]

        opt = minimize(_fun, _IDENTITY[free], method='L-BFGS-B', jac=True,
                       options={'maxfun': int(max_fun), 'gtol': 1e-6, 'ftol': 1e-10})
        theta = _IDENTITY.copy()
        theta[free] = opt.x
        return theta, opt


def _normalize(img:np.ndarray):
    img = np.nan_to_num(np.asarray(img, dtype=np.float64))
    low, high = np.min(img), np.max(img)
    if high <= low:
        raise ValueError('Images for registration must not be constant!')
    return (img - low) / (high - low)


def apply_affine(img:np.ndarray, matrix:np.ndarray, order:int=1):
    """ Resamples a 2D image with an affine matrix.

    The matrix maps pixel indices of the output to pixel indices of the
    input (the scipy.ndimage.affine_transform convention). Pixels without
    data are NaN.

    Returns
    -------
    output_img: ndarray [x,y]
        float32

    """
    return ndi.affine_transform(np.asarray(img, dtype=np.float32), matrix, order=order,
                                output=np.float32, mode='constant', cval=np.nan)


def apply_affine_stack(stack:np.ndarray, matrix:np.ndarray, order:int=1, workers:int=None):
    """ Resamples every frame of a stack with an affine matrix in parallel.

    Returns
    -------
    output_img: ndarray [t,x,y]
        float32, pixels without data are NaN

    """
    return map_frames(lambda frame: apply_affine(frame, matrix, order),
                      stack, dtype=np.float32, workers=workers)


class AffineMap():
    """ Result of the registration.

    Attributes
    ----------
    affine: ndarray [3,3]
        matrix mapping pixel indices of the static image to pixel indices
        of the moving one
    info: dict
        'mi' and 'mi_identity' - mutual information at full resolution with
        the estimated and with the identity transform, 'accepted' - False if
        the optimization failed to improve the identity (affine is identity
        then), 'levels' - list of (nfev, message) of every optimization

    """
    def __init__(self, affine:np.ndarray, info:dict=None):
        self.affine = np.array(affine, dtype=np.float64)
        self.info = {} if info is None else info

    def transform(self, image:np.ndarray, order:int=1):
        """ Transforms an image from the moving grid to the static grid.

        """
        return apply_affine(image, self.affine, order)

    def __str__(self):
        return str(self.affine)


class AffineRegistration():
    """ Multi-resolution affine registration of two images.

    Parameters
    ----------
    nbins: int
        number of bins of the intensity histograms
    level_iters: list of int
        maximal number of the objective function evaluations at each scale,
        from the coarsest to the finest
    sigmas: list of float
        Gaussian smoothing of each scale
    factors: list of int
        sub-sampling factor of the static image grid at each scale
    max_samples: int
        maximal number of pixels used for the metric estimation at one
        scale, a fixed random subset is used if the grid is larger;
        all pixels if None
    min_level_size: int
        minimal size of the shorter side of the sub-sampled grid, coarser
        scales are skipped
    seed: int
        seed of the pixels sampling

    """
    def __init__(self, nbins:int=32, level_iters:list=None, sigmas:list=None,
                 factors:list=None, max_samples:int=200000, min_level_size:int=64,
                 seed:int=0):
        self.nbins = nbins
        self.level_iters = [1000, 100, 30] if level_iters is None else list(level_iters)
        self.sigmas = [3, 1, 0] if sigmas is None else list(sigmas)
        self.factors = [4, 2, 1] if factors is None else list(factors)
        if not (len(self.level_iters) == len(self.sigmas) == len(self.factors) > 0):
            raise ValueError('Iterations, sigmas and factors must have the same non-zero length!')
        self.max_samples = max_samples
        self.min_level_size = min_level_size
        self.seed = seed

    def optimize(self, static:np.ndarray, moving:np.ndarray, starting_affine:np.ndarray=None):
        """ Estimates the transform aligning the moving image to the static one.

        Parameters
        ----------
        static: ndarray [x,y]
            reference image
        moving: ndarray [x,y]
            image to be aligned to the reference one, the same shape
        starting_affine: ndarray [3,3]
            pre-aligning matrix, identity if None

        Returns
        -------
        affine_map: AffineMap

        """
        if np.ndim(static) != 2 or np.shape(static) != np.shape(moving):
            raise ValueError('Static and moving images must be 2D and have the same shape!')
        static, moving = _normalize(static), _normalize(moving)
        rng = np.random.default_rng(self.seed)
        center = (np.array(static.shape) - 1) / 2.0
        scale = float(max(static.shape)) / 2.0
        # pixel -> normalized coordinates and back
        to_norm = np.array([[1 / scale, 0, -center[0] / scale],
                            [0, 1 / scale, -center[1] / scale],
                            [0, 0, 1]])
        to_px = np.linalg.inv(to_norm)

        def _smooth(img, sigma):
            if sigma <= 0:
                return img
            img = ndi.gaussian_filter(img, sigma)
            return (img - img.min()) / (img.max() - img.min())

        matrix = np.eye(3) if starting_affine is None else np.array(starting_affine, dtype=np.float64)
        # a level is used only if its grid is large enough for a reliable
        # estimation of the joint histogram, the finest one is always used
        schedule = [(max_fun, sigma, factor) for max_fun, sigma, factor
                    in zip(self.level_iters, self.sigmas, self.factors)
                    if min(static.shape) // factor >= self.min_level_size]
        if not schedule:
            schedule = [(self.level_iters[-1], self.sigmas[-1], self.factors[-1])]
        levels = []
        for n, (max_fun, sigma, factor) in enumerate(schedule):
            static_s, moving_s = _smooth(static, sigma), _smooth(moving, sigma)
            stages = [_TRANSLATION, np.arange(6)] if n == 0 else [np.arange(6)]
            for free in stages:
                metric = _LevelMetric(static_s, moving_s, int(factor), center, scale,
                                      matrix, self.nbins, self.max_samples, rng)
                theta, opt = metric.minimize(free, max_fun)
                if np.all(np.isfinite(theta)) and opt.fun <= metric.value(_IDENTITY):
                    matrix = to_px @ _param_to_matrix(theta) @ to_norm @ matrix
                levels.append((int(opt.nfev), str(opt.message)))

        # acceptance test at full resolution
        final = _LevelMetric(static, moving, 1, center, scale, np.eye(3),
                             self.nbins, self.max_samples, rng)
        theta_final = (to_norm @ matrix @ to_px)[:2, :].ravel()
        mi = -final.value(theta_final)
        mi_identity = -final.value(_IDENTITY)
        accepted = bool(np.isfinite(mi) and mi > mi_identity)
        info = {'mi': float(mi), 'mi_identity': float(mi_identity),
                'accepted': accepted, 'levels': levels}
        return AffineMap(matrix if accepted else np.eye(3), info)

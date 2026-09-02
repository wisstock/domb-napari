""" Affine image registration by mutual information.

Stand-alone module for domb-napari plugin, provides 2D affine registration of
two images with different intensity distributions (e.g. two spectral channels
of a dual-view splitter).

The similarity measure is the mutual information of the joint intensity
distribution, estimated with Parzen windowing by a cubic B-spline kernel over
a 32-bin histogram. The registration is multi-resolution: the images are
represented as a three-level isotropic Gaussian scale space, and at every
level the six parameters of the affine transform are optimized by L-BFGS-B
with the analytic gradient of the mutual information.

Only 2D images are supported, the intensity histograms are estimated over the
whole image without masking, and image grids are assumed isotropic with unit
pixel size. Anything else raises NotImplementedError.

References
----------
- Parzen, 1962. "On estimation of a probability density function and mode". doi: 10.1214/aoms/1177704472
- Mattes et al., 2003. "PET-CT image registration in the chest using free-form deformations". doi: 10.1109/TMI.2003.809072

"""
import numpy as np
import numpy.linalg as npl

from scipy import ndimage as ndi
from scipy.optimize import minimize


_FLOATING = np.float32       # scale space is kept in single precision
_EPS = 2.2204460492503131e-016


class AffineInversionError(Exception):
    """ Raised when an affine matrix cannot be inverted.

    """
    pass


class AffineInvalidValuesError(Exception):
    """ Raised when an affine matrix contains invalid values.

    """
    pass


def _grid(shape:tuple):
    """ Row and column index grids of the given shape.

    """
    return np.indices(shape, dtype=np.float64)


def _apply_affine_2d(affine:np.ndarray, i:np.ndarray, j:np.ndarray):
    """ Maps grid indices through a homogeneous 3x3 matrix.

    """
    x0 = affine[0, 0] * i + affine[0, 1] * j + affine[0, 2]
    x1 = affine[1, 0] * i + affine[1, 1] * j + affine[1, 2]
    return x0, x1


def _interpolate_scalar_2d(image:np.ndarray, di:np.ndarray, dj:np.ndarray):
    """ Bilinear interpolation of an image at arbitrary positions.

    Positions outside of the image grid contribute zero. A position counts as
    inside only when all four of its neighbours exist, i.e. 0 <= di < R-1 and
    0 <= dj < C-1, so the last row and column are treated as border.

    Parameters
    ----------
    image: ndarray [x,y]
        image to interpolate
    di: ndarray
        row coordinates of the interpolating positions
    dj: ndarray
        column coordinates of the interpolating positions

    Returns
    -------
    values: ndarray
        interpolated intensities
    inside: ndarray of bool
        True where the position lies inside of the image grid

    """
    nr, nc = image.shape
    values = ndi.map_coordinates(image, np.stack((di, dj)), order=1,
                                 mode='grid-constant', cval=0.0)
    inside = ((di >= 0) & (di < nr - 1) & (dj >= 0) & (dj < nc - 1))
    return values, inside


def _transform_2d_affine(image:np.ndarray, out_shape:tuple, affine:np.ndarray,
                         interpolation:str='linear'):
    """ Resamples an image on a grid mapped by an affine matrix.

    Parameters
    ----------
    image: ndarray [x,y]
        image to transform
    out_shape: tuple
        shape of the sampling grid
    affine: ndarray [3,3]
        matrix mapping sampling grid indices to input image indices, identity
        if None
    interpolation: str
        'linear' or 'nearest'

    Returns
    -------
    output_img: ndarray [x,y]
        transformed image of the shape out_shape

    """
    ii, jj = _grid(tuple(int(s) for s in out_shape))
    if affine is None:
        di, dj = ii, jj
    else:
        di, dj = _apply_affine_2d(affine, ii, jj)

    if interpolation == 'linear':
        return _interpolate_scalar_2d(np.asarray(image, dtype=np.float64),
                                      di, dj)[0]

    nr, nc = image.shape
    ri = np.clip(np.rint(di).astype(np.intp), 0, nr - 1)
    rj = np.clip(np.rint(dj).astype(np.intp), 0, nc - 1)
    inside = (di >= -0.5) & (di <= nr - 0.5) & (dj >= -0.5) & (dj <= nc - 0.5)
    return np.where(inside, image[ri, rj], 0).astype(image.dtype)


def _image_gradient_2d(img:np.ndarray, img_world2grid:np.ndarray,
                       img_spacing:np.ndarray, out_shape:tuple,
                       out_grid2world:np.ndarray):
    """ Gradient of an image sampled on an arbitrary grid.

    Central differences taken with a half-pixel step along each axis, the
    image being bilinearly interpolated at the displaced positions. Positions
    whose neighbourhood falls outside of the image get a zero gradient.

    Parameters
    ----------
    img: ndarray [x,y]
        image to differentiate
    img_world2grid: ndarray [3,3]
        matrix mapping physical coordinates to img grid indices
    img_spacing: ndarray [2,]
        pixel size of img along each axis
    out_shape: tuple
        shape of the sampling grid
    out_grid2world: ndarray [3,3]
        matrix mapping sampling grid indices to physical coordinates

    Returns
    -------
    output_grad: ndarray [x,y,2]
        image gradient at every point of the sampling grid

    """
    out_shape = tuple(int(s) for s in out_shape)
    ii, jj = _grid(out_shape)
    x0, x1 = _apply_affine_2d(out_grid2world, ii, jj)
    h = 0.5 * np.asarray(img_spacing, dtype=np.float64)

    out = np.zeros(out_shape + (2,), dtype=np.float64)
    x = [x0, x1]
    # the displaced point is carried over between the two axes and is restored
    # only when both probes succeeded, so a probe that fell outside of the
    # image shifts the evaluation of the next axis as well
    dx = [x0.copy(), x1.copy()]
    for p in range(2):
        probe = list(dx)
        probe[p] = x[p] - h[p]
        q0, q1 = _apply_affine_2d(img_world2grid, probe[0], probe[1])
        v_minus, in_minus = _interpolate_scalar_2d(img, q0, q1)

        probe = list(dx)
        probe[p] = x[p] + h[p]
        q0, q1 = _apply_affine_2d(img_world2grid, probe[0], probe[1])
        v_plus, in_plus = _interpolate_scalar_2d(img, q0, q1)

        good = in_minus & in_plus
        out[..., p] = np.where(good, (v_plus - v_minus) / img_spacing[p], 0.0)
        dx[p] = np.where(~in_minus, x[p] - h[p],
                         np.where(~in_plus, x[p] + h[p], x[p]))
    return out


def _spacings(grid2world:np.ndarray, dim:int=2):
    """ Pixel size along each axis of a grid-to-physical matrix.

    """
    if grid2world is None:
        return np.ones(dim)
    return np.sqrt(np.sum(np.asarray(grid2world)[:dim, :dim] ** 2, axis=0))


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


def _bin_normalize(x:np.ndarray, mval:float, delta:float):
    """ Maps intensities to the range covered by the histogram.

    """
    if delta == 0:
        return np.zeros_like(np.asarray(x, dtype=np.float64))
    return x / delta - mval


def _bin_index(normalized:np.ndarray, nbins:int, padding:int):
    """ Histogram bin of a normalized intensity, clipped to the padded range.

    """
    if not np.all(np.isfinite(normalized)):
        raise AffineInvalidValuesError('Non-finite intensities in histogram!')
    return np.clip(np.floor(normalized), padding,
                   nbins - 1 - padding).astype(np.intp)


def compute_parzen_mi(joint:np.ndarray, joint_gradient:np.ndarray,
                      smarginal:np.ndarray, mmarginal:np.ndarray,
                      mi_gradient:np.ndarray):
    """ Mutual information of a joint distribution and its gradient.

    Parameters
    ----------
    joint: ndarray [nbins,nbins]
        joint intensity distribution
    joint_gradient: ndarray [nbins,nbins,n]
        gradient of the joint distribution w.r.t. the transform parameters
    smarginal: ndarray [nbins,]
        marginal intensity distribution of the static image
    mmarginal: ndarray [nbins,]
        marginal intensity distribution of the moving image
    mi_gradient: ndarray [n,]
        buffer to write the gradient of the mutual information to, the
        gradient is not computed if None

    Returns
    -------
    metric_value: float
        mutual information of the two images

    """
    valid = (joint >= _EPS) & (mmarginal[None, :] >= _EPS)
    factor = np.zeros_like(joint)
    ratio = np.divide(joint, mmarginal[None, :], out=np.ones_like(joint),
                      where=valid)
    factor[valid] = np.log(ratio[valid])

    if mi_gradient is not None:
        mi_gradient[:] = np.einsum('ij,ijk->k', np.where(valid, factor, 0.0),
                                   joint_gradient)

    keep = valid & (smarginal[:, None] > _EPS)
    log_smarginal = np.zeros_like(smarginal)
    positive = smarginal > _EPS
    log_smarginal[positive] = np.log(smarginal[positive])
    metric_value = np.sum(joint[keep] * (factor - log_smarginal[:, None])[keep])
    return float(metric_value)


class AffineTransform2D():
    """ 2D affine transform parametrized by six coefficients.

    The transform is given by the matrix

    T = |t0, t1, t2|
        |t3, t4, t5|
        | 0,  0,  1|

    and the identity corresponds to the parameters [1, 0, 0, 0, 1, 0].

    """
    def __init__(self):
        self.dim = 2
        self.number_of_parameters = 6

    def get_dim(self):
        """ Dimensionality of the transform.

        """
        return self.dim

    def get_number_of_parameters(self):
        """ Number of the transform parameters.

        """
        return self.number_of_parameters

    def get_identity_parameters(self):
        """ Parameters corresponding to the identity transform.

        """
        return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def param_to_matrix(self, theta:np.ndarray):
        """ Homogeneous matrix of the given parameters.

        """
        theta = np.asarray(theta, dtype=np.float64)
        matrix = np.eye(3)
        matrix[0, :] = theta[0:3]
        matrix[1, :] = theta[3:6]
        return matrix

    def jacobian(self, theta:np.ndarray, x:np.ndarray):
        """ Jacobian of T(x) w.r.t. the transform parameters at a single point.

        """
        jac = np.zeros((2, 6))
        jac[0, 0], jac[0, 1], jac[0, 2] = x[0], x[1], 1.0
        jac[1, 3], jac[1, 4], jac[1, 5] = x[0], x[1], 1.0
        return jac

    def jacobian_times_gradient(self, theta:np.ndarray, x0:np.ndarray,
                                x1:np.ndarray, grad:np.ndarray):
        """ Product of the transposed Jacobian and an image gradient.

        Evaluates J(x).T @ grad on a whole grid at once, which is the only
        way the Jacobian enters the gradient of the mutual information.

        Parameters
        ----------
        theta: ndarray [6,]
            transform parameters
        x0: ndarray [x,y]
            first physical coordinate of every grid point
        x1: ndarray [x,y]
            second physical coordinate of every grid point
        grad: ndarray [x,y,2]
            image gradient at every grid point

        Returns
        -------
        output_prod: ndarray [x,y,6]
            derivative of the moving intensity w.r.t. each parameter

        """
        g0 = grad[..., 0]
        g1 = grad[..., 1]
        return np.stack((x0 * g0, x1 * g0, g0,
                         x0 * g1, x1 * g1, g1), axis=-1)


class AffineMap():
    """ Affine transform between the grids of two images.

    Wraps an affine matrix together with the sampling information of the
    domain (static image) and the co-domain (moving image), and applies it to
    an image. This is the object returned by AffineRegistration.optimize, its
    affine attribute holds the estimated matrix.

    Parameters
    ----------
    affine: ndarray [3,3]
        matrix of the transform, identity if None
    domain_grid_shape: tuple
        shape of the static image
    domain_grid2world: ndarray [3,3]
        grid-to-physical matrix of the static image, identity if None
    codomain_grid_shape: tuple
        shape of the moving image
    codomain_grid2world: ndarray [3,3]
        grid-to-physical matrix of the moving image, identity if None

    """
    def __init__(self, affine:np.ndarray,
                 domain_grid_shape:tuple=None, domain_grid2world:np.ndarray=None,
                 codomain_grid_shape:tuple=None, codomain_grid2world:np.ndarray=None):
        self.set_affine(affine)
        self.domain_shape = domain_grid_shape
        self.domain_grid2world = domain_grid2world
        self.codomain_shape = codomain_grid_shape
        self.codomain_grid2world = codomain_grid2world

    def set_affine(self, affine:np.ndarray):
        """ Sets the transform matrix and its inverse.

        """
        if affine is None:
            self.affine = None
            self.affine_inv = None
            return

        affine = np.array(affine)
        if affine.ndim != 2 or affine.shape[0] != affine.shape[1]:
            raise AffineInversionError('Affine transform must be a square 2D matrix!')
        if not np.all(np.isfinite(affine)):
            raise AffineInvalidValuesError('Affine transform contains invalid elements!')
        if not np.all(affine[-1, :-1] == 0.0) or affine[-1, -1] != 1.0:
            raise AffineInvalidValuesError('Malformed homogeneous matrix!')

        self.affine = affine.copy()
        try:
            self.affine_inv = npl.inv(affine)
        except npl.LinAlgError as err:
            raise AffineInversionError('Affine cannot be inverted!') from err

    def _apply_transform(self, image:np.ndarray, interpolation:str='linear',
                         image_grid2world:np.ndarray=None,
                         sampling_grid_shape:tuple=None,
                         sampling_grid2world:np.ndarray=None,
                         resample_only:bool=False, apply_inverse:bool=False):
        """ Applies the transform, or its inverse, to the input image.

        """
        if interpolation not in ('linear', 'nearest'):
            raise ValueError(f'Unknown interpolation method: {interpolation}')

        if sampling_grid_shape is None:
            sampling_grid_shape = self.codomain_shape if apply_inverse else self.domain_shape
        if sampling_grid_shape is None:
            raise ValueError('Unknown sampling info, provide a valid sampling_grid_shape!')

        dim = len(sampling_grid_shape)
        if dim != 2 or np.asarray(image).ndim != 2:
            raise NotImplementedError('Only 2D images are supported!')

        if sampling_grid2world is None:
            sampling_grid2world = self.codomain_grid2world if apply_inverse else self.domain_grid2world
        if sampling_grid2world is None:
            sampling_grid2world = np.eye(dim + 1)

        if image_grid2world is None:
            image_grid2world = self.domain_grid2world if apply_inverse else self.codomain_grid2world
            if image_grid2world is None:
                image_grid2world = np.eye(dim + 1)
        image_world2grid = npl.inv(image_grid2world)

        aff = self.affine_inv if apply_inverse else self.affine
        if (aff is None) or resample_only:
            comp = image_world2grid.dot(sampling_grid2world)
        else:
            comp = image_world2grid.dot(aff.dot(sampling_grid2world))

        return _transform_2d_affine(image, sampling_grid_shape, comp,
                                    interpolation=interpolation)

    def transform(self, image:np.ndarray, interpolation:str='linear',
                  image_grid2world:np.ndarray=None,
                  sampling_grid_shape:tuple=None,
                  sampling_grid2world:np.ndarray=None,
                  resample_only:bool=False):
        """ Transforms an image from the moving grid to the static grid.

        Parameters
        ----------
        image: ndarray [x,y]
            image to transform, sampled on the moving grid
        interpolation: str
            'linear' or 'nearest'
        image_grid2world: ndarray [3,3]
            grid-to-physical matrix of the input image, identity if None
        sampling_grid_shape: tuple
            shape of the output grid, the static image shape if None
        sampling_grid2world: ndarray [3,3]
            grid-to-physical matrix of the output grid, identity if None
        resample_only: bool
            resample the input image without applying the affine transform

        Returns
        -------
        output_img: ndarray [x,y]
            transformed image sampled on the static grid

        """
        return np.array(self._apply_transform(
            image, interpolation=interpolation,
            image_grid2world=image_grid2world,
            sampling_grid_shape=sampling_grid_shape,
            sampling_grid2world=sampling_grid2world,
            resample_only=resample_only, apply_inverse=False))

    def transform_inverse(self, image:np.ndarray, interpolation:str='linear',
                          image_grid2world:np.ndarray=None,
                          sampling_grid_shape:tuple=None,
                          sampling_grid2world:np.ndarray=None,
                          resample_only:bool=False):
        """ Transforms an image from the static grid to the moving grid.

        Parameters are the same as for the transform method.

        """
        return np.array(self._apply_transform(
            image, interpolation=interpolation,
            image_grid2world=image_grid2world,
            sampling_grid_shape=sampling_grid_shape,
            sampling_grid2world=sampling_grid2world,
            resample_only=resample_only, apply_inverse=True))

    def __str__(self):
        return str(self.affine)


class ParzenJointHistogram():
    """ Joint intensity histogram smoothed by Parzen windowing.

    Estimates the joint and marginal probability density functions of two
    images and the derivatives of the joint one w.r.t. the parameters of a
    transform. The histogram is padded by two bins at each side, because the
    cubic B-spline kernel has a radius of two bins and samples at the extreme
    intensities spread beyond the observed range.

    Parameters
    ----------
    nbins: int
        number of bins of the joint and marginal distributions

    """
    def __init__(self, nbins:int):
        self.nbins = nbins
        self.padding = 2
        self.setup_called = False
        self.joint_grad = None
        self.metric_grad = None

    def setup(self, static:np.ndarray, moving:np.ndarray):
        """ Computes the bin size and offset from the observed intensities.

        """
        self.smin = np.min(static)
        self.smax = np.max(static)
        self.mmin = np.min(moving)
        self.mmax = np.max(moving)

        denominator = self.nbins - 2 * self.padding
        self.sdelta = (self.smax - self.smin) / denominator if denominator else 0.0
        self.mdelta = (self.mmax - self.mmin) / denominator if denominator else 0.0
        self.smin = (self.smin / self.sdelta if self.sdelta else 0.0) - self.padding
        self.mmin = (self.mmin / self.mdelta if self.mdelta else 0.0) - self.padding

        self.joint_grad = None
        self.metric_grad = None
        self.metric_val = 0
        self.joint = np.zeros((self.nbins, self.nbins))
        self.smarginal = np.zeros(self.nbins, dtype=np.float64)
        self.mmarginal = np.zeros(self.nbins, dtype=np.float64)
        self.setup_called = True

    def _bins(self, static:np.ndarray, moving:np.ndarray):
        """ Bin indexes of both images and the kernel argument of each pixel.

        """
        rn = _bin_normalize(static, self.smin, self.sdelta)
        r = _bin_index(rn, self.nbins, self.padding).ravel()
        cn = _bin_normalize(moving, self.mmin, self.mdelta)
        c = _bin_index(cn, self.nbins, self.padding).ravel()
        spline_arg = ((c - 2) - np.ravel(cn))  # kernel argument of the leftmost bin
        return r, c, spline_arg

    def update_pdfs_dense(self, static:np.ndarray, moving:np.ndarray):
        """ Computes the joint and marginal distributions of two images.

        The joint distribution is written to the joint attribute, the marginal
        ones to smarginal and mmarginal.

        Parameters
        ----------
        static: ndarray [x,y]
            static image
        moving: ndarray [x,y]
            moving image resampled on the static grid

        """
        if static.shape != moving.shape:
            raise ValueError('Images must have the same shape!')
        if static.ndim != 2:
            raise NotImplementedError('Only 2D images are supported!')
        if not self.setup_called:
            self.setup(static, moving)

        nbins = self.nbins
        r, c, spline_arg = self._bins(static, moving)
        valid_points = static.size

        joint = np.zeros(nbins * nbins, dtype=np.float64)
        total_sum = 0.0
        row_offset = r * nbins
        for offset in range(-2, 3):
            val = _cubic_spline(spline_arg + (offset + 2))
            joint += np.bincount(row_offset + (c + offset), weights=val,
                                 minlength=nbins * nbins)
            total_sum += val.sum()

        self.joint = joint.reshape(nbins, nbins)
        self.smarginal = np.bincount(r, minlength=nbins).astype(np.float64)
        if total_sum > 0:
            self.joint /= valid_points
            self.smarginal /= valid_points
            self.mmarginal = self.joint.sum(axis=0)

    def update_gradient_dense(self, theta:np.ndarray, transform:AffineTransform2D,
                              static:np.ndarray, moving:np.ndarray,
                              grid2world:np.ndarray, mgradient:np.ndarray):
        """ Computes the gradient of the joint distribution.

        The vector of partial derivatives of the joint histogram w.r.t. each
        transform parameter is written to the joint_grad attribute.

        Parameters
        ----------
        theta: ndarray [n,]
            transform parameters to compute the gradient at
        transform: AffineTransform2D
            transform with respect to whose parameters the gradient is taken
        static: ndarray [x,y]
            static image
        moving: ndarray [x,y]
            moving image resampled on the static grid
        grid2world: ndarray [3,3]
            matrix mapping static grid indices to the pre-aligned physical
            coordinates at which the Jacobian is evaluated
        mgradient: ndarray [x,y,2]
            gradient of the moving image

        """
        if static.shape != moving.shape:
            raise ValueError('Images must have the same shape!')
        if static.ndim != 2:
            raise NotImplementedError('Only 2D images are supported!')
        if mgradient.shape != moving.shape + (2,):
            raise ValueError('Invalid gradient field dimensions!')
        if not self.setup_called:
            self.setup(static, moving)

        nbins = self.nbins
        nparams = len(theta)
        r, c, spline_arg = self._bins(static, moving)
        valid_points = static.size

        ii, jj = _grid(static.shape)
        x0, x1 = _apply_affine_2d(grid2world, ii, jj)
        prod = transform.jacobian_times_gradient(theta, x0, x1, mgradient)
        prod = prod.reshape(-1, nparams)

        grad = np.zeros((nbins * nbins, nparams), dtype=np.float64)
        row_offset = r * nbins
        for offset in range(-2, 3):
            val = _cubic_spline_derivative(spline_arg + (offset + 2))
            idx = row_offset + (c + offset)
            for k in range(nparams):
                grad[:, k] -= np.bincount(idx, weights=val * prod[:, k],
                                          minlength=nbins * nbins)

        norm_factor = valid_points * self.mdelta
        if norm_factor > 0:
            grad /= norm_factor
        self.joint_grad = grad.reshape(nbins, nbins, nparams)


class MutualInformationMetric():
    """ Mutual information similarity metric and its gradient.

    Provides the objective function driving the optimizer: the negative
    mutual information of the static image and the moving one transformed by
    the current parameters, together with its analytic gradient.

    Parameters
    ----------
    nbins: int
        number of bins of the intensity histograms

    """
    def __init__(self, nbins:int=32):
        self.histogram = ParzenJointHistogram(nbins)
        self.metric_val = None
        self.metric_grad = None

    def setup(self, transform:AffineTransform2D, static:np.ndarray, moving:np.ndarray,
              static_grid2world:np.ndarray=None, moving_grid2world:np.ndarray=None,
              starting_affine:np.ndarray=None):
        """ Prepares the metric for a pair of images at one scale space level.

        Parameters
        ----------
        transform: AffineTransform2D
            transform with respect to whose parameters the gradient is taken
        static: ndarray [x,y]
            static image
        moving: ndarray [x,y]
            moving image
        static_grid2world: ndarray [3,3]
            grid-to-physical matrix of the static image, identity if None
        moving_grid2world: ndarray [3,3]
            grid-to-physical matrix of the moving image, identity if None
        starting_affine: ndarray [3,3]
            pre-aligning matrix, identity if None

        """
        nparams = transform.get_number_of_parameters()
        self.metric_grad = np.zeros(nparams, dtype=np.float64)
        self.dim = static.ndim
        if self.dim != 2:
            raise NotImplementedError('Only 2D images are supported!')

        if moving_grid2world is None:
            moving_grid2world = np.eye(self.dim + 1)
        if static_grid2world is None:
            static_grid2world = np.eye(self.dim + 1)

        self.transform = transform
        self.static = np.asarray(static).astype(np.float64)
        self.moving = np.asarray(moving).astype(np.float64)
        self.static_grid2world = static_grid2world
        self.static_world2grid = npl.inv(static_grid2world)
        self.moving_grid2world = moving_grid2world
        self.moving_world2grid = npl.inv(moving_grid2world)
        self.static_spacing = _spacings(static_grid2world, self.dim)
        self.moving_spacing = _spacings(moving_grid2world, self.dim)
        self.starting_affine = starting_affine

        prealign = np.eye(self.dim + 1) if starting_affine is None else starting_affine
        self.affine_map = AffineMap(prealign,
                                    domain_grid_shape=static.shape,
                                    domain_grid2world=static_grid2world,
                                    codomain_grid_shape=moving.shape,
                                    codomain_grid2world=moving_grid2world)
        self.histogram.setup(self.static, self.moving)

    def _update_histogram(self):
        """ Updates the distributions for the currently set transform.

        """
        static_values = self.static
        moving_values = self.affine_map.transform(self.moving)
        self.histogram.update_pdfs_dense(static_values, moving_values)
        return static_values, moving_values

    def _update_mutual_information(self, params:np.ndarray, update_gradient:bool=True):
        """ Updates the metric value, and its gradient, at the given parameters.

        """
        current_affine = self.transform.param_to_matrix(params)
        static2prealigned = self.static_grid2world
        if self.starting_affine is not None:
            current_affine = current_affine.dot(self.starting_affine)
            static2prealigned = self.starting_affine.dot(static2prealigned)
        self.affine_map.set_affine(current_affine)

        static_values, moving_values = self._update_histogram()

        hist = self.histogram
        grad = None
        if update_gradient:
            grad = self.metric_grad
            # the moving image gradient is evaluated at the moved points, while
            # the Jacobian is evaluated at the pre-aligned ones
            grid_to_world = current_affine.dot(self.static_grid2world)
            mgrad = _image_gradient_2d(self.moving, self.moving_world2grid,
                                       self.moving_spacing, self.static.shape,
                                       grid_to_world)
            hist.update_gradient_dense(params, self.transform, static_values,
                                       moving_values, static2prealigned, mgrad)

        self.metric_val = compute_parzen_mi(hist.joint, hist.joint_grad,
                                            hist.smarginal, hist.mmarginal, grad)

    def distance(self, params:np.ndarray):
        """ Negative mutual information at the given parameters.

        """
        try:
            self._update_mutual_information(params, update_gradient=False)
        except (AffineInversionError, AffineInvalidValuesError):
            return np.inf
        return -1 * self.metric_val

    def gradient(self, params:np.ndarray):
        """ Gradient of the negative mutual information at the given parameters.

        """
        try:
            self._update_mutual_information(params, update_gradient=True)
        except (AffineInversionError, AffineInvalidValuesError):
            return 0 * self.metric_grad
        return -1 * self.metric_grad

    def distance_and_gradient(self, params:np.ndarray):
        """ Negative mutual information and its gradient in a single pass.

        """
        try:
            self._update_mutual_information(params, update_gradient=True)
        except (AffineInversionError, AffineInvalidValuesError):
            return np.inf, 0 * self.metric_grad
        return -1 * self.metric_val, -1 * self.metric_grad


class IsotropicScaleSpace():
    """ Multi-resolution representation of an image.

    A list of images produced by smoothing the input with Gaussian kernels of
    decreasing width, each one associated with a sub-sampled grid. Level 0 is
    the finest one, i.e. the unsmoothed image on the original grid.

    Parameters
    ----------
    image: ndarray [x,y]
        input image
    factors: list of float
        sub-sampling factor of each scale, from the coarsest to the finest
    sigmas: list of float
        smoothing parameter of each scale, from the coarsest to the finest
    image_grid2world: ndarray [3,3]
        grid-to-physical matrix of the input image, identity if None
    input_spacing: ndarray [2,]
        pixel size of the input image, unit if None

    """
    def __init__(self, image:np.ndarray, factors:list, sigmas:list,
                 image_grid2world:np.ndarray=None, input_spacing:np.ndarray=None):
        self.dim = image.ndim
        self.num_levels = len(factors)
        if len(sigmas) != self.num_levels:
            raise ValueError('Sigmas and factors must have the same length!')

        input_size = np.array(image.shape)
        img = (image.astype(np.float64) - np.min(image)) / (np.max(image) - np.min(image))

        self.images = [img.astype(_FLOATING)]
        self.domain_shapes = [input_size.astype(np.int32)]
        if input_spacing is None:
            input_spacing = np.ones(self.dim, dtype=np.int32)
        self.spacings = [input_spacing]
        self.affines = [image_grid2world]

        min_index = int(np.argmin(input_spacing))
        for i in range(1, self.num_levels):
            factor = factors[self.num_levels - 1 - i]
            shrink_factors = np.zeros(self.dim)
            new_spacing = np.zeros(self.dim)
            shrink_factors[min_index] = factor
            new_spacing[min_index] = input_spacing[min_index] * factor
            for j in range(self.dim):
                if j == min_index:
                    continue
                shrink_factors[j] = factor
                new_spacing[j] = input_spacing[j] * factor
                min_diff = np.abs(new_spacing[j] - new_spacing[min_index])
                for f in range(1, int(factor)):
                    diff = np.abs(input_spacing[j] * f - new_spacing[min_index])
                    if diff < min_diff:
                        shrink_factors[j] = f
                        new_spacing[j] = input_spacing[j] * f
                        min_diff = diff

            extended = np.append(shrink_factors, [1])
            if image_grid2world is not None:
                affine = image_grid2world.dot(np.diag(extended))
            else:
                affine = np.diag(extended)
            output_size = (input_size / shrink_factors).astype(np.int32)
            new_sigmas = np.ones(self.dim) * sigmas[self.num_levels - i - 1]

            filtered = ndi.gaussian_filter(image.astype(np.float64), new_sigmas)
            filtered = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered))

            self.images.append(filtered.astype(_FLOATING))
            self.domain_shapes.append(output_size)
            self.spacings.append(new_spacing)
            self.affines.append(affine)

    def get_image(self, level:int):
        """ Smoothed image of the given level.

        """
        return self.images[level]

    def get_domain_shape(self, level:int):
        """ Grid shape of the given level.

        """
        return self.domain_shapes[level]

    def get_affine(self, level:int):
        """ Grid-to-physical matrix of the given level.

        """
        return self.affines[level]

    def get_spacing(self, level:int):
        """ Pixel size of the given level.

        """
        return self.spacings[level]


class AffineRegistration():
    """ Multi-resolution affine registration of two images.

    Estimates the affine transform that brings a moving image onto a static
    one by maximizing their mutual information. The optimization starts at the
    coarsest scale and the solution of each level is used as the pre-alignment
    of the next one.

    Parameters
    ----------
    metric: MutualInformationMetric
        similarity metric, mutual information with 32 bins if None
    level_iters: list of int
        maximal number of the objective function evaluations at each scale,
        from the coarsest to the finest, [1000, 500, 100] if None
    sigmas: list of float
        smoothing parameter of each scale, [3, 1, 0] if None
    factors: list of float
        sub-sampling factor of each scale, [4, 2, 1] if None
    method: str
        optimization method, any gradient-based method of scipy.optimize
    options: dict
        extra options for the optimizer, {'gtol': 1e-4} if None

    """
    def __init__(self, metric:MutualInformationMetric=None, level_iters:list=None,
                 sigmas:list=None, factors:list=None, method:str='L-BFGS-B',
                 options:dict=None):
        self.metric = metric if metric is not None else MutualInformationMetric()

        if level_iters is None:
            level_iters = [1000, 500, 100]
        self.level_iters = level_iters
        self.levels = len(level_iters)
        if self.levels == 0:
            raise ValueError('The iterations sequence cannot be empty!')

        self.options = options
        self.method = method
        self.factors = [4, 2, 1] if factors is None else factors
        self.sigmas = [3, 1, 0] if sigmas is None else sigmas

    def _init_optimizer(self, static:np.ndarray, moving:np.ndarray,
                        transform:AffineTransform2D, params0:np.ndarray,
                        static_grid2world:np.ndarray, moving_grid2world:np.ndarray,
                        starting_affine:np.ndarray):
        """ Normalizes the input images and builds their scale spaces.

        """
        self.dim = static.ndim
        if self.dim != 2:
            raise NotImplementedError('Only 2D images are supported!')
        self.transform = transform
        self.nparams = transform.get_number_of_parameters()

        if params0 is None:
            params0 = transform.get_identity_parameters()
        self.params0 = params0

        if starting_affine is None:
            self.starting_affine = np.eye(self.dim + 1)
        elif isinstance(starting_affine, np.ndarray):
            self.starting_affine = starting_affine
        else:
            raise NotImplementedError('Starting affine must be a matrix or None!')

        static_spacing = _spacings(static_grid2world, self.dim)
        moving_spacing = _spacings(moving_grid2world, self.dim)

        smin, smax = np.min(static), np.max(static)
        static = (static.astype(np.float64) - smin) / (smax - smin)
        mmin, mmax = np.min(moving), np.max(moving)
        moving = (moving.astype(np.float64) - mmin) / (mmax - mmin)

        self.moving_ss = IsotropicScaleSpace(moving, self.factors, self.sigmas,
                                             image_grid2world=moving_grid2world,
                                             input_spacing=moving_spacing)
        self.static_ss = IsotropicScaleSpace(static, self.factors, self.sigmas,
                                             image_grid2world=static_grid2world,
                                             input_spacing=static_spacing)

    def optimize(self, static:np.ndarray, moving:np.ndarray,
                 transform:AffineTransform2D, params0:np.ndarray,
                 static_grid2world:np.ndarray=None, moving_grid2world:np.ndarray=None,
                 starting_affine:np.ndarray=None, ret_metric:bool=False):
        """ Estimates the transform aligning the moving image to the static one.

        Parameters
        ----------
        static: ndarray [x,y]
            reference image
        moving: ndarray [x,y]
            image to be aligned to the reference one
        transform: AffineTransform2D
            transform to be estimated
        params0: ndarray [n,]
            initial transform parameters, identity if None
        static_grid2world: ndarray [3,3]
            grid-to-physical matrix of the static image, identity if None
        moving_grid2world: ndarray [3,3]
            grid-to-physical matrix of the moving image, identity if None
        starting_affine: ndarray [3,3]
            pre-aligning matrix, identity if None
        ret_metric: bool
            also return the optimal parameters and the metric value

        Returns
        -------
        affine_map: AffineMap
            estimated transform, its affine attribute holds the matrix and its
            transform method applies it to an image

        """
        self._init_optimizer(static, moving, transform, params0,
                             static_grid2world, moving_grid2world,
                             starting_affine)

        original_static_shape = self.static_ss.get_image(0).shape
        original_static_grid2world = self.static_ss.get_affine(0)
        original_moving_shape = self.moving_ss.get_image(0).shape
        original_moving_grid2world = self.moving_ss.get_affine(0)
        affine_map = AffineMap(None,
                               domain_grid_shape=original_static_shape,
                               domain_grid2world=original_static_grid2world,
                               codomain_grid_shape=original_moving_shape,
                               codomain_grid2world=original_moving_grid2world)

        opt = None
        for level in range(self.levels - 1, -1, -1):
            self.current_level = level
            max_iter = self.level_iters[-1 - level]

            # the smoothed static image is resampled on the grid of this level,
            # while the moving one stays at full resolution
            smooth_static = self.static_ss.get_image(level)
            current_static_shape = self.static_ss.get_domain_shape(level)
            current_static_grid2world = self.static_ss.get_affine(level)
            current_affine_map = AffineMap(None,
                                           domain_grid_shape=current_static_shape,
                                           domain_grid2world=current_static_grid2world,
                                           codomain_grid_shape=original_static_shape,
                                           codomain_grid2world=original_static_grid2world)
            current_static = current_affine_map.transform(smooth_static)
            current_moving = self.moving_ss.get_image(level)
            current_moving_grid2world = original_moving_grid2world

            self.metric.setup(transform, current_static, current_moving,
                              static_grid2world=current_static_grid2world,
                              moving_grid2world=current_moving_grid2world,
                              starting_affine=self.starting_affine)

            if self.options is None:
                self.options = {'gtol': 1e-4}
            if self.method == 'L-BFGS-B':
                self.options['maxfun'] = max_iter
            else:
                self.options['maxiter'] = max_iter

            opt = minimize(self.metric.distance_and_gradient, self.params0,
                           method=self.method, jac=True, options=self.options)

            # the level solution becomes the pre-alignment of the next one
            self.starting_affine = self.transform.param_to_matrix(opt.x).dot(self.starting_affine)
            self.params0 = self.transform.get_identity_parameters()

        affine_map.set_affine(self.starting_affine)
        if ret_metric:
            return affine_map, opt.x, opt.fun
        return affine_map

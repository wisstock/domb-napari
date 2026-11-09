import numpy as np
import pytest
from scipy import ndimage as ndi

from domb_napari.core import registration as rg

MATRIX = np.array([[1.004, 0.003, 2.3], [-0.002, 0.997, -3.1], [0, 0, 1]])


def _pair(shape, seed, n_dots):
    rng = np.random.default_rng(seed)
    img = np.zeros(shape)
    img[rng.integers(20, shape[0] - 20, n_dots), rng.integers(20, shape[1] - 20, n_dots)] = rng.uniform(500, 3000, n_dots)
    img = ndi.gaussian_filter(img, 4) * 200
    rows, cols = np.mgrid[:shape[0], :shape[1]]
    img += 300 * np.exp(-((rows - shape[0] / 2) ** 2 + (cols - shape[1] / 2) ** 2) / (2 * (shape[0] / 4) ** 2))
    moving = ndi.affine_transform(img, MATRIX, order=1)
    return rng.poisson(img + 100).astype(float), rng.poisson(moving * 0.6 + 100).astype(float)


def _corner_error(matrix, shape):
    h, w = shape
    corners = np.array([[0, 0, 1], [0, w, 1], [h, 0, 1], [h, w, 1], [h / 2, w / 2, 1]]).T
    return np.abs((matrix @ corners - np.linalg.inv(MATRIX) @ corners)[:2]).max()


@pytest.mark.parametrize('shape,seed', [((256, 256), 0), ((256, 256), 1), ((384, 192), 0), ((384, 192), 2)])
def test_registration_recovers_affine(shape, seed):
    static, moving = _pair(shape, seed, 30)
    affine_map = rg.AffineRegistration().optimize(static, moving)
    assert affine_map.info['accepted']
    assert affine_map.info['mi'] > affine_map.info['mi_identity']
    assert _corner_error(affine_map.affine, shape) < 0.3


def test_unrelated_images_are_not_transformed():
    rng = np.random.default_rng(0)
    static, moving = rng.random((64, 64)), rng.random((64, 64))
    affine_map = rg.AffineRegistration().optimize(static, moving)
    if not affine_map.info['accepted']:
        assert np.array_equal(affine_map.affine, np.eye(3))


def test_apply_affine_stack():
    rng = np.random.default_rng(1)
    stack = rng.integers(0, 4000, (6, 40, 40)).astype(np.uint16)
    out = rg.apply_affine_stack(stack, MATRIX)
    assert out.dtype == np.float32 and out.shape == stack.shape
    ref = ndi.affine_transform(stack[3].astype(np.float64), MATRIX, order=1, cval=np.nan)
    np.testing.assert_allclose(out[3], ref, rtol=1e-5, equal_nan=True)
    assert np.isnan(out).any()                       # pixels without data
    np.testing.assert_allclose(rg.apply_affine_stack(stack, np.eye(3)), stack)
    assert np.array_equal(rg.AffineMap(MATRIX).transform(stack[3]), out[3], equal_nan=True)


def test_shape_validation():
    with pytest.raises(ValueError):
        rg.AffineRegistration().optimize(np.zeros((8, 8)), np.zeros((8, 9)))
    with pytest.raises(ValueError):
        rg.AffineRegistration().optimize(np.ones((8, 8)), np.ones((8, 8)))

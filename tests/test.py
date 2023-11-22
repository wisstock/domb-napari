from unittest.mock import Mock, patch
import pytest
import numpy as np
from domb_napari._widget import split_channels

@pytest.fixture
def viewer_mock(make_napari_viewer):
    viewer = make_napari_viewer()
    return viewer

@pytest.fixture
def image_mock():
    img_mock = Mock()
    img_mock.name = "test_image"
    img_mock.data = np.random.rand(2, 3, 10, 10)  # Replace with your actual data shape
    return img_mock

@patch('skimage.filters.gaussian')
@patch('domb.utils.masking.pb_exp_correction')
def test_split_channels(mock_pb_correction, mock_gaussian, viewer_mock, image_mock):
    # Mock the return values of the functions you're patching
    mock_gaussian.return_value = image_mock.data
    mock_pb_correction.return_value = (image_mock.data, None, 0.9)

    # Call the function to test
    split_channels(viewer_mock, image_mock, gaussian_blur=True, photobleaching_correction=True)

    # Assert that the expected functions were called
    mock_gaussian.assert_called_once_with(image_mock.data, sigma=0.75, channel_axis=0)
    mock_pb_correction.assert_called_once()

    # Assert that the viewer was updated or the layer was added
    assert viewer_mock.layers[image_mock.name + '_ch0'].data is not None
    assert np.any(np.isclose(image_mock.data, viewer_mock.layers[image_mock.name + '_ch0'].data))

if __name__ == '__main__':
    pytest.main()
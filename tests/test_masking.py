import numpy as np

from domb_napari.core import masking


def test_mask_segmentation_areas():
    mask = np.zeros((256, 256), dtype=bool)
    mask[40:200, 60:180] = True
    seg = masking.mask_segmentation(mask, 30)
    assert np.array_equal(seg > 0, mask)
    areas = np.bincount(seg.ravel())[1:]
    assert len(areas) == 30
    assert areas.max() / areas.min() < 3
    assert np.array_equal(seg, masking.mask_segmentation(mask, 30))


def test_in_rois_up_mask_keeps_labels_of_adjacent_rois():
    img = np.zeros((40, 40))
    img[5:15, 5:15] = 10.0       # bright site in ROI 3
    img[22:32, 22:32] = 10.0     # bright site in ROI 7
    rois = np.zeros((40, 40), dtype=int)
    # L-shaped ROI 7 whose bounding box covers the whole ROI 3
    rois[0:40, 20:40] = 7
    rois[35:40, 0:20] = 7
    rois[2:18, 2:18] = 3
    labels = masking.up_mask(img, det_th=0.5, rois_mask=rois, in_rois_method='threshold',
                             in_rois_th_corr=1.0, opening_fp=0, dilation_fp=0)
    assert set(np.unique(labels)) == {0, 3, 7}
    assert np.all(rois[labels == 3] == 3) and np.all(rois[labels == 7] == 7)
    assert labels[10, 10] == 3 and labels[27, 27] == 7


def test_global_up_mask_and_intensity_mask():
    img = np.zeros((40, 40))
    img[5:15, 5:15] = 10.0
    img[25:35, 25:35] = -10.0
    img[0, 0] = np.nan
    up = masking.up_mask(img, det_th=0.25)
    assert up.max() == 1 and up[10, 10] == 1 and up[30, 30] == 0
    down = masking.intensity_mask(img, masking_mode='down', down_threshold=-0.5)
    assert down.max() == 1 and down[30, 30] == 1 and down[10, 10] == 0


def test_dots_mask():
    img = np.zeros((3, 64, 64))
    for r, c in ((16, 16), (16, 48), (48, 32)):
        img[:, r, c] = 100.0
    labels = masking.dots_mask(img, background_level=75, detection_level=25,
                               minimal_distance=2, mask_radius=4)
    assert labels.max() == 3
    assert len({labels[16, 16], labels[16, 48], labels[48, 32]}) == 3
    assert labels[16, 21] == 0 and labels[16, 20] == labels[16, 16]

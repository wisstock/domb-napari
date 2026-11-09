""" Segmentation of bright dots and of intensity changes.

"""
import numpy as np

from scipy import ndimage as ndi
from scipy.cluster.vq import kmeans2

from skimage import feature
from skimage import filters
from skimage import measure
from skimage import morphology
from skimage import segmentation


# scale of the det_th threshold of the global up masking
GLOBAL_TH_SCALE = 0.1


def _abs_max(img:np.ndarray):
    val = np.nanmax(np.abs(img))
    return float(val) if np.isfinite(val) else 0.0


def _finite(img:np.ndarray):
    """ Image with NaN replaced by zeros, NaN pixels are never detected.

    """
    return np.nan_to_num(np.asarray(img, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def mask_segmentation(input_mask:np.ndarray, fragment_num:int=30, seed:int=42):
    """ Splits a binary mask into compact fragments of similar area.

    Pixels of the mask are clustered by their coordinates with k-means.

    Parameters
    ----------
    input_mask: ndarray [x,y]
        input binary mask
    fragment_num: int
        number of fragments

    Returns
    -------
    output_mask: ndarray [x,y]
        labels of fragments, 1..fragment_num

    """
    input_mask = np.asarray(input_mask) != 0
    if input_mask.ndim != 2:
        raise ValueError('Input mask must be 2D array!')
    coords = np.argwhere(input_mask).astype(np.float64)
    if coords.shape[0] < fragment_num:
        raise ValueError('The mask is smaller than the number of fragments!')
    _, cluster = kmeans2(coords, fragment_num, minit='++', seed=seed, iter=25)
    _, cluster = np.unique(cluster, return_inverse=True)   # drop empty clusters
    rois = np.zeros(input_mask.shape, dtype=np.int32)
    rois[input_mask] = cluster + 1
    return rois


def dots_mask(input_img:np.ndarray, background_level:float=75.0,
              detection_level:float=25.0, minimal_distance:int=2, mask_radius:int=5):
    """ Labels of bright dots detected on the maximal intensity projection.

    1. background removal: MIP minus its background_level percentile;
    2. local maxima brighter than detection_level % of the maximum;
    3. disk of mask_radius around every maximum;
    4. watershed split of merged disks.

    Parameters
    ----------
    input_img: ndarray [t,x,y]
        image series
    background_level: float
        percentile of MIP intensity subtracted as background
    detection_level: float
        minimal intensity of a dot, % of the maximal MIP intensity
    minimal_distance: int
        minimal distance between the dots and between the centers of labels, px
    mask_radius: int
        radius of the round mask of a dot, px

    Returns
    -------
    labels: ndarray [x,y]

    """
    input_mip = _finite(np.nanmax(input_img, axis=0))
    detection_mip = np.clip(input_mip - np.percentile(input_mip, background_level), 0, None)

    peaks_coord = feature.peak_local_max(detection_mip,
                                         min_distance=minimal_distance,
                                         threshold_rel=detection_level / 100.0)
    peaks_img = np.zeros(input_mip.shape, dtype=bool)
    peaks_img[tuple(peaks_coord.T)] = True
    peaks_mask = morphology.dilation(peaks_img, footprint=morphology.disk(mask_radius))

    mask_dist_img = ndi.distance_transform_edt(peaks_mask)
    centers_coord = feature.peak_local_max(mask_dist_img, min_distance=minimal_distance)
    mask_centers = np.zeros(input_mip.shape, dtype=bool)
    mask_centers[tuple(centers_coord.T)] = True

    return segmentation.watershed(-mask_dist_img,
                                  markers=measure.label(mask_centers),
                                  mask=peaks_mask,
                                  compactness=10)


def threshold_mask(img:np.ndarray, threshold:float, mode:str='up'):
    """ Binary mask of intensity gain or loss with fixed post-processing.

    The threshold is relative to the maximal absolute intensity of the image.
    Post-processing: erosion (disk 2), dilation (disk 1), holes filling, so
    objects thinner than 5 px are removed and the rest are 1 px smaller.

    """
    img = _finite(img)
    level = _abs_max(img) * threshold
    mask = img >= level if mode == 'up' else img <= level
    mask = morphology.erosion(mask, footprint=morphology.disk(2))
    mask = morphology.dilation(mask, footprint=morphology.disk(1))
    return ndi.binary_fill_holes(mask)


def _up_detection(img, method, th, op_f, d_f):
    if method == 'threshold':
        up_m = threshold_mask(img, th, mode='up')
    elif method == 'otsu':
        img = _finite(img)
        up_m = img > filters.threshold_otsu(img) if np.ptp(img) > 0 else np.zeros(img.shape, dtype=bool)
    else:
        raise ValueError("Unknown detection method! Use 'threshold' or 'otsu'.")
    if op_f != 0:
        up_m = morphology.opening(up_m, footprint=morphology.disk(op_f))
    if d_f != 0:
        up_m = morphology.dilation(up_m, footprint=morphology.disk(d_f))
    return up_m.astype(bool)


def up_mask(detection_img:np.ndarray, det_th:float=0.25,
            rois_mask:np.ndarray=None, in_rois_method:str='otsu',
            in_rois_th_corr:float=0.1,
            opening_fp:int=1, dilation_fp:int=0):
    """ Labels of regions with high intensity.

    Global mode (rois_mask is None): threshold at
    max|img| * det_th * GLOBAL_TH_SCALE, connected components are labeled.

    In-ROIs mode: detection is done separately inside of the bounding box of
    every ROI and is restricted to the pixels of this ROI, detected sites
    keep the label of their ROI. Method 'threshold' uses
    max|box| * det_th * in_rois_th_corr, method 'otsu' - Otsu threshold.

    Returns
    -------
    labels: ndarray [x,y]

    """
    if rois_mask is None:
        mask = _up_detection(detection_img, 'threshold', det_th * GLOBAL_TH_SCALE,
                             opening_fp, dilation_fp)
        return measure.label(mask)

    rois_mask = np.asarray(rois_mask)
    if rois_mask.shape != detection_img.shape:
        raise ValueError('ROIs mask and the detection frame should have the same shape!')
    up_labels = np.zeros_like(rois_mask)
    for region in measure.regionprops(rois_mask):
        r0, c0, r1, c1 = region.bbox
        box = (slice(r0, r1), slice(c0, c1))
        one_roi_mask = _up_detection(detection_img[box], in_rois_method,
                                     det_th * in_rois_th_corr, opening_fp, dilation_fp)
        selected = one_roi_mask & (rois_mask[box] == region.label)
        up_labels[box][selected] = region.label
    return up_labels


def intensity_mask(detection_img:np.ndarray, masking_mode:str='up',
                   up_threshold:float=0.2, down_threshold:float=-0.9,
                   opening_footprint:int=0):
    """ Labels of regions with intensity gain ('up') or loss ('down').

    Thresholds are relative to the maximal absolute intensity of the frame.

    """
    if masking_mode == 'up':
        mask = threshold_mask(detection_img, up_threshold, mode='up')
    elif masking_mode == 'down':
        mask = threshold_mask(detection_img, down_threshold, mode='down')
    else:
        raise ValueError("Unknown masking mode! Use 'up' or 'down'.")
    if opening_footprint != 0:
        mask = morphology.opening(mask, footprint=morphology.disk(opening_footprint))
        mask = morphology.dilation(mask, footprint=morphology.disk(1))
    return measure.label(mask)

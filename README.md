domb-napari
===========

[![Stand With Ukraine](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/banner-direct-single.svg)](https://stand-with-ukraine.pp.ua)

[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/domb-napari)](https://napari-hub.org/plugins/domb-napari)
![PyPI - Version](https://img.shields.io/pypi/v/domb-napari)
![PyPI - License](https://img.shields.io/pypi/l/domb-napari)
![Website](https://img.shields.io/website?up_message=domb.bio%2Fnapari&up_color=%2323038C93&url=https%3A%2F%2Fdomb.bio%2Fnapari%2F)

__napari Toolkit of Department of Molecular Biophysics <br /> Bogomoletz Institute of Physiology of NAS of Ukraine, Kyiv,  Ukraine__

napari plugin for analyzing fluorescence-labeled proteins redistribution. Offers widgets designed for analyzing the redistribution of fluorescence-labeled proteins in widefield epifluorescence time-lapse acquisitions. Particularly useful for studying various phenomena such as calcium-dependent translocation of neuronal calcium sensors, synaptic receptor traffic during long-term plasticity induction, and membrane protein tracking.

![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/translocation.gif)
__Hippocalcin (neuronal calcium sensor) redistributes in dendritic branches upon NMDA application__



# Detection of fluorescence redistributions
A set of widgets designed for detecting fluorescence intensity redistribution through the analysis of differential image series (red-green detection).

Inspired by [Dovgan et al., 2010](https://pubmed.ncbi.nlm.nih.gov/20704590/) and [Osypenko et al., 2019](https://www.sciencedirect.com/science/article/pii/S0969996119301974?via%3Dihub).

## Image preprocessing
Provides functions for preprocessing multi-channel fluorescence acquisitions:
- If the input image has 4 dimensions (time, channel, x-axis, y-axis), channels will be split into individual 3 dimensions images (time, x-axis, y-axis) with the `_ch%index%` suffix.
- If the `gaussian blur` option is selected, the image will be blurred with a Gaussian filter using sigma=`gaussian sigma`.
- If the `photobleaching correction` option is selected, the image will undergo correction with exponential (method `exp`) or bi-exponential (method `bi_exp`) fitting.

![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/pic_0.png)

## Red-green series
Primary method for detecting fluorescent-labeled targets redistribution in time. Returns a series of differential images representing the intensity difference between the current frame and the previous one as new image with the `_red-green` suffix.

Parameters:

- `left frames` - number of previous frames for pixel-wise averaging.
- `space frames` - number of frames between the last left and first right frames.
- `right frames` - number of subsequent frames for pixel-wise averaging.
- `save mask series` - if selected, a series of labels will be created for each frame of the differential image with the threshold `insertion threshold`.

![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/pic_1.png)

## Up masking
Generates labels for insertion sites (regions with increasing intensity) based on `-red-green` images. Returns labels layer with `_up-labels` suffix.

Parameters:

- `detection img index` - index of the frame from `-red-green` image used for insertion sites detection.
- `insertion threshold` - threshold value for insertion site detection, intensity on selected `_red-green` frame normalized in -1 - 0 range.
- `save mask` - if selected, a total up mask (containing all ROIs) will be created with the `_up-mask` suffix.

![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/pic_2.png)

## Intensity masking
Extension of __Up Masking__ widget. Detects regions with increasing (`masking mode` - `up`) or decreasing (`masking mode` - `down`) intensity in `-red-green` images. Returns a labels layer with either `_up-labels` or `_down-labels` suffix, depending on the mode.



# Traffic monitoring with pH-sensitive tag
A collection of widgets designed for the analysis of image series containing the pH-sensitive fluorescence protein Superecliptic pHluorin (SEP).

Insipred by [Fujii et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28474392/) and [Sposini et al., 2020](https://www.nature.com/articles/s41596-020-0371-z).

## SEP image preprocessing
Processes image series obtained through repetitive pH exchange methods (such as U-tube or ppH approaches). Frames with odd indexes, including index 0, are interpreted as images acquired at pH 7.0, representing total fluorescence intensity (saved with the suffix `_total`). Even frames are interpreted as images obtained at acidic pH (5.5-6.0), representing intracellular fluorescence only (saved with the suffix `_intra`).

If `calc surface img` is selected, an additional total fluorescence image with subtracted intracellular intensity will be saved as the cell surface fluorescence fraction (suffix `_surface`). The input image should be a 3-dimensional single-channel time-lapse.



# Intensty profiles building and data frame saving
## Individual labels profiles
Builds a plot with mean intensity profiles for each ROI in `labels` using absolute intensity (if `raw intensity` is selected) or relative intensities (ΔF/F0).

The `time scale` sets the number of seconds between frames for x-axis scaling.

The baseline intensity for ΔF/F0 profiles is estimated as the mean intensity of the initial profile points (`ΔF win`).

Filters ROIs by minimum (`min amplitude`) and maximum (`max amplitude`) intensity amplitudes.

_Note: Intensity filtering is most relevant for ΔF/F0 profiles._

Additionally, you can save ROI intensity profiles as .csv using the `save data frame` option and specifying the `saving path`. The output data frames `%img_name%_lab_prof.csv` will contain the following columns:

- __id__ - unique image ID, the name of the input `napari.Image` object.
- __roi__ - ROI number, consecutively numbered starting from 1.
- __int__ - ROI mean intensity, raw or ΔF/F0 according to the `raw intensity` option.
- __index__ - frame index
- __time__ - frame time point according to the `time scale`.

_Note: The data frame will contain information for all ROIs; filtering options pertain to plotting only._

![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/pic_3.png)

## Labels profile
Builds a plot with the averaged intensity of all ROIs in `labels`. Can take two images (`img 0` and `img 1`) as input if `two profiles` are selected.

The `time scale` and `ΔF win` are the same as in the __Individual Labels Profiles__.

The `stat method` provides methods for calculating intensity errors:

- `se` - standard error of mean.
- `iqr` - interquartile range.
- `ci` - 95% confidence interval for t-distribution.

![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/pic_4.png)
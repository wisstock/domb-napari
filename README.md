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

---

## Detection of fluorescence redistributions
A set of widgets designed for detecting fluorescence intensity redistribution through the analysis of differential image series (red-green detection).

Inspired by [Dovgan et al., 2010](https://pubmed.ncbi.nlm.nih.gov/20704590/) and [Osypenko et al., 2019](https://www.sciencedirect.com/science/article/pii/S0969996119301974?via%3Dihub).

### Image preprocessing
Provides functions for preprocessing multi-channel fluorescence acquisitions:
- If the input image has 4 dimensions (time, channel, x-axis, y-axis), channels will be split into individual 3 dimensions images (time, x-axis, y-axis) with the `_ch%index%` suffix.
- If the `gaussian blur` option is selected, the image will be blurred with a Gaussian filter using sigma=`gaussian sigma`.
- If the `photobleaching correction` option is selected, the image will undergo correction with exponential (method `exp`) or bi-exponential (method `bi_exp`) fitting.
- If the `crop ch` option is selected, only a selected range of channel frames will be saved (corresponding to start and stop indexes from `crop range`).

![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/pic_00.png)

### Red-green series
Primary method for detecting fluorescent-labeled targets redistribution in time. Returns a series of differential images representing the intensity difference between the current frame and the previous one as new image with the `_red-green` suffix.

Parameters:

- `left frames` - number of previous frames for pixel-wise averaging.
- `space frames` - number of frames between the last left and first right frames.
- `right frames` - number of subsequent frames for pixel-wise averaging.

![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/pic_11.png)

### Up masking
Generates labels for insertion sites (regions with increasing intensity) based on `-red-green` images. Returns labels layer with `_up-labels` suffix.

Parameters:

- `detection img index` - index of the frame from `-red-green` image used for insertion sites detection.
- `insertion threshold` - threshold value for insertion site detection, intensity on selected `_red-green` frame normalized in -1 - 0 range.
- `opening footprint` - footprint size in pixels for mask filtering with morphology opening (disabled if 0).
- `save mask` - if selected, a total up mask (containing all ROIs) will be created with the `_up-mask` suffix.

![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/pic_22.png)

### Intensity masking
Extension of __Up Masking__ widget. Detects regions with increasing (`masking mode` - `up`) or decreasing (`masking mode` - `down`) intensity in `-red-green` images. Returns a labels layer with either `_up-labels` or `_down-labels` suffix, depending on the mode.

![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/pic_33.png)

---

## Exo-/endo-cytosis monitoring with pH-sensitive tag
A collection of widgets designed for the analysis of image series containing the pH-sensitive fluorescence protein Superecliptic pHluorin (SEP).

Insipred by [Fujii et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28474392/) and [Sposini et al., 2020](https://www.nature.com/articles/s41596-020-0371-z).

### SEP image preprocessing
Processes image series obtained through repetitive pH exchange methods (such as U-tube or ppH approaches). `pH 1st frame` option indicates the 1st frame pH. By default frames with odd indexes, including index 0, are interpreted as images acquired at pH 7.0, representing total fluorescence intensity (saved with the suffix `_total`). Even frames are interpreted as images obtained at acidic pH (5.5-6.0), representing intracellular fluorescence only (saved with the suffix `_intra`).

If `calc surface img` is selected, an additional total fluorescence image with subtracted intracellular intensity will be saved as the cell surface fluorescence fraction (suffix `_surface`). The input image should be a 3-dimensional single-channel time-lapse.

The `calc projections` option allows obtaining individual pH series projections (pixel-wise series MIP - pixel-wise series average) for the detection of individual exo/endocytosis events.

---

## Intensty profiles building and data frame saving
### Individual labels profiles
Builds a plot with mean intensity profiles for each ROI in `labels` using absolute intensity (if `absolute intensity` is selected) or relative intensities (ΔF/F0).

The `time scale` sets the number of seconds between frames for x-axis scaling.

The baseline intensity for ΔF/F0 profiles is estimated as the mean intensity of the initial profile points (`ΔF win`). You could filter ROIs by minimum and maximum ΔF/F0 amplitudes with the `ΔF aplitude lim` option.

_Note: amplitude filtering working with ΔF/F0 profiles only._

If the `profiles crop` option is selected, only a selected range of intensity profiles indexes will be plotted (corresponding to start and stop indexes from `profiles range`).

Additionally, you can save ROI intensity profiles as .csv using the `save data frame` option and specifying the `saving path`. The output data frames `%img_name%_lab_prof.csv` will contain the following columns:

- __id__ - unique image ID, the name of the input `napari.Image` object.
- __roi__ - ROI number, consecutively numbered starting from 1.
- __int__ - ROI mean intensity, raw or ΔF/F0 according to the `raw intensity` option.
- __index__ - frame index
- __time__ - frame time point according to the `time scale`.

_Note: The data frame will contain information for all ROIs; amplitude filtering and crop options pertain to plotting only._

Absolute intensity         | ![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/pic_44.png)
:-------------------------:|:-------------------------:
__ΔF/F0__|![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/pic_55.png)


### Labels stat profiles
Builds a plot with the averaged intensity of all ROIs in `labels`. Can take two images (`img 0` and `img 1`) as input if `two profiles` are selected.

The `time scale` and `ΔF win` are the same as in the __Individual Labels Profiles__.

The `stat method` provides methods for estimation intensity and errors:

- `se` - standard error of mean.
- `iqr` - interquartile range.
- `ci` - 95% confidence interval for t-distribution.

![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/pic_66.png)
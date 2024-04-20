domb-napari
===========

[![Stand With Ukraine](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/banner-direct-single.svg)](https://stand-with-ukraine.pp.ua)

[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/domb-napari)](https://napari-hub.org/plugins/domb-napari)
![PyPI - Version](https://img.shields.io/pypi/v/domb-napari)
![PyPI - License](https://img.shields.io/pypi/l/domb-napari)
![Website](https://img.shields.io/website?up_message=domb.bio%2Fnapari&up_color=%2323038C93&url=https%3A%2F%2Fdomb.bio%2Fnapari%2F)

__napari Toolkit of Department of Molecular Biophysics <br /> Bogomoletz Institute of Physiology of NAS of Ukraine, Kyiv,  Ukraine__

This plugin offers widgets specifically designed to analyze the redistribution of fluorescence-labeled proteins in widefield epifluorescence time-lapse acquisitions. It is particularly useful for studying various phenomena, including:
- Calcium-dependent translocation of neuronal calcium sensors.
- Synaptic receptor traffic during long-term plasticity induction.
- Membrane protein tracking.

![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/translocation.gif)
__Hippocalcin (neuronal calcium sensor) redistributes in dendritic branches upon NMDA application__

---

## Detection of fluorescence redistributions
A set of widgets designed for preprocessing multispectral image stacks and detecting redistributions in fluorescence intensity. These widgets specifically analyze differential "red-green" image series to identify changes in fluorescence intensity.

Inspired by [Dovgan et al., 2010](https://pubmed.ncbi.nlm.nih.gov/20704590/) and [Osypenko et al., 2019](https://www.sciencedirect.com/science/article/pii/S0969996119301974?via%3Dihub).

### Dual-view stack registration
Registration of four-channel image stacks, including two excitation wavelengths and two emission pathbands, acquired with a dual-view beam splitter. This setup detects different spectral pathbands using distinct sides of the CCD matrix.

- `offset img` - input for a four-channel time-lapse image stack.
- `reference img` - an optional four-channel reference image (e.g., fluorescence beads image), used for offset estimation if `use reference img` is selected.
- `input crop` - number of pixels that will be deleted from each side of input stack frames to discard misalignment artifacts from the dual-view system.
- `output crop` - number of pixels that will be deleted from each side of output stack frames to discard registration artifacts.


### Multichannel stack preprocessing
- `stack order` - Represents the order of axes in the input data array (T - time, C - color, X and Y - image dimensions). If the input image stack has four dimensions (time, channel, x-axis, y-axis), channels will be split into individual three-dimensional images (time, x-axis, y-axis), each with the `_ch%index%` suffix.
- `median filter` - provides frame-by-frame image smoothing using a kernel of size specified in `median kernel`.
- `background subtraction` - compensates for background fluorescence intensity. Background intensity is estimated frame by frame as the mean intensity value outside of a simple Otsu mask.
- If the `photobleaching correction` option is selected, the image will undergo correction using either an exponential (method `exp`) or bi-exponential (method `bi_exp`) fitting.
- Image stacks can be cropped according to start and stop indexes specified in `frames range` if `drop frames` is selected.

![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/stack_preprocessing.png)


### Red-green series
Primary method for detecting fluorescence-labeled targets redistribution. This widget returns a series of differential images, each representing the intensity difference between the current frame and the previous one, output image labeled with the `_red-green` suffix.

Parameters:

- `left frames` - specifies the number of previous frames used for pixel-wise averaging.
- `space frames` - determines the number of frames between the last left frame and the first right frame.
- `right frames` - specifies the number of subsequent frames used for pixel-wise averaging.

`normalize by int`  function normalizes the differential images relative to the absolute intensity of the input image stack, which helps to reduce background noise amplitude.

If `save MIP` is selected, the maximal intensity projection (MIP) of the differential image stack will be saved with the `_red-green-MIP` suffix.

![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/rg_series.png)

### Up masking
Generates labels for insertion sites (regions with increasing intensity) based on `-red-green` images. Returns labels layer with `_up-labels` suffix.

Parameters:

- `detection img index` - index of the frame from `-red-green` image used for insertion sites detection.
- `insertion threshold` - threshold value for insertion site detection, intensity on selected `_red-green` frame normalized in -1 - 0 range.
- `opening footprint` - footprint size in pixels for mask filtering with morphology opening (disabled if 0).
- `save mask` - if selected, a total up mask (containing all ROIs) will be created with the `_up-mask` suffix.

![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/up_masking.png)

### Intensity masking
Extension of __Up Masking__ widget. Detects regions with increasing (`masking mode` - `up`) or decreasing (`masking mode` - `down`) intensity in `-red-green` images. Returns a labels layer with either `_up-labels` or `_down-labels` suffix, depending on the mode.

![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/int_masking.png)

---

## FRET detection
Widgets for detection and analysis of Förster resonance energy transfer multispectral image stacks.

Based on [Zal and Gascoigne, 2004](https://pubmed.ncbi.nlm.nih.gov/15189889/), [Chen et al., 2006](https://pubmed.ncbi.nlm.nih.gov/16815904/) and [Kamino et al., 2023](https://pubmed.ncbi.nlm.nih.gov/37014867/)

_Under development: calculation of crosstalk coefficients and G-factor, B-FRET estimation._

### E-FRET estimation
E-FRET estimation with 3-cube approach.

This method utilizes default values for `a` and `d` coefficients and the `G`-factor, optimized for the pair EYFP and ECFP in our experimental setup:
- Microscope Olympus IX71
- Cube Chroma 69008
- Dual-view system with Chroma 505DCXR beam splitter
- Donor excitation wavelength 435 nm
- Acceptor excitation wavelength 505 nm

Parameters:
- `DD img` - donor emission channel image acquired with the donor excitation wavelength.
- `AD img` - Acceptor emission channel image acquired with the donor excitation wavelength.
- `AA img` - Acceptor emission channel image acquired with the acceptor excitation wavelength.
- `output type` - Type of output image: sensitized emission (Fc), apparent FRET efficiency (Eapp), or FRET efficiency with photobleaching correction (Ecorr).

If the `save normalized` option is selected, an additional image will be saved. This image is normalized to the absolute intensity of the `AA img`, resulting in reduced background noise amplitude.

Raw Eapp| Normalized Eapp
:-:|:-:
![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/fret_raw.png)|![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/fret_norm.png)

---

## Exo-/endo-cytosis monitoring with pH-sensitive tag
A set of widgets designed for the analysis of image series containing the pH-sensitive fluorescence protein Superecliptic pHluorin (SEP).

Insipred by [Fujii et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28474392/) and [Sposini et al., 2020](https://www.nature.com/articles/s41596-020-0371-z).

### SEP image preprocessing
Processes image series obtained through repetitive pH exchange methods (such as U-tube or ppH approaches). `pH 1st frame` option indicates the 1st frame pH. By default frames with odd indexes, including index 0, are interpreted as images acquired at pH 7.0, representing total fluorescence intensity (saved with the suffix `_total`). Even frames are interpreted as images obtained at acidic pH (5.5-6.0), representing intracellular fluorescence only (saved with the suffix `_intra`).

If `calc surface img` is selected, an additional total fluorescence image with subtracted intracellular intensity will be saved as the cell surface fluorescence fraction (suffix `_surface`). The input image should be a 3-dimensional single-channel time-lapse.

The `calc projections` option allows obtaining individual pH series projections (pixel-wise series MIP - pixel-wise series average) for the detection of individual exo/endocytosis events.

---

## Intensty profiles building and data frame saving
### ROIs profiles
This widget builds a plot with mean intensity profiles for each Region of Interest (ROI) in labels. It uses either absolute intensity (if `absolute intensity` is selected) or relative intensities (ΔF/F0).

- `time scale` - sets the number of seconds between frames for x-axis scaling.
- `ΔF win` - the baseline intensity for ΔF/F0 profiles is estimated as the mean intensity of the specified number of initial profile points.
- `ΔF amplitude lim` - allows filtering of ROIs by minimum and maximum ΔF/F0 amplitudes. Note: Amplitude filtering works with ΔF/F0 profiles only.
- `profiles crop` - if selected, only a specified range of intensity profile indexes will be plotted, corresponding to the start and stop indexes from `profiles range`.

Additionally, you can save ROI intensity profiles as .csv files using the save data frame option and specifying the `saving path`. The output data frames named %img_name%_lab_prof.csv will include the following columns:

- `id` - Unique image ID, the name of the input napari.Image object.
- `roi` - ROI number, consecutively numbered starting from 1.
- `int` - ROI mean intensity, either raw or ΔF/F0, according to the selected intensity option.
- `index` - Frame index.
- `time` - Frame time point, adjusted according to the time scale.

_Note: The data frame will contain information for all ROIs; amplitude filtering and crop options pertain to plotting only._

Absolute intensity         | ![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/rois_abs.png)
:-------------------------:|:-------------------------:
__ΔF/F0__|![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/rois_df.png)


### Stat profiles
This widget builds a plot displaying the averaged intensity of all Regions of Interest (ROI) specified in labels. It can handle up to three images (img 0, img 1, and img 2) as inputs, depending on the selected profiles num.

`time scale`, `ΔF win`, and `absolute intensity` parameters are identical as described in the __ROIs profiles__ widget.

The `stat method` allows for the estimation of intensity and associated errors through the following methods:
- `se` - mean +/- standard error of the mean.
- `iqr` - median +/- interquartile range.
- `ci` - mean +/- 95% confidence interval based on the t-distribution.

Absolute intensity         | ![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/stat_abs.png)
:-------------------------:|:-------------------------:
__ΔF/F0__|![](https://raw.githubusercontent.com/wisstock/domb-napari/master/images/stat_df.png)
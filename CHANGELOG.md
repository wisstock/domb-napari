# Changelog

## 0.7.0

Results of FRET, red-green, ΔF and statistical widgets differ from 0.6.0, see "Changed" and "Fixed".

### Changed (numerical results)
- **No clipping anywhere.** Background subtraction, `Fc`, `E_D`, `E_A`, ΔF and red-green outputs are signed `float32`. Undefined values (non-positive denominator, no data after registration) are `NaN` instead of `0` or `1e-12`.
- **Spectral unmixing** uses the exact solution with optional `b` and `c` coefficients; identical to the previous equations if `b = c = 0`.
- **Ecorr** is corrected with the acceptor bleaching curve averaged over a mask, not pixel-wise.
- **Cross-talk coefficients**: primary value is the Deming regression slope of time-averaged images (was OLS of single frames). Ratio and OLS estimates are reported as well.
- **G-factor (Zal)**: primary value is the area-weighted ratio of sums over ROIs (was OLS slope with free intercept); frames can be averaged; mask is split into fragments of similar area by k-means (was watershed from random seeds).
- **G-factor (Chen)**: ratios of sums instead of means of ROI ratios; bootstrap standard error is no longer divided by `sqrt(n_boot)`; 95% interval added.
- **Red-green series**: `out[i] = mean(img[i+l+s : i+l+s+r]) - mean(img[i : i+l])`. Previously window lengths were one frame longer and `space_frames`/`right_frames` were swapped. `normalize_by_int` is off by default and uses a global normalization.
- **ΔF series**: pure `F - F0` and `(F - F0)/F0`. Previously the output was multiplied by a normalized intensity image (twice for ΔF/F0). The weighting is the optional `intensity_weighted`.
- **Statistical profiles**: SE and CI use the number of ROIs (was the number of frames) and `ddof=1`; `iqr` shows the quartiles (was ± full IQR).
- **Registration**: output is `float32` (was truncated to the input integer type); both align modes use linear interpolation (`load matrix` was cubic); pixels without data are `NaN`.
- **Photobleaching correction**: models have a constant term; reported R² is the coefficient of determination.

### Fixed
- Chen G-factor estimation crashed (`mask_l` keyword, wrong method call, undefined `roi_mask`).
- `r` was reported as `r²` in all calibration tables; p-values were saved as strings.
- `output_crop` of the registration used `input_crop`.
- `CTXY` stacks: channel loop used the number of frames.
- Otsu mask of the photobleaching correction was calculated from a wrong slice.
- `manual_channels`: the matrix was applied to channels 0 and 2 regardless of the selection.
- Background subtraction modified `int32` layers in place.
- In-ROIs up masking overwrote labels of neighbouring ROIs.
- Data frames: `index` column (`linspace` artefact), `roi` column is the label value (was the ordinal number), first label was lost if labels had no background.
- `save_df` raised `UnboundLocalError` instead of the shape error.
- `G: 0` / `xi: 0` in the configuration gave silent `inf` / zeros; now `null` and validated.

### Added
- Widget **E-FRET ROIs profiles**: FRET of ROIs from mean intensities, CSV export.
- `write_to_config` option of the G-factor widget.
- `background_percentile`, `frames_to_average`, `fragments_num`, `b`, `c`, `bleach_mask`, `F0_min`, `intensity_weighted` parameters.
- Registration: acceptance test by mutual information, diagnostics, crop and shape in the header of the saved matrix.
- `domb_napari.core` package without napari/Qt dependencies; `FRETCoefs`, `linear_fits`, `roi_profiles`, `apply_affine_stack`.
- Test suite and CI.

### Performance
- FRET, ΔF and red-green kernels: single-pass parallel numba loops with `float32` output.
- Registration: normalized parametrization, translation pre-step, pixel subsampling (about 3-6x faster, converges where 0.6.0 returned identity).
- Frame-wise median filter and affine resampling run in a thread pool.
- Data frame is assembled without Python loops; Dietrich baseline is calculated once.
- Profile widgets and data frame saving run in worker threads; a widget can not be started twice.

### Renamed / removed
- `mask_diamets` -> `mask_radius` (it always was the radius).
- `_widget.py`, `_utils.py`, `_affine.py` are replaced by `core/` and `_widgets/`; `_e_fret.py` is an alias of `core/fret.py`.
- Columns of calibration CSV files, see README.
- Default saving path is the home directory (was the working directory at import time).
- Dependency `tifffile` removed; `pyyaml`, `magicgui`, `vispy` declared.

---
title: 'domb-napari: a python package for intensity-based FRET estimation and a napari toolkit for express analysis of 2D live-cell imaging data'
tags:
  - Python
  - napari
  - live-cell imaging
  - fluorescent microscopy
  - FRET
authors:
  - name: Borys Olifirov
    orcid: 0000-0001-9915-7769
    affiliation: "1, 2"
    corresponding: true
  - name: Dana Biruk
    orcid:  0009-0001-9812-1040
    affiliation: 3
  - name: Oleksandra Hrubiian
    orcid: 0000-0001-5886-2605
    affiliation: "1, 2"
affiliations:
 - name: Department of Biophysics of Sensory Signalling, Bogomoletz Institute of Physiology of NAS of Ukraine, Ukraine
   index: 1
   ror: 01r6bzv07
 - name: Laboratory of Molecular Assays and Imaging, Institute of Bioorganic Chemistry of PAS, Poland
   index: 2
   ror: 04ejdtr48
 - name: Kyiv Academic University, Ukraine
   index: 3
   ror: 02vrpj575
date: 16 February 2026
bibliography: paper.bib
---

# Summary

In today's high-throughput-focused research environment, simple and reliable exploratory analysis of experimental data remains important. We have developed a toolset for the express analysis of live-cell fluorescence imaging data, with a specific focus on visualising and detecting dynamic intracellular processes.

This toolset's core functionality includes two main areas. First, it offers simple tools for detecting fluorescence redistribution using derivative images for visualisation/quantification of the fluorescence intensity changes over time series. Second, it includes an end-to-end pipeline for Förster Resonance Energy Transfer (FRET) experiments, providing a set of functions for calibration and estimation using two well-described ratiometric methods. Combined with basic functions for segmentation, plotting, and data export, this creates a complete working environment for analysing live-cell imaging experiments. The implemented methods are simple and robust, making them broadly applicable across various biological domains, including neuroscience, biophysics, and cytology.

The main tool, `domb-napari`, is developed as a plugin for napari, an open-source multidimensional image viewer, ensuring accessibility for biologists without coding expertise. Additionally, the functions specifically for FRET calibration and estimation are available as a fully independent Python module, which can be used separately for greater flexibility.


# Statement of need

Napari, the open-source multidimensional viewer, has very limited built-in functionality and is developed through community-created plugins [@Chiu2022]. However, there are very limited set of plugins for basic analysis of live-imaging data, which often may be poorly documented [@Haase2021; @Nauroth2023; @Soltwedel2024; @Gradel2025]. Our main aim was to bridge the gap between highly specialised solutions and just a set of basic functions (filtering, morphology operations, plotting, etc.), create a general-purpose toolset for fast analysis of fluorescence live-cell imaging data, from the primary detection of regions of interest up to plotting and exporting data in a tidy format. The core functionality of our plugin relies on a straightforward, pixel-wise derivative approach, allowing for rapid inspection and detection of changes or redistribution in fluorescence intensity. This method is broadly applicable across various experimental designs. Possible areas of application include:

- __Fluorescence probe signal detection:__ Calcium imaging, voltage-sensing probes imaging, pH-sensing probes imaging, etc. [@Taylor2011; @Reese2015].
- __Tracking and redistribution detection:__ Detection of membrane protein complexes, intracellular trafficking detection, endocytosis and exocytosis events detection [@McVicker2016; @Scheefhals2019; @Mendoza2024].

This primary feature is integrated with additional functionality for detecting and estimating Förster resonance energy transfer (FRET). FRET is a non-radiative phenomenon in which energy is transferred from one fluorophore (donor) to another (acceptor) over a distance of 1–10 nm. This energy transfer can be interpreted as a protein–protein interaction (if the fluorophores are attached to different proteins) or as a conformational change (if the fluorophores are on different parts of the same protein). This makes FRET a valuable tool for studying molecular processes in living cells without requiring super-resolution microscopy. There are many methods for FRET quantification. Some frequently used approaches belong to bulk measurement techniques such as ratiometric FRET, fluorescence lifetime imaging FRET, and photobleaching FRET. Ratiometric FRET, or intensity-based FRET estimation, is a widely used and accessible technique. It primarily relies on measuring the sensitised emission ($F_{c}$), which is the acceptor's fluorescence resulting from energy transfer from the donor [@Shrestha2015]. Despite this, ratiometric FRET lacks open-source, user-friendly tools for automated data processing and visualisation. Existing pipelines often rely on MATLAB [@Nagy2016; @Muller2025], which is extensible but proprietary software, or ImageJ [@Feige2005; @Hachet2006; @Rebenku2023], which is open-source but Java-based software, making it difficult to modify or create analysis pipelines. Although ImageJ supports Python integration through Jython, this is limited to Python 2. Existing Python-based tools for FRET analysis, often distributed as packages or scripts [@Kamino2023], present a significant barrier to entry for the typical biologist. As of early 2026, only one solution for FRET analysis is currently available within the napari ecosystem [@Leblanc2025; @Dupont2025]. Rather than reinventing methodologies, our work focuses on implementing well-described and widely adopted FRET estimation techniques: E-FRET [@Zal2004; @Chen2006] and 3^3^-FRET [@Erickson2001; @Butz2016].


# Fluorescence redistribution analysis toolset


The key element of the `domb-napari` plugin is the "red-green" images ($I_{RG}$), which are named according to a specific lookup table where red indicates positive changes and green represents negative changes. This image is a pixel-by-pixel difference between the averaged intensity of a right (later in time, $\bar{I}_{right}$) and a left (earlier in time, $\bar{I}_{left}$) time window.

Users can adjust the detection sensitivity to specific event kinetics by changing three parameters: the sizes of the averaging windows (frame intervals $\[r_{0}:r\]$ and $\[l_{0}:l\]$) and the frame shift ($s$) between the windows. Fast events are best detected using short, or single-frame, windows with no spacer, which effectively turns the estimation into a temporal derivative; a long shift and larger window sizes are suitable for detecting slow transient changes [^1]:

[^1]: In this context, the terms "fast" and "slow" refer to the relationship between the speed of the biological process and the speed of data acquisition, rather than the actual speed of the process itself.

$$I_{RG} = \bar{I}_{right} - \bar{I}_{left}  = \frac{1}{r - r_0}\sum_{t=r_0+s}^{r+s} I_{t} - \frac{1}{l - l_0}\sum_{t=l_0}^{l} I_{t}$$

This approach has been previously implemented and successfully applied in our laboratory's research [@Dovgan2010; @Osypenko2019]. The plugin's additional functionality, which leverages the `scipy` and `scikit-image`, includes:

- __Data Preprocessing:__ Simple multichannel time-series preprocessing, including background correction and filtering.
- __Feature Detection:__ Intensity-based segmentation and local maxima detection methods.
- __Visualisation:__ Plotting widgets for generating intensity profiles of regions of interest (ROIs) across the time series.
- __Data Export:__ Exporting ROI intensity values as tidy CSV data frames.

By combining all these steps on multichannel data (Fig. 1A), the plugin enables simultaneous analysis of the dynamics of the target of interest using "red-green" images (Fig. 1B). This analysis can be performed in combination with features detected on an optional reference channels (Fig. 1C). The resulting output may serve as the initial input for subsequent, more specialised analysis workflows. Furthermore, the use of Numba's Just-In-Time (JIT) compilation optimises array operations, leading to improved performance in fast, repetitive time-series analysis [@Lam2015].



![Fig. 1. Express analysis of the fluorescence redistribution in the live-cell imaging data. Adapted from [@Olifirov2025]](fig1.png)


# Quantitative FRET analysis with `e_fret` module

The raw signal measured in the acceptor's channel under donor excitation ($I_{DA}$) needs correction to accurately estimate $F_{c}$. This correction is essential for subtracting unwanted contributions of the signals from direct acceptor excitation crosstalk and donor signal bleedthrough into the acceptor channel. These necessary corrections utilize the signals from the acceptor ($I_{AA}$) and donor ($I_{DD}$) spectral channels, along with their corresponding coefficients, $a$ and $d$:

$$F_{c} = I_{DA} - a I_{AA} - d I_{DD}$$

The `e_fret` module offers a set of methods for estimating coefficients, based on both donor-only and acceptor-only calibration sample imaging. Estimation of FRET using the absolute value of $F_{C}$ is strongly dependent on the specific imaging setup and FRET pair characteristics. To perform a comparison of results across different studies, various methods for estimating _relative_ FRET values have been developed. Our module offers two primary techniques:

- __FRET efficiency__ ($E_{D}$) estimation: This is achieved using the E-FRET approach, which normalises the FRET signal to the total donor intensity ($I_{D}^{tot}$):

$$E_{D} = \frac{F_{c}}{I_{D}^{tot}} = \frac{F_c / G}{F_c / G + I_{DD}}$$

- __FRET ratio__ ($E_{A}$) estimation: This is achieved using the 3^3^-FRET approach, which is based on the total acceptor intensity ($I_{A}^{tot}$):

$$E_{A} = \frac{F_{c}}{I_{A}^{tot}} = \frac{\Xi F_c}{aI_{AA}}$$

Precise calculation of ths $I_{D}^{tot}$ and $I_{A}^{tot}$ values from the spectral imaging data alsou requires estimation of method-dependent proportionality coefficient, the $G$ factor for $E_{D}$ estimation and excitation ratio $\Xi$ for $E_{A}$ estimation [^2].

[^2]: For detailed, extended definitions and calculation methods of coefficients, see the corresponding papers on E-FRET [@Zal2004; @Chen2006] and 3^3^-FRET [@Erickson2001; @Butz2016].

Our package offers two methods for estimating the $G$ factor, both relying on imaging calibration samples. The original approach, developed by Zal & Gascoigne, is based on the relationship between sensitised emission depletion ($\Delta F_{c}$) and donor dequenching ($I_{DD}$) following acceptor photodestruction [@Zal2004]. This method uses calibration samples containing a linked donor-acceptor tandem. The initial image acquisition shows high FRET values (indicated by lower donor intensity). Subsequent acquisition of the same field of view, performed after partial acceptor bleaching, results in lower FRET values ($F_{c}^{post}$) and higher donor intensity ($I_{DD}^{post}$):

$$G = \frac{\Delta F_{c}}{\Delta I_{DD}} = \frac{F_{c} - F_{c}^{post}}{I_{DD}^{post} - I_{DD}}$$  

Chen et al. proposed an alternative method for estimating the $G$ factor based on comparing FRET values from two calibration donor-acceptor tandems [@Chen2006]. These tandems are characterised by distinct distances between the FRET pair components, resulting in different FRET efficiencies. By relating the sample parameters—specifically the corrected FRET signals ($F_c$), donor intensities in the absence of acceptor ($I_{DD}$), and acceptor intensities in the absence of donor ($I_{AA}$) – for the tandem with higher ($F_{c}^{high}$, $I_{DD}^{high}$, and $I_{AA}^{high}$) and lower ($F_{c}^{low}$, $I_{DD}^{low}$, and $I_{AA}^{low}$) FRET values, it is possible to accurately estimate the $G$ factor for a specific FRET pair:

$$G = \frac{F_{c}^{high} / I_{AA}^{high} - F_{c}^{low} / I_{AA}^{low}}{I_{DD}^{low} / I_{AA}^{low} - I_{DD}^{high} / I_{AA}^{high}}$$  

Module performance is optimised for pixel-wise operations using JIT compilation functions provided by Numba. For non-coding researchers, all the described functionality is conveniently accessible as a set of corresponding widgets within the `domb-napari` plugin. 


# Package development status

The plugin is currently was used in published research [@Olifirov2025] and has been successfully tested on mid-scale spectral imaging data acquired using widefield, laser scanning confocal, and spinning disk confocal microscopes.

Ongoing development focuses on implementing more advanced intensity-based FRET estimation techniques to address common issues in live-cell imaging with fluorescence protein tags, such as photobleaching and low signal-to-noise ratio. 


# AI usage disclosure

A generative AI model (Gemini 3 Pro) was used for the final code review, formatting and proofreading of the documentation and docstrings. A specialised AI model (DeepL) was used to proofread the main text. The authors declare that all code and text were manually inspected and verified by humans, and that they bear full responsibility for the content.


# Ethics statement

All animal experiments were approved by the host institution and conducted in compliance with Directive 2010/63/EU and Law of Ukraine No. 3447-IV “On the Protection of Animals from Cruelty”.


# Acknowledgements

This work was funded by the long-term program of support of the Ukrainian research teams at the Polish Academy of Sciences, carried out in collaboration with the U.S. National Academy of Sciences, with the financial support of external partners (PAN.BFB.S.BWZ.405.022.2023), and the National Academy of Science of Ukraine grants (0124U001556, 0124U001557).


# References


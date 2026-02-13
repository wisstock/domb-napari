---
title: 'domb-napari: a python package for quantitative FRET estimation and a napari toolkit for express analysis of live-cell imaging data'
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
date: 10 January 2026
bibliography: paper.bib
---

# Summary

Test-test-test


# Statement of need

Test-test-test

Despite this, ratiometric FRET lacks open-source, user-friendly tools for automated data processing and visualization. Existing pipelines often rely on MATLAB `[@Nagy:2016; @Muller:2025]`, which is extensible but proprietary software, or ImageJ `[@Feige:2005; @Hachet:2006; @Rebenku:2023]`, which is open-source but Java-based software, making it difficult to modify or create analysis pipelines. Although ImageJ supports Python integration through Jython, this is limited to Python 2 and does not allow the use of external Python packages such as numpy, scipy, pandas, etc.


# Quantitative FRET analysis with `e_fret` module

$$F_{c} = I_{DA} - a I_{AA} - d I_{DD} \label{eq:f_c}$$

$$E_{D} = \frac{I_{FRET}}{I_{D}^{tot}} = \frac{F_c / G}{F_c / G + I_{DD}} \label{eq:e_d}$$

$$E_{A} = \frac{I_{FRET}}{I_{A}^{tot}} = \frac{\Xi F_c}{aI_{AA}} \label{eq:e_a}$$


# Fluorescence redistribution analysis toolset


# AI usage disclosure

A generative AI model (Gemini 3 Pro) was used for the final code review, formatting and proofreading of the documentation and docstrings. A specialised AI model (DeepL) was used to proofread the main text. The authors declare that all code and text were manually inspected and verified by humans, and that they bear full responsibility for the content.



# Acknowledgements

This work was funded by the long-term program of support of the Ukrainian research teams at the Polish Academy of Sciences, carried out in collaboration with the U.S. National Academy of Sciences, with the financial support of external partners (PAN.BFB.S.BWZ.405.022.2023), and the National Academy of Science of Ukraine grant (0124U001556, 0124U001557).

# References

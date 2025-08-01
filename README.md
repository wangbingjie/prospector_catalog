## Fitting photometry only with Prospector

[![arXiv](https://img.shields.io/badge/arXiv-2310.01276-b31b1b.svg)](https://arxiv.org/abs/2310.01276)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.8401181.svg)](http://dx.doi.org/10.5281/zenodo.8401181)

This repo contains the scripts used for creating the Prospector catalogs, as outlined in [Wang+24](https://ui.adsabs.harvard.edu/abs/2024ApJS..270...12W/abstract).

The catalogs are deposited to Zenodo at [doi:10.5281/zenodo.8401181](http://dx.doi.org/10.5281/zenodo.8401181). The latest catalogs are based on all the broad + medium-band observations (instead of broad bands alone)!

Note that lensing magnification is treated consistently during SED-fitting; see [Eq. 1-2](https://arxiv.org/pdf/2310.01276). For general usage that does not require the lensing magnification term in the likelihood, simply remove the `lnprobfn` function in `uncover_*_params.py`, and use the default `lnprobfn` function from Prospector.

If, however, lensing magnification is relevant, then the convergence and shear maps are required as inputs for the script `mu_from_maps.py`, which can be obstained from the UNCOVER DR webpage.

`submit.py` contains the scripts used for submitting cluster jobs using slurm; also it outlines the sequence for calling the related scripts in this repository for creating a Prospector catalog.

Please cite [Wang+24](https://ui.adsabs.harvard.edu/abs/2024ApJS..270...12W/abstract) and the relavant Prospector paper(s) if you find these useful in your research.

## Fitting photometry + spectrum with Prospector
[![arXiv](https://img.shields.io/badge/arXiv-2308.03745-b31b1b.svg)](https://arxiv.org/abs/2308.03745)

`pv1_prism_fsps_gal_params.py` is an example parameter file for fitting photometry and spectrum simultaneously, using either the [prospector-alpha](https://ui.adsabs.harvard.edu/abs/2017ApJ...837..170L/abstract) or [prospector-beta](https://ui.adsabs.harvard.edu/abs/2023ApJ...944L..58W/abstract) model.

Please cite [Wang+23](https://ui.adsabs.harvard.edu/abs/2023ApJ...957L..34W/abstract) and the modeling paper(s) if you find this code useful in your research.
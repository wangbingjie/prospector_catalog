These are the scripts used for creating the Prospector catalogs, as outlined in [Wang+24](https://ui.adsabs.harvard.edu/abs/2024ApJS..270...12W/abstract).

The catalogs are deposited to Zenodo at [doi:10.5281/zenodo.8401181](https://zenodo.org/records/10223792). (Broad + medium-band catalogs will be available soon.)

Note that lensing magnification is treated consistently during SED-fitting; see [Eq. 1-2](https://arxiv.org/pdf/2310.01276). For general usage that does not require the lensing magnification term in the likelihood, simply remove the `lnprobfn` function in `uncover_*_params.py`, and use the default `lnprobfn` function from Prospector.

If, however, lensing magnification is relevant, then the convergence and shear maps are required as inputs for the script `mu_from_maps.py`, which can be obstained from the UNCOVER DR webpage.

'''some of the functions are borrowed from cigale.
'''
import numpy as np
from scipy.constants import c, pi
from astropy import units as u

def lambda_flambda_to_fnu(wavelength, flambda):
    """
    Convert a Fλ vs λ spectrum to Fν vs λ

    Parameters
    ----------
    wavelength: list-like of floats
        The wavelengths in nm.
    flambda: list-like of floats
        Fλ flux density in W/m²/nm (or Lλ luminosity density in W/nm).

    Returns
    -------
    fnu: array of floats
        The Fν flux density in mJy (or the Lν luminosity density in
        1.e-29 W/Hz).

    """
    wavelength = np.array(wavelength, dtype=float)
    flambda = np.array(flambda, dtype=float)

    # Factor 1e+29 is to switch from W/m²/Hz to mJy
    # Factor 1e-9 is to switch from nm to m (only one because the other nm
    # wavelength goes with the Fλ in W/m²/nm).
    fnu = 1e+29 * 1e-9 * flambda * wavelength * wavelength / c

    return fnu


def lambda_flambda_to_fnu_aa_maggies(wavelength, flambda):
    """
    wavelength in angstrom; flambda in erg/cm²/s/A

    Returns
    -------
    fnu: array of floats
        Fν flux density in maggies

    """
    _flam = flambda*u.erg/u.cm**2/u.second/u.angstrom
    fnu = lambda_flambda_to_fnu(wavelength*u.angstrom.to(u.nanometer), _flam.to(u.watt/u.meter**2/u.nanometer))
    fnu = fnu * u.millijansky.to(u.jansky)
    fnu = fnu/3631
    
    return fnu


def lambda_fnu_to_flambda(wavelength, fnu):
    """
    Convert a Fν vs λ spectrum to Fλ vs λ

    Parameters
    ----------
    wavelength: list-like of floats
        The wavelengths in nm.
    fnu: list-like of floats
        The Fν flux density in mJy (or the Lν luminosity density in
        1.e-29 W/Hz).

    Returns
    -------
    flambda: array of floats
        Fλ flux density in W/m²/nm (or Lλ luminosity density in W/nm).

    """
    wavelength = np.array(wavelength, dtype=float)
    fnu = np.array(fnu, dtype=float)

    # Factor 1e-29 is to switch from Jy to W/m²/Hz
    # Factor 1e+9 is to switch from m to nm
    flambda = 1e-29 * 1e+9 * fnu / (wavelength * wavelength) * c

    return flambda


def lambda_fnu_to_flambda_aa_maggies(wavelength, fnu):
    """
    wavelength in angstrom; fnu in maggies

    Returns
    -------
    flambda: array of floats
        Fλ flux density in erg/cm²/s/A

    """

    flambda = lambda_fnu_to_flambda(wavelength*u.angstrom.to(u.nanometer), fnu*3631*u.jansky.to(u.millijansky))

    flambda = flambda * u.watt/u.meter**2/u.nanometer
    flambda = flambda.to(u.erg/u.cm**2/u.second/u.angstrom)

    return flambda.value


def redshift_spectrum(wavelength, flux, redshift, is_fnu=False):
    """Redshit a spectrum

    Parameters
    ----------
    wavelength: array like of floats
        The wavelength in nm.
    flux: array like of floats
        The flux or luminosity density.
    redshift: float
        The redshift.
    is_fnu: boolean
        If false (default) the flux is a Fλ density in W/m²/nm (or a Lλ
        luminosity density in W/nm). If true, the flux is a Fν density in mJy
        (or a Lν luminosity density in 1.e-29 W/Hz).

    Results
    -------
    wavelength, flux: tuple of numpy arrays of floats
        The redshifted spectrum with the same kind of flux (or luminosity)
        density as the input.

    """
    wavelength = np.array(wavelength, dtype=float)
    flux = np.array(flux, dtype=float)
    redshift = float(redshift)

    redshift_factor = 1. + redshift

    if is_fnu:
        # Switch to Fλ
        flux = lambda_fnu_to_flambda(wavelength, flux)

    wavelength *= redshift_factor
    flux /= redshift_factor

    if is_fnu:
        # Switch back to Fλ
        flux = lambda_flambda_to_fnu(wavelength, flux)

    return wavelength, flux


def redshift_spectrum_prosp_units(wavelength, flux, redshift, is_fnu=False):
    """Redshit a spectrum

    Parameters
    ----------
    wavelength: array like of floats
        The wavelength in angstrom.
    flux: array like of floats
        Flux density.
    redshift: float
        The redshift.
    is_fnu: boolean
        If false (default) the flux is a Fλ density in erg/cm²/s/A. 
        If true, the flux is a Fν density in maggies.

    Results
    -------
    wavelength, flux: tuple of numpy arrays of floats
        The redshifted spectrum with the same kind of flux (or luminosity)
        density as the input.

    """
    wavelength = np.array(wavelength, dtype=float)
    flux = np.array(flux, dtype=float)
    redshift = float(redshift)

    if redshift < 0:
        redshift_factor = 1. / (1. - redshift)
    else:
        redshift_factor = 1. + redshift

    if is_fnu:
        # Switch to Fλ
        flux = lambda_fnu_to_flambda_aa_maggies(wavelength, flux)

    wavelength *= redshift_factor
    flux /= redshift_factor

    if is_fnu:
        # Switch back to Fλ
        flux = lambda_flambda_to_fnu_aa_maggies(wavelength, flux)

    return wavelength, flux


def deredshift_spectrum_prosp_units(wavelength, flux, redshift, is_fnu=False):
    """Redshit a spectrum

    Parameters
    ----------
    wavelength: array like of floats
        The wavelength in angstrom.
    flux: array like of floats
        Flux density.
    redshift: float
        The redshift.
    is_fnu: boolean
        If false (default) the flux is a Fλ density in erg/cm²/s/A. 
        If true, the flux is a Fν density in maggies.

    Results
    -------
    wavelength, flux: tuple of numpy arrays of floats
        The redshifted spectrum with the same kind of flux (or luminosity)
        density as the input.

    """
    wavelength = np.array(wavelength, dtype=float)
    flux = np.array(flux, dtype=float)
    redshift = float(redshift)

    redshift_factor = 1. + redshift

    if is_fnu:
        # Switch to Fλ
        flux = lambda_fnu_to_flambda_aa_maggies(wavelength, flux)

    wavelength /= redshift_factor
    flux *= redshift_factor

    if is_fnu:
        # Switch back to Fλ
        flux = lambda_flambda_to_fnu_aa_maggies(wavelength, flux)

    return wavelength, flux


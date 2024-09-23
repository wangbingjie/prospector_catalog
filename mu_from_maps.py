import time, sys, os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

def rt_dir():
    rt_roar = '/storage/home/bbw5389/group/'
    rt_mac = '/Users/bwang/research/'
    dat_dirs = [rt_roar, rt_mac]
    for _dir in dat_dirs:
        if os.path.isdir(_dir): return _dir

gamma_fname = rt_dir()+'uncover_sps_gen1/phot_catalog/low_res/hlsp_uncover_model_a2744_zitrin-dPIE-PIEMD_gamma.fits'
if not os.path.exists(gamma_fname):
    gamma_fname = rt_dir()+'uncover_sps_gen1/phot_catalog/low_res/hlsp_uncover_model_a2744_zitrin-dPIE-PIEMD_gamma.fits.gz'
gamma_map = fits.getdata(gamma_fname, ext=0)
wcs = WCS(fits.getheader(gamma_fname))
kappa_fname = rt_dir()+'uncover_sps_gen1/phot_catalog/low_res/hlsp_uncover_model_a2744_zitrin-dPIE-PIEMD_kappa.fits'
if not os.path.exists(kappa_fname):
    kappa_fname = rt_dir()+'uncover_sps_gen1/phot_catalog/low_res/hlsp_uncover_model_a2744_zitrin-dPIE-PIEMD_kappa.fits.gz'
kappa_map = fits.getdata(kappa_fname, ext=0)

def mu_from_kappa_gamma(kappa, gamma):
    res = (1-kappa)**2 - gamma**2
    res = np.absolute(1.0/res)
    return res

def xy_in_kappa_and_gamma(ra, dec, verbose=False):
    '''get pixel coords for the maps
    '''
    px, py = wcs.wcs_world2pix(ra, dec, 0)
    px = int(np.round(px, 0))
    py = int(np.round(py, 0))
    if verbose:
        print('px, py', px, py)
    return (px, py)

def scale_mu(zred, px, py, lens_zred=0.308, verbose=False):

    if zred <= lens_zred:
        if verbose:
            print('scale_mu:', zred, '<=', lens_zred)
        return 1

    try:
        _kappa = kappa_map[py, px]
        _gamma = gamma_map[py, px]
        if verbose:
            print(_kappa, _gamma)

        dds = cosmo.angular_diameter_distance_z1z2(lens_zred, zred)
        ds = cosmo.angular_diameter_distance(zred)
        dist_ratio = dds/ds

        kappa_scaled = _kappa * dist_ratio
        gamma_scaled = _gamma * dist_ratio
        if verbose:
            print(kappa_scaled, gamma_scaled)

        mu = mu_from_kappa_gamma(kappa=kappa_scaled, gamma=gamma_scaled)

        if verbose:
            print('mu', mu)

        return mu.value

    except:
        if verbose:
            print('outside lensing model FoV')
        return 1


if __name__ == "__main__":
    # example
    idx = 41611 - 1
    cat = Table.read(rt_dir()+'uncover_sps_gen1/sps_catalog/UNCOVER_v5.2.0_LW_SUPER_SPScatalog_spsv0.1.fits')
    obj_from_cat = cat[idx]
    ra, dec = obj_from_cat['ra']*u.deg, obj_from_cat['dec']*u.deg
    px, py = xy_in_kappa_and_gamma(ra, dec)
    mu = scale_mu(obj_from_cat['z_50'], px, py, verbose=True)
    print(obj_from_cat['id'], 'mu = ', mu)

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import numpy.ma as ma
import astropy.units as u
from astropy.table import Table
from astropy.cosmology import WMAP9 as cosmo

def rt_dir():

    dat_dirs = ['/storage/home/bbw5389/group/',
                '/Users/bwang/research/']

    for _dir in dat_dirs:
        if os.path.isdir(_dir): return _dir

def data_dir(data='cwd'):

    rt_roar = '/storage/home/bbw5389/group/'
    rt_mac = '/Users/bwang/research/'

    if data == 'cwd':
        dat_dirs = [rt_roar + 'uncover_sps_gen1/stellar_pop_catalog_mb/',
                    rt_mac + 'uncover_sps_gen1/stellar_pop_catalog_mb/'
                   ]
    elif data == 'pbeta':
        dat_dirs = [rt_roar + 'uncover_sps_gen1/prospector_beta/',
                    rt_mac + 'uncover_sps_gen1/prospector_beta/'
                   ]
    elif data == 'gen1':
        dat_dirs = [rt_roar + 'uncover_sps_gen1/',
                    rt_mac + 'uncover_sps_gen1/'
                   ]
    elif data == 'pirate':
        dat_dirs = [rt_roar + 'sed/pirate/pirate/data/nn/',
                    rt_mac + 'software/MathewsEtAl2023/data/']
    else:
        return None

    for _dir in dat_dirs:
        if os.path.isdir(_dir): return _dir
        

def finished_id(indir=None, prior='phisfh', dtype='phot', rt_fname=False, verbose=True, **extra):
    '''if exists id_{}_phot_phisfh.npz, then id_{}_mcmc_phisfh.h5 must be completed.
    '''
    if indir is None:
        fnames = os.listdir()
    else:
        fnames = os.listdir(indir)

    i_str = 1 # id

    _idxs = []
    _fnames = []
    for i in range(len(fnames)):
        this_fname = fnames[i]
        if prior in this_fname and dtype in this_fname:
            _idxs.append(int(this_fname.split('_')[i_str]))
            _fnames.append(this_fname)
    _idxs = np.sort(_idxs)

    if verbose:
        print('finished in', indir, dtype, len(_idxs))

    if rt_fname:
        return _fnames
    else:
        return _idxs

# ----------------- catalog specific

def filter_dictionary(mb=True, alma=True):
    '''alma bands includes upper limits, which requires special treatment;
    will be dealt with in get_cat_fnu() & get_cat_enu()
    '''
    if not mb:
        filter_dict = {'f435w': 'acs_wfc_f435w',
                       'f606w': 'acs_wfc_f606w',
                       'f814w': 'acs_wfc_f814w',
                       'f090w': 'jwst_f090w',
                       'f105w': 'wfc3_ir_f105w',
                       'f115w': 'jwst_f115w',
                       'f125w': 'wfc3_ir_f125w',
                       'f140w': 'wfc3_ir_f140w',
                       'f150w': 'jwst_f150w',
                       'f160w': 'wfc3_ir_f160w',
                       'f200w': 'jwst_f200w',
                       'f277w': 'jwst_f277w',
                       'f356w': 'jwst_f356w',
                       'f410m': 'jwst_f410m',
                       'f444w': 'jwst_f444w',
                       'alma':  '1d3mm'
                       }
    else:
        filter_dict = {'f435w': 'acs_wfc_f435w',
                       'f606w': 'acs_wfc_f606w',
                       'f814w': 'acs_wfc_f814w',
                       'f070w': 'jwst_f070w',
                       'f090w': 'jwst_f090w',
                       'f105w': 'wfc3_ir_f105w',
                       'f115w': 'jwst_f115w',
                       'f125w': 'wfc3_ir_f125w',
                       'f140w': 'wfc3_ir_f140w',
                       'f140m': 'jwst_f140m',
                       'f150w': 'jwst_f150w',
                       'f160w': 'wfc3_ir_f160w',
                       'f162m': 'jwst_f162m',
                       'f182m': 'jwst_f182m',
                       'f200w': 'jwst_f200w',
                       'f210m': 'jwst_f210m',
                       'f250m': 'jwst_f250m',
                       'f277w': 'jwst_f277w',
                       'f300m': 'jwst_f300m',
                       'f335m': 'jwst_f335m',
                       'f356w': 'jwst_f356w',
                       'f360m': 'jwst_f360m',
                       'f410m': 'jwst_f410m',
                       'f430m': 'jwst_f430m',
                       'f444w': 'jwst_f444w',
                       'f460m': 'jwst_f460m',
                       'f480m': 'jwst_f480m',
                       'alma':  '1d3mm'
                       }
    if not alma:
        filter_dict.pop('alma')
    return filter_dict

def get_cat_fnu(idx, catalog, filts):
    '''Given a row idx in the catalog,
    return all flux in ABmag ZP, i.e., ABmag = ZP - 2.5*log10(f_f444w).
    '''
    fnus = []
    cols_fnu = []
    for filt in filts:
        cols_fnu.append('f_{}'.format(filt))
    for i in cols_fnu:
        this = catalog[idx][i]
        if ma.is_masked(this):
            this = np.nan
        fnus.append(this)

    if 'f_alma' in catalog.colnames:
        f_alma = catalog[idx]['f_alma']
        e_alma = catalog[idx]['e_alma']
        # alma detected
        if (not ma.is_masked(f_alma)) and (not ma.is_masked(e_alma)):
            fnus[-1] = f_alma
        # upper limit
        elif ma.is_masked(f_alma) and (not ma.is_masked(e_alma)):
            fnus[-1] = 0.0
        # nothing
        else:
            fnus[-1] = np.nan

    return np.array(fnus)

def get_cat_enu(idx, catalog, filts):
    '''Given a row idx in the catalog,
    return all flux in ABmag ZP, i.e., ABmag = ZP - 2.5*log10(f_f444w).
    '''
    enus = []
    cols_enu = []
    for filt in filts:
        cols_enu.append('e_{}'.format(filt))
    for i in cols_enu:
        this = catalog[idx][i]
        if ma.is_masked(this):
            this = np.nan
        enus.append(this)

    if 'f_alma' in catalog.colnames:
        f_alma = catalog[idx]['f_alma']
        e_alma = catalog[idx]['e_alma']
        # alma detected
        if (not ma.is_masked(f_alma)) and (not ma.is_masked(e_alma)):
            enus[-1] = e_alma
        # upper limit
        elif ma.is_masked(f_alma) and (not ma.is_masked(e_alma)):
            enus[-1] = e_alma
        # nothing
        else:
            enus[-1] = np.nan

    return np.array(enus)

def get_fnu_maggies(idx, catalog, filts, abzp=28.9):
    fnu = get_cat_fnu(idx, catalog, filts)
    return abzp_to_maggies(fnu, abzp)

def get_enu_maggies(idx, catalog, filts, abzp=28.9):
    fnu = get_cat_enu(idx, catalog, filts)
    return abzp_to_maggies(fnu, abzp)

# -----------------

lightspeed = 2.998e18  # AA/s
jansky_cgs = 1e-23
def maggie_to_cgs(wave_angstrom, spec_maggies):
    # maggies to erg/s/cm^2/AA
    fnu_jy = spec_maggies * 3631.0
    wave_um = wave_angstrom/1e4
    flam = 3e-13 * fnu_jy / wave_um**2
    return flam

def maggies_to_njy(fnu):
    return fnu * 3631.0 * 1e9

def jy_to_maggies(flux):
    '''1 maggie is the flux density in Janskys divided by 3631
    microJy = 1e-6 Jy
    '''
    return flux/3631

def abzp_to_maggies(fnu, abzp=28.9):
    maggies = fnu*10**(-0.4*abzp)
    return maggies

def maggie_to_abmag(fnu):
    res = np.log10(fnu)/(-0.4)
    return res

def abzp_to_abmag(fnu, abzp=28.9):
    _maggie = abzp_to_maggies(fnu=fnu, abzp=abzp)
    return maggie_to_abmag(fnu=_maggie)

def abmag_to_maggie(mag):
    return 10**(mag*(-0.4))

def abzp_to_njy(fnu, abzp=28.9):
    _fnu = abzp_to_maggies(fnu=fnu, abzp=abzp)
    return maggies_to_njy(fnu=_fnu)

def microJy_to_abmag(x):
    '''uJy to AB mag
    '''
    return 23.9 - np.log10(x) * 2.5

def mapp_to_Muv(m_apparent, zred, mu=1):
    '''apparent AB mag to absolute AB mag
    '''
    dl = cosmo.luminosity_distance(zred).to(u.pc)
    return m_apparent - 5*np.log10(dl.value) + 5 + 2.5*np.log10(1+zred) + 2.5*np.log10(mu)

def quantile(data, percents=[16,50,84], weights=None):
    ''' percents in units of 1%
    weights specifies the frequency (count) of data.
    '''
    if weights is None:
        return np.percentile(data, percents)
    ind = np.argsort(data)
    d = data[ind]
    w = weights[ind]
    p = 1.*w.cumsum()/w.sum()*100
    y = np.interp(percents, p, d)
    return y

def elum_to_cgs(model):
    """call after model.predict()
    convert model._eline_lum in fsps units to erg/s/cm^2
    """
    elams = model._ewave_obs[model._use_eline]
    # We have to remove the extra (1+z) since this is flux, not a flux density
    # Also we convert to cgs
    model.line_norm = model.flux_norm() / (1 + model._zred) * (3631*jansky_cgs)
    elums = model._eline_lum[model._use_eline] * model.line_norm

    return elums

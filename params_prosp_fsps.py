import sys, os
import numpy as np
from astropy.cosmology import WMAP9 as cosmo
from prospect.models import priors, transforms

def params_fsps_phisfh(obs=None, free_gas_logu=False, **extra):
    '''mass + metallicity + sfh priors from p-beta
    '''

    from prospect.models import priors_beta as PZ

    if free_gas_logu:
        fit_order = ['nzsfh',
                     'dust2', 'dust_index', 'dust1_fraction',
                     'log_fagn', 'log_agn_tau', 'gas_logz',
                     'duste_qpah', 'duste_umin', 'log_duste_gamma',
                     'gas_logu']
    else:
        fit_order = ['nzsfh',
                     'dust2', 'dust_index', 'dust1_fraction',
                     'log_fagn', 'log_agn_tau', 'gas_logz',
                     'duste_qpah', 'duste_umin', 'log_duste_gamma']

    # -------------
    # MODEL_PARAMS
    model_params = {}

    # --- BASIC PARAMETERS ---
    model_params['nzsfh'] = {'N': 9,
                             'isfree': True,
                             'init': np.array([0.5,8,0.0, 0,0,0,0,0,0]),
                             'prior': PZ.PhiSFH(zred_mini=1e-3, zred_maxi=20.0,
                                                mass_mini=6.0, mass_maxi=12.5,
                                                z_mini=-1.98, z_maxi=0.19,
                                                logsfr_ratio_mini=-5.0, logsfr_ratio_maxi=5.0,
                                                logsfr_ratio_tscale=0.3, nbins_sfh=7,
                                                const_phi=True)}

    model_params['zred'] = {'N': 1, 'isfree': False,
                            'depends_on': transforms.nzsfh_to_zred,
                            'init': 0.5,
                            'prior': priors.Uniform(mini=1e-3, maxi=20.0)}

    model_params['logmass'] = {'N': 1, 'isfree': False,
                              'depends_on': transforms.nzsfh_to_logmass,
                              'init': 8.0,
                              'units': 'Msun',
                              'prior': priors.Uniform(mini=6.0, maxi=12.5)}

    model_params['logzsol'] = {'N': 1, 'isfree': False,
                               'init': -0.5,
                               'depends_on': transforms.nzsfh_to_logzsol,
                               'units': r'$\log (Z/Z_\odot)$',
                               'prior': priors.Uniform(mini=-1.98, maxi=0.19)}

    model_params['imf_type'] = {'N': 1, 'isfree': False,
                                'init': 1, #1 = chabrier
                                'units': "FSPS index",
                                'prior': None}

    model_params['add_igm_absorption'] = {'N': 1, 'isfree': False, 'init': True}
    model_params["add_agb_dust_model"] = {'N': 1, 'isfree': False, 'init': True}
    model_params["pmetals"] = {'N': 1, 'isfree': False, 'init': -99}

    # --- SFH ---
    nbins_sfh = 7
    model_params["sfh"] = {'N': 1, 'isfree': False, 'init': 3}
    model_params['logsfr_ratios'] = {'N': nbins_sfh-1, 'isfree': False,
                                     'init': 0.0,
                                     'depends_on': transforms.nzsfh_to_logsfr_ratios,
                                     'prior': priors.StudentT(mean=np.full(nbins_sfh-1,0.0),
                                              scale=np.full(nbins_sfh-1,0.3),
                                              df=np.full(nbins_sfh-1,2))}

    model_params["mass"] = {'N': 7,
                            'isfree': False,
                            'init': 1e6,
                            'units': r'M$_\odot$',
                            'depends_on': logsfr_ratios_to_masses}

    model_params['agebins'] = {'N': 7, 'isfree': False,
                               'init': zred_to_agebins(np.atleast_1d(0.5)),
                               'prior': None,
                               'depends_on': zred_to_agebins}

    # --- Dust Absorption ---
    model_params['dust_type'] = {"N": 1, "isfree": False, "init": 4, "units": "FSPS index"}
    model_params['dust1_fraction'] = {'N': 1, 'isfree': True,
                                      'init': 1.0,
                                      'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3),
                                      }

    model_params['dust2'] = {'N': 1, 'isfree': True,
                             'init': 0.0,
                             'units': '',
                             'prior': priors.ClippedNormal(mini=0.0, maxi=4.0, mean=0.3, sigma=1.0),
                             }

    model_params['dust1'] = {"N": 1,
                             "isfree": False,
                             'depends_on': to_dust1,
                             "init": 0.0, "units": "optical depth towards young stars",
                             "prior": None}
    model_params['dust_index'] = {'N': 1, 'isfree': True,
                                  'init': 0.7,
                                  'units': '',
                                  'prior': priors.Uniform(mini=-1.0, maxi=0.4)}

    # --- Nebular Emission ---
    model_params['add_neb_emission'] = {'N': 1, 'isfree': False, 'init': True}
    model_params['add_neb_continuum'] = {'N': 1, 'isfree': False, 'init': True}
    model_params['gas_logz'] = {'N': 1, 'isfree': True,
                                'init': -0.5,
                                'units': r'log Z/Z_\odot',
                                'prior': priors.Uniform(mini=-2.0, maxi=0.5)}
    if free_gas_logu:
        model_params['gas_logu'] = {"N": 1, 'isfree': True,
                                    'init': -1.0, 'units': r"Q_H/N_H",
                                    'prior': priors.Uniform(mini=-4, maxi=-1)}
    else:
        model_params['gas_logu'] = {"N": 1, 'isfree': False,
                                    'init': -1.0, 'units': r"Q_H/N_H",
                                    'prior': priors.Uniform(mini=-4, maxi=-1)}

    # --- AGN dust ---
    model_params['add_agn_dust'] = {"N": 1, "isfree": False, "init": True}

    model_params['log_fagn'] = {'N': 1, 'isfree': True,
                                'init': -7.0e-5,
                                'prior': priors.Uniform(mini=-5.0, maxi=np.log10(3.0))}
    model_params['fagn'] = {"N": 1, "isfree": False, "init": 0, "depends_on": to_fagn}

    model_params['log_agn_tau'] = {'N': 1, 'isfree': True,
                                   'init': np.log10(20.0),
                                   'prior': priors.Uniform(mini=np.log10(5.0), maxi=np.log10(150.0))}
    model_params['agn_tau'] = {"N": 1, "isfree": False, "init": 0, "depends_on": to_agn_tau}

    # --- Dust Emission ---
    model_params['duste_qpah'] = {'N':1, 'isfree':True,
                                  'init': 2.0,
                                  'prior': priors.ClippedNormal(mini=0.0, maxi=7.0, mean=2.0, sigma=2.0)}

    model_params['duste_umin'] = {'N':1, 'isfree':True,
                                  'init': 1.0,
                                  'prior': priors.ClippedNormal(mini=0.1, maxi=25.0, mean=1.0, sigma=10.0)}

    model_params['log_duste_gamma'] = {'N':1, 'isfree':True,
                                       'init': -2.0,
                                       'prior': priors.ClippedNormal(mini=-4.0, maxi=0.0, mean=-2.0, sigma=1.0)}
    model_params['duste_gamma'] = {"N": 1, "isfree": False, "init": 0, "depends_on": to_duste_gamma}

    #---- Units ----
    model_params['peraa'] = {'N': 1, 'isfree': False, 'init': False}

    model_params['mass_units'] = {'N': 1, 'isfree': False, 'init': 'mformed'}

    model_params['sigma_smooth'] = {'N': 1, 'isfree': False, 'init': 0.0}

    extra = [k for k in model_params.keys() if k not in fit_order]
    fit_order = fit_order + list(extra)

    tparams = {}
    for i in fit_order:
        tparams[i] = model_params[i]
    for i in list(model_params.keys()):
        if i not in fit_order:
            tparams[i] = model_params[i]
    model_params = tparams

    return (model_params, fit_order)


def params_fsps_alpha(obs=None, free_gas_logu=False, **extra):
    '''p-alpha priors
    '''

    if free_gas_logu:
        fit_order = ['zred', 'logmass', 'logzsol', 'logsfr_ratios',
                     'dust2', 'dust_index', 'dust1_fraction',
                     'log_fagn', 'log_agn_tau', 'gas_logz',
                     'duste_qpah', 'duste_umin', 'log_duste_gamma', 'gas_logu']
    else:
        fit_order = ['zred', 'logmass', 'logzsol', 'logsfr_ratios',
                     'dust2', 'dust_index', 'dust1_fraction',
                     'log_fagn', 'log_agn_tau', 'gas_logz',
                     'duste_qpah', 'duste_umin', 'log_duste_gamma']

    # -------------
    # MODEL_PARAMS
    model_params = {}

    # --- BASIC PARAMETERS ---
    model_params['zred'] = {'N': 1, 'isfree': True,
                            'init': 0.5,
                            'prior': priors.Uniform(mini=1e-3, maxi=20.0)}

    model_params['logmass'] = {'N': 1, 'isfree': True,
                              'init': 8.0,
                              'units': 'Msun',
                              'prior': priors.Uniform(mini=6.0, maxi=12.5)}

    model_params['logzsol'] = {'N': 1, 'isfree': True,
                               'init': -0.5,
                               'units': r'$\log (Z/Z_\odot)$',
                               'prior': priors.Uniform(mini=-1.98, maxi=0.19)}

    model_params['imf_type'] =  {'N': 1,
                                'isfree': False,
                                'init': 1, #1 = chabrier
                                'units': None,
                                'prior': None}
    model_params['add_igm_absorption'] = {'N': 1, 'isfree': False, 'init': True}
    model_params["add_agb_dust_model"] = {'N': 1, 'isfree': False, 'init': True}
    model_params["pmetals"] = {'N': 1, 'isfree': False, 'init': -99}

    # --- SFH ---
    nbins_sfh = 7
    model_params["sfh"] = {'N': 1, 'isfree': False, 'init': 3}
    model_params['logsfr_ratios'] = {'N': nbins_sfh-1, 'isfree': True,
                                     'init': 0.0,
                                     'prior': priors.StudentT(mean=np.full(nbins_sfh-1,0.0),
                                              scale=np.full(nbins_sfh-1,0.3),
                                              df=np.full(nbins_sfh-1,2))}

    model_params["mass"] = {'N': 7,
                            'isfree': False,
                            'init': 1e6,
                            'units': r'M$_\odot$',
                            'depends_on': logsfr_ratios_to_masses}

    model_params['agebins'] = {'N': 7, 'isfree': False,
                               'init': zred_to_agebins(np.atleast_1d(0.5)),
                               'prior': None,
                               'depends_on': zred_to_agebins}

    # --- Dust Absorption ---
    model_params['dust_type'] = {"N": 1, "isfree": False, "init": 4, "units": "FSPS index"}
    model_params['dust1_fraction'] = {'N': 1, 'isfree': True,
                                      'init': 1.0,
                                      'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}

    model_params['dust2'] = {'N': 1, 'isfree': True,
                             'init': 0.3,
                             'units': '',
                             'prior': priors.ClippedNormal(mini=0.0, maxi=4.0, mean=0.3, sigma=1.0)}

    model_params['dust1'] = {"N": 1,
                             "isfree": False,
                             'depends_on': to_dust1,
                             "init": 0.0, "units": "optical depth towards young stars",
                             "prior": None}
    model_params['dust_index'] = {'N': 1, 'isfree': True,
                                  'init': -0.3,
                                  'units': '',
                                  'prior': priors.Uniform(mini=-1.0, maxi=0.4)}

    # --- Nebular Emission ---
    model_params['add_neb_emission'] = {'N': 1, 'isfree': False, 'init': True}
    model_params['add_neb_continuum'] = {'N': 1, 'isfree': False, 'init': True}
    model_params['gas_logz'] = {'N': 1, 'isfree': True,
                                'init': -0.5,
                                'units': r'log Z/Z_\odot',
                                'prior': priors.Uniform(mini=-2.0, maxi=0.5)}
    model_params['gas_logu'] = {"N": 1, 'isfree': False,
                                'init': -1.0, 'units': r"Q_H/N_H",
                                'prior': priors.Uniform(mini=-4.0, maxi=-1.0)}
    if free_gas_logu:
        model_params['gas_logu'] = {"N": 1, 'isfree': True,
                                    'init': -1.0, 'units': r"Q_H/N_H",
                                    'prior': priors.Uniform(mini=-4.0, maxi=-1.0)}

    # --- AGN dust ---
    model_params['add_agn_dust'] = {"N": 1, "isfree": False, "init": True}

    model_params['log_fagn'] = {'N': 1, 'isfree': True,
                                'init': -7.0e-5,
                                'prior': priors.Uniform(mini=-5.0, maxi=np.log10(3.0))}
    model_params['fagn'] = {"N": 1, "isfree": False, "init": 0, "depends_on": to_fagn}

    model_params['log_agn_tau'] = {'N': 1, 'isfree': True,
                                   'init': np.log10(20.0),
                                   'prior': priors.Uniform(mini=np.log10(5.0), maxi=np.log10(150.0))}
    model_params['agn_tau'] = {"N": 1, "isfree": False, "init": 0, "depends_on": to_agn_tau}

    # --- Dust Emission ---
    model_params['duste_qpah'] = {'N':1, 'isfree':True,
                                  'init': 2.0,
                                  'prior': priors.ClippedNormal(mini=0.0, maxi=7.0, mean=2.0, sigma=2.0)}

    model_params['duste_umin'] = {'N':1, 'isfree':True,
                                  'init': 1.0,
                                  'prior': priors.ClippedNormal(mini=0.1, maxi=25.0, mean=1.0, sigma=10.0)}

    model_params['log_duste_gamma'] = {'N':1, 'isfree':True,
                                       'init': -2.0,
                                       'prior': priors.ClippedNormal(mini=-4.0, maxi=0.0, mean=-2.0, sigma=1.0)}
    model_params['duste_gamma'] = {"N": 1, "isfree": False, "init": 0, "depends_on": to_duste_gamma}

    #---- Units ----
    model_params['peraa'] = {'N': 1, 'isfree': False, 'init': False}

    model_params['mass_units'] = {'N': 1, 'isfree': False, 'init': 'mformed'}

    model_params['sigma_smooth'] = {'N': 1, 'isfree': False, 'init': 0.0}

    extra = [k for k in model_params.keys() if k not in fit_order]
    fit_order = fit_order + list(extra)

    tparams = {}
    for i in fit_order:
        tparams[i] = model_params[i]
    for i in list(model_params.keys()):
        if i not in fit_order:
            tparams[i] = model_params[i]
    model_params = tparams

    return (model_params, fit_order)

def zred_to_agebins(zred=None,**extras):
    """parrot v3 and after

    Set the nonparameteric SFH age bins depending on the age of the universe
    at ``zred``. The first bin is not altered and the last bin is always 10% of
    the upper edge of the oldest bin, but the intervening bins are evenly
    spaced in log(age).

    Parameters
    ----------
    zred : float
        Cosmological redshift.  This sets the age of the universe.

    agebins :  ndarray of shape ``(nbin, 2)``
        The SFH bin edges in log10(years).

    Returns
    -------
    agebins : ndarray of shape ``(nbin, 2)``
        The new SFH bin edges.
    """
    amin = 7.1295
    nbins_sfh = 7
    tuniv = cosmo.age(zred)[0].value*1e9 # because input zred is atleast_1d
    tbinmax = (tuniv*0.9)
    if (zred <= 3.):
        agelims = [0.0,7.47712] + np.linspace(8.0,np.log10(tbinmax),nbins_sfh-2).tolist() + [np.log10(tuniv)]
    else:
        agelims = np.linspace(amin,np.log10(tbinmax),nbins_sfh).tolist() + [np.log10(tuniv)]
        agelims[0] = 0

    agebins = np.array([agelims[:-1], agelims[1:]])
    return agebins.T

def logsfr_ratios_to_masses(logmass=None, logsfr_ratios=None, agebins=None, **extras):
    """This converts from an array of log_10(SFR_j / SFR_{j+1}) and a value of
    log10(\Sum_i M_i) to values of M_i.  j=0 is the most recent bin in lookback
    time.
    """
    nbins = agebins.shape[0]
    sratios = 10**np.clip(logsfr_ratios, -10, 10)  # numerical issues...
    dt = (10**agebins[:, 1] - 10**agebins[:, 0])
    coeffs = np.array([ (1. / np.prod(sratios[:i])) * (np.prod(dt[1: i+1]) / np.prod(dt[: i]))
                        for i in range(nbins)])
    m1 = (10**logmass) / coeffs.sum()

    return m1 * coeffs

def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    return dust1_fraction*dust2

def to_fagn(log_fagn=None, **extras):
    return 10**log_fagn

def to_agn_tau(log_agn_tau=None, **extras):
    return 10**log_agn_tau

def to_duste_gamma(log_duste_gamma=None, **extras):
    return 10**log_duste_gamma


from prospect.sources import FastStepBasis
def build_sps_fsps(zcontinuous=2, compute_vega_mags=False, **extras):
    sps = FastStepBasis(zcontinuous=zcontinuous, compute_vega_mags=compute_vega_mags)
    return sps


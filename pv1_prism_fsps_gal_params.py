'''
works with prospector v1;
fitting photometry + spectrum simultaneously
'''
import time, sys, os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from scipy import interpolate

from astropy.table import Table
from astropy.io import ascii
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
from astropy.io import fits

import sedpy

import prospect
from prospect.utils.obsutils import fix_obs
from prospect.sources import FastStepBasis
from prospect.models import transforms, priors
from prospect.models.templates import TemplateLibrary
from prospect.likelihood import NoiseModel
from prospect.likelihood.kernels import Uncorrelated
from prospect.models.sedmodel import PolySpecModel

from prospect.models import priors_beta as pb

import utils as ut_phot

# - Parser with default arguments -
parser = prospect.prospect_args.get_parser("dynesty")
parser.add_argument('--objid', type=int, default=40579, help="ID of the object to fit")
parser.add_argument('--cat_phot', type=str, default='mask/primer_uds_prism.fits', help='photometry catalog')
parser.add_argument('--zspec', type=float, default=3.11)
parser.add_argument('--spec_fits', type=str, default="data/rubies-uds1-v2_prism-clear_4233_40579.spec.fits")
parser.add_argument('--outdir', type=str, default='test/', help="Output folder name.")

parser.add_argument('--pmodel', type=str, default='palpha', help="prospector model: palpha, pbeta")

parser.add_argument('--dyn', type=int, default=0, 
                    help="If 0, std run; if 1, quick dynesty run; if 2, debug, max=1100")
parser.add_argument('--run', type=str, default='phisfh', help="appended to the output filename")

parser.add_argument('--neb_marg', type=int, default=1, help="If 1, turn on nebular_marginalization")
parser.add_argument('--rm_cont', type=int, default=1, help="If 1, turn on polynomial correction to rm continuum")
parser.add_argument('--npoly', type=int, default=7, help="polynomial order")

args = parser.parse_args()
run_params = vars(args)

run_params.update({
'verbose': True,
'dyn': args.dyn,
'outdir': args.outdir,
'nofork': True,
# model
'neb_marg': bool(args.neb_marg),
'rm_cont': bool(args.rm_cont),
'model_noise': True,
# dynesty params
'dynesty': True,
'nested_maxcall': None,
'nested_maxcall_init': None,
'nested_maxiter': None,
'nested_maxbatch': None, # maximum number of dynamic patches
'nested_bound': 'multi', # bounding method
'nested_sample': 'rwalk', # sampling method
# 'nested_walks': 32,  # original-64 sampling gets very inefficient w/ high S/N spectra
'nested_nlive_batch': 400, # size of live point "batches"
'nested_nlive_init': 1600, # number of initial live points
'nested_weight_kwargs': {'pfrac': 1.0}, # weight posterior over evidence by 100%
'nested_dlogz_init': 0.01,
'nested_target_n_effective': 10000,
# 'nested_posterior_thresh': 0.02,
# Model info - not much of this is actually needed
'zcontinuous': 2,
'compute_vega_mags': False,
'initial_disp':0.1,
'interp_type': 'logarithmic',
'nbins_sfh': 7,
'sigma': 0.3,
'df': 2,
'agelims': [0.0,7.4772,8.0,8.5,9.0,9.5,9.8,10.0]
})
run_params['nested_first_update'] = {'min_ncall': 20000, 'min_eff': 7.5}

if run_params['dyn'] == 1:
    # quick dynesty fits
    run_params.update({
    'nested_nlive_init': 600,
    'nested_dlogz_init': 0.1,
    'nested_target_n_effective': 10000,
    'nested_maxcall': 500000,
    'nested_maxcall_init': 500000,
    })
if run_params['dyn'] == 2:
    # debug
    run_params.update({
    'nested_maxcall': 2000,
    'nested_maxcall_init': 2000,
    })
    
run_params["param_file"] = __file__

assert os.path.exists(run_params['outdir']) == True, "no output folder, {} !".format(run_params['outdir'])
print('run_params:', run_params)


model_agn = False
    
# ----------------------
try:
    SPS_HOME = os.getenv('SPS_HOME')
    info = np.genfromtxt(os.path.join(SPS_HOME, 'data', 'emlines_info.dat'),
                         dtype=[('wave', 'f8'), ('name', '<U20')],
                         delimiter=',')
except(OSError, KeyError, ValueError) as e:
    print("Could not read and cache emission line info from $SPS_HOME/data/emlines_info.dat")
    raise(OSError)

nelines = info['name'].shape[0]


def build_obs(objid, zspec,
              err_floor=0.05, calib_sn_cut=5, **extras):
    """Build a dictionary of observational data

    :param objid:
        The ID of the galaxy to be fit.

    :returns obs:
        A dictionary of observational data to use in the fit.
    """

    obs = {}
    obs['id'] = objid
    obs['spec_file'] = args.spec_fits
    obs['zspec'] = zspec
    zred = obs['zspec']

    idx_phot = np.where(cat_phot['id']==obs['id'])[0][0]
    filter_dict = ut_phot.filter_dictionary()
    filter_code = list(filter_dict.keys())
    filter_name = list(filter_dict.values())
    
    obs['x_pixel'] = 0; obs['y_pixel'] = 0

    # ------- photometry -------
    ph = cat_phot[idx_phot]

    obs['filters'] = sedpy.observate.load_filters(filter_name)
    obs['phot_wave'] = np.array([f.wave_effective for f in obs['filters']])

    flux = ut_phot.get_fnu_maggies(idx_phot, cat_phot, filter_code)
    unc = ut_phot.get_enu_maggies(idx_phot, cat_phot, filter_code)
    
    obs["maggies"] = flux
    obs["maggies_unc"] = unc

    phot_mask = (unc > 0) & (np.isfinite(flux))
    _mask = np.ones_like(unc, dtype=bool)
    for k in range(len(flux)):
        if unc[k] > 0:
            if flux[k] < 0 and flux[k] + 5*unc[k] < 0:
                _mask[k] = False
    phot_mask &= _mask

    obs['phot_mask'] = phot_mask
    obs['maggies_unc'] = np.clip(obs['maggies_unc'], a_min=obs['maggies']*err_floor, a_max=None)

    # -------- spectrum --------
    spec = fits.open(obs['spec_file'])

    obs['wavelength'] = spec[1].data['wave'] * 10000 # convert microns to AA
    obs['spectrum'] = ut_phot.jy_to_maggies(spec[1].data['flux']*1e-6)
    obs['unc_raw'] = ut_phot.jy_to_maggies(spec[1].data['err']*1e-6)
    obs['unc'] = np.clip(obs['unc_raw'], a_min=obs['spectrum']*err_floor, a_max=None)
    
    # LSF
    ## from jdox
    ## -------- read resolution data
    nirspec = fits.open('data/jwst_nirspec_prism_disp.fits')

    nirspec_wave = nirspec[1].data['WAVELENGTH'] # in micron
    nirspec_R = nirspec[1].data['R']
    # nirspec_R *= 1.3 # per Gabe's suggestion
    func_nirspec = interpolate.interp1d(nirspec_wave, 3e5/(nirspec_R*2.355))

    # interpolate nirspec_wave vs. nirspec_sigma_v to get sigma_v for observed wavelength points.
    obs['sigma_v'] = func_nirspec(spec[1].data['wave']) # wavelength should be in micron
    
    '''
    ## from msafit, for a point source
    nirspec = ascii.read('data/lsf/point_source_lsf_clear_prism_QD3_i185_j85.csv')
    nirspec_wave = nirspec['wave'] / 1e4 # in micron
    nirspec_sigmav = 3e5*nirspec['sigma']/nirspec['wave']
    func_nirspec = interpolate.interp1d(nirspec_wave, nirspec_sigmav, bounds_error=False, fill_value='extrapolate')
    obs['sigma_v'] = func_nirspec(spec[1].data['wave'])
    '''

    # -------- mask out regions
    # TBD: data defects, spectral lines, etc.
    mask = np.isfinite(obs['spectrum'])
    mask &= np.isfinite(obs['unc'])
    mask &= obs['unc'] > 0
    mask &= np.isfinite(obs['sigma_v'])
    mask &= np.isfinite(obs['wavelength'])

    obs['mask'] = mask
    
    obs = prospect.utils.obsutils.fix_obs(obs)

    # plt.step(obs['wavelength'], obs['spectrum'], color='k', alpha=0.5)
    # plt.step(obs['wavelength'][mask], obs['spectrum'][mask], alpha=1)
    # plt.step(obs['wavelength'], obs['unc_raw'], alpha=0.2)
    # plt.scatter(obs['phot_wave'], obs['maggies'])
    # plt.show()
    # sys.exit()

    return obs


def build_model(obs=None, waverange=None, neb_marg=True, 
                rm_cont=True, npoly=7,
                model_agn=True,
                **extras):

    """Build a prospect.models.SedModel object

    :param zred: (optional, default: None)
        approximate value for the redshift, which is left as a free parameter.

    :param waverange: (optional, default: None)
        rest-frame wavelength range in angstrom; used to calculate polyorder.

    :returns model:
        An instance of prospect.models.SedModel
    """

    import params_prosp_fsps as pfile

    model_params = None
    fit_order = None
    if run_params['pmodel'] == 'palpha':
        model_params, fit_order = pfile.params_fsps_alpha(obs=obs, free_gas_logu=True)
        model_params['zred'] = {'N': 1, 'isfree': True, 'init': obs['zspec'],
                                'prior': priors.Uniform(mini=obs['zspec']-0.1, maxi=obs['zspec']+0.1)}

    elif run_params['pmodel'] == 'pbeta':
        model_params, fit_order = pfile.params_fsps_phisfh(obs=obs, free_gas_logu=True)
        # or, can also use `pb.PhiSFH` to allow a little bit of freedom around z_spec
        model_params['nzsfh'] = {'N': 9, 'isfree': True, 'init': np.array([0.5,8,0.0,0,0,0,0,0,0]),
                                 'prior': pb.PhiSFHfixZred(zred=obs['zspec'],
                                                           mass_mini=6.0, mass_maxi=12.5,
                                                           z_mini=-1.98, z_maxi=0.19,
                                                           logsfr_ratio_mini=-5.0, logsfr_ratio_maxi=5.0,
                                                           logsfr_ratio_tscale=0.3, nbins_sfh=7,
                                                           const_phi=True)}
        
    # velocity dispersion
    # we ignore any smoothing beyond instrumental; fine for most cases
    # there are, however, edge cases where sigma_smooth could matter, e.g., very massive galaxies or AGN
    model_params.update(TemplateLibrary['spectral_smoothing'])
    model_params['sigma_smooth']['isfree'] = False
    model_params['sigma_smooth']['init'] = 0

    # This removes the continuum from the spectroscopy.
    # Highly recommend using when modeling both photometry & spectroscopy.
    # Do _not_ use polynomial correction when there is no continuum detection;
    # if this is some but not all of the spectrum, may need to mask some regions.
    if rm_cont:
        model_params.update(TemplateLibrary['optimize_speccal'])
        model_params['spec_norm']['isfree'] = False
        model_params['polyorder']['init']   = npoly

    # This is a pixel outlier model. It helps to marginalize over poorly modeled noise,
    # such as residual sky lines or even missing absorption lines
    model_params['f_outlier_spec'] = {"N": 1,
                                      "isfree": True,
                                      "init": 0.01,
                                      "prior": priors.TopHat(mini=1e-5, maxi=0.2)}

    model_params['nsigma_outlier_spec'] = {"N": 1,
                                          "isfree": False,
                                          "init": 50.0}

    model_params['f_outlier_phot'] = {"N": 1,
                                      "isfree": False,
                                      "init": 0.00,
                                      "prior": priors.TopHat(mini=0, maxi=0.5)}

    model_params['nsigma_outlier_phot'] = {"N": 1,
                                          "isfree": False,
                                          "init": 10.0}

    # This is a multiplicative noise inflation term. It inflates the noise in
    # all spectroscopic pixels as necessary to get a good fit.
    model_params['spec_jitter'] = {"N": 1,
                                   "isfree": True,
                                   "init": 1.0,
                                   "prior": priors.TopHat(mini=0.5, maxi=5.0)}

    model_params['nebemlineinspec'] = {'N': 1, 'isfree': False, 'init': False}

    fit_order += ['f_outlier_spec', 'spec_jitter']
        
    if neb_marg:
        print('neb marginalization')
        # if marginalizing, nebemlineinspec needs to be False
        model_params.update(TemplateLibrary['nebular_marginalization'])
        model_params['use_eline_prior'] = {'N': 1, 'is_free': False, 'init': False}
        
        # change these according to the spectrum
        model_params['elines_to_fit']['init'] = ['Ba-beta 4861', 'Ba-alpha 6563', 'Ba-gamma 4341', 'Pa-5 9546', 
                                                 'Pa-delta 1.00494um', 'He I 1.08303um', 'Pa-gamma 1.09381um', 'Pa-beta 1.28181um', 
                                                 '[O III] 4959', '[O III] 5007'
                                                ]

        # eline_sigma is usually a fit parameter alongside sigma_smooth;
        # lets you in principle have different velocity widths for the emission lines vs continuum

    else:
        print('no neb marginalization')
        

    def sigmav_to_total(eline_sigma_velocity=None, zred=None, sigma_v=None, spec_obs_wave=None,
                        eline_rest_wave=None, **extras):
        ### instantiate
        eline_sigma_total = np.zeros_like(eline_rest_wave) + eline_sigma_velocity
        ### find which lines need broadening
        eline_obs_wave = eline_rest_wave * (1+zred)
        idx = (eline_obs_wave > spec_obs_wave.min()) & (eline_obs_wave < spec_obs_wave.max())
        ### interpolate instrumental resolution to observed emission line wavelength
        eline_sigma_velocity_instrumental = np.interp(eline_obs_wave[idx],spec_obs_wave,sigma_v)
        ### combine
        eline_sigma_total[idx] = np.hypot(eline_sigma_velocity, eline_sigma_velocity_instrumental)
        ### return combined
        return eline_sigma_total

    ### Here we ensure the emission line widths are inflated by the instrumental resolution
    # `eline_sigma_velocity` is a SINGLE velocity dispersion for all emission lines
    # `eline_sigma` is the TOTAL line width for all emission lines (instrumental + velocity)
    model_params['eline_sigma_velocity'] = {'N': 1, 'isfree': True, 'init': 100.0,
                                            'prior':priors.TopHat(mini=0,maxi=5000)}

    model_params['eline_sigma'] = {'N': nelines, # number of lines
                                   'isfree': False, 'depends_on':sigmav_to_total,
                                   'init': 600.0, 'units': r'km/s',
                                   'prior': priors.TopHat(mini=100, maxi=1000)}


    fit_order += ['eline_sigma_velocity']

    extra = [k for k in model_params.keys() if k not in fit_order]
    fit_order = fit_order + list(extra)

    tparams = {}
    for i in fit_order:
        tparams[i] = model_params[i]
    for i in list(model_params.keys()):
        if i not in fit_order:
            tparams[i] = model_params[i]
    model_params = tparams
    
    model = PolySpecModel(model_params)

    return model


# --------------
# SPS Object
# --------------

def build_sps(zred, zcontinuous=2, smooth_instrument=False, obs=None, model_agn=True, dell=0, **extras):
    
    if model_agn:
        sps = AGNSSPBasis(zcontinuous=zcontinuous, compute_vega_mags=False)
    else:
        sps = FastStepBasis(zcontinuous=zcontinuous, compute_vega_mags=False)

    if (obs is not None) and (smooth_instrument):
        #from exspect.utils import get_lsf
        print('---- wave-dependent resolution ----')
        wave_obs = obs["wavelength"]
        sigma_v  = obs["sigma_v"]
        speclib  = sps.ssp.libraries[1].decode("utf-8")
        # print('speclib', speclib)
        wave, delta_v = get_lsf(wave_obs, sigma_v, speclib=speclib, zred=zred, **extras)
        sps.ssp.params['smooth_lsf'] = True
        sps.ssp.set_lsf(wave, delta_v)

    if dell != 0:
        sps.ssp.params['tpagb_norm_type'] = 1
        sps.ssp.params['dell'] = 1

    print("sps.ssp.params['dell']", sps.ssp.params['dell'])

    return sps

def get_lsf(wave_obs, sigma_v, speclib, zred, **extras):
    """This method takes an instrimental resolution curve and returns the
    quadrature difference between the instrumental dispersion and the library
    dispersion, in km/s, as a function of restframe wavelength
    :param wave_obs: ndarray
        Observed frame wavelength (AA)
    :param sigma_v: ndarray
        Instrumental spectral resolution in terms of velocity dispersion (km/s)
    :param speclib: string
        The spectral library.  One of 'miles' or 'c3k_a', returned by
        `sps.ssp.libraries[1]`
    """
    lightspeed = 2.998e5  # km/s
    # filter out some places where sdss reports zero dispersion
    good = sigma_v > 0
    wave_obs, sigma_v = wave_obs[good], sigma_v[good]
    wave_rest = wave_obs / (1 + zred)

    # Get the library velocity resolution function at the corresponding
    # *rest-frame* wavelength
    if speclib == "miles":
        miles_fwhm_aa = 2.54
        sigma_v_lib = lightspeed * miles_fwhm_aa / 2.355 / wave_rest
        # Restrict to regions where MILES is used
        good = (wave_rest > 3525.0) & (wave_rest < 7500)

    elif speclib == "c3k_a":
        R_c3k = 3000
        sigma_v_lib = lightspeed / (R_c3k * 2.355)
        # Restrict to regions where C3K is used
        good = (wave_rest > 2750.0) & (wave_rest < 9100.0)

    # elif speclib == "c3k_hr":
    #     data_lib = np.loadtxt('FSPS/model/fsps/SPECTRA/C3K/c3k_hr.lambda',
    #                             dtype=[('wave_lib', '<f8'), ('sigma_v_lib', '<f8')])
    #     sigma_v_lib = data_lib['sigma_v_lib'][np.digitize(wave_rest, data_lib['wave_lib'])-1]
    #     good = (wave_rest > 0)

    else:
        sigma_v_lib = sigma_v
        good = slice(None)
        raise ValueError("speclib of type {} not supported".format(speclib))

    # Get the quadrature difference
    # (Zero and negative values are skipped by FSPS)
    dsv = np.sqrt(np.clip(sigma_v**2 - sigma_v_lib**2, 0, np.inf))

    # return the broadening of the rest-frame library spectra required to match
    # the observed frame instrumental lsf
    return wave_rest[good], dsv[good]


# ------------------
# Noise Model
# ------------------

def build_noise(model_noise, **extras):
    if model_noise:
        jitter = Uncorrelated(parnames = ['spec_jitter'])
        spec_noise = NoiseModel(kernels=[jitter],metric_name='unc',weight_by=['unc'])
        return spec_noise, None
    else:
        return None, None

# ------------------------------------

# build observations
obs = build_obs(**run_params)
# print(obs)

# build sps
sps = build_sps(zred=obs['zspec'], smooth_instrument=True, obs=obs, model_agn=model_agn, **run_params)
obs['rest_wavelength'] = sps.wavelengths
# print(obs)

# calculate rest-frame wavelength range
run_params['waverange'] = (np.max(obs['wavelength'][obs['mask']]) -
    np.min(obs['wavelength'][obs['mask']]))/(1+obs['zspec'])

# build model
model = build_model(obs=obs, model_agn=model_agn, **run_params)

### save instrumental resolution and emission line rest-frame wavelengths in run_params
model.params['sigma_v'] = obs['sigma_v']
model.params['spec_obs_wave'] = obs['wavelength']
model.params['eline_rest_wave'] = info['wave']
print(model)

# build noise
noise = build_noise(**run_params)

print("\nFitting {}".format(obs['id']))
print("------------------\n")

output = prospect.fitting.fit_model(obs, model, sps, noise, #lnprobfn=lnprobfn, 
                                    **run_params)

######
fname = "id_{}_mcmc_{}.h5".format(obs['id'], run_params['run'])
hfile = os.path.join(run_params['outdir'], fname)

prospect.io.write_results.write_hdf5(hfile, run_params, model, obs,
                  output["sampling"][0], output["optimization"][0],
                  tsample=output["sampling"][1],
                  toptimize=output["optimization"][1],
                  sps=sps)

try:
    hfile.close()
except(AttributeError):
    pass

print('Finished. Saved to {}'.format(hfile))

print((output["sampling"][0])['samples'][-1])

import time, sys, os
import numpy as np
import numpy.ma as ma
from astropy.table import Table
from astropy import units as u

import sedpy
import prospect
from prospect.fitting import fit_model
from prospect.sources import FastStepBasis
from prospect.models.sedmodel import PolySpecModel

import dynesty

import utils as ut_cwd
import mu_from_maps as lens_mu

import emulator as Emu

pdir = ut_cwd.data_dir(data='pirate')
multiemul_file = os.path.join(pdir, "parrot_v4_obsphot_512n_5l_24s_00z24.npy")

# - Parser with default arguments -
parser = prospect.prospect_args.get_parser()
parser.add_argument('--catalog', type=str, default="UNCOVER_v5.0.1_LW_SUPER_CATALOG.fits")
parser.add_argument('--idx0', type=int, default=0,
                    help="Range of galaxies to fit, from idx0 to idx1-1; zero-index row number of the catalog.")
parser.add_argument('--idx1', type=int, default=1,
                    help="Range of galaxies to fit, from idx0 to idx1-1; zero-index row number of the catalog.")
parser.add_argument('--outdir', type=str, default='chains_parrot/', help="Output folder name.")
parser.add_argument('--dyn', type=int, default=0, 
                    help="If 0, std run; if 1, quick dynesty run; if 2, debug, max=1100")
args = parser.parse_args()
catalog_file = args.catalog

run_params = vars(args)
run_params.update({
'free_gas_logu': False, # parrot is trained with fixed gas_logu
'verbose': True,
'dyn': args.dyn,
'outdir': args.outdir,
'nofork': True,
# dynesty params
'dynesty': True,
'nested_maxcall': None,
'nested_maxcall_init': None,
'nested_maxiter': None,
'nested_maxbatch': None, # maximum number of dynamic patches
'nested_bound': 'multi', # bounding method
'nested_sample': 'rwalk', # sampling method
#'nested_walks': 50, # MC walks
'nested_nlive_batch': 400, # size of live point "batches"
'nested_nlive_init': 1600, # number of initial live points
'nested_weight_kwargs': {'pfrac': 1.0}, # weight posterior over evidence by 100%
'nested_dlogz_init': 0.01,
'nested_target_n_effective': 20000,
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

if run_params['dyn'] == 1:
    # quick dynesty fits, for testing purpose
    run_params.update({
    'nested_nlive_init': 800,
    # 'nested_dlogz_init': 0.1,
    'nested_target_n_effective': 10000,
    'nested_maxcall': 500000,
    'nested_maxcall_init': 500000,
    })
if run_params['dyn'] == 2:
    # debug
    run_params.update({
    'nested_maxcall': 1100,
    'nested_maxcall_init': 1100,
    })
    
run_params["param_file"] = __file__

if not run_params['outdir'].endswith('/'):
    run_params['outdir'] = run_params['outdir'] + '/'
if not os.path.exists(run_params['outdir']):
    os.makedirs(run_params['outdir'])
    print("new directory created:", run_params['outdir'])
print(run_params)

mdir = ut_cwd.data_dir('gen1') + 'phot_catalog/'
cat = Table.read(mdir+catalog_file)

if 'f_alma' in cat.colnames:
    alma = True
else:
    alma = False
if 'f_f480m' in cat.colnames:
    mb = True
else:
    mb = False
filter_dict = ut_cwd.filter_dictionary(mb=mb, alma=alma)
filts = list(filter_dict.keys())
filternames = list(filter_dict.values())
# print(len(filts))

def load_obs(idx=None, err_floor=0.05, **extras):
    '''
    idx: obj idx in the catalog
    '''

    from prospect.utils.obsutils import fix_obs

    flux = ut_cwd.get_fnu_maggies(idx, cat, filts)
    unc = ut_cwd.get_enu_maggies(idx, cat, filts)

    obs = {}
    obs["filters"] = sedpy.observate.load_filters(filternames)
    obs["wave_effective"] = np.array([f.wave_effective for f in obs["filters"]])
    obs["maggies"] = flux
    obs["maggies_unc"] = unc
    # define photometric mask
    # mask out fluxes with negative errors, and high-confidence negative flux
    phot_mask = (unc > 0) & (np.isfinite(flux))
    _mask = np.ones_like(unc, dtype=bool)
    for k in range(len(flux)):
        if unc[k] > 0:
            if flux[k] < 0 and flux[k] + 5*unc[k] < 0:
                _mask[k] = False
    phot_mask &= _mask
    obs['phot_mask'] = phot_mask
    # impose minimum error floor
    obs['maggies_unc'] = np.clip(obs['maggies_unc'], a_min=obs['maggies']*err_floor, a_max=None)

    obs["wavelength"] = None
    obs["spectrum"] = None
    obs['unc'] = None
    obs['mask'] = None

    # other useful info
    obs['objid'] = cat['id'][idx]
    obs['catalog'] = catalog_file

    obs = fix_obs(obs)

    return obs

def build_model(obs=None, emulfp=multiemul_file, **extras):

    import params_prosp_parrot as pfile
    model_params, fit_order = pfile.params_parrot_phisfh(obs=obs)

    return Emu.EmulatorBeta(model_params, fp=emulfp, obs=obs, param_order=fit_order)


def load_sps(**extras):

    return None


# ---------------- lensing
from copy import deepcopy
from prospect.likelihood import lnlike_spec, lnlike_phot, chi_spec, chi_phot, write_log

def lnprobfn(theta, model=None, obs=None, sps=None, noise=(None, None),
             residuals=False, nested=False, negative=False, verbose=False):

    _obs = deepcopy(obs)
    mu = lens_mu.scale_mu(zred=theta[0], px=obs['x_lensmap'], py=obs['y_lensmap'], verbose=verbose)

    if residuals:
        lnnull = np.zeros(_obs["ndof"]) - 1e18  # np.infty
        #lnnull = -np.infty
    else:
        lnnull = -np.infty

    # --- Calculate prior probability and exit if not within prior ---
    lnp_prior = model.prior_product(theta, nested=nested)
    if not np.isfinite(lnp_prior):
        return lnnull

    #  --- Update Noise Model ---
    spec_noise, phot_noise = noise
    vectors, sigma_spec = {}, None
    model.set_parameters(theta)
    if spec_noise is not None:
        spec_noise.update(**model.params)
        vectors.update({"unc": _obs.get('unc', None)})
        sigma_spec = spec_noise.construct_covariance(**vectors)
    if phot_noise is not None:
        phot_noise.update(**model.params)
        vectors.update({'phot_unc': _obs.get('maggies_unc', None),
                        'phot': _obs.get('maggies', None)})

    # --- Generate mean model ---
    try:
        t1 = time.time()
        spec, phot, x = model.predict(theta, _obs, sps=sps, sigma_spec=sigma_spec)
        spec *= mu
        phot *= mu
        d1 = time.time() - t1
    except(ValueError):
        return lnnull
    except:
        print("There was an error during the likelihood call at parameters {}".format(theta))
        raise

    # --- Optionally return chi vectors for least-squares ---
    # note this does not include priors!
    if residuals:
        chispec = chi_spec(spec, _obs)
        chiphot = chi_phot(phot, _obs)
        return np.concatenate([chispec, chiphot])

    #  --- Mixture Model ---
    f_outlier_spec = model.params.get('f_outlier_spec', 0.0)
    if (f_outlier_spec != 0.0):
        sigma_outlier_spec = model.params.get('nsigma_outlier_spec', 10)
        vectors.update({'nsigma_outlier_spec': sigma_outlier_spec})
    f_outlier_phot = model.params.get('f_outlier_phot', 0.0)
    if (f_outlier_phot != 0.0):
        sigma_outlier_phot = model.params.get('nsigma_outlier_phot', 10)
        vectors.update({'nsigma_outlier_phot': sigma_outlier_phot})

    # --- Emission Lines ---

    # --- Calculate likelihoods ---
    t1 = time.time()
    lnp_spec = lnlike_spec(spec, obs=_obs,
                           f_outlier_spec=f_outlier_spec,
                           spec_noise=spec_noise,
                           **vectors)
    lnp_phot = lnlike_phot(phot, obs=_obs,
                           f_outlier_phot=f_outlier_phot,
                           phot_noise=phot_noise, **vectors)
    lnp_eline = getattr(model, '_ln_eline_penalty', 0.0)

    d2 = time.time() - t1
    if verbose:
        write_log(theta, lnp_prior, lnp_spec, lnp_phot, d1, d2)

    lnp = lnp_prior + lnp_phot + lnp_spec + lnp_eline
    if negative:
        lnp *= -1

    return lnp

# ---------------- fit !
badobs_ids_list = []
for ifit in np.arange(args.idx0, args.idx1, 1):

    # run on the full catalog
    objid = cat['id'][ifit]
    print("\nFitting {}".format(objid))
    print("------------------\n")
    run_params['idx'] = ifit # choose a galaxy
    _can_fit = False
    try:
        obs = load_obs(**run_params)
        _can_fit = True
    except(AssertionError):
        # all NaNs, etc.
        _can_fit = False
        badobs_ids_list.append(objid)
        print('no phot')

    if _can_fit:
        obs['x_pixel'] = 0; obs['y_pixel'] = 0
        obs['ra'] = cat[ifit]['ra']; obs['dec'] = cat[ifit]['dec']
        ra = obs['ra']*u.deg
        dec = obs['dec']*u.deg
        obs['x_lensmap'], obs['y_lensmap'] = lens_mu.xy_in_kappa_and_gamma(ra, dec)

        model = build_model(obs=obs, **run_params)
        sps = load_sps(**run_params)
        print(obs)
        print(model)
        # print(model.theta_index)

        ts = time.strftime("%y%b%d-%H.%M", time.localtime())
        hfile = os.path.join(run_params['outdir'], "id_{0}_mcmc_phisfh.h5".format(objid))

        if obs['x_lensmap'] < 0 or obs['y_lensmap'] < 0:
            # outside lens model FoV
            output = fit_model(obs, model, sps, **run_params)
        else:
            output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
        print('done in {0}s'.format(output["sampling"][1]))

        prospect.io.write_results.write_hdf5(hfile, run_params, model, obs,
                          output["sampling"][0], output["optimization"][0],
                          tsample=output["sampling"][1],
                          toptimize=output["optimization"][1],
                          sps=sps, write_model_params=False)
        try:
            hfile.close()
        except(AttributeError):
            pass

        print('Finished. Saved to {}'.format(hfile))
        
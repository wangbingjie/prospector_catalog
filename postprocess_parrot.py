import os, sys, random
import numpy as np

from astropy.table import Table
from astropy.cosmology import WMAP9 as cosmo
from astropy.io import fits
from astropy import units as u

import prospect.io.read_results as reader
from prospect.models.transforms import logsfr_ratios_to_masses
from dynesty.utils import resample_equal

import sedpy
from sedpy import observate

import utils as ut_cwd
import utils_unit as ut_u
import mu_from_maps as lens_mu

percents = [0.1,2.3,15.9,50,84.1,97.7,99.9]
nsamp = 1000

def theta_dict_from_run(fit, rtn_dict=True):
    '''
    `theta` _has_ to match the model.free_params
    `scalar` contains those from `theta` that do not require transformation
    '''

    if fit == 'fid':
        theta = ['zred', 'total_mass', 'logzsol',
                 'logsfr_ratios_1', 'logsfr_ratios_2', 'logsfr_ratios_3',
                 'logsfr_ratios_4', 'logsfr_ratios_5', 'logsfr_ratios_6',
                 'dust2', 'dust_index', 'dust1_fraction',
                 'log_fagn', 'log_agn_tau', 'gas_logz',
                 'duste_qpah', 'duste_umin', 'log_duste_gamma']

        scalar = ['zred', 'total_mass', 'logzsol',
                  'dust2', 'dust_index', 'dust1_fraction',
                  'log_fagn', 'log_agn_tau', 'gas_logz',
                  'duste_qpah', 'duste_umin', 'log_duste_gamma']

    elif fit == 'fsps_sfhgas':
        theta = ['zred', 'total_mass', 'logzsol',
                 'logsfr_ratios_1', 'logsfr_ratios_2', 'logsfr_ratios_3',
                 'logsfr_ratios_4', 'logsfr_ratios_5', 'logsfr_ratios_6',
                 'dust2', 'dust_index', 'dust1_fraction',
                 'log_fagn', 'log_agn_tau', 'gas_logz',
                 'duste_qpah', 'duste_umin', 'log_duste_gamma', 'gas_logu']

        scalar = ['zred', 'total_mass', 'logzsol',
                  'dust2', 'dust_index', 'dust1_fraction',
                  'log_fagn', 'log_agn_tau', 'gas_logz',
                  'duste_qpah', 'duste_umin', 'log_duste_gamma', 'gas_logu']

    else:
        raise Exception("wrong fit specified!")

    if rtn_dict:
        theta_dict = dict(zip(theta, np.arange(0,len(theta))))
        return theta_dict
    else:
        return scalar


def getPercentiles(chain, quantity='zred', theta_index=None, percents=percents):
    ''' get the percentiles for a scalar output that does not need transform functions
        (e.g., mass, dust, etc).
    '''
    p = np.percentile(chain[:,theta_index[quantity]], q=percents, axis=0)
    return p.T

def z_to_agebins(zred=None, agebins=None, nbins_sfh=7, amin=7.1295, **extras):
    tuniv = cosmo.age(zred).value*1e9
    tbinmax = (tuniv*0.9)
    if (zred <= 3.):
        agelims = [0.0,7.47712] + np.linspace(8.0,np.log10(tbinmax),nbins_sfh-2).tolist() + [np.log10(tuniv)]
    else:
        agelims = np.linspace(amin,np.log10(tbinmax),nbins_sfh).tolist() + [np.log10(tuniv)]
        agelims[0] = 0

    agebins = np.array([agelims[:-1], agelims[1:]])
    return agebins.T

def stepInterp(ab, val, ts):
    '''ab: agebins vector
    val: the original value (sfr, etc) that we want to interpolate
    ts: new values we want to interpolate to '''
    newval = np.zeros_like(ts) + np.nan
    for i in range(0,len(ab)):
        newval[(ts >= ab[i,0]) & (ts < ab[i,1])] = val[i]
    newval[-1] = 0
    return newval

def get_mwa(agebins, sfr):
    ages = 10**agebins
    dt = np.abs(ages[:,1] - ages[:,0])
    return(np.sum(np.mean(ages, axis=1) * sfr * dt) / np.sum(sfr * dt) / 1e9) # in Gyr

def getSFH(chain, nagebins=7, sfrtimes=[10,30,100], tbins=100, theta_index=None,
           rtn_chains=False, percents=percents):
    ''' get the percentiles of the SFH for each pixel.

    Returns
    __________

    age_interp : list of ages -- lookback time in Gyr
    sfh : 3 x npix x len(age_interp) array that gives the 16/50/84% interval of the SFH at each lookback time
    mwa : 3 x npix array-- 16/50/84th percentile mass-weighted age
    sfr : 3 x npix x n_sfrtimes -- 16/50/84% SFR over each timescale
    '''

    # run the transforms to get the SFH at each step in the chain for every pixel:
    # (pixel, agebin, chain step)
    nsamples = chain.shape[0]
    allmasses = np.zeros((nsamples, nagebins)) # mass formed as a func of time
    allsfhs = np.zeros((nsamples, nagebins))
    allagebins = np.zeros((nsamples, nagebins, 2))
    allMWA = np.zeros((nsamples))

    _idx_logsfr_ratios = []
    for t in list(theta_index.keys()):
        if 'logsfr_ratios' in t:
            _idx_logsfr_ratios.append(theta_index[t])

    # make the SFR for each draw in each pixel
    for iteration in range(nsamples):
        # get agebins and time spacing
        allagebins[iteration,:,:] = z_to_agebins(zred=chain[iteration, theta_index['zred']])
        dt = 10**allagebins[iteration,:, 1] - 10**allagebins[iteration,:, 0]

        # get mass per bin
        masses = logsfr_ratios_to_masses(logsfr_ratios=chain[iteration, _idx_logsfr_ratios],
            agebins=allagebins[iteration,:,:], logmass=chain[iteration, theta_index['total_mass']])

        allmasses[iteration, :] = masses

        # convert to sfr
        allsfhs[iteration, :] = masses / dt

        # go ahead and get the mass-weighted age too
        allMWA[iteration] = get_mwa(allagebins[iteration,:,:], allsfhs[iteration,:])

    # interpolate everything onto the same time grid
    # define the grid as going from lookback time = 0 to the oldest time in all the samples
    # with 1000 log-spaced samples in between (might want to update this later!)
    allagebins_ago = 10**allagebins/1e9
    age_interp = np.logspace(1, np.log10(np.max(allagebins_ago*1e9)), tbins)/1e9 # lookback time in Gyr
    age_interp_linear = np.linspace(0, 100, 51) / 1e3 # for <sfr> only; Myr -> Gyr
    allmasses_interp = np.zeros((nsamples, len(age_interp)))
    allsfhs_interp = np.zeros((nsamples, len(age_interp)))
    allsfhs_interp_linear = np.zeros((nsamples, len(age_interp_linear)))
    allsfrs = np.zeros((nsamples, len(sfrtimes)))
    for iteration in range(nsamples):
        allmasses_interp[iteration, :] = stepInterp(allagebins_ago[iteration,:], allmasses[iteration,:], age_interp)
        allsfhs_interp[iteration, :] = stepInterp(allagebins_ago[iteration,:], allsfhs[iteration,:], age_interp)

        allsfhs_interp_linear[iteration, :] = stepInterp(allagebins_ago[iteration,:], allsfhs[iteration,:], age_interp_linear)
        # because we use linear scale for getting the average sfrs, we start from 0 yr,
        # and fill in this bin using the most recent sfr in the log-spaced bin
        nan_idx = np.isnan(allsfhs_interp_linear[iteration, :])
        foo = allsfhs_interp_linear[iteration, ~nan_idx]
        allsfhs_interp_linear[iteration, nan_idx] = foo[0]

        # also get SFR averaged over all the timescales we want
        for i, time in enumerate(sfrtimes):
            allsfrs[iteration, i] = np.mean(allsfhs_interp_linear[iteration, (age_interp_linear<=time*1e-3)])

    if rtn_chains:
        return (np.log10(np.max(allagebins_ago*1e9)), allmasses_interp, allsfhs_interp, allMWA, allsfrs)
    else:
        # sfr and MWA percentiles
        sfh = np.percentile(allsfhs_interp, percents, axis=0)
        mwa = np.percentile(allMWA, percents)
        sfr = np.percentile(allsfrs, percents, axis=0)
        return (np.log10(np.max(allagebins_ago*1e9)), np.percentile(allmasses_interp, percents, axis=0).T,
                sfh.T, mwa, sfr.T)

def get_all_outputs(res=None, rtn_chains=False, run_params=None):

    ''' Get all the outputs '''

    # load the output file and get the chain
    chain = res['chain']
    theta_index = res['theta_index']
    keys = res['scalar']

    # get the basic quantities
    percentiles = {}
    for key in keys:
        percentiles[key] = getPercentiles(chain, key, theta_index)

    if rtn_chains:
        return getSFH(chain, theta_index=theta_index, rtn_chains=rtn_chains)[1:]
    else:
        # get sfh params and turn into maps
        age_interp, mt, sfh, mwa, sfr = getSFH(chain, theta_index=theta_index) # sfh and sfr already transposed

        # each of these keys is a (xpix x ypix x nparam x 16/50/84%) map
        # percentiles['age_interp'] = age_interp
        percentiles['mt'] = mt
        sfh[np.isnan(sfh)] = 0
        percentiles['sfh'] = sfh
        percentiles['mwa'] = mwa
        percentiles['sfr'] = sfr

        return percentiles

def get_all_outputs_and_chains(res=None, run_params=None, percents=percents, nsamp=nsamp):

    ''' Get all the outputs '''

    # load the output file and get the unweighted chain
    chain = res['chain']
    theta_index = res['theta_index']
    keys = res['scalar']

    # get the basic quantities
    percentiles = {}
    for key in keys:
        percentiles[key] = getPercentiles(chain, key, theta_index)

    age_interp, allmasses_interp, allsfhs_interp, allMWA, allsfrs = getSFH(chain, theta_index=theta_index, rtn_chains=True)
    # sfr and MWA percentiles
    # rtn_chains is defaulted to False: so need to transpose sfh and sfr
    allsfhs_interp[np.isnan(allsfhs_interp)] = 0
    mt = np.percentile(allmasses_interp, percents, axis=0)
    sfh = np.percentile(allsfhs_interp, percents, axis=0)
    mwa = np.percentile(allMWA, percents)
    sfr = np.percentile(allsfrs, percents, axis=0)

    # each of these keys is a (xpix x ypix x nparam x 16/50/84%) map
    # percentiles['age_interp'] = age_interp
    percentiles['mformed_t'] = mt.T
    percentiles['sfh'] = sfh.T
    percentiles['mwa'] = mwa
    percentiles['sfr'] = sfr.T

    # saved chains are subsampled, so that we can plot stellar mass on the corner plot
    sub_idx = random.sample(range(res['chain'].shape[0]), nsamp)
    sub_idx.append(-1)
    chain = res['chain'][sub_idx]
    chains = {#'age_interp':age_interp,
              'agebins_max':age_interp,
              'mformed_t': allmasses_interp,
              'sfh':allsfhs_interp, 'mwa':allMWA[sub_idx], 'sfr':allsfrs[sub_idx,:]}

    # all untransformed thetas
    for k in list(theta_index.keys()):
        chains[k] = chain[:,theta_index[k]]

    return percentiles, chains, sub_idx

def run_all(h5_fname=None,
            objid=None, fit=None, prior=None, mod_fsps=None, sps=None,
            input_folder='chains_parrot', output_folder='chains_parrot',
            percents=percents, **extra):

    _in_dir = input_folder[:]
    _out_dir = output_folder[:]
    if h5_fname is None:
        _h5_fname = 'id_{}_mcmc_{}.h5'.format(objid, prior)
        fname = os.path.join(_in_dir, _h5_fname)
        #print(fname)
    else:
        _h5_fname = h5_fname[:]
        fname = os.path.join(_in_dir, h5_fname)
    print('post-processing:', fname)

    res, obs, _ = reader.results_from(fname, dangerous=False)
    chain_ml = res['chain'][-1]
    res['theta_index'] = theta_dict_from_run(fit)
    res['scalar'] = theta_dict_from_run(fit, rtn_dict=False)
    unweighted_chain = resample_equal(res['chain'], res['weights']) # unweighted_chain
    res['chain'] = np.vstack([unweighted_chain, chain_ml]) # last chain is always ml

    percentiles, chains, sub_idx = get_all_outputs_and_chains(res)

    # total mass formed -> stellar mass
    stellarmass = []
    ssfr = []

    # also get uvj colors
    uvj_names = ['bessell_U', 'bessell_V', 'twomass_J', 'synthetic_u', 'synthetic_g', 'synthetic_i']
    filts_uvj_sedpy = observate.load_filters(uvj_names)
    weff_uvj = np.array([f.wave_effective for f in filts_uvj_sedpy])

    gz_names = ['sdss_g0', 'sdss_z0']
    filts_gz_sedpy = observate.load_filters(gz_names)
    weff_gz = np.array([f.wave_effective for f in filts_gz_sedpy])
    
    # NUV - r, r - J
    nuv_names = ['galex_NUV', 'sdss_r0', 'twomass_J' ]
    filts_nuv_sedpy = observate.load_filters(nuv_names)
    weff_nuv = np.array([f.wave_effective for f in filts_nuv_sedpy])

    uvj_abmag = []
    uvj_color = []    
    gz_abmag = []
    gz_color = []
    nuv_abmag = []
    nuv_color = []

    mu_all = []
    modmags_all = []
    modspecs_all = []
    
    # ML photometry, and ML spectrum from fsps
    # call twice to make sure fsps catches the correct grid
    modspec_map, modmags_map, sm_map = mod_fsps.predict(chain_ml, sps=sps, obs=obs)
    modspec_map, modmags_map, sm_map = mod_fsps.predict(chain_ml, sps=sps, obs=obs)
    
    # modspec_map is in observed frame
    modspec_flam = ut_u.lambda_fnu_to_flambda_aa_maggies(wavelength=sps.wavelengths*(1+chain_ml[res['theta_index']['zred']]), fnu=modspec_map)
    restwave, modspec_flam = ut_u.deredshift_spectrum_prosp_units(wavelength=sps.wavelengths*(1+chain_ml[res['theta_index']['zred']]), 
                                                                  flux=modspec_flam, redshift=chain_ml[res['theta_index']['zred']],
                                                                  is_fnu=False)
    _uvj_abmag = observate.getSED(sps.wavelengths, modspec_flam, filterlist=filts_uvj_sedpy, linear_flux=False)
    uvj_abmag_map = _uvj_abmag
    uvj_color_map = [_uvj_abmag[0]-_uvj_abmag[1], _uvj_abmag[1]-_uvj_abmag[2],
                     _uvj_abmag[4]-_uvj_abmag[5], _uvj_abmag[3]-_uvj_abmag[4]]

    # g-z
    _gz_abmag = observate.getSED(sps.wavelengths, modspec_flam, filterlist=filts_gz_sedpy, linear_flux=False)
    gz_abmag_map = _gz_abmag
    gz_color_map = _gz_abmag[0]-_gz_abmag[1]
    
    # NUV
    _nuv_abmag = observate.getSED(sps.wavelengths, modspec_flam, filterlist=filts_nuv_sedpy, linear_flux=False)
    nuv_abmag_map = _nuv_abmag
    nuv_color_map = [_nuv_abmag[0]-_nuv_abmag[1], _nuv_abmag[1]-_nuv_abmag[2]]        

    modspec, modmags, sm = mod_fsps.predict(res['chain'][0], sps=sps, obs=obs)
    for i, _subidx in enumerate(sub_idx):
        modspec, modmags, sm = mod_fsps.predict(res['chain'][int(_subidx)], sps=sps, obs=obs)

        # magnification
        _mu = lens_mu.scale_mu(zred=res['chain'][int(_subidx)][res['theta_index']['zred']],
                               px=obs['x_lensmap'], py=obs['y_lensmap'], verbose=False)
        mu_all.append(_mu)
        modmags_all.append(modmags*_mu)
        modspecs_all.append(modspec*_mu)

        _mass = res['chain'][int(_subidx)][res['theta_index']['total_mass']]
        stellarmass.append(np.log10(10**_mass*sm))
        ssfr.append(chains['sfr'][i]/ (10**_mass*sm) ) # sfr chains are already sub-sampled

        # UVJ & ugj
        _zred = res['chain'][int(_subidx)][res['theta_index']['zred']] * 1
        modspec_flam = ut_u.lambda_fnu_to_flambda_aa_maggies(wavelength=sps.wavelengths*(1+_zred), fnu=modspec)
        restwave, modspec_flam = ut_u.deredshift_spectrum_prosp_units(wavelength=sps.wavelengths*(1+_zred), 
                                                                      flux=modspec_flam, redshift=_zred,
                                                                      is_fnu=False)
        _uvj_abmag = observate.getSED(sps.wavelengths, modspec_flam, filterlist=filts_uvj_sedpy, linear_flux=False)
        uvj_abmag.append(_uvj_abmag)
        # U-V, V-J, g-i, u-g
        # UVJugi
        uvj_color.append([_uvj_abmag[0]-_uvj_abmag[1], _uvj_abmag[1]-_uvj_abmag[2],
                          _uvj_abmag[4]-_uvj_abmag[5], _uvj_abmag[3]-_uvj_abmag[4]])

        # g-z
        _gz_abmag = observate.getSED(sps.wavelengths, modspec_flam, filterlist=filts_gz_sedpy, linear_flux=False)
        gz_abmag.append(_gz_abmag)
        gz_color.append(_gz_abmag[0]-_gz_abmag[1])
        
        # NUV
        _nuv_abmag = observate.getSED(sps.wavelengths, modspec_flam, filterlist=filts_nuv_sedpy, linear_flux=False)
        nuv_abmag.append(_nuv_abmag)
        nuv_color.append([_nuv_abmag[0]-_nuv_abmag[1], _nuv_abmag[1]-_nuv_abmag[2]])

    stellarmass = np.array(stellarmass)
    ssfr = np.array(ssfr)

    uvj_abmag = np.array(uvj_abmag)
    uvj_abmag_map = np.array(uvj_abmag_map)
    uvj_color = np.array(uvj_color)
    uvj_color_map = np.array(uvj_color_map)
    
    gz_abmag = np.array(gz_abmag)
    gz_abmag_map = np.array(gz_abmag_map)
    gz_color = np.array(gz_color)
    gz_color_map = np.array(gz_color_map)
    
    nuv_abmag = np.array(nuv_abmag)
    nuv_abmag_map = np.array(nuv_abmag_map)
    nuv_color = np.array(nuv_color)
    nuv_color_map = np.array(nuv_color_map)

    mu_all = np.array(mu_all)
    modmags_all = np.array(modmags_all)
    modspecs_all = np.array(modspecs_all)

    percentiles['stellar_mass'] = np.percentile(stellarmass, percents)
    percentiles['ssfr'] = np.percentile(ssfr, percents, axis=0).T

    percentiles['rest_UVJugi'] = np.percentile(uvj_abmag, percents, axis=0).T
    percentiles['rest_UVJugi_map'] = uvj_abmag_map
    percentiles['rest_UVJugi_colors'] = np.percentile(uvj_color, percents, axis=0).T
    percentiles['rest_UVJugi_colors_map'] = uvj_color_map

    percentiles['rest_gz'] = np.percentile(gz_abmag, percents, axis=0).T
    percentiles['rest_gz_map'] = gz_abmag_map
    percentiles['rest_gz_colors'] = np.percentile(gz_color, percents, axis=0).T
    percentiles['rest_gz_colors_map'] = gz_color_map
    
    percentiles['rest_NUVrJ'] = np.percentile(nuv_abmag, percents, axis=0).T
    percentiles['rest_NUVrJ_map'] = nuv_abmag_map
    percentiles['rest_NUVrJ_colors'] = np.percentile(nuv_color, percents, axis=0).T
    percentiles['rest_NUVrJ_colors_map'] = nuv_color_map
        
    # mu
    _mu = []
    for _z in percentiles['zred']:
        _mu.append(lens_mu.scale_mu(zred=_z, px=obs['x_lensmap'], py=obs['y_lensmap'], verbose=False))
    percentiles['mu'] = np.array(_mu)

    perc_fname = _h5_fname.replace('mcmc', 'perc')
    perc_fname = perc_fname.replace('.h5', '.npz')
    sname = os.path.join(_out_dir, perc_fname)
    np.savez(sname, percentiles=percentiles, chain_ml=chain_ml, theta_lbs=res['theta_index'])
    print('saved percentiles to', sname)

    unw_fname = _h5_fname.replace('mcmc', 'unw')
    unw_fname = unw_fname.replace('.h5', '.npz')
    chains['stellar_mass'] = stellarmass
    chains['ssfr'] = ssfr
    sname = os.path.join(_out_dir, unw_fname)
    np.savez(sname, chains=chains, chain_ml=chain_ml, sub_idx=sub_idx)
    print('saved chains to', sname)

    mu_map = lens_mu.scale_mu(zred=chain_ml[res['theta_index']['zred']], px=obs['x_lensmap'], py=obs['y_lensmap'], verbose=False)
    
    spec_fname = _h5_fname.replace('mcmc', 'spec')
    spec_fname = spec_fname.replace('.h5', '.npz')
    sname = os.path.join(_out_dir, spec_fname)
    np.savez(sname,
             modspec_map=modspec_map, modmags_map=modmags_map, mu_map=mu_map,
             modmags_perc=np.percentile(modmags_all, percents, axis=0).T,
             modspecs_perc=np.percentile(modspecs_all, percents, axis=0).T
            )
             # modmags_all & modspecs_all is magnified!
    print('saved model spec to', sname)

'''creates *_spec_*.npz
'''
import os, sys, time
import numpy as np
import numpy.ma as ma
from astropy.table import Table
import prospect.io.read_results as reader

import utils as ut_cwd
import emulator as Emu

pdir = ut_cwd.data_dir(data='pirate')
multiemul_file = os.path.join(pdir, "parrot_v4_obsphot_512n_5l_24s_00z24.npy")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--prior', type=str, default='phisfh', help='phisfh, phisfhzfixed')
parser.add_argument('--catalog', type=str, default="UNCOVER_v5.0.1_LW_SUPER_CATALOG.fits")
parser.add_argument('--indir', type=str, default='chains_parrot', help='input folder storing inidividual results')
parser.add_argument('--outdir', type=str, default='results', help='output folder storing unweighted chains and quantiles')
args = parser.parse_args()
print(args)

which_prior = args.prior
catalog_file = args.catalog

foo = args.indir[:]
if foo.endswith('/'):
    foo = foo[:-1]
sname = os.path.join(args.outdir, 'spec_{}_{}'.format(args.prior, foo)+'.npz')
print('will be saved to', sname)

perc_dir = args.indir
chain_dir = args.indir

mdir = ut_cwd.data_dir('gen1') + 'phot_catalog/'
cat = Table.read(mdir+catalog_file)
if 'f_alma' in cat.colnames:
    alma = True
else:
    alma = False
filter_dict = ut_cwd.filter_dictionary(alma=alma)
filts = list(filter_dict.keys())
filternames = list(filter_dict.values())


objid_list = []
list_chi2_fsps = []

list_spec_map = []
list_modmag_map = []
list_mu_map = []
list_mu_med = []

list_obsmag = []
list_obsmag_unc = []

list_nbands = []

print(perc_dir)
all_files = os.listdir(perc_dir)
cnt = 0

def chi2(modmags, obsmags, obsunc):
    _obsunc = np.clip(obsunc, a_min=obsmags*0.05, a_max=None)
    return (((modmags-obsmags)/_obsunc)**2).sum()


def build_model(obs=None, emulfp=multiemul_file, **extras):

    import params_prosp_parrot as pfile
    model_params, fit_order = pfile.params_parrot_phisfh(obs=obs)

    return Emu.EmulatorBeta(model_params, fp=emulfp, obs=obs, param_order=fit_order)

h5_foo = ut_cwd.finished_id(indir=chain_dir, prior=which_prior, dtype='mcmc', rt_fname=True, verbose=True)
h5_foo = os.path.join(chain_dir, h5_foo[0])
print('loaded h5 for mod_fsps():', h5_foo)
res, obs, _ = reader.results_from(h5_foo, dangerous=False)
model = build_model(obs=obs)

for this_file in all_files:
    if this_file.endswith('_spec_{}.npz'.format(which_prior)):
        mid = int(this_file.split('_')[1])
        dat = np.load(os.path.join(perc_dir, this_file), allow_pickle=True)
        list_spec_map.append(dat['modspec_map'])
        list_modmag_map.append(dat['modmags_map'])
        list_mu_map.append(dat['mu_map'])
        list_mu_med.append(dat['mu_med'])

        _idx = np.where(cat['id']==mid)[0][0]

        obs_fnu = ut_cwd.get_fnu_maggies(idx=_idx, catalog=cat, filts=filts)
        obs_enu = ut_cwd.get_enu_maggies(idx=_idx, catalog=cat, filts=filts)
        list_obsmag.append(obs_fnu)
        list_obsmag_unc.append(obs_enu)

        # chi2
        phot_mask = (obs_enu > 0) & (np.isfinite(obs_fnu))
        _mask = np.ones_like(obs_fnu, dtype=bool)
        for k in range(len(obs_fnu)):
            if obs_enu[k] > 0:
                if obs_fnu[k] < 0 and obs_fnu[k] + 5*obs_enu[k] < 0:
                    _mask[k] = False
        phot_mask &= _mask
        mask = phot_mask
    
        obsmags = obs_fnu[mask]
        obsunc = obs_enu[mask]
        list_nbands.append(len(obsmags))
        fsps_mags = dat['modmags_map'][mask]*dat['mu_map']
        
        chi2_fsps = chi2(fsps_mags, obsmags, obsunc)
        list_chi2_fsps.append(chi2_fsps)

        objid_list.append(mid)
        cnt += 1

        if cnt % 100 == 0:
            print(cnt)
        
np.savez(sname, objid=objid_list, obsmag=list_obsmag, obsmag_unc=list_obsmag_unc,
         modspec_map=list_spec_map, modmag_map=list_modmag_map, mu_map=list_mu_map,
         chi2_fsps=list_chi2_fsps,
         nbands=list_nbands,
        )
print('length:', len(objid_list))
print('saved to', sname)

'''creates *_perc_*.npz
ALL prospector model paramters are saved
'''
import os, sys, time
import numpy as np
from astropy.table import Table
import prospect.io.read_results as reader

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--prior', type=str, default='phisfh')
parser.add_argument('--catalog', type=str, default="UNCOVER_v5.0.1_LW_SUPER_CATALOG.fits")
parser.add_argument('--indir', type=str, default='chains_parrot', help='input folder storing chains')
parser.add_argument('--outdir', type=str, default='results', help='output folder storing unweighted chains and quantiles')
args = parser.parse_args()
print(args)

which_prior = args.prior
catalog_file = args.catalog

foo = args.indir
if foo.endswith('/'):
    foo = foo[:-1]
sname = os.path.join(args.outdir, 'quant_{}_{}'.format(args.prior, foo)+'.npz')
print('will be saved to', sname)

perc_dir = args.indir

mdir = 'phot_catalog/'
cat = Table.read(mdir+catalog_file)

objid_list = []

zred_spec = []

zred_ml = []

zred = []
total_mass = []
stellar_mass = []
met = []

dust2 = []
dust_index = []
dust1_fraction = []

log_fagn = []
log_agn_tau = []
gas_logz = []

duste_qpah = []
duste_umin = []
log_duste_gamma = []

mwa = []
sfr10 = []
sfr30 = []
sfr100 = []
ssfr10 = []
ssfr30 = []
ssfr100 = []

mu = []

rest_UVJugi = []
rest_UVJugi_map = []
rest_UVJugi_colors = []
rest_UVJugi_colors_map = []

rest_gz = []
rest_gz_map = []
rest_gz_colors = []
rest_gz_colors_map = []

rest_NUVrJ = []
rest_NUVrJ_map = []
rest_NUVrJ_colors = []
rest_NUVrJ_colors_map = []

print(perc_dir)
all_files = os.listdir(perc_dir)

cnt = 0
missed = []
for this_file in all_files:
    if this_file.endswith('perc_{}.npz'.format(which_prior)):
        mid = int(this_file.split('_')[1])
        # print(mid)

        ffnpz = os.path.join(perc_dir, this_file)
        dat = np.load(ffnpz, allow_pickle=True)
        perc = dat['percentiles'][()]
        # print(perc.keys())

        zred.append(perc['zred'])
        total_mass.append(perc['total_mass'])
        stellar_mass.append(perc['stellar_mass'])
        met.append(perc['logzsol'])

        mwa.append(perc['mwa'])
        sfr10.append(perc['sfr'][0])
        sfr30.append(perc['sfr'][1])
        sfr100.append(perc['sfr'][2])

        ssfr10.append(perc['ssfr'][0])
        ssfr30.append(perc['ssfr'][1])
        ssfr100.append(perc['ssfr'][2])

        dust2.append(perc['dust2'])
        dust_index.append(perc['dust_index'])
        dust1_fraction.append(perc['dust1_fraction'])

        log_fagn.append(perc['log_fagn'])
        log_agn_tau.append(perc['log_agn_tau'])
        gas_logz.append(perc['gas_logz'])

        duste_qpah.append(perc['duste_qpah'])
        duste_umin.append(perc['duste_umin'])
        log_duste_gamma.append(perc['log_duste_gamma'])

        mu.append(perc['mu'])

        rest_UVJugi.append(perc['rest_UVJugi'])
        rest_UVJugi_map.append(perc['rest_UVJugi_map'])
        rest_UVJugi_colors.append(perc['rest_UVJugi_colors'])
        rest_UVJugi_colors_map.append(perc['rest_UVJugi_colors_map'])
        
        rest_gz.append(perc['rest_gz'])
        rest_gz_map.append(perc['rest_gz_map'])
        rest_gz_colors.append(perc['rest_gz_colors'])
        rest_gz_colors_map.append(perc['rest_gz_colors_map'])
        
        rest_NUVrJ.append(perc['rest_NUVrJ'])
        rest_NUVrJ_map.append(perc['rest_NUVrJ_map'])
        rest_NUVrJ_colors.append(perc['rest_NUVrJ_colors'])
        rest_NUVrJ_colors_map.append(perc['rest_NUVrJ_colors_map'])
        
        # make sure all theta idx are the same
        zred_ml.append(dat['chain_ml'][0])

        idx_ftrue = np.where(cat['id']==mid)[0][0]
        zred_spec.append(cat['z_spec'][idx_ftrue])

        objid_list.append(mid)
        cnt += 1

        if cnt % 100 == 0:
            print(cnt)

np.savez(sname, objid=objid_list,
         zred_spec=zred_spec, zred=zred, zred_ml=zred_ml,
         total_mass=total_mass, stellar_mass=stellar_mass, met=met,
         dust2=dust2, dust_index=dust_index, dust1_fraction=dust1_fraction,
         log_fagn=log_fagn, log_agn_tau=log_agn_tau, gas_logz=gas_logz,
         duste_qpah=duste_qpah, duste_umin=duste_umin, log_duste_gamma=log_duste_gamma,
         mwa=mwa,
         sfr10=sfr10, sfr30=sfr30, sfr100=sfr100,
         ssfr10=ssfr10, ssfr30=ssfr30, ssfr100=ssfr100,
         mu=mu,
         rest_UVJugi=rest_UVJugi, rest_UVJugi_map=rest_UVJugi_map, rest_UVJugi_colors=rest_UVJugi_colors, rest_UVJugi_colors_map=rest_UVJugi_colors_map,
         rest_gz=rest_gz, rest_gz_map=rest_gz_map, rest_gz_colors=rest_gz_colors, rest_gz_colors_map=rest_gz_colors_map,
         rest_NUVrJ=rest_NUVrJ, rest_NUVrJ_map=rest_NUVrJ_map, rest_NUVrJ_colors=rest_NUVrJ_colors, rest_NUVrJ_colors_map=rest_NUVrJ_colors_map,
        )

print('length:', len(objid_list))
print('saved to', sname)

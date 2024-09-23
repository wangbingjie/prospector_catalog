'''creates *_sfh_*.npz
ML chain is the last entry
'''
import os, sys, time
import numpy as np
from astropy.table import Table

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--prior', type=str, default='phisfh', help='phisfh, phisfhzfixed')
parser.add_argument('--indir', type=str, default='post_parrot', help='input folder storing chains')
parser.add_argument('--outdir', type=str, default='results', help='output folder storing unweighted chains and quantiles')
args = parser.parse_args()
print(args)

which_prior = args.prior

foo = args.indir[:]
if foo.endswith('/'):
    foo = foo[:-1]
sname = os.path.join(args.outdir, 'sfh_{}_{}'.format(args.prior, foo)+'.npz')
print('sfhs will be saved to', sname)

objid_list = []
tmax_all = []
sfh_all = []
all_files = os.listdir(args.indir)

# percentiles for SFH
perc = np.array([0.1, 2.3, 15.9, 50, 84.1, 97.7, 99.9]) * 0.01
        
cnt = 0
for this_file in all_files:
    if this_file.endswith('unw_{}.npz'.format(which_prior)):
        mid = int(this_file.split('_')[1])
        _ffile = os.path.join(args.indir, this_file)

        dat = np.load(_ffile, allow_pickle=True)
        chains = dat['chains'][()]
        # print(chains.keys())

        tmax_all.append(chains['agebins_max'])
        sfh_all.append(np.quantile(chains['sfh'], perc, axis=0))
        objid_list.append(mid)

        cnt += 1
        if cnt % 10000 == 0:
            np.savez(sname, objid=objid_list, #age_interp=chains['age_interp'],
                     agebins_max=tmax_all, sfh=sfh_all)
            print(cnt)

np.savez(sname, objid=objid_list, #age_interp=chains['age_interp'],
         agebins_max=tmax_all, sfh=sfh_all)
# age_interp [lookback time in Gyr] can be reconstructed as
# sfh = fsfh['sfh'][idx_in_sfh]
# tbins = sfh.shape[1]
# age = np.logspace(1, fsfh['agebins_max'][idx_in_sfh], tbins)/1e9

print('length:', len(objid_list))
print('saved to', sname)

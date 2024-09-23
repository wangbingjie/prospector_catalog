'''creates *_chains_*.npz
'''
import os, sys, time
import numpy as np
from astropy.table import Table
import prospect.io.read_results as reader

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--prior', type=str, default='phisfh')
parser.add_argument('--indir', type=str, default='chains_parrot', help='input folder storing chains')
parser.add_argument('--outdir', type=str, default='results', help='output folder storing unweighted chains and quantiles')
args = parser.parse_args()
print(args)

which_prior = args.prior
_indir = args.indir[:]

objid_list = []
chains_all = []

all_files = os.listdir(_indir)

keys = ['zred', 'total_mass', 'stellar_mass', 'logzsol', 'mwa',
        'sfr10', 'sfr30', 'sfr100',
        'ssfr10', 'ssfr30', 'ssfr100',
        'dust2', 'dust_index', 'dust1_fraction',
        'log_fagn', 'log_agn_tau', 'gas_logz',
        'duste_qpah', 'duste_umin', 'log_duste_gamma']

foo = args.indir
if foo.endswith('/'):
    foo = foo[:-1]
sname = os.path.join(args.outdir, 'chains_{}_{}'.format(args.prior, foo)+'.npz')
print('will be saved to', sname)

cnt = 0
missed = []
for this_file in all_files:
    if this_file.endswith('unw_{}.npz'.format(which_prior)):
        mid = int(this_file.split('_')[1])
        _ffile = os.path.join(_indir, this_file)
        
        dat = np.load(_ffile, allow_pickle=True)
        chains = dat['chains'][()]
        # print(chains.keys())

        chain_eqwt = np.stack([chains['zred'], chains['total_mass'], chains['stellar_mass'],
                               chains['logzsol'], chains['mwa'],
                               chains['sfr'][:,0], chains['sfr'][:,1], chains['sfr'][:,2],
                               chains['ssfr'][:,0], chains['ssfr'][:,1], chains['ssfr'][:,2],
                               chains['dust2'], chains['dust_index'], chains['dust1_fraction'],
                               chains['log_fagn'], chains['log_agn_tau'], chains['gas_logz'],
                               chains['duste_qpah'], chains['duste_umin'], chains['log_duste_gamma']
                               ]).T

        # chains_all.append(np.vstack([chain_eqwt, chain_ml]))
        chains_all.append(chain_eqwt)
        objid_list.append(mid)

        cnt += 1
        if cnt % 5000 == 0:
            print(cnt)
            np.savez(sname, objid=objid_list, chains=chains_all, theta_labels=keys)

np.savez(sname, objid=objid_list, chains=chains_all, theta_labels=keys)

print('length:', len(objid_list))
print('saved to', sname)

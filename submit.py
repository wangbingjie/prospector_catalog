import os, sys, time
import numpy as np
from astropy.table import Table

def data_dir():

    dat_dirs = ['/storage/home/bbw5389/group/',
                '/Users/bwang/research/']

    for _dir in dat_dirs:
        if os.path.isdir(_dir): return _dir

def run_params(pycmd, log_dir='log', acc='bc', i=0, jobname='p', wtime=48, env='prosp-dev'):
    
    jname = '{}_{}'.format(jobname, i)
    
    ts = time.strftime("%y%b%d-%H.%M", time.localtime())

    if acc == 'bc':
        txt_acc = '\n'.join(["#!/bin/bash -l",
                             "#SBATCH --account=jql6565_bc\n"])
    elif acc == 'sc':
        txt_acc = '\n'.join(["#!/bin/bash -l",
                             "#SBATCH --account=jql6565_sc\n",
                             "#SBATCH --mem=10G\n"])
    else:
        txt_acc = '\n'.join(["#!/bin/bash -l",
                             "#SBATCH --account=open\n"])
        
    txt_acc += "#SBATCH --time={:d}:00:00\n".format(wtime)

    txt_2 = '\n'.join([
        "#SBATCH --nodes=1",
        "#SBATCH --job-name={}".format(jname[:16]),
        "#SBATCH --output={}/{}_{}.out".format(log_dir, jname, ts),
        "#SBATCH --error={}/{}_{}.err".format(log_dir, jname, ts),
        "",
        'now=$(date +"%T")',
        'echo "start time ... $now"',
        "",
        'module load anaconda3',
        "source activate {}".format(env),
        "",
        "cd /storage/home/bbw5389/group/uncover_sps_gen1/stellar_pop_catalog_mb",
        "python {}".format(pycmd),
        "",
        'now=$(date +"%T")',
        'echo "end time ... $now"',
        ""])

    txt = txt_acc + txt_2

    f = open('_params.slurm','w')
    f.write(txt)
    f.close()
    os.system('sbatch _params.slurm')
    os.system('rm _params.slurm')
    return None


if __name__ == '__main__':
    
    ver = 'v5.0.1_LW_SUPER'
    spsver = 'spsv0.0'

    ################################## step 1. sed fit ####################################

    catalog = 'UNCOVER_{}_CATALOG.fits'.format(ver)

    cat = Table.read('../phot_catalog/' + catalog)
    tot = np.arange(len(cat))

    '''
    # the below is for fitting the missed objects, one per core.
    tot = []
    for _id in cat['id'].data:
        if _id not in nfiles_phot:
            tot.append(_id)
    tot = np.array(tot)
    print(tot)
    tot = tot - 1 # id to idx # this only works if using the full phot catalog
    ncores = len(tot)
    '''

    acc = 'bc'
    ncores = 840 # number of cores to request
    wtime = int(24*7) # time

    groups = np.array_split(tot, ncores) # divide the total number into xxx cores

    outdir = 'chains_parrot_{}_{}'.format(ver, spsver)

    isExist = os.path.exists(outdir)
    if not isExist:
        os.makedirs(outdir)
        print("new output directory created:", outdir)
    logdir = 'log/{}'.format(outdir)
    isExist = os.path.exists(logdir)
    if not isExist:
        os.makedirs(logdir)
        print("new log directory created:", logdir)
   
    for igroup in range(len(groups)):
        idx0 = groups[igroup][0]
        idx1 = groups[igroup][-1] + 1 # +1 b/c id1 is not included when running the fit
        if 'zspec' in catalog:
            _cmd = 'uncover_gen1_parrot_phisfhzspec_params.py --catalog {} --idx0 {} --idx1 {} --outdir {}'.format(catalog, idx0, idx1, outdir)
        else:
            _cmd = 'uncover_gen1_parrot_phisfh_params.py --catalog {} --idx0 {} --idx1 {} --outdir {}'.format(catalog, idx0, idx1, outdir)
        if igroup == 0:
            print(_cmd)
        run_params(_cmd, jobname='mb', log_dir=logdir, acc=acc, i=idx0, wtime=wtime, env='prosp-dev')
        time.sleep(0.05)


    ################################ step 2. post-prsocessing ################################

    wtime = 48
    catalog = 'UNCOVER_{}_CATALOG.fits'.format(ver)

    sps = 'parrot'
    if 'zspec' in catalog:
        indir = 'chains_parrot_zspec_{}_{}'.format(ver, spsver)
        prior = 'phisfhzspec'
    else:
        indir = 'chains_parrot_{}_{}'.format(ver, spsver)
        prior = 'phisfh'

    run = 'std'
    
    outdir = indir
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print("new output directory created:", outdir)
    logdir = 'log/{}'.format(outdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
        print("new log directory created:", logdir)

    acc = 'bc'

    ids_file  = 'None' # can also pass a .txt file that contains the ids of the sources that need to perform the post-prsocessing on

    ## have to be matched to that in postprocess_parrot_wrap.py
    n_split_arr = 800 # number of cores

    for i in range(n_split_arr):
        _cmd = "postprocess_parrot_wrap.py --prior {} --fit 'fid' --catalog {} --indir {} --outdir {} --narr {} --iarr {} --ids_file {} --run {}".format(prior, catalog, indir, outdir, n_split_arr, i, ids_file, run)
        if i == 0:
            print(_cmd)
        run_params(_cmd, log_dir=logdir, acc=acc, i=i, jobname='p', wtime=wtime, env='prosp-dev')
        time.sleep(0.05)


    ########################## step 3. parse individual results into summary files ##########################

    prior = 'phisfh' # prospector-beta mass + sfh priors

    # saves posterior moments
    _cmd = 'save_perc.py --catalog UNCOVER_{}_CATALOG.fits --indir post_parrot_{}_{} --prior {}'.format(ver, ver, spsver, prior)
    print(_cmd)
    run_params(_cmd, jobname='perc', oe_dir='log', acc='sc', i=0, wtime=10)

    # saves transformed chains (i.e., those published in the data release)
    _cmd = 'save_chain.py --catalog UNCOVER_{}_CATALOG.fits --indir post_parrot_{}_{}'.format(ver, ver, spsver)
    print(_cmd)
    run_params(_cmd, jobname='chain', log_dir='log', acc='sc', i=0, wtime=10)

    # saves zred, total_mass, logsfr_ratios
    _cmd = 'save_chain_untrans.py --catalog UNCOVER_{}_CATALOG.fits --indir post_parrot_{}_{} --prior {}'.format(ver, ver, spsver, prior)
    print(_cmd)
    run_params(_cmd, jobname='chainu', log_dir='log', acc='sc', i=0, wtime=10)
    
    _cmd = 'save_sfh.py --catalog UNCOVER_{}_CATALOG.fits --indir post_parrot_{}_{}'.format(ver, ver, spsver)
    print(_cmd)
    run_params(_cmd, jobname='sfh', log_dir='log', acc='sc', i=0, wtime=10)
    
    _cmd = 'save_spec.py --catalog UNCOVER_{}_CATALOG.fits --chain_indir chains_parrot_{}_{} --perc_indir chains_parrot_{}_{} --outdir results'.format(ver, ver, spsver, ver, spsver)
    print(_cmd)
    run_params(_cmd, jobname='spec', log_dir='log/', acc='sc', i=0, wtime=10)

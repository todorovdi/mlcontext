import os
import os.path as op
import numpy as np
import mne
from mne.io import read_raw_fif
from mne.decoding import (cross_val_multiscore, LinearModel)

from csp_my import SPoC

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import StratifiedKFold, KFold, ShuffleSplit
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.metrics import make_scorer
from jr.gat import scorer_spearman, AngularRegression
from mne import Epochs
import pandas as pd
from Levenshtein import editops
import warnings
import sys
from base2 import (int_to_unicode, point_in_circle,getXGBparams,
                   calc_target_coordinates_centered, radius_target,
                   radius_cursor, B2B_SPoC,target_angs)
from config2 import n_jobs as n_jobs_def
from config2 import (path_data,path_data_tmp,freq_name2freq,
                     stage2event_ids,stim_channel_name,delay_trig_photodi)
from xgboost import XGBRegressor

from datetime import datetime  as dt



target_coords = calc_target_coordinates_centered(target_angs)

n_jobs            = int( par.get('n_jobs', n_jobs_def) )
subject           = par['subject']
env_to_run        = par['env_to_run']; env = env_to_run
regression_type   = par['regression_type']
freq_name         = par['freq_name']
freq_limits       = par.get('freq_limits',freq_name2freq[freq_name] )
if isinstance(freq_limits,str):
    freq_limits = eval(freq_limits)
hpass             = par['hpass']  # '0.1', no_hpass, no_filter

output_folder = par.get('output_folder',f'spoc_home2_{hpass}')
ICAstr = par.get('ICAstr','with_ICA'  )  # empty string is allowed
time_locked = par.get('time_locked','target')
# home position
tmin = float( par.get('tmin',-0.5)    )
tmax = float( par.get('tmax',0)       )
# task = 'LocaError'  # 'VisuoMotor' or 'LocaError'
task = par.get('task','VisuoMotor')

do_classic_dec            = int( par.get('do_classic_dec',1)            )
do_partial_dec            = int( par.get('do_partial_dec',1)            )
est_parallel_across_dims  = int( par.get('est_parallel_across_dims',1)  )
est_parallel_within_dim   = int( par.get('est_parallel_within_dim',0)   )
b2b_each_fit_is_parllel   = int( par.get('b2b_each_fit_is_parllel',0)   )
classic_dec_verbose       = int( par.get('classic_dec_verbose',3)       )

nb_fold                   = int( par.get('nb_fold',6)                   )
n_splits_B2B              = int( par.get('n_splits_B2B',30)             )
SPoC_n_components         = int( par.get('SPoC_n_components',5)         )

analysis_name = par.get('analysis_name', 'prevmovement_preverrors_errors_prevbelief')
use_preloaded_raw = int( par.get('use_preloaded_raw', 0) )
mne_fit_log_level         = par.get('mne_fit_log_level', 'warning')

load_epochs  = int( par.get('load_epochs',0)  ) # only reg type and env would be optimized
save_epochs  = int( par.get('save_epochs',0)  ) # only reg type and env
load_flt_raw = int( par.get('load_flt_raw',1) )
save_flt_raw = int( par.get('save_flt_raw',1) )

##########################################################################
##########################################################################

print(f'__START: {__file__} subj={subject}, hpass={hpass}, '
      f'regression_type={regression_type}, freq_name={freq_name}, '
      f'env={env_to_run} at {dt.now()}')
#mne_fit_log_level = 'debug'

# both on will give errors, both off are fine
assert not (est_parallel_across_dims and est_parallel_within_dim)

allow_CUDA = 1
if allow_CUDA:
    mne.cuda.init_cuda()
    n_jobs_MNE='cuda'
else:
    n_jobs_MNE=n_jobs
n_jobs_XGB=n_jobs

results_folder = op.join(path_data,subject,'results',output_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

if b2b_each_fit_is_parllel:
    n_jobs_SPoC = n_jobs
else:
    n_jobs_SPoC = 1

# if we parallelize across dims, we want it to be 1
if est_parallel_across_dims:
    n_jobs_per_dim_classical_dec = 1
else:
    if est_parallel_within_dim:
        n_jobs_per_dim_classical_dec = n_jobs
    else:
        n_jobs_per_dim_classical_dec = 1
add_clf_creopts_est = getXGBparams(n_jobs = n_jobs_per_dim_classical_dec)

#regression_type = 'Ridge'
#regression_type = 'xgboost'

if b2b_each_fit_is_parllel:
    add_clf_creopts = getXGBparams(n_jobs = None)
else:
    add_clf_creopts = getXGBparams(n_jobs = 1)

##############################

fname = op.join(path_data, subject, 'behavdata', f'err_sens_{task}.npz')
f = np.load(fname, allow_pickle=True)['arr_0'][()]
#env2corr      = f['env2corr'][()]
#env2err_sens      = f['env2err_sens'][()]
#env2pre_err_sens  = f['env2pre_err_sens'][()]
env2pre_dec_data  = f['env2pre_dec_data']
env2non_hit  = f['env2non_hit']

# read behav data
fname = op.join(path_data, subject, 'behavdata',
                f'behav_{task}_df.pkl' )
behav_df = pd.read_pickle(fname)

# time_locked = 'target'  # 'target' or 'reach'
# Read raw to extra MEG event triggers
#for freq_name, freq in freq_names_freqs:
freq = freq_limits
print(f'---------- Starting freq = {freq_name}')
# read raw data
run = list()
files = os.listdir(op.join(path_data, subject))
run.extend(([op.join(
            path_data, subject + '/')
            + f for f in files if task in f]))
fname_raw = run[0]

print(f'use_preloaded_raw = {use_preloaded_raw}')
#import sys; sys.exit(1)
# get raw and events from raw file
# raw = read_raw_ctf(fname_raw, preload=True, system_clock='ignore')
try:
    len(ep)
    preloaded_raw_is_present = True
except NameError as e:
    preloaded_raw_is_present = False



# fdsf
# pattern to get pars    s/^\(\w\+\)\s*=\s*\([^#]*\).*$/\1 = par.get('\1',\2)/gc

bsl = None
from config2 import min_event_duration
if (not use_preloaded_raw) or (not preloaded_raw_is_present):

    fn = f'raw_{task}_{hpass}_{ICAstr}_{freq_name}_{env_to_run}_{time_locked}_{tmin:.2f},{tmax:.2f}.fif'
    fn_epochs_full = op.join( path_data, subject, fn)

    fn_flt_raw = f'raw_{task}_{hpass}_{ICAstr}_{freq_name}.fif'
    fn_flt_raw_full = op.join(path_data_tmp, subject, fn_flt_raw )

    fn_events = f'events_{task}_{hpass}_{ICAstr}.txt'
    fn_events_full = op.join(path_data, subject, fn_events )

    if load_epochs and os.path.exists(fn_epochs_full):
        print(f'INFO: Found epochs, loading {fn_epochs_full}')
        ep = mne.read_epochs(fn_epochs_full)
    else:
        if load_flt_raw and os.path.exists(fn_flt_raw_full) and os.path.exists(fn_events_full):
            print(f'INFO: Found filtered raw, loading {fn_flt_raw_full}')
            raw = read_raw_fif(fn_flt_raw_full, preload=True)
            events = mne.read_events(fn_events_full)
        else:
            raw = read_raw_fif(op.join(path_data, subject, f'raw_{task}_{hpass}_{ICAstr}.fif' ),
                    preload=True)
            events = mne.find_events(raw, stim_channel=stim_channel_name,
                                        min_duration=min_event_duration)
            events[:, 0] += delay_trig_photodi

            raw.filter(freq[0], freq[1], n_jobs=n_jobs_MNE)

            if save_flt_raw:
                print(f'Saved filtered raw, loading {fn_flt_raw_full}')
                try:
                    raw.save(fn_flt_raw_full,overwrite=True)
                except OSError as e:
                    print(f'ERROR: got OSerror during saving raw {str(e)}')

                mne.write_events(fn_events_full, events, overwrite=True)

        epochs = Epochs(raw, events, event_id=stage2event_ids[time_locked],
                        tmin=tmin, tmax=tmax, preload=True,
                        baseline=bsl, decim=2)

        del raw

        env2epochs=dict(stable=epochs['20', '21', '22', '23', '30'],
                random=epochs['25', '26', '27', '28', '35'] )
        ep = env2epochs[env_to_run]
        if save_epochs:
            print(f'Saved epochs to {fn_epochs_full}')
            ep.save(fn_epochs_full,overwrite=True)

#times = epochs.times
times = ep.times
import gc; gc.collect()

##########################################################################
########################## Prepare ML       ##############################
##########################################################################

# Regression for classic decoding
alphas = np.logspace(-5, 5, 12)
spoc = SPoC(n_components=SPoC_n_components, log=True, reg='oas',
                                rank='full', n_jobs=n_jobs_SPoC)
spoc_est = SPoC(n_components=SPoC_n_components, log=True, reg='oas',
                                rank='full', n_jobs=n_jobs_per_dim_classical_dec,
                                fit_log_level=mne_fit_log_level)

# G is a much slower operation

if regression_type == 'Ridge':
    est = make_pipeline(spoc_est, RidgeCV(alphas=alphas))
    # Regressions for the B2B
    G = make_pipeline(spoc, RidgeCV(alphas=alphas, fit_intercept=False))
elif regression_type == 'xgboost':
    xgb = XGBRegressor(**add_clf_creopts)

    add_clf_creopts['n_jobs'] = n_jobs_per_dim_classical_dec
    xgb_est = XGBRegressor(**add_clf_creopts_est)
    est = make_pipeline(spoc_est, xgb_est)
    # Regressions for the B2B
    G = make_pipeline(spoc, xgb)
else:
    raise ValueError('wrong regression value')

H = LinearRegression(fit_intercept=False, n_jobs= n_jobs_SPoC)
#G = direct pipeline (spoc + regerssor)
#H = back pipeline
b2b = B2B_SPoC(G=G, H=H, n_splits=n_splits_B2B,
        parallel_type='across_splits_and_dims')
# Cross-validation
cv = KFold(nb_fold, shuffle=True)

########################  Run decoding
#for env in ['stable', 'random']:
print(f'----- starting env = {env}')
non_hit = env2non_hit[env]
analysis_value = env2pre_dec_data[env]


X = ep.pick_types(meg=True, ref_meg=False)._data
wh = (times > -0.45) & (times < 0.05)
X = X[non_hit]  # trial x channel x time
X = X[:, :, wh]
Y = np.array(analysis_value)
Y = Y[:, non_hit].T  # TRANSPOSE!!!

dim = Y.shape[1]

##########################################################################
########################## Classic decoding ##############################
##########################################################################
if do_classic_dec:
    print(f'-------- Start classic decoding '
          f'est_parallel_across_dims={est_parallel_across_dims} for {n_jobs} jobs')
    scoring = make_scorer(scorer_spearman)
    scores = list()
    # over all dims
    def _est_run(est,X,y,cv,scoring,n_jobs):
        with mne.use_log_level(mne_fit_log_level):
            score = cross_val_multiscore(est, X, y=y, cv=cv, scoring=scoring,
                                            n_jobs=n_jobs, verbose=classic_dec_verbose)
        score = score.mean(axis=0)
        return score

    from joblib import Parallel, delayed
    if est_parallel_across_dims:
        scores = Parallel(n_jobs=n_jobs)(
            delayed(_est_run)(est,X,Y[:,dimi],cv,scoring,n_jobs_per_dim_classical_dec ) \
                for dimi in range(dim ))
    else:
        for dimi in range(dim):
            y = Y[:, dimi]
            score = _est_run(est,X,y,cv,scoring,n_jobs_per_dim_classical_dec)
            scores.append(score)
    scores = np.array(scores)

    # save scores
    fn = f'{env}_{regression_type}_scores_{analysis_name}_{freq_name}.npy'
    fname = op.join(results_folder, fn)
    print(f'Finished saving {fname}')

#X: trialx x MEG x time
#Y: trials x vals

##########################################################################
########################## Partial decoding ##############################
##########################################################################
if do_partial_dec:
    print('-------- Start partial decoding ')
    Y = scale(Y)
    b2b.fit(X, Y)

    partial_scores = np.diag(b2b.E_)
    np.save(fname, np.array(scores))
    fname = op.join(results_folder,
        f'{env}_{regression_type}_partial_scores_{analysis_name}_{freq_name}.npy')
    # NOTE: here he use spaRtial scores, was written wrong
    #                '%s_%spartial_scores_%s_%s.npy' % (subject, env,
    #                                                   analysis_name,
    #                                                   freq_name))
    np.save(fname, np.array(partial_scores))

    print(f'Finished saving {fname}')

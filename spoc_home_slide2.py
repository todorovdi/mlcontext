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
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import make_scorer
from jr.gat import scorer_spearman, AngularRegression
from mne import Epochs
import pandas as pd
from Levenshtein import editops
import warnings
import sys
from base2 import (int_to_unicode, point_in_circle,getXGBparams,
                   calc_target_coordinates_centered, radius_target,
                   radius_cursor, B2B_SPoC)
from config2 import path_data
from config2 import event_ids_tgt,event_ids_feedback
from xgboost import XGBRegressor

from datetime import datetime  as dt
print(f'__START: {__file__} subj={subject}, hpass={hpass}, '
      f'regression_type={regression_type}, freq_name={freq_name}, '
      f'env={env_to_run} at {dt.now()}')


mne.cuda.init_cuda()
n_jobs_MNE='cuda'
n_jobs_XGB=n_jobs

#subject = sys.argv[1]
#hpass = 'no_filter'  # '0.1', no_hpass, no_filter
output_folder = f'spoc_home2_{hpass}'

nb_fold = 6
n_splits_B2B = 30


results_folder = op.join(path_data,subject,'results',output_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

ICAstr = 'with_ICA'  # or empty string
b2b_each_fit_is_parallel = False

if b2b_each_fit_is_parallel:
    n_jobs_SPoC = n_jobs
    add_clf_creopts = getXGBparams(n_jobs = None)
else:
    n_jobs_SPoC = 1
    add_clf_creopts = getXGBparams(n_jobs = 1)

# if we parallelize across dims, we want it to be 1
est_parallel_across_dims = 1
if est_parallel_across_dims:
    n_jobs_per_dim_classical_dec = 1
else:
    n_jobs_per_dim_classical_dec = n_jobs
add_clf_creopts_est = getXGBparams(n_jobs = n_jobs_per_dim_classical_dec)

#time_locked = 'target'
## home position
#tmin = -0.5; tmax = 0

# NEED PARAMS time_locked, tmin,tmax

assert regression_type in  ['Ridge', 'xgboost']

##############################
# task = 'LocaError'  # 'VisuoMotor' or 'LocaError'
task = 'VisuoMotor'

fname = op.join(path_data, subject, 'behavdata', f'err_sens_{task}.npz')
f = np.load(fname, allow_pickle=True)['arr_0'][()]
#env2corr      = f['env2corr'][()]
#env2err_sens      = f['env2err_sens'][()]
#env2pre_err_sens  = f['env2pre_err_sens'][()]
#env2pre_dec_data  = f['env2pre_dec_data']
#env2non_hit  = f['env2non_hit']

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

print(use_preloaded_raw)
#import sys; sys.exit(1)
# get raw and events from raw file
# raw = read_raw_ctf(fname_raw, preload=True, system_clock='ignore')
if not use_preloaded_raw:
    raw = read_raw_fif(op.join(path_data, subject, f'raw_{task}_{hpass}_{ICAstr}.fif' ),
            preload=True)
    #events = mne.find_events(raw, stim_channel='UPPT001',
    #                            min_duration=0.02)
    #events[:, 0] += 18  # to account for delay between trig. & photodi.

    raw.filter(freq[0], freq[1], n_jobs=n_jobs_MNE)

    bsl = None  # (start,end)
    if time_locked == 'target':
        epochs_type = getEpochs_custom(raw, event_ids_tgt, tmin=tmin,tmax=tmax, bsl=bsl)
    elif time_locked == 'target':
        epochs_type = getEpochs_custom(raw, event_ids_feedback, tmin=tmin,tmax=tmax, bsl=bsl)
    else:
        raise ValueError(f'wrong time_locked={time_locked}')
    #epochs = Epochs(raw, events, event_id=[20, 21, 22, 23, 25, 26, 27, 28],
    #                tmin=tmin, tmax=tmax, preload=True,
    #                baseline=None, decim=2)

env2epochs=dict(stable=epochs['20', '21', '22', '23', '30'],
        random=epochs['25', '26', '27', '28', '35'] )

times = epochs.times

#analysis_name = 'prevmovement_preverrors_errors_prevbelief'


#epochs_type = zip(['feedback', 'target'],

r = getAnalysisData(env, time_locked, control_type, behav_df)
analysis_name, analysis_value, non_hit_cur = r



# Regression for classic decoding
alphas = np.logspace(-5, 5, 12)
spoc = SPoC(n_components=5, log=True, reg='oas',
                                rank='full', n_jobs=n_jobs_SPoC)
spoc_est = SPoC(n_components=5, log=True, reg='oas',
                                rank='full', n_jobs=n_jobs_per_dim_classical_dec)

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
env = env_to_run
print(f'----- starting env = {env}')
non_hit = env2non_hit[env]
analysis_value = env2pre_dec_data[env]

ep = env2epochs[env]

X = ep.pick_types(meg=True, ref_meg=False)._data
wh = (times > -0.45) & (times < 0.05)
X = X[non_hit]  # trial x channel x time
X = X[:, :, wh]
Y = np.array(analysis_value)
Y = Y[:, non_hit].T  # TRANSPOSE!!!

dim = Y.shape[1]

# Classic decoding
print(f'-------- Start classic decoding est_parallel_across_dims={est_parallel_across_dims}')
scoring = make_scorer(scorer_spearman)
scores = list()
# over all dims
def _est_run(est,X,y,cv,scoring,n_jobs):
    with mne.use_log_level('warning'):
        score = cross_val_multiscore(est, X, y=y, cv=cv, scoring=scoring,
                                        n_jobs=n_jobs)
    score = score.mean(axis=0)
    return scores

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

#X: trialx x MEG x time
#Y: trials x vals

# run partial decoding
print('-------- Start partial decoding ')
Y = scale(Y)
b2b.fit(X, Y)
partial_scores = np.diag(b2b.E_)
# save scores
fn = f'{env}_{regression_type}_scores_{analysis_name}_{freq_name}.npy'
fname = op.join(results_folder, fn)
np.save(fname, np.array(scores))
fname = op.join(results_folder,
    f'{env}_{regression_type}_spatial_scores_{analysis_name}_{freq_name}.npy')
# NOTE: here he use spaRtial scores, was written wrong
#                '%s_%spartial_scores_%s_%s.npy' % (subject, env,
#                                                   analysis_name,
#                                                   freq_name))
np.save(fname, np.array(partial_scores))

print(f'Finished saving {fname}')

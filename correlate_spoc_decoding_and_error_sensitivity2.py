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
import xgboost
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr
from mne import Epochs
import pandas as pd
import warnings
import sys
from base2 import (int_to_unicode, point_in_circle,getXGBparams,
                   calc_target_coordinates_centered, radius_target,
                   radius_cursor, partial_reg, B2B_SPoC,
                   event_ids, env2envcode, env2subtr, target_angs)
from config2 import path_data
from error_sensitivity import enforceTargetTriggerConsistency
from xgboost import XGBRegressor

###########
from datetime import datetime  as dt
print(f'__START: {__file__} subj={subject}, hpass={hpass}, '
      f'regression_type={regression_type}, freq_name={freq_name}, '
      f'env={env_to_run} at {dt.now()}')

mne.cuda.init_cuda()
n_jobs_SPoC = n_jobs
n_jobs_MNE='cuda'
n_splits_B2B = 30
each_fit_is_parallel = False

if b2b_each_fit_is_parallel:
    add_clf_creopts = getXGBparams(n_jobs = None)
else:
    add_clf_creopts = getXGBparams(n_jobs = 1)

#target_angs = (np.array([157.5, 112.5, 67.5, 22.5]) + 90) * \
#              (np.pi/180)
target_coords = calc_target_coordinates_centered(target_angs)

#subject = sys.argv[1]
#TODO: use no filter
#hpass = '0.1'  # '0.1', no_hpass
#hpass = 'no_filter'  # '0.1', no_hpass
output_folder = f'corr_spoc_es2_{hpass}'


#regression_type = 'Ridge'
#regression_type = 'xgboost'

task = 'VisuoMotor'
fname = op.join(path_data, subject, 'behavdata', f'err_sens_{task}.npz')
f = np.load(fname, allow_pickle=True)['arr_0'][()]
#env2corr      = f['env2corr'][()]
#env2err_sens      = f['env2err_sens'][()]
#env2pre_err_sens  = f['env2pre_err_sens'][()]
env2pre_dec_data  = f['env2pre_dec_data']
env2non_hit       = f['env2non_hit']


nb_fold = 6

freq_names = ['broad']
freqs = [(4, 60)]

results_folder = op.join(path_data,subject,'results',output_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

ICA = 'with_ICA'  # or empty string

time_locked = 'target'
tmin = -0.5
tmax = 0

#use_preloaded_raw = True

fname = op.join(path_data, subject, 'behavdata',
                f'behav_{task}_df.pkl')

behav_df = pd.read_pickle(fname)
# Read behav files and Compare behav and MEG events
# Compare meg events to behav events
targets      = np.array(behav_df['target'])
environment  = np.array(behav_df['environment'])
behav_df_full = behav_df


#meg_targets = epochs.copy().events[:, 2]
#for env,envcode in env2envcode.items():
#    trial_inds = np.where(environment == envcode)[0]
#    meg_targets[trial_inds] = meg_targets[trial_inds] - env2subtr[env]
#
#if do_delete_trials:
#    from Levenshtein import editops
#    changes = editops(int_to_unicode(targets), int_to_unicode(meg_targets))
#    # we have missing triggers in MEG file so we delete stuff from behav file
#    delete_trials = [change[1] for change in changes]
#    # read behav.pkl and remove bad_trials if bad_trials is not empty
#    if delete_trials:
#        behav_df = behav_df_full.copy().drop(delete_trials, errors='ignore')
#        behav_df.to_pickle(fname)
#
#    if not np.array_equal(meg_targets, targets):
#        warnings.warn('MEG events and behavior file do not match')

# task = 'LocaError_'  # 'VisuoMotor_' or 'LocaError_'
# time_locked = 'target'  # 'target' or 'reach'
# Read raw to extra MEG event triggers
# cycle over frequencies
#for freq_name, freq in zip(freq_names, freqs):
print(freq_name + '--------------------------')
# read behav data
# read raw data
run = list()
files = os.listdir(op.join(path_data, subject))
run.extend(([op.join(
            path_data, subject + '/')
            + f for f in files if task in f]))
fname_raw = run[0]
# get raw and events from raw file
# raw = read_raw_ctf(fname_raw, preload=True, system_clock='ignore')
fname_raw_full = op.join(path_data, subject, f'raw_{task}_{hpass}_{ICA}.fif')

if not use_preloaded_raw:
    if not os.path.exists(fname_raw_full):
        print(f'{fname_raw_full} does not exist, exiting')
        sys.exit(1)

    raw = read_raw_fif(fname_raw_full, preload=True)
    events = mne.find_events(raw, stim_channel='UPPT001',
                                min_duration=0.02)
    events[:, 0] += 18  # to account for delay between trig. & photodi.
    # check that a target trigger is always followed by a reach trigger
    t = -1
    bad_trials = list()
    bad_events = list()
    for ii, event in enumerate(events):
        if event[2] in [20, 21, 22, 23]:
            t += 1
            if events[ii+1, 2] == 100:
                if events[ii+2, 2] != 30:
                    bad_trials.append(t)
                    warnings.warn('Bad sequence of triggers')
                    # Delete bad events until the next beginning of a trial (10)
                    bad_events.append(ii - 1)
                    for iii in range(5):
                        if events[ii + iii, 2] == 10:
                            break
                        else:
                            bad_events.append(ii+iii)
            elif events[ii+1, 2] != 30:
                bad_trials.append(t)
                warnings.warn('Bad sequence of triggers')
                # Delete bad events until the next beginning of a trial (10)
                bad_events.append(ii - 1)
                for iii in range(5):
                    if events[ii + iii, 2] == 10:
                        break
                    else:
                        bad_events.append(ii+iii)
    events = np.delete(events, bad_events, 0)

    raw.filter(freq[0], freq[1], n_jobs=n_jobs_MNE)

epochs = Epochs(raw, events, event_id = event_ids,
                tmin=tmin, tmax=tmax, preload=True,
                baseline=None, decim=2)


environment = np.array(behav_df_full['environment'])
behav_df_curfreq = behav_df_full.copy()
enforceTargetTriggerConsistency(behav_df_curfreq, epochs,environment, save_fname=fname)

env2epochs=dict(stable=epochs['20', '21', '22', '23', '30'],
        random=epochs['25', '26', '27', '28', '35'] )

#for env,envcode in env2envcode.items():
trial_inds = np.where(environment == envcode)[0]
meg_targets_cur = epochs.copy().events[:, 2]
meg_targets_cur[trial_inds] = meg_targets_cur[trial_inds] - env2subtr[env]
targets_cur = targets[trial_inds]

environment = np.array(behav_df_curfreq['environment'])
feedback     = np.array(behav_df_curfreq['feedback'])
feedbackX    = np.array(behav_df_curfreq['feedbackX'])
feedbackY    = np.array(behav_df_curfreq['feedbackY'])
feedbackY    = np.array(behav_df_curfreq['feedbackY'])
errors       = np.array(behav_df_curfreq['error'])

feedbackX_cur = feedbackX[trial_inds]
feedbackY_cur = feedbackY[trial_inds]
errors_cur = errors[trial_inds]

## keep only non_hit trials
#non_hit_cur = point_in_circle(targets_cur, target_coords,
#                                 feedbackX_cur, feedbackY_cur,
#                                 radius_target + radius_cursor)
#non_hit = non_hit_cur
#non_hit = np.array(non_hit)
## remove trials following hit (because no previous error)
## non_hit = ~(~non_hit | ~np.insert(non_hit, 0, 1)[:-1])
#non_hit = np.insert(non_hit, 0, 0)[:-1]
#non_hit[[0, 192]] = False  # Removing first trial of each block

non_hit = env2non_hit[env]

abs_errors_cur = np.abs(errors_cur)
times = epochs.times
# analysis_name = 'target_movement_errors_preverrors'

spoc = SPoC(n_components=5, log=True, reg='oas',
                                rank='full', n_jobs=n_jobs_SPoC)
# Regression for classic decoding
if regression_type == 'Ridge':
    alphas = np.logspace(-5, 5, 12)
    est = make_pipeline(spoc, RidgeCV())
    # Regressions for the B2B
    G = make_pipeline(spoc,
                        RidgeCV(alphas=alphas, fit_intercept=False))
elif regression_type == 'xgboost':
    xgb = XGBRegressor(**add_clf_creopts)
    est = make_pipeline(spoc, xgb)
    # Regressions for the B2B
    G = make_pipeline(spoc, xgb)

    #param_grid = {
    #    'pca__n_components': [5, 10, 15, 20, 25, 30],
    #    'model__max_depth': [2, 3, 5, 7, 10],
    #    'model__n_estimators': [10, 100, 500],
    #}
    #grid = GridSearchCV(pipeline, param_grid,
    #  cv=5, n_jobs=-1, scoring='roc_auc')
else:
    raise ValueError('wrong regression value')
H = LinearRegression(fit_intercept=False, n_jobs=n_jobs)
b2b = B2B_SPoC(G=G, H=H, n_splits=n_splits_B2B,
        each_fit_is_parallel=b2b_each_fit_is_parallel)
# Cross-validation
cv = KFold(nb_fold, shuffle=True)


ep = env2epochs[env]
analysis_value = env2pre_dec_data[env]

#values_for_es,analysis_value = getErrSensVals(errors_cur,targets_cur,movement_cur)
#Y_es = np.array(values_for_es)
#Y_es = Y_es[:, non_hit]
#Y_es

X = ep.pick_types(meg=True, ref_meg=False)._data
wh = (times > -0.48) & (times < 0.01)
X = X[non_hit]
X = X[:, :, wh]
Y = np.array(analysis_value)
Y = Y[:, non_hit].T
#es = ((Y_es[0]-Y_es[2]) - (Y_es[1]-Y_es[3]))/Y_es[4]
#corr = ((Y_es[0]-Y_es[2]) - (Y_es[1]-Y_es[3]))
# Classic decoding
# Decod only prev_errors
y = Y[:, 3]
y_preds = np.zeros(len(y))
scores = list()
print('Starting CV')
for train, test in cv.split(X, y):
    est.fit(X[train], y[train])
    y_preds[test] = est.predict(X[test])
    score = spearmanr(y_preds[test], y[test])
    scores.append(score[0])
diff = np.abs(y - y_preds)

# run partial decoding
# Y = scale(Y)
# ensemble = ShuffleSplit(n_splits=20, test_size=.5)
# H_hats = list()
# for G_set, H_set in ensemble.split(Y[:, 0], X):
#     Y_hats = list()
#     for ii in range(Y.shape[1]):
#         y = Y[:, ii]
#         Y_hat = G.fit(X[G_set], y[G_set]).predict(X)
#         Y_hats.append(Y_hat)
#     Y_hats = np.array(Y_hats).T
#     H_hat = H.fit(Y[H_set], Y_hats[H_set]).coef_
#     H_hats.append(H_hat)
# partial_scores = np.diag(np.mean(H_hats, 0))
# save scores
fname = op.join(results_folder,
                f'{regression_type}_{env}_preverror_{freq_name}.npy')
np.save(fname, np.array(y))
fname = op.join(results_folder,
                f'{regression_type}_{env}_diff_{freq_name}.npy')
np.save(fname, np.array(diff))
fname = op.join(results_folder,
                f'{regression_type}_{env}_scores_{freq_name}.npy')
np.save(fname, np.array(scores))
print(f'Saved {fname}')

###########################################


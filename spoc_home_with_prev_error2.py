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
from config2 import path_data
from xgboost import XGBRegressor

from datetime import datetime  as dt
print(f'__START: {__file__} subj={subject}, hpass={hpass}, regression_type={regression_type} at {dt.now()}')


mne.cuda.init_cuda()
n_jobs_MNE='cuda'
n_jobs_XGB=n_jobs

target_coords = calc_target_coordinates_centered(target_angs)

#subject = sys.argv[1]
#hpass = 'no_filter'  # '0.1', no_hpass, no_filter
output_folder = f'spoc_home2_{hpass}'

nb_fold = 6
n_splits_B2B = 30

freq_names = ['broad', 'theta', 'alpha', 'beta', 'gamma']
freqs = [(4, 60), (4, 7), (8, 12), (13, 30), (31, 60)]

results_folder = op.join(path_data,subject,'results',output_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

ICAstr = 'with_ICA'  # or empty string
b2b_each_fit_is_parallel = False

time_locked = 'target'
# home position
tmin = -0.5; tmax = 0

#regression_type = 'Ridge'
#regression_type = 'xgboost'


if b2b_each_fit_is_parallel:
    add_clf_creopts = getXGBparams(n_jobs = None)
else:
    add_clf_creopts = getXGBparams(n_jobs = 1)

##############################
task = 'VisuoMotor'

fname = op.join(path_data, subject, 'behavdata', f'err_sens_{task}.npz')
f = np.load(fname, allow_pickle=True)['arr_0'][()]
#env2corr      = f['env2corr'][()]
#env2err_sens      = f['env2err_sens'][()]
#env2pre_err_sens  = f['env2pre_err_sens'][()]
env2pre_dec_data  = f['env2pre_dec_data']
env2non_hit  = f['env2non_hit']


# task = 'LocaError_'  # 'VisuoMotor_' or 'LocaError_'
# time_locked = 'target'  # 'target' or 'reach'
# Read raw to extra MEG event triggers
for freq_name, freq in zip(freq_names, freqs):
    print(freq_name + '--------------------------')
    # read behav data
    fname = op.join(path_data, subject, 'behavdata',
                    f'behav_{task}_df.pkl' )
    behav_df = pd.read_pickle(fname)
    # read raw data
    run = list()
    files = os.listdir(op.join(path_data, subject))
    run.extend(([op.join(
                path_data, subject + '/')
                + f for f in files if task in f]))
    fname_raw = run[0]
    # get raw and events from raw file
    # raw = read_raw_ctf(fname_raw, preload=True, system_clock='ignore')
    raw = read_raw_fif(op.join(path_data, subject, f'raw_{task}_{hpass}_{ICAstr}.fif' ),
            preload=True)
    events = mne.find_events(raw, stim_channel='UPPT001',
                             min_duration=0.02)
    events[:, 0] += 18  # to account for delay between trig. & photodi.

    raw.filter(freq[0], freq[1], n_jobs=n_jobs_MNE)

    epochs = Epochs(raw, events, event_id=[20, 21, 22, 23, 25, 26, 27, 28],
                    tmin=tmin, tmax=tmax, preload=True,
                    baseline=None, decim=2)

    env2epochs=dict(stable=epochs['20', '21', '22', '23', '30'],
            random=epochs['25', '26', '27', '28', '35'] )

    # Define variables ---------------------
    # Targets position
    #environment = np.array(behav_df['environment'])
    #stable = np.where(environment == 0)[0]
    #random = np.where(environment == 1)[0]
    #targets = np.array(behav_df['target'])
    #targets_stable = targets[stable]
    #targets_random = targets[random]
    ## Feedback positions
    #feedback = np.array(behav_df['feedback'])
    #feedback_stable = feedback[stable]
    #feedback_random = feedback[random]
    #feedbackX = np.array(behav_df['feedbackX'])
    #feedbackX_stable = feedbackX[stable]
    #feedbackX_random = feedbackX[random]
    #feedbackY = np.array(behav_df['feedbackY'])
    #feedbackY_stable = feedbackY[stable]
    #feedbackY_random = feedbackY[random]
    ## Movement positions
    #movement = np.array(behav_df['org_feedback'])
    #movement_stable = movement[stable]
    #movement_random = movement[random]
    ## Error positions
    #errors = np.array(behav_df['error'])
    #errors_stable = errors[stable]
    #errors_random = errors[random]
    #abs_errors_stable = np.abs(errors_stable)
    #abs_errors_random = np.abs(errors_random)
    ## Belief
    #belief = np.array(behav_df['belief'])
    #belief_stable = belief[stable]
    #belief_random = belief[random]
    # keep only non_hit trials
    #non_hit_stable = point_in_circle(targets_stable, target_coords,
    #                                 feedbackX_stable, feedbackY_stable,
    #                                 radius_target + radius_cursor)
    #non_hit_random = point_in_circle(targets_random, target_coords,
    #                                 feedbackX_random, feedbackY_random,
    #                                 radius_target + radius_cursor)

    times = epochs.times

    analysis_name = 'prevmovement_preverrors_errors_prevbelief'

    n_jobs_SPoC = n_jobs
    # Regression for classic decoding
    alphas = np.logspace(-5, 5, 12)
    spoc = SPoC(n_components=5, log=True, reg='oas',
                                 rank='full', n_jobs=n_jobs_SPoC)
    if regression_type == 'Ridge':
        est = make_pipeline(spoc, RidgeCV(alphas=alphas))
        # Regressions for the B2B
        G = make_pipeline(spoc, RidgeCV(alphas=alphas, fit_intercept=False))
    elif regression_type == 'xgboost':
        xgb = XGBRegressor(**add_clf_creopts)
        est = make_pipeline(spoc, xgb)
        # Regressions for the B2B
        G = make_pipeline(spoc, xgb)
    else:
        raise ValueError('wrong regression value')

    H = LinearRegression(fit_intercept=False)
    #G = direct pipeline (spoc + regerssor)
    #H = back pipeline
    b2b = B2B_SPoC(G=G, H=H, n_splits=n_splits_B2B, 
            each_fit_is_parallel=b2b_each_fit_is_parallel)
    # Cross-validation
    cv = KFold(nb_fold, shuffle=True)

    #  Run decoding
    for env in ['stable', 'random']:
        non_hit = env2non_hit[env]
        analysis_value = env2pre_dec_data[env]

        ep = env2epochs[env]

        X = ep.pick_types(meg=True, ref_meg=False)._data
        wh = (times > -0.45) & (times < 0.05)
        X = X[non_hit]  # trial x channel x time
        X = X[:, :, wh]
        Y = np.array(analysis_value)
        Y = Y[:, non_hit].T

        # Classic decoding
        scoring = make_scorer(scorer_spearman)
        scores = list()
        # over all trials
        for ii in range(Y.shape[1]):
            y = Y[:, ii]
            score = cross_val_multiscore(est, X, y=y, cv=cv, scoring=scoring,
                                         n_jobs=n_jobs)
            score = score.mean(axis=0)
            scores.append(score)
        scores = np.array(scores)

        #X: trialx x MEG x time
        #Y: trials x vals

        # run partial decoding
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

        print(f'Finished saving {fn}')

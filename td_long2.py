import os
import os.path as op
import numpy as np
import mne
from mne.io import read_raw_fif
from mne.decoding import (cross_val_multiscore, LinearModel, SlidingEstimator)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.metrics import make_scorer
from jr.gat import scorer_spearman, AngularRegression
from mne import Epochs
from mne.time_frequency import psd_multitaper
import pandas as pd
from Levenshtein import editops
import warnings
from tqdm import tqdm
import sys
from base2 import (int_to_unicode, point_in_circle, getXGBparams,
                   calc_target_coordinates_centered, radius_target,
                   radius_cursor, B2B, 
                   event_ids, env2envcode, env2subtr, target_angs)
from error_sensitivity import getAnalysisData
from config2 import path_data, subjects
from xgboost import XGBRegressor
task = 'VisuoMotor'

#target_angs = (np.array([157.5, 112.5, 67.5, 22.5]) + 90) * \
#              (np.pi/180)
target_coords = calc_target_coordinates_centered(target_angs)

#subject = sys.argv[1]
#hpass = '0.1'  # '0.1', '0.05', no_hpass
is_short = True

ICA = 'with_ICA'  # or empty string
if hpass == 'no_hpass':
    bsl = True
else:
    bsl = False

if is_short:
    output_folder = f'td_long2_{hpass}_short_{ICA}'
else:
    output_folder = f'td_long2_{hpass}_{ICA}'

nb_fold = 6
n_splits_B2B = 30

add_clf_creopts = getXGBparams()

control_type = 'movement'  # 'movement', 'feedback', 'target' or 'belief'

results_folder = op.join(path_data,subject,'results',output_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# read behav and meg data
fname = op.join(path_data, subject, 'behavdata',
                f'behav_{task}_df.pkl' )
behav_df = pd.read_pickle(fname)
raw = read_raw_fif(op.join(path_data, subject,
                   f'raw_{task}_{hpass}_{ICA}.fif' ),
                   preload=True)
events = mne.find_events(raw, stim_channel='UPPT001',
                         min_duration=0.02)
events[:, 0] += 18  # to account for delay between trig. & photodi.
if is_short:
    if bsl:
        epochs_feedback = Epochs(raw, events, event_id=[30, 35],
                                 tmin=-0.2, tmax=3, preload=True,
                                 baseline=(-0.2, 0), decim=6)
    else:
        epochs_feedback = Epochs(raw, events, event_id=[30, 35],
                                 tmin=-0.2, tmax=3, preload=True,
                                 baseline=None, decim=6)
    epochs_feedback.pick_types(meg=True, misc=False)
else:
    if bsl:
        epochs_feedback = Epochs(raw, events, event_id=[30, 35],
                                 tmin=-2, tmax=5, preload=True,
                                 baseline=None, decim=6)
        epochs_target = Epochs(raw, events, event_id=[20, 21, 22, 23, 25, 26, 27, 28],
                               tmin=-5, tmax=2, preload=True,
                               baseline=None, decim=6)
        epochs_bsl = Epochs(raw, events, event_id=[20, 21, 22, 23, 25, 26, 27, 28],
                            tmin=-0.46, tmax=-0.05, preload=True,
                            baseline=None, decim=6)
        # Apply baseline before the target to the epochs time-locked on feedback
        bsl_channels = mne.pick_types(epochs_feedback.info, meg=True)
        bsl_data = epochs_bsl.get_data()[:, bsl_channels, :]
        bsl_data = np.mean(bsl_data, axis=2)
        epochs_feedback._data[:, bsl_channels, :] -= bsl_data[:, :, np.newaxis]
        # Apply baseline before the target to the epochs time-locked on targets
        # we use the baseline from trials n-1 (the first trial is remove in
        # subsequent analysis)
        epochs_target._data[1:, bsl_channels, :] -= bsl_data[:-1, :, np.newaxis]
    else:
        epochs_feedback = Epochs(raw, events, event_id=[30, 35],
                                 tmin=-2, tmax=5, preload=True,
                                 baseline=None, decim=6)
        epochs_target = Epochs(raw, events, event_id=[20, 21, 22, 23, 25, 26, 27, 28],
                               tmin=-5, tmax=2, preload=True,
                               baseline=None, decim=6)
    epochs_feedback.pick_types(meg=True, misc=False)
    epochs_target.pick_types(meg=True, misc=False)


clf  = RidgeCV()
clf2 = RidgeCV(fit_intercept=False)

# Estimators for classic decoding
kwargs = dict()
pipeline = make_pipeline(StandardScaler(), LinearModel(clf))
est = SlidingEstimator(pipeline,
                       scoring=make_scorer(scorer_spearman),
                       n_jobs=n_jobs, **kwargs)
# Estimators for partial decoding
G = make_pipeline(clf2)
H = LinearRegression(fit_intercept=False)

b2b = B2B(G=G, H=H, n_splits=n_splits_B2B)

cv = KFold(nb_fold, shuffle=True)


if is_short:
    epochs_type = zip(['feedback'],
                      [epochs_feedback])
else:
    epochs_type = zip(['feedback', 'target'],
                      [epochs_feedback, epochs_target])

#  Run decoding
for (time_locked, epochs) in epochs_type:
    env2epochs={'all':epochs['20', '21', '22', '23', '30', 
        '25', '26', '27', '28', '35'],
            'stable':epochs['20', '21', '22', '23', '30'],
            'random':epochs['25', '26', '27', '28', '35'] }

    for env in ['all', 'stable', 'random']:
        print(f'Starting {env} --------------------------------------------')

        print(f'starting time_locked={time_locked}')
        ep = env2epochs[env]

        r = getAnalysisData(env, time_locked, control_type, behav_df)
        analysis_name, analysis_value, non_hit_cur = r

        X = ep.pick_types(meg=True, ref_meg=False)._data
        X = X[non_hit_cur]
        Y = np.array(analysis_value)
        Y = Y[:, non_hit_cur].T

        scores = list()
        # Classic decoding using SlidingEstimator
        for ii in tqdm(range(Y.shape[1])):
            y = Y[:, ii]
            score = cross_val_multiscore(est, X, y=y, cv=cv, n_jobs=n_jobs)
            score = score.mean(axis=0)
            scores.append(score)
        scores = np.array(scores)
        # Partial decoding
        partial_scores = list()
        Y = scale(Y)

        print('---- Starting computing partial scores')
        for ii in tqdm(range(X.shape[-1])):
            Xii = scale(X[:, :, ii])
            b2b.fit(Xii, Y)
            partial_scores.append(np.diag(b2b.E_))
        partial_scores = np.array(partial_scores).T
        # save scores
        fname = op.join(results_folder,
                f'{env}_{regression_type}_scores_{time_locked}_{analysis_name}.npy')
        np.save(fname, np.array(scores))

        fname = op.join(results_folder, 
                f'{env}_{regression_type}_partial_scores_{time_locked}_{analysis_name}.npy')
        #% (subject, env, , analysis_name))
        np.save(fname, np.array(partial_scores))

        print(f'Saved {fname}')

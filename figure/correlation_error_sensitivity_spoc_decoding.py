import os
import os.path as op
import numpy as np
from scipy.stats import ttest_rel, linregress
import mne
from mne.io import read_raw_fif
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from mne import Epochs
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from base2 import (int_to_unicode,
                   init_target_positions, point_in_circle,
                   calc_target_coordinates_centered, radius, radius_target,
                   radius_cursor, partial_reg, B2B)
from config2 import path_data, subjects
sns.set_palette('colorblind')
plt.style.use('seaborn')
color_palette = sns.color_palette("colorblind", 8).as_hex()
colors = [color_palette[1], color_palette[7]]
my_pal = {'Stable': colors[0], 'Random': colors[1]}

target_angs = (np.array([157.5, 112.5, 67.5, 22.5]) + 90) * \
              (np.pi/180)
target_coords = calc_target_coordinates_centered(target_angs)
path_data = '/Volumes/data/MemErrors/data2/'

save_folder = 'corr_es_SPoC_decod'
if not op.exists('/Users/quentinra/Desktop/figs_memerror/%s' % save_folder):
    os.mkdir('/Users/quentinra/Desktop/figs_memerror/%s' % save_folder)

# Get the error sensitivity measure
ind_es_stable = list()
ind_es_random = list()
all_es_stable = list()
all_es_random = list()
mean_es_stable = list()
mean_es_random = list()
ind_errors_stable = list()
ind_errors_random = list()
mean_errors_stable = list()
mean_errors_random = list()
nb_sub = len(subjects)
for subject in subjects:
    task = 'VisuoMotor'  # 'VisuoMotor_' or 'LocaError_'

    folder = op.join(path_data, subject, 'behavdata')
    fname = op.join(folder, f'behav_{task}_df.pkl' )
    behav_df = pd.read_pickle(fname)

    # Perturbations
    perturbations = np.array(behav_df['perturbation'])
    # Environment
    environment = np.array(behav_df['environment']).astype(int)
    # Targets position
    targets = np.array(behav_df['target'])
    prev_targets = np.insert(targets, 0, 0)[:-1]
    # Feedback positions
    feedback = np.array(behav_df['feedback'])
    prev_feedback = np.insert(feedback, 0, 0)[:-1]
    feedbackX = np.array(behav_df['feedbackX'])
    feedbackY = np.array(behav_df['feedbackY'])
    # Movement positions
    movement = np.array(behav_df['org_feedback'])
    prev_movement = np.insert(movement, 0, 0)[:-1]
    # Error positions
    errors = np.array(behav_df['error'])
    prev_errors = np.insert(errors, 0, 0)[:-1]
    # keep only non_hit trials
    non_hit = point_in_circle(targets, target_coords, feedbackX,
                              feedbackY,
                              radius_target + radius_cursor)
    abs_errors = np.abs(errors)

    analyses_value = [target_angs[targets], target_angs[prev_targets],
                      movement, prev_movement,
                      prev_errors, environment, perturbations, errors]

    Y = np.array(analyses_value)
    non_hit = np.array(non_hit)
    # remove trials following hit (because no previous error)
    # non_hit = ~(~non_hit | ~np.insert(non_hit, 0, 1)[:-1])
    non_hit = np.insert(non_hit, 0, 0)[:-1]
    # remove first trials (because no previous error)
    first_trials = np.where(behav_df['trials'] == 0)
    non_hit[first_trials] = False
    Y = Y[:, non_hit]
    # Error positions
    errors_stable = Y[7][np.where(Y[5] == 0)]
    errors_random = Y[7][np.where(Y[5] == 1)]
    prev_errors_stable = Y[4][np.where(Y[5] == 0)]
    prev_errors_random = Y[4][np.where(Y[5] == 1)]
    ind_errors_stable.append(errors_stable)
    ind_errors_random.append(errors_random)
    mean_errors_stable.append(errors_stable.mean())
    mean_errors_random.append(errors_random.mean())
    # Compute error sensitivity
    es = ((Y[0]-Y[2]) - (Y[1]-Y[3]))/Y[4]
    es_stable = es[np.where(Y[5] == 0)]
    es_random = es[np.where(Y[5] == 1)]
    all_es_stable.extend(es_stable)
    all_es_random.extend(es_random)
    ind_es_stable.append(es_stable)
    ind_es_random.append(es_random)
    mean_es_stable.append(es_stable.mean())
    mean_es_random.append(es_random.mean())
all_es_stable = np.array(all_es_stable)
all_es_random = np.array(all_es_random)
mean_es_stable = np.array(mean_es_stable)
mean_es_random = np.array(mean_es_random)
mean_errors_stable = np.array(mean_errors_stable)
mean_errors_random = np.array(mean_errors_random)

# Get the SPoC decoding measure
hpass = 'no_filter'  # '0.1', 'detrend', no_hpass
output_folder = 'spoc_home2_%s' % hpass
analysis_name = 'prevmovement_preverrors_errors_prevbelief'
analyses = ['Prev_movement', 'Prev_errors', 'Errors', 'Prev_belief']

time_locked = 'target'
freqs = ['broad', 'theta', 'alpha', 'beta', 'gamma']
environment = ['stable', 'random']
control = 'b2b'  # 'classic'
for freq_name in freqs:
    all_scores_stable = list()
    all_scores_random = list()
    for subject in subjects:
        # results_folder = 'decoding_no_hpass_no_bsl'
        results_folder = output_folder
        for env in environment:
            if control == 'classic':
                fname = '%s_%sscores_%s_%s.npy' % (subject,
                                                   env,
                                                   analysis_name,
                                                   freq_name)
            if control == 'b2b':
                fname = '%s_%spartial_scores_%s_%s.npy' % (subject,
                                                           env,
                                                           analysis_name,
                                                           freq_name)
            sc = np.load(op.join(path_data, subject, 'results/',
                                 results_folder, fname))
            if env == 'stable':
                all_scores_stable.append(sc)
            elif env == 'random':
                all_scores_random.append(sc)
    all_scores_stable = np.array(all_scores_stable)
    all_scores_random = np.array(all_scores_random)

    nb_sub = len(subjects)
    scores_stable = np.ravel(all_scores_stable, order='F')
    scores_random = np.ravel(all_scores_random, order='F')
    scores = np.concatenate((scores_stable, scores_random), axis=0)
    type = 2 * ([analyses[0]] * nb_sub + [analyses[1]] * nb_sub + [analyses[2]] *
                nb_sub + [analyses[3]] * nb_sub)
    cond = ['Stable'] * 4 * nb_sub + ['Random'] * 4 * nb_sub
    data = pd.DataFrame({'Decoding Performance': scores,
                        'Condition': cond, 'Type': type})
    my_pal = {'Stable': colors[0], 'Random': colors[1]}

    prev_error_data = data[data['Type'] == 'Prev_errors']
    mean_preverror_stable = prev_error_data['Decoding Performance'][prev_error_data['Condition']=='Stable']
    mean_preverror_random = prev_error_data['Decoding Performance'][prev_error_data['Condition']=='Random']

    # Plot correlation
    all_preverror = np.concatenate((np.array(mean_preverror_stable),
                                   np.array(mean_preverror_random)))
    all_es = np.concatenate((mean_es_stable, mean_es_random))
    # Compute linear regression
    slope_all, intercept_all, r_all, p_all, _ = linregress(all_preverror,
                                                           all_es)
    line_all = slope_all*all_preverror+intercept_all
    slope_stable, intercept_stable, r_stable, p_stable, _ = linregress(mean_preverror_stable,
                                                                       mean_es_stable)
    line_stable = slope_stable*mean_preverror_stable+intercept_stable
    slope_random, intercept_random, r_random, p_random, _ = linregress(mean_preverror_random,
                                                                       mean_es_random)
    line_random = slope_random*mean_preverror_random+intercept_random
    # Plot dot for stable and random
    plt.plot(mean_preverror_stable, mean_es_stable, 'o', color=colors[0])
    plt.plot(mean_preverror_random, mean_es_random, 'o', color=colors[1])
    # Plot line for stable and random
    plt.plot(mean_preverror_stable, line_stable, color=colors[0])
    plt.plot(mean_preverror_random, line_random, color=colors[1])
    # Plot line for all
    min = np.min(all_preverror)
    max = np.max(all_preverror)
    aa = np.arange(min, max, (max-min)/10.)
    np.ravel([colors]*5)
    for aa1, aa2, color in zip(aa[:-1], aa[1:], np.ravel([colors]*5)):
        wh = np.where((all_preverror > aa1) & (all_preverror < aa2))
        plt.plot(all_preverror[wh], line_all[wh], color=color)

import os
import os.path as op
import numpy as np
from scipy.stats import ttest_rel, linregress
from scipy.signal import savgol_filter
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


# Get the error sensitivity measure
all_es = list()
all_perturbations = list()
all_environment = list()
mean_es_stable = list()
mean_es_random = list()
all_es_stable = list()
all_es_random = list()
all_es_stable2 = list()
all_es_random2 = list()
for ii, subject in enumerate(subjects):
    task = 'VisuoMotor_'  # 'VisuoMotor_' or 'LocaError_'

    fname = op.join(path_data, subject, 'behavdata',
                    'behav_%sdf.pkl' % task)
    behav_df = pd.read_pickle(fname)
    trials = behav_df['trials']
    # Perturbations
    perturbations = np.array(behav_df['perturbation'])
    # Environment
    environment = np.array(behav_df['environment']).astype(int)
    # Targets position
    targets = np.array(behav_df['target_inds'])
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
    # Compute error sensitivity
    Y[4][0] = np.nan
    es = ((Y[0]-Y[2]) - (Y[1]-Y[3]))/Y[4]
    perturbations = perturbations * (np.pi/90)
    environment_for_plot = environment.copy()
    perturbations_for_plot = perturbations.copy()
    environment = environment.astype(float)

    es[~non_hit] = np.nan
    perturbations[~non_hit] = np.nan
    environment[~non_hit] = np.nan


    es_stable = es[np.where(environment==0)]
    es_random = es[np.where(environment==1)]
    mean_es_stable.append(es_stable.mean())
    mean_es_random.append(es_random.mean())


    sys.exit(0)

    #mean_es_stable.append(np.nanmean(es_stable))
    #mean_es_random.append(np.nanmean(es_random))

    all_es_stable.extend(es_stable)
    all_es_random.extend(es_random)

    all_es_stable2.append(es_stable)
    all_es_random2.append(es_random)

    all_es.append(es)
    all_perturbations.append(perturbations)
    all_environment.append(environment)

mean_es_stable = np.array(mean_es_stable)
mean_es_random = np.array(mean_es_random)
all_es_stable = np.array(all_es_stable)
all_es_random = np.array(all_es_random)

mean_es = np.nanmean(all_es, axis=0)

# plot mean error-sensitivity (av. accross time then participants)
data = {'stable': mean_es_stable, 'random': mean_es_random}
data_df = pd.DataFrame(data)
sns.violinplot(data=data_df)
t, p = ttest_rel(mean_es_stable, mean_es_random)
#plt.savefig('/Users/romainquentin/Desktop/data/MemErrors/behavioral_figures/es_mean.png')
#plt.close()

plt.figure()
# plot error sensitivity accross trials (average accross participants)
plt.plot(trials, mean_es, linewidth=0.6, label='err_sens')
plt.plot(trials, perturbations_for_plot, linewidth=0.8, label='perturb')
plt.plot(trials, environment_for_plot, linewidth=0.8, label='env')
# Smooth es
kernel_size = 10
kernel = np.ones(kernel_size) / kernel_size
mean_es_smooth = np.convolve(mean_es, kernel, mode='same')
plt.plot(trials, mean_es_smooth, linewidth=1, label='err_sens_smooth')
plt.ylim(-4, 4)
plt.tight_layout()
plt.legend()
#plt.savefig('/Users/romainquentin/Desktop/data/MemErrors/behavioral_figures/es_mean_dynamic.png')
#plt.close()


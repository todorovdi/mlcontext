import os
import os.path as op
import numpy as np
from scipy.stats import ttest_rel
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
from config2 import path_data, path_fig, subjects
sns.set_palette('colorblind')
plt.style.use('seaborn')
color_palette = sns.color_palette("colorblind", 8).as_hex()
colors = [color_palette[1], color_palette[7]]
my_pal = {'Stable': colors[0], 'Random': colors[1]}

target_angs = (np.array([157.5, 112.5, 67.5, 22.5]) + 90) * \
              (np.pi/180)
target_coords = calc_target_coordinates_centered(target_angs)
#path_data = '/Volumes/data/MemErrors/data2/'


save_folder = 'error_sensitivity'
if not op.exists( op.join(path_fig, save_folder) ):
    os.mkdir(op.join(path_fig, save_folder) )

task = 'VisuoMotor'  # 'VisuoMotor_' or 'LocaError_'

all_es_stable = list()
all_es_random = list()
mean_es_stable = list()
mean_es_random = list()
nb_sub = len(subjects)
for subject in subjects:
    folder = op.join(path_data, subject, 'behavdata')
    fname = op.join(folder,f'behav_{task}_df.pkl')
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
                      prev_errors, environment, perturbations]

    Y = np.array(analyses_value)
    non_hit = np.array(non_hit)
    # remove trials following hit (because no previous error)
    # non_hit = ~(~non_hit | ~np.insert(non_hit, 0, 1)[:-1])
    non_hit = np.insert(non_hit, 0, 0)[:-1]
    # remove first trials (because no previous error)
    non_hit[[0, 192, 384, 576]] = False  # Removing first trial of each block
    Y = Y[:, non_hit]
    # Compute error sensitivity
    es = ((Y[0]-Y[2]) - (Y[1]-Y[3]))/Y[4]
    es_stable = es[np.where(Y[5] == 0)]
    es_random = es[np.where(Y[5] == 1)]
    all_es_stable.extend(es_stable)
    all_es_random.extend(es_random)
    mean_es_stable.append(es_stable.mean())
    mean_es_random.append(es_random.mean())
all_es_stable = np.array(all_es_stable)
all_es_random = np.array(all_es_random)
mean_es_stable = np.array(mean_es_stable)
mean_es_random = np.array(mean_es_random)

all_es = np.append(all_es_stable, all_es_random)
mean_es = np.append(mean_es_stable, mean_es_random)

type = ['error sensitivity'] * len(mean_es)
cond = ['Stable'] * len(mean_es_stable) + ['Random'] * len(mean_es_random)
data = pd.DataFrame({'Decoding Performance': mean_es,
                    'Condition': cond, 'Type': type})
fig, ax = plt.subplots(figsize=(2.5, 2.35))
ax = sns.violinplot(x='Type', y='Decoding Performance', hue='Condition',
                    data=data, saturation=1, palette=my_pal, split=True,
                    linewidth=0.5, inner='stick')
ax.axhline(y=0, linestyle='dotted', color='k', linewidth=0.8)
ax.set_ylim(-0.4, 0.8)
ax.yaxis.set_label_text('Error Sensitivity')
ax.xaxis.set_label_text('')
ax.get_legend().set_visible(False)
ypos = data.groupby(['Type'])['Decoding Performance'].mean().tolist()
yadd = data.groupby(['Type'])['Decoding Performance'].std().tolist()
xpos = range(len(ypos))
# ax.set_title('Decoding previous error')
plt.tight_layout()
fname_fig = op.join(path_fig,save_folder,'es_violin')
ax.figure.savefig(fname_fig , dpi=400)
plt.close()

print(f'Plotting finished, saved to {fname_fig}')

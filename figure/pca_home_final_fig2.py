import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from config2 import subjects, path_fig, path_data
from base2 import decod_stats, gat_stats
from jr.plot import share_clim
from jr import OnlineReport
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import ttest_1samp
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_rel
sns.set_palette('colorblind')
plt.style.use('seaborn')
color_palette = sns.color_palette("colorblind", 8).as_hex()
colors = [color_palette[1], color_palette[7]]

hpass = '0.1'  # '0.1', 'detrend', no_hpass
output_folder = 'spoc_home2_%s' % hpass
analys = 'tmep'  # or 'fmep'
save_folder = 'pca2_%s' % analys
if not op.exists( op.join(path_fig, save_folder) ):
    os.mkdir(op.join(path_fig, save_folder) )

task = 'VisuoMotor'
time_locked = 'target'
environment = ['stable', 'random']
control = 'b2b'  # 'classic'
# control = 'classic'  # 'classic'
if analys is 'tmep':
    analysis_name = 'target_movement_errors_preverrors'
if analys is 'fmep':
    analysis_name = 'feedback_movement_errors_preverrors'
all_scores_stable = list()
all_scores_random = list()
for subject in subjects:
    # results_folder = 'decoding_no_hpass_no_bsl'
    results_folder = output_folder
    for env in environment:
        if control is 'classic':
            fname = '%s_%sscores_%s.npy' % (subject,
                                            env,
                                            analysis_name)
        if control is 'b2b':
            fname = '%s_%spartial_scores_%s.npy' % (subject,
                                                    env,
                                                    analysis_name)
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
type = 2 * (['Target'] * nb_sub + ['Movement'] * nb_sub + ['Error'] *
            nb_sub + ['Previous Errors'] * nb_sub)
cond = ['Stable'] * 4 * nb_sub + ['Random'] * 4 * nb_sub
data = pd.DataFrame({'Decoding Performance': scores,
                    'Condition': cond, 'Type': type})
my_pal = {'Stable': colors[0], 'Random': colors[1]}
fig, ax = plt.subplots()
ax = sns.violinplot(x='Type', y='Decoding Performance', hue='Condition',
                    data=data,
                    inner='stick', palette=my_pal, split=True)
ax.axhline(y=0, linestyle='dotted', color='k', linewidth=0.8)
ax.set_ylim(-0.08, 0.2)
ypos = data.groupby(['Type'])['Decoding Performance'].mean().tolist()
yadd = data.groupby(['Type'])['Decoding Performance'].std().tolist()
xpos = range(len(ypos))
for ii in range(all_scores_stable.shape[1]):
    t, p = ttest_1samp(all_scores_stable[:, ii], 0)
    if p < 0.001:
        ax.text(xpos[ii] - 0.5, 0.08, 'p < 0.001', color=colors[0],
                fontsize=9)
    else:
        ax.text(xpos[ii] - 0.5, 0.08, 'p = %.3f' % p, color=colors[0],
                fontsize=9)
    t, p = ttest_1samp(all_scores_random[:, ii], 0)
    if p < 0.001:
        ax.text(xpos[ii] + 0.05, 0.08, 'p < 0.001', color=colors[1],
                fontsize=9)
    else:
        ax.text(xpos[ii] + 0.05, 0.08, 'p = %.3f' % p, color=colors[1],
                fontsize=9)
    t, p = ttest_rel(all_scores_stable[:, ii], all_scores_random[:, ii])
    if p < 0.001:
        ax.text(xpos[ii], -0.07, 'p < 0.001', fontsize=9)
    else:
        ax.text(xpos[ii], -0.07, 'p = %.3f' % p,
                fontsize=9)
ax.set_title(env)
ax.figure.savefig(op.join(path_fig, save_folder, control)

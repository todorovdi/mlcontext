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
from scipy.stats import ttest_1samp, ttest_rel
import seaborn as sns
import pandas as pd
sns.set_palette('colorblind')
plt.style.use('seaborn')
color_palette = sns.color_palette("colorblind", 8).as_hex()
colors = [color_palette[1], color_palette[7]]

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma']
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

path_data = '/Volumes/data/MemErrors/data2/'
hpass = '0.1'  # '0.1', 'detrend', no_hpass
output_folder = 'psd_long2_%s' % hpass
save_folder = 'Exp2_psd_long2_%s' % hpass
if not op.exists( op.join(path_fig, save_folder) ):
    os.mkdir(op.join(path_fig, save_folder) )

fname = op.join(path_data, subjects[0], 'results', output_folder,
                'times_feedback.npy')
times_feedback = np.load(fname)
fname = op.join(path_data, subjects[0], 'results', output_folder,
                'times_target.npy')
times_target = np.load(fname)

envs = ['all', 'stable', 'random']
control_type = 'feedback'  # 'movement' or 'feedback'
if control_type == 'feedback':
    analyses_name_feedback = 'feedback_errors_nexterrors_belief'
    analyses_name_target = 'prevfeedback_preverrors_errors_prevbelief'
if control_type == 'movement':
    analyses_name_feedback = 'movement_errors_nexterrors_belief'
    analyses_name_target = 'prevmovement_preverrors_errors_prevbelief'
time_lockeds = [['feedback', analyses_name_feedback,
                 times_feedback],
                ['target', analyses_name_target,
                times_target]]
controls = ['classic', 'b2b']
for env in envs:
    for time_locked in time_lockeds:
        times = time_locked[2]
        for control in controls:
            all_scores = list()
            for subject in subjects:
                results_folder = output_folder
                if control == 'classic':
                    fname = '%s_scores_%s_%s.npy' % (env,
                                                       time_locked[0],
                                                       time_locked[1])
                elif control == 'b2b':
                    fname = '%s_partial_scores_%s_%s.npy' % (env,
                                                               time_locked[0],
                                                               time_locked[1])
                sc = np.load(op.join(path_data, subject, 'results',
                                     results_folder, fname))
                all_scores.append(sc)
            all_scores = np.array(all_scores)
            all_scores = np.moveaxis(all_scores, 1, 0)
            # Plot
            colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C6']
            for ii, scores in enumerate(all_scores):
                plt.title('%s %s %s' % (env, control, time_locked[1]))
                plt.plot(times, scores.mean(0), color=colors[ii],
                         label=str(ii))
                sig = decod_stats(scores) < 0.05
                plt.fill_between(times, scores.mean(0),
                                 where=sig, color=colors[ii], alpha=0.3)
                plt.legend()
            plt.savefig(path_fig,save_folder,'Exp2_%s_%s_%s_%s' % (env, control, time_locked[0], time_locked[1]),
                        dpi=400)
            plt.close()

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

#path_data = '/Volumes/data/MemErrors/data2/'
hpass = 'no_hpass'  # '0.1', 'detrend', no_hpass
output_folder = 'td_with_prev_errors2_%s' % hpass
analys = 'tmep'  # or 'fmep'
save_folder = 'td_baselined2_%s' % analys
if not op.exists( op.join(path_fig, save_folder) ):
    os.mkdir(op.join(path_fig, save_folder) )

# output_folder = 'decoding_partial_source_%s_%s' % (baseline, hpass)
envs = ['stable', 'random']
time_lockeds = ['target', 'reach']
report = OnlineReport()
sfreq = 120
tmin = -0.3
time_locked = 'reach'  # or 'target'
n_stample_times = 20
if time_locked == 'reach':
    tmax, num = 1.8, 8
    sample_times = np.linspace(0, (tmax-tmin)*sfreq, int((tmax-tmin)*sfreq + 1) )
elif time_locked == 'target':
    tmax = 3.5
    sample_times = np.linspace(0, (tmax-tmin)*sfreq, int((tmax-tmin)*sfreq + 1) )
    tmax, num = 3.3, 13  # Use for plotting xticks
times = sample_times/sfreq + tmin
all_scores = list()
results_folder = output_folder
for env in envs:
    if analys == 'tmep':
        analysis_name = 'targets_movement_errors_preverrors'
    if analys == 'fmep':
        analysis_name = 'feedback_movement_errors_preverrors'
    for subject in subjects:
        fname = '%s_scores_%s_%s.npy' % (env,
                                            time_locked,
                                            analysis_name)
        scores = np.load(op.join(path_data, subject, 'results/',
                         results_folder, fname))
        all_scores.append(dict(scores=scores, env=env, partial='no'))
        fname = '%s_partial_scores_%s_%s.npy' % (env,
                                                    time_locked,
                                                    analysis_name)
        scores = np.load(op.join(path_data, subject, 'results/',
                         results_folder, fname))
        all_scores.append(dict(scores=scores, env=env, partial='e_hat'))
        fname = '%s_partial_scores_pred_%s_%s.npy' % (env,
                                                        time_locked,
                                                        analysis_name)
        scores = np.load(op.join(path_data, subject, 'results/',
                         results_folder, fname))
        all_scores.append(dict(scores=scores, env=env, partial='pred'))

all_scores = pd.DataFrame(all_scores)
chance = 0

# scores for VisuoMotor_
visuo = all_scores.loc[(all_scores['partial'] == 'no') &
                       (all_scores['env'] == 'stable')].scores
all_visuo = list()
all_visuo.extend(ii[1] for ii in visuo.items())
all_visuo = np.array(all_visuo)
# scores for LocaError_
loca = all_scores.loc[(all_scores['partial'] == 'no') &
                      (all_scores['env'] == 'random')].scores
all_loca = list()
all_loca.extend(ii[1] for ii in loca.items())
all_loca = np.array(all_loca)
# scores for VisuoMotor_ partial
visuo_partial = all_scores.loc[(all_scores['partial'] == 'e_hat') &
                               (all_scores['env'] == 'stable')].scores
all_visuo_partial = list()
all_visuo_partial.extend(ii[1] for ii in visuo_partial.items())
all_visuo_partial = np.array(all_visuo_partial)
# scores for LocaError_ partial
loca_partial = all_scores.loc[(all_scores['partial'] == 'e_hat') &
                              (all_scores['env'] == 'random')].scores
all_loca_partial = list()
all_loca_partial.extend(ii[1] for ii in loca_partial.items())
all_loca_partial = np.array(all_loca_partial)
# scores for VisuoMotor_ partial_pred
visuo_partial_pred = all_scores.loc[(all_scores['partial'] == 'pred') &
                                    (all_scores['env'] == 'stable')].scores
all_visuo_partial_pred = list()
all_visuo_partial_pred.extend(ii[1] for ii in visuo_partial_pred.items())
all_visuo_partial_pred = np.array(all_visuo_partial_pred)
# scores for LocaError_ partial
loca_partial_pred = all_scores.loc[(all_scores['partial'] == 'pred') &
                              (all_scores['env'] == 'random')].scores
all_loca_partial_pred = list()
all_loca_partial_pred.extend(ii[1] for ii in loca_partial_pred.items())
all_loca_partial_pred = np.array(all_loca_partial_pred)

for decoding_type, scores in zip(['classic', 'b2b', 'b2bpred'],
                       [[all_visuo, all_loca],
                        [all_visuo_partial, all_loca_partial],
                        [all_visuo_partial_pred, all_loca_partial_pred]]):
    # Plot errors
    chance = 0
    visuo = scores[0][:, 2, :]
    loca = scores[1][:, 2, :]
    fig_diag, axes = plt.subplots(figsize=(3.5, 2.5))
    if decoding_type == 'classic':
        axes.set_title('Decoding Error')
        plt.ylabel('Decoding Performance (r)')
    elif decoding_type == 'b2b':
        axes.set_title('Decoding Error (partial)')
        plt.ylabel('Decoding Performance (Ê)')
    plt.xlabel('Time (s)')
    plt.xticks(np.linspace(-.3, tmax, num))
    plt.ylim(-0.01, 0.17)
    axes.axhline(y=chance, linewidth=0.7, color='k', ls='dashed')
    axes.axvline(x=0, linewidth=0.7, color='k', ls='dashed')
    for name, color, ii in zip(['Stable (learning)', 'Random'],
                               colors[:2],
                               [visuo, loca]):
        sig = decod_stats(ii - chance) < 0.05
        sem = np.std(ii, axis=0)/np.sqrt(len(subjects))
        axes.fill_between(times,
                          np.array(np.mean(ii, axis=0))+(np.array(sem)),
                          np.array(np.mean(ii, axis=0))-(np.array(sem)),
                          color=color, alpha=0.4)
        axes.fill_between(times,
                          np.array(np.mean(ii, axis=0))+(np.array(sem)),
                          np.array(np.mean(ii, axis=0))-(np.array(sem)),
                          where=sig, color=color, alpha=0.8, label=name)
        axes.fill_between(times,
                          np.array(np.mean(ii, axis=0))+(np.array(sem)),
                          chance,
                          where=sig, color=color, alpha=0.4)
    # plt.legend()
    plt.tight_layout()
    plt.savefig(op.join(path_fig,save_folder,f'error_{decoding_type}_{time_locked}'),
                dpi=400)
    #plt.savefig('/Users/quentinra/Desktop/figs_memerror/%s/error_%s_%s.png' % (save_folder, decoding_type, time_locked),
    plt.close()
    # Plot target
    chance = 0
    visuo = scores[0][:, 0, :]
    loca = scores[1][:, 0, :]
    fig_diag, axes = plt.subplots()
    if decoding_type == 'classic':
        axes.set_title('Decoding Target')
        plt.ylabel('Decoding Performance (r)', fontsize='x-large')
    elif decoding_type == 'b2b':
        axes.set_title('Decoding Target (partial)')
        plt.ylabel('Decoding Performance (E_hat)', fontsize='x-large')
    plt.xlabel('Time', fontsize='x-large')
    plt.xticks(np.linspace(-.3, tmax, num))

    axes.axhline(y=chance, linewidth=0.7, color='k', ls='dashed')
    axes.axvline(x=0, linewidth=0.7, color='k', ls='dashed')
    for name, color, ii in zip(['Stable (learning)', 'Random'],
                               colors[:2],
                               [visuo, loca]):
        sig = decod_stats(ii - chance) < 0.05
        sem = np.std(ii, axis=0)/np.sqrt(len(subjects))
        axes.fill_between(times,
                          np.array(np.mean(ii, axis=0))+(np.array(sem)),
                          np.array(np.mean(ii, axis=0))-(np.array(sem)),
                          color=color, alpha=0.4)
        axes.fill_between(times,
                          np.array(np.mean(ii, axis=0))+(np.array(sem)),
                          np.array(np.mean(ii, axis=0))-(np.array(sem)),
                          where=sig, color=color, alpha=0.8, label=name)
        axes.fill_between(times,
                          np.array(np.mean(ii, axis=0))+(np.array(sem)),
                          chance,
                          where=sig, color=color, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(op.join(path_fig,save_folder,f'target_{decoding_type}_{time_locked}'))
    plt.close()
    # Plot movement
    chance = 0
    visuo = scores[0][:, 1, :]
    loca = scores[1][:, 1, :]
    fig_diag, axes = plt.subplots()
    if decoding_type == 'classic':
        axes.set_title('Decoding Movement')
        plt.ylabel('Decoding Performance (r)', fontsize='x-large')
    elif decoding_type == 'b2b':
        axes.set_title('Decoding Movement (partial)')
        plt.ylabel('Decoding Performance (E_hat)', fontsize='x-large')
    plt.xlabel('Time', fontsize='x-large')
    plt.xticks(np.linspace(-.3, tmax, num))

    axes.axhline(y=chance, linewidth=0.7, color='k', ls='dashed')
    axes.axvline(x=0, linewidth=0.7, color='k', ls='dashed')
    for name, color, ii in zip(['Stable (learning)', 'Random'],
                               colors[:2],
                               [visuo, loca]):
        sig = decod_stats(ii - chance) < 0.05
        sem = np.std(ii, axis=0)/np.sqrt(len(subjects))
        axes.fill_between(times,
                          np.array(np.mean(ii, axis=0))+(np.array(sem)),
                          np.array(np.mean(ii, axis=0))-(np.array(sem)),
                          color=color, alpha=0.4)
        axes.fill_between(times,
                          np.array(np.mean(ii, axis=0))+(np.array(sem)),
                          np.array(np.mean(ii, axis=0))-(np.array(sem)),
                          where=sig, color=color, alpha=0.8, label=name)
        axes.fill_between(times,
                          np.array(np.mean(ii, axis=0))+(np.array(sem)),
                          chance,
                          where=sig, color=color, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(op.join(path_fig,save_folder,f'movement_{decoding_type}_{time_locked}'))
    plt.close()
    # Plot previous errors
    chance = 0
    visuo = scores[0][:, 3, :]
    loca = scores[1][:, 3, :]
    fig_diag, axes = plt.subplots()
    if decoding_type == 'classic':
        axes.set_title('Decoding Previous Error')
        plt.ylabel('Decoding Performance (r)', fontsize='x-large')
    elif decoding_type == 'b2b':
        axes.set_title('Decoding Previous Error (partial)')
        plt.ylabel('Decoding Performance (E_hat)', fontsize='x-large')
    plt.xlabel('Time', fontsize='x-large')
    plt.xticks(np.linspace(-.3, tmax, num))

    axes.axhline(y=chance, linewidth=0.7, color='k', ls='dashed')
    axes.axvline(x=0, linewidth=0.7, color='k', ls='dashed')
    for name, color, ii in zip(['Stable (learning)', 'Random'],
                               colors[:2],
                               [visuo, loca]):
        sig = decod_stats(ii - chance) < 0.05
        sem = np.std(ii, axis=0)/np.sqrt(len(subjects))
        axes.fill_between(times,
                          np.array(np.mean(ii, axis=0))+(np.array(sem)),
                          np.array(np.mean(ii, axis=0))-(np.array(sem)),
                          color=color, alpha=0.4)
        axes.fill_between(times,
                          np.array(np.mean(ii, axis=0))+(np.array(sem)),
                          np.array(np.mean(ii, axis=0))-(np.array(sem)),
                          where=sig, color=color, alpha=0.8, label=name)
        axes.fill_between(times,
                          np.array(np.mean(ii, axis=0))+(np.array(sem)),
                          chance,
                          where=sig, color=color, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(op.join(path_fig,save_folder,f'prev_error_{decoding_type}_{time_locked}'))
    plt.close()

    # Plot diff
    chance = 0
    visuo = scores[0][:, 2, :]
    loca = scores[1][:, 2, :]
    diff = visuo - loca
    fig_diag, axes = plt.subplots(figsize=(3.5, 2))
    if decoding_type == 'classic':
        axes.set_title('Difference between Stable and Passive Random')
        plt.ylabel('Decoding Performance (r)')
    elif decoding_type == 'b2b':
        axes.set_title('Decoding Error (partial)')
        plt.ylabel('Decoding Performance (Ê)')
    plt.xlabel('Time (s)')
    plt.xticks(np.linspace(-.3, tmax, num))
    plt.ylim(-0.01, 0.25)
    axes.axhline(y=chance, linewidth=0.7, color='k', ls='dashed')
    axes.axvline(x=0, linewidth=0.7, color='k', ls='dashed')

    name = 'difference'
    color = colors[0]
    ii = diff
    sig = decod_stats(ii - chance) < 0.05
    sem = np.std(ii, axis=0)/np.sqrt(len(subjects))
    axes.fill_between(times,
                      np.array(np.mean(ii, axis=0))+(np.array(sem)),
                      np.array(np.mean(ii, axis=0))-(np.array(sem)),
                      color=color, alpha=0.4)
    axes.fill_between(times,
                      np.array(np.mean(ii, axis=0))+(np.array(sem)),
                      np.array(np.mean(ii, axis=0))-(np.array(sem)),
                      where=sig, color=color, alpha=1, label=name)
    axes.fill_between(times,
                      np.array(np.mean(ii, axis=0))+(np.array(sem)),
                      chance,
                      where=sig, color=color, alpha=0.4)
    # plt.legend()
    plt.tight_layout()
    fig_fname_full =op.join(path_fig,save_folder,f'diff_{decoding_type}_{time_locked}')
    plt.savefig(fig_fname_full, dpi=400)
    print(f'Fig saved to {fig_fname_full}')
    plt.close()

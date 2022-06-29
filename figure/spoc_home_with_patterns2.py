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
from mne import read_epochs, EvokedArray
from scipy.stats import ttest_1samp
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_rel
sns.set_palette('colorblind')
plt.style.use('seaborn')
color_palette = sns.color_palette("colorblind", 8).as_hex()
colors = [color_palette[1], color_palette[7]]

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

#pattern used to replace
#%s/'\/Users.*\/\([a-zA-Z_]*\)%.*'\s*%\s*(\(\w*\),\s*\(\w*\),\s*\(\w*\))/op.join(path_fig,\2,f'\1_{\3}_{\4}')/gc

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
hpass = '0.1'  # '0.1', 'detrend', no_hpass
output_folder = 'spoc_patterns2_%s' % hpass
analys = 'tmep'  # or 'fmep'
save_folder = 'SPoC_patterns2_%s' % analys
if not op.exists( op.join(path_fig, save_folder) ):
    os.mkdir(op.join(path_fig, save_folder) )

task = 'VisuoMotor_'
time_locked = 'target'
freqs = ['broad', 'theta', 'alpha', 'beta', 'gamma']
environment = ['stable', 'random']
control = 'b2b'  # 'classic'
control = 'classic'  # 'classic'
for freq_name in freqs:
    if analys == 'tmep':
        analysis_name = 'target_movement_errors_preverrors'
    if analys == 'fmep':
        analysis_name = 'feedback_movement_errors_preverrors'
    all_scores_stable = list()
    all_scores_random = list()
    all_patterns_stable = list()
    all_patterns_random = list()

    for subject in subjects:
        # results_folder = 'decoding_no_hpass_no_bsl'
        results_folder = output_folder
        for env in environment:
            fname = '%s_%s_scores_%s_%s.npy' % ( env,
                                               analysis_name,
                                               freq_name)
            sc = np.load(op.join(path_data, subject, 'results',
                                 results_folder, fname))
            fname = '%s_%s_patterns_%s_%s.npy' % ( env,
                                                 analysis_name,
                                                 freq_name)
            pat = np.load(op.join(path_data, subject, 'results',
                                  results_folder, fname))
            if env == 'stable':
                all_scores_stable.append(sc)
                all_patterns_stable.append(pat)
            elif env == 'random':
                all_scores_random.append(sc)
                all_patterns_random.append(pat)
    all_scores_stable = np.array(all_scores_stable)
    all_scores_random = np.array(all_scores_random)
    all_patterns_stable = np.array(all_patterns_stable)
    all_patterns_random = np.array(all_patterns_random)

    nb_sub = len(subjects)
    scores_stable = np.ravel(all_scores_stable, order='F')
    scores_random = np.ravel(all_scores_random, order='F')
    scores = np.concatenate((scores_stable, scores_random), axis=0)
    type = 2 * (['Target'] * nb_sub + ['Movement'] * nb_sub + ['Error'] *
                nb_sub + ['Prev_error'] * nb_sub)
    cond = ['Stable'] * 4 * nb_sub + ['Random'] * 4 * nb_sub
    data = pd.DataFrame({'Decoding Performance': scores,
                        'Condition': cond, 'Type': type})
    my_pal = {'Stable': colors[0], 'Random': colors[1]}

    # Plot prev Error
    prev_error_data = data[data['Type'] == 'Prev_error']
    fig, ax = plt.subplots(figsize=(2.5, 2.35))
    ax = sns.violinplot(x='Type', y='Decoding Performance', hue='Condition',
                        data=prev_error_data, saturation=1,
                        inner='stick', palette=my_pal, split=True,
                        linewidth=0.5)
    ax.axhline(y=0, linestyle='dotted', color='k', linewidth=0.8)
    ax.set_ylim(-0.25, 0.6)
    ax.yaxis.set_label_text('Decoding Performance (r)')
    ax.xaxis.set_label_text('')
    ax.get_legend().set_visible(False)
    ypos = data.groupby(['Type'])['Decoding Performance'].mean().tolist()
    yadd = data.groupby(['Type'])['Decoding Performance'].std().tolist()
    xpos = range(len(ypos))
    ax.set_title('Decoding previous error')
    plt.tight_layout()
    ax.figure.savefig(op.join(path_fig,save_folder,f'f{freq_name}_{control}'),
                      dpi=400)
    plt.close()

    yticks = [-0.1, 0, 0.1, 0.2, 0.3, 0.4]


    # Plot prev Target
    target_data = data[data['Type'] == 'Target']
    fig, ax = plt.subplots(figsize=(1.4, 2.2))
    ax = sns.violinplot(x='Type', y='Decoding Performance', hue='Condition',
                        data=target_data, saturation=1,
                        inner='stick', palette=my_pal, split=True,
                        linewidth=0.5)
    ax.axhline(y=0, linestyle='dotted', color='k', linewidth=0.8)
    ax.set_ylim(-0.25, 0.6)
    ax.set_yticks(yticks)
    ax.yaxis.set_label_text('Decoding Performance (Ê)')
    ax.xaxis.set_label_text('')
    ax.get_legend().set_visible(False)
    ypos = data.groupby(['Type'])['Decoding Performance'].mean().tolist()
    yadd = data.groupby(['Type'])['Decoding Performance'].std().tolist()
    xpos = range(len(ypos))
    # ax.set_title('Decoding previous error')
    plt.tight_layout()
    ax.figure.savefig(op.join(path_fig,save_folder,f'targ_{freq_name}_{control}'),
                      dpi=400)
    plt.close()

    # Plot prev movement
    movement_data = data[data['Type'] == 'Movement']
    fig, ax = plt.subplots(figsize=(1.4, 2.2))
    ax = sns.violinplot(x='Type', y='Decoding Performance', hue='Condition',
                        data=movement_data, saturation=1,
                        inner='stick', palette=my_pal, split=True,
                        linewidth=0.5)
    ax.axhline(y=0, linestyle='dotted', color='k', linewidth=0.8)
    ax.set_ylim(-0.25, 0.6)
    ax.set_yticks(yticks)
    ax.yaxis.set_label_text('Decoding Performance (Ê)')
    ax.xaxis.set_label_text('')
    ax.get_legend().set_visible(False)
    ypos = data.groupby(['Type'])['Decoding Performance'].mean().tolist()
    yadd = data.groupby(['Type'])['Decoding Performance'].std().tolist()
    xpos = range(len(ypos))
    # ax.set_title('Decoding previous error')
    plt.tight_layout()
    ax.figure.savefig(op.join(path_fig,save_folder,f'mov_{freq_name}_{control}'),
                      dpi=400)
    plt.close()

    # Plot next error
    error_data = data[data['Type'] == 'Error']
    fig, ax = plt.subplots(figsize=(1.4, 2.2))
    ax = sns.violinplot(x='Type', y='Decoding Performance', hue='Condition',
                        data=error_data, saturation=1,
                        inner='stick', palette=my_pal, split=True,
                        linewidth=0.5)
    ax.axhline(y=0, linestyle='dotted', color='k', linewidth=0.8)
    ax.set_ylim(-0.25, 0.6)
    ax.set_yticks(yticks)
    ax.yaxis.set_label_text('Decoding Performance (r)')
    ax.xaxis.set_label_text('')
    ax.get_legend().set_visible(False)
    ypos = data.groupby(['Type'])['Decoding Performance'].mean().tolist()
    yadd = data.groupby(['Type'])['Decoding Performance'].std().tolist()
    xpos = range(len(ypos))
    # ax.set_title('Decoding previous error')
    plt.tight_layout()
    ax.figure.savefig(op.join(path_fig,save_folder,f'next_error_{freq_name}_{control}'),
                      dpi=400)
    plt.close()

# ---------- Plot patterns
epochs = read_epochs('/Users/quentinra/Desktop/epochs_example.fif',
                     preload=True)
epochs.pick_types(meg=True, ref_meg=False)
info = epochs.info
info['sfreq'] = 1

prev_error_pats = all_patterns_random[:, 3, :, :]
pat = prev_error_pats.mean(0)

pp = EvokedArray(pat, info, tmin=0)
pp.plot_topomap(times=[0, 1, 2, 3, 4])

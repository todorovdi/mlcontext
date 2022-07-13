import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from config2 import subjects, path_fig, path_data
from base2 import decod_stats, gat_stats
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import ttest_1samp, ttest_rel
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
#hpass = 'no_filter'  # '0.1', 'detrend', no_hpass
output_folder = f'spoc_home2_{hpass}'
analysis_name = 'prevmovement_preverrors_errors_prevbelief'
analyses = ['Prev_movement', 'Prev_errors', 'Errors', 'Prev_belief']

save_folder = 'SPoC2_%s' % analysis_name
if not op.exists( op.join(path_fig, save_folder) ):
    os.mkdir(op.join(path_fig, save_folder) )


regression_types_all = ['Ridge', 'xgboost']
assert regression_type in regression_types_all

saved_successfully = []

import shutil

task = 'VisuoMotor'
time_locked = 'target'
#freqs = ['broad']
environment = ['stable', 'random']
rt = regression_type
fnames_zero_size = []
fnames_missing = []
for decoding_type in ['classic', 'b2b']:
    # for decoding_type in ['b2b']:
    #    for freq_name in freqs:
    all_scores_stable = list()
    all_scores_random = list()
    for subject in subjects:
        print(f'Starting subject {subject}')
        # results_folder = 'decoding_no_hpass_no_bsl'
        results_folder = output_folder
        for env in environment:
            sc = None
            if decoding_type == 'classic':
                fname = f'{env}_{rt}_scores_{analysis_name}_{freq_name}.npy'
            if decoding_type == 'b2b':
                fname = f'{env}_{rt}_partial_scores_{analysis_name}_{freq_name}.npy'
            fname_full = op.join(path_data, subject, 'results',
                    results_folder, fname)

            ###### renaming
            if not os.path.exists(fname_full):
                fname_alt = f'{env}_{rt}_spatial_scores_{analysis_name}_{freq_name}.npy'
                fname_alt_full = op.join(path_data, subject, 'results',
                        results_folder, fname_alt)

                if os.path.exists(fname_alt_full):
                    shutil.move(fname_alt_full,fname_full)
                    print(f'moved {fname_alt_full} to {fname_full}')

            if not os.path.exists(fname_full):
                print(f'WARNING: file does not exist: {fname_full}')
                fnames_missing += [fname_full]
                continue
            else:
                sc = np.load(fname_full)
                if not sc.size:
                    print(f'WARNING: corrupted scores for {fname_full}')
                    fnames_zero_size += [fname_full]
                    continue
                if env == 'stable':
                    all_scores_stable.append(sc)
                elif env == 'random':
                    all_scores_random.append(sc)
    all_scores_stable = np.array(all_scores_stable)
    all_scores_random = np.array(all_scores_random)

    nb_sub = len(all_scores_random)
    scores_stable = np.ravel(all_scores_stable, order='F')
    scores_random = np.ravel(all_scores_random, order='F')
    scores = np.concatenate((scores_stable, scores_random), axis=0)
    type = 2 * ([analyses[0]] * nb_sub + [analyses[1]] * nb_sub + [analyses[2]] *
                nb_sub + [analyses[3]] * nb_sub)
    cond = ['Stable'] * 4 * nb_sub + ['Random'] * 4 * nb_sub
    try:
        data = pd.DataFrame({'Decoding Performance': scores,
                            'Condition': cond, 'Type': type})
        my_pal = {'Stable': colors[0], 'Random': colors[1]}
    except ValueError as e:
        print('----- ERROR: There was exception during data frame creation\n', e)
        continue

    if not data.size:
        continue


    # Plot
    fig, axs = plt.subplots(1, len(analyses))
    for ii, analysis in enumerate(analyses):
        print(ii, analysis)
        data_sel = data[data['Type'] == analysis]
        m_stable = data_sel[data_sel['Condition'] == 'Stable']['Decoding Performance'].mean(0)
        m_random = data_sel[data_sel['Condition'] == 'Random']['Decoding Performance'].mean(0)
        std_stable = data_sel[data_sel['Condition'] == 'Stable']['Decoding Performance'].std(0)
        std_random = data_sel[data_sel['Condition'] == 'Random']['Decoding Performance'].std(0)
        _, p_stable = ttest_1samp(data_sel[data_sel['Condition'] == 'Stable']['Decoding Performance'], 0)
        _, p_random = ttest_1samp(data_sel[data_sel['Condition'] == 'Random']['Decoding Performance'], 0)
        _, p_diff = ttest_rel(data_sel[data_sel['Condition'] == 'Stable']['Decoding Performance'],
                                data_sel[data_sel['Condition'] == 'Random']['Decoding Performance'])

        ax = axs[ii]
        sns.violinplot(x='Type', y='Decoding Performance', hue='Condition',
                        data=data_sel, saturation=1,
                        inner='stick', palette=my_pal, split=True,
                        linewidth=0.5, ax=ax)
        ax.axhline(y=0, linestyle='dotted', color='k', linewidth=0.8)

        ax.axhline(y=m_stable, linestyle='--', color='red', linewidth=1.5)
        ax.axhline(y=m_random, linestyle='--', color='blue', linewidth=1.5)

        ax.axhline(y=m_stable-std_stable, linestyle='--', color='red',  linewidth=0.7)
        ax.axhline(y=m_random-std_random, linestyle='--', color='blue', linewidth=0.7)
        ax.axhline(y=m_stable+std_stable, linestyle='--', color='red',  linewidth=0.7)
        ax.axhline(y=m_random+std_random, linestyle='--', color='blue', linewidth=0.7)

        if p_stable < 0.05:
            ax.text(-0.4, -0.1, f'p={p_stable:.3f}', fontsize=7, color='red')
        else:
            ax.text(-0.4, -0.1, f'p={p_stable:.3f}', fontsize=7)
        ax.text(-0.4, -0.11, f'm={m_stable:.3f}', fontsize=7)
        if p_random < 0.05:
            ax.text(0.1, -0.1, f'p={p_random:.3f}', fontsize=7, color='red')
        else:
            ax.text(0.1, -0.1, f'p={p_random:.3f}', fontsize=7)
        ax.text(0.1, -0.11, f'm={m_random:.3f}', fontsize=7)
        if p_diff < 0.05:
            ax.text(0, 0.3, f'p={p_diff:.3f}', fontsize=7, color='red')
        else:
            ax.text(0, 0.3, f'p={p_diff:.3f}', fontsize=7)
        ax.set_ylim(-0.2, 0.4)
        ax.yaxis.set_label_text('Decoding Performance (r)')
        ax.xaxis.set_label_text('')
        ax.get_legend().set_visible(False)
        ax.set_title('Decoding %s' % analysis)

    plt.tight_layout()
    fname_fig = op.join(path_fig,save_folder,
        f'{rt}_{analysis_name}_{freq_name}_{decoding_type}')
    fig.savefig(fname_fig ,dpi=400)
    plt.close()

    saved_successfully += [fname_fig]

    print(f'---------- Figure saved to {fname_fig}')

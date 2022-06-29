import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from config2 import subjects, path_fig, path_data 
from base2 import decod_stats, gat_stats
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import ttest_1samp, ttest_rel
from scipy.stats import spearmanr
import seaborn as sns
import pandas as pd
from pingouin import partial_corr

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

hpass = '0.1'  # '0.1', 'detrend', no_hpass
output_folder = 'corr_spoc_es2_%s' % hpass
save_folder = 'SPoC_behavior_correlation'
if not op.exists( op.join(path_fig, save_folder) ):
    os.mkdir(op.join(path_fig, save_folder) )

freqs = ['broad']

regression_types = ['Ridge', 'xgboost']

task = 'VisuoMotor'
for rt in regression_types[:1]:
    for freq_name in freqs:
        all_scores_stable = list()
        all_diff_stable = list()
        all_es_stable = list()
        all_corr_stable = list()
        all_scores_random = list()
        all_diff_random = list()
        all_es_random = list()
        all_corr_random = list()
        mean_es_stable = list()
        mean_es_random = list()
        mean_corr_stable = list()
        mean_corr_random = list()
        all_preverror_stable = list()
        all_preverror_random = list()
        for env in ['stable', 'random']:
            for subject in subjects:
                print(f'Starting subject {subject}')
                results_folder = output_folder
                fname = op.join(results_folder,
                                f'{rt}_{env}_scores_{freq_name}.npy' )
                sc = np.load(op.join(path_data, subject, 'results',
                                     fname))
                fname = op.join(results_folder,
                                f'{rt}_{env}_diff_{freq_name}.npy')
                diff = np.load(op.join(path_data, subject, 'results',
                                       fname))
                fname = op.join(results_folder,
                                f'{rt}_{env}_corr_{freq_name}.npy')
                corr = np.load(op.join(path_data, subject, 'results',
                                     fname))
                fname = op.join(results_folder,
                                f'{rt}_{env}_preverror_{freq_name}.npy')
                preverror = np.load(op.join(path_data, subject, 'results',
                                    fname))

                fname = op.join(path_data, subject, 'behavdata', f'err_sens_{task}.npz')
                f = np.load(fname, allow_pickle=True)['arr_0'][()]
                env2err_sens      = f['env2err_sens'][()]
                es = env2err_sens[env]

                #fname = op.join(results_folder,
                #                '%ses_%s.npy' % (env, freq_name))
                #es = np.load(op.join(path_data, subject, 'results/',
                #                     fname))

                if env == 'stable':
                    all_scores_stable.append(sc)
                    all_diff_stable.append(diff)
                    all_es_stable.append(es)
                    all_corr_stable.append(corr)
                    all_preverror_stable.append(preverror)
                    mean_es_stable.append(es.mean())
                    mean_corr_stable.append(corr.mean())
                elif env == 'random':
                    all_scores_random.append(sc)
                    all_diff_random.append(diff)
                    all_es_random.append(es)
                    all_corr_random.append(corr)
                    all_preverror_random.append(preverror)
                    mean_es_random.append(es.mean())
                    mean_corr_random.append(corr.mean())
    all_scores_stable = np.array(all_scores_stable)
    all_diff_stable = np.array(all_diff_stable)
    all_es_stable = np.array(all_es_stable)
    all_corr_stable = np.array(all_corr_stable)
    all_scores_random = np.array(all_scores_random)
    all_diff_random = np.array(all_diff_random)
    all_es_random = np.array(all_es_random)
    all_corr_random = np.array(all_corr_random)
    all_preverror_stable = np.array(all_preverror_stable)
    all_preverror_random = np.array(all_preverror_random)

    mean_es_stable = np.array(mean_es_stable)
    mean_es_random = np.array(mean_es_random)
    mean_corr_stable = np.array(mean_corr_stable)
    mean_corr_random = np.array(mean_corr_random)

    all_r_stable = list()
    all_p_stable = list()
    for ii in range(len(all_diff_stable)):
        diff = all_diff_stable[ii]
        es = all_es_stable[ii]
        r, p = spearmanr(diff, es)
        all_r_stable.append(r)
        all_p_stable.append(p)
    all_r_stable = np.array(all_r_stable)
    all_p_stable = np.array(all_p_stable)
    ttest_1samp(all_r_stable, 0)


    all_r_random = list()
    all_p_random = list()
    for ii in range(len(all_diff_random)):
        diff = all_diff_random[ii]
        es = all_es_random[ii]
        r, p = spearmanr(diff, es)
        all_r_random.append(r)
        all_p_random.append(p)
    all_r_random = np.array(all_r_random)
    all_p_random = np.array(all_p_random)
    ttest_1samp(all_r_random, 0)

    all_r_stable = list()
    all_p_stable = list()
    for ii in range(len(all_diff_stable)):
        diff = all_diff_stable[ii]
        corr = all_corr_stable[ii]
        r, p = spearmanr(diff, np.abs(corr))
        all_r_stable.append(r)
        all_p_stable.append(p)
    all_r_stable = np.array(all_r_stable)
    all_p_stable = np.array(all_p_stable)
    ttest_1samp(all_r_stable, 0)

    all_r_random = list()
    all_p_random = list()
    for ii in range(len(all_diff_random)):
        diff = all_diff_random[ii]
        corr = all_corr_random[ii]
        r, p = spearmanr(diff, np.abs(corr))
        all_r_random.append(r)
        all_p_random.append(p)
    all_r_random = np.array(all_r_random)
    all_p_random = np.array(all_p_random)
    ttest_1samp(all_r_random, 0)

    all_r_stable = list()
    all_p_stable = list()
    for ii in range(len(all_diff_stable)):
        pe = np.abs(all_preverror_stable[ii])
        diff = all_diff_stable[ii]
        es = all_es_stable[ii]
        r, p = spearmanr(pe, es)
        all_r_stable.append(r)
        all_p_stable.append(p)
    all_r_stable = np.array(all_r_stable)
    all_p_stable = np.array(all_p_stable)
    ttest_1samp(all_r_stable, 0)

    all_r_stable = list()
    all_p_stable = list()
    for ii in range(len(all_diff_stable)):
        data = np.vstack((all_diff_stable[ii],
                          all_es_stable[ii],
                          np.abs(all_preverror_stable[ii]))).T
        data = pd.DataFrame(data=data, columns=['diff', 'es', 'pe'])
        results = partial_corr(data=data, x='diff', y='es', covar=['pe'], method='spearman')
        all_r_stable.append(results.values[0, 1])
        all_p_stable.append(results.values[0, 3])
    all_r_stable = np.array(all_r_stable)
    all_p_stable = np.array(all_p_stable)
    ttest_1samp(all_r_stable, 0)


    all_r_random = list()
    all_p_random = list()
    for ii in range(len(all_diff_random)):
        data = np.vstack((all_diff_random[ii],
                          all_es_random[ii],
                          all_preverror_random[ii])).T
        data = pd.DataFrame(data=data, columns=['diff', 'es', 'pe'])
        results = partial_corr(data=data, x='diff', y='es', covar=['pe'], method='spearman')
        all_r_random.append(results.values[0, 1])
        all_p_random.append(results.values[0, 3])
    all_r_random = np.array(all_r_random)
    all_p_random = np.array(all_p_random)
    ttest_1samp(all_r_random, 0)


    fig, axs = plt.subplots(10, 3)
    axs = axs.ravel()
    for ii in range(len(all_diff)):
        diff = all_diff[ii]
        es = all_es[ii]
        axs[ii].plot(diff, es, 'o', alpha=0.1, color=colors[1])
        axs[ii].set_xticks([])
        axs[ii].set_yticks([])

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from config2 import subjects
from base2 import decod_stats, gat_stats
from jr.plot import share_clim
from jr import OnlineReport
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

path_data = '/Volumes/data/MemErrors/data2/'
hpass = 'no_filter'  # '0.1', 'detrend', no_hpass
output_folder = 'spoc_home2_%s' % hpass
analysis_name = 'prevmovement_preverrors_errors_prevbelief'
analyses = ['Prev_movement', 'Prev_errors', 'Errors', 'Prev_belief']

save_folder = 'SPoC2_%s' % analysis_name
if not op.exists( op.join(path_fig, save_folder) ):
    os.mkdir(op.join(path_fig, save_folder) )

task = 'VisuoMotor_'
time_locked = 'target'
freqs = ['6', '10', '14', '18', '22', '26', '30', '34', '38', '42',
         '46', '50', '54', '58', '62', '66', '70', '74', '78', '82', '86',
         '90', '94', '98', '102', '106', '110', '114', '118', '122', '126',
         '130', '134', '138', '142', '146', '150', '154',
         '158', '162', '166', '170', '174', '178', '182', '186', '190',
         '194', '198', '202', '206', '210', '214', '218', '222', '226', '230', '234', '238',
         '242', '246', '250', '254', '258', '262', '266', '270', '274',
         '278', '282', '286', '290', '294', '298', '302', '306', '310', '314', '318', '322', '326',
         '330', '334', '338', '342', '346', '350', '354', '358', '362']

regression_types = ['Ridge', 'xgboost']
environment = ['stable', 'random']

for rt in regression_types[:1]:
    for control in ['classic', 'b2b']:
        all_freqs_scores_stable = list()
        all_freqs_scores_random = list()
        for freq_name in freqs:
            all_scores_stable = list()
            all_scores_random = list()
            for subject in subjects:
                # results_folder = 'decoding_no_hpass_no_bsl'
                results_folder = output_folder
                for env in environment:
                    if control == 'classic':
                        fname = f'{env}_{rt}_scores_{analysis_name}_{freq_name}.npy'
                    if control == 'b2b':
                        fname = f'{env}_{rt}_partial_scores_{analysis_name}_{freq_name}.npy' 
                    sc = np.load(op.join(path_data, subject, 'results/',
                                         results_folder, fname))
                    if env == 'stable':
                        all_scores_stable.append(sc)
                    elif env == 'random':
                        all_scores_random.append(sc)
            all_scores_stable = np.array(all_scores_stable)
            all_scores_random = np.array(all_scores_random)
            print('append one set of scores of shape %s' % all_scores_stable[:, 1].shape)
            all_freqs_scores_stable.append(all_scores_stable)
            all_freqs_scores_random.append(all_scores_random)
    all_freqs_scores_stable = np.array(all_freqs_scores_stable)
    all_freqs_scores_random = np.array(all_freqs_scores_random)

    for i, name in enumerate(analyses):
        plt.figure(figsize=[35, 10])
        plt.plot(freqs, all_freqs_scores_stable.mean(1)[:, i], color='C1')
        plt.plot(freqs, all_freqs_scores_random.mean(1)[:, i], color='C2')
        for ii, _ in enumerate(freqs):
            m_stable = all_freqs_scores_stable[ii, :, i].mean()
            _, p_stable = ttest_1samp(all_freqs_scores_stable[ii, :, i], 0)
            m_random = all_freqs_scores_random[ii, :, i].mean()
            _, p_random = ttest_1samp(all_freqs_scores_random[ii, :, i], 0)
            plt.ylim(-0.01, 0.05)
            plt.text(ii, 0.02, f'm={m_stable:.3f}', fontsize=6, color='C1')
            plt.text(ii, 0.015, f'p={p_stable:.3f}', fontsize=6, color='C1')
            plt.text(ii, 0.005, f'm={m_random:.3f}', fontsize=6, color='C2')
            plt.text(ii, 0, f'p={p_random:.3f}', fontsize=7, color='C2')
        plt.savefig(path_fig,save_folder,'lotsoffreqs_%s_%s' % (name, control),
                    dpi=400)

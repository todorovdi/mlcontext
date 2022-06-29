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

# BUG: does not produce a figure :(

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
hpass = '0.05'  # '0.1', 'detrend', no_hpass
output_folder = 'psd_decoding2%s' % hpass
save_folder = 'psd_decoding2'
if not op.exists( op.join(path_fig, save_folder) ):
    os.mkdir(op.join(path_fig, save_folder) )

envs = ['all', 'stable', 'random']
control = 'b2b'  # 'classic' or 'b2b'
all_scores_all = list()
all_scores_stable = list()
all_scores_random = list()
for subject in subjects:
    # results_folder = 'decoding_no_hpass_no_bsl'
    results_folder = output_folder
    for env in envs:
        analysis_name = 'prevfeedback_preverrors_errors'
        if control == 'classic':
            fname = '%s_scores_all_freqs_%s.npy' % (env, analysis_name)
        elif control == 'b2b':
            fname = '%s_partial_scores_all_freqs_%s.npy' % (env, analysis_name)
        sc = np.load(op.join(path_data, subject, 'results',
                             results_folder, fname))
        if env == 'all':
            all_scores_all.append(sc)
        elif env == 'stable':
            all_scores_stable.append(sc)
        elif env == 'random':
            all_scores_random.append(sc)
all_scores_all = np.array(all_scores_all)
all_scores_stable = np.array(all_scores_stable)
all_scores_random = np.array(all_scores_random)

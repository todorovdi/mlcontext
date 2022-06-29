import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from config2 import subjects, path_fig, path_data
from base2 import decod_stats
import seaborn as sns

sns.set_palette('colorblind')
plt.style.use('seaborn')
color_palette = sns.color_palette("colorblind", 8).as_hex()
colors = [color_palette[1], color_palette[7]]
# colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C6']
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
hpass = '0.1'  # '0.1', 'detrend', no_hpass
control_type = 'target'  # 'movement', 'feedback', 'target' or 'belief'
controls = ['classic', 'b2b']
is_short = True
ICA = 'with_ICA'
if control_type == 'feedback':
    analyses_name_feedback = 'feedback_errors_next_errors_belief'
    analyses_name_target = 'prevfeedback_preverrors_errors_prevbelief'
elif control_type == 'movement':
    analyses_name_feedback = 'movement_errors_next_errors_belief'
    analyses_name_target = 'prevmovement_preverrors_errors_prevbelief'
elif control_type == 'target':
    analyses_name_feedback = 'target_errors_nexterrors_belief'
    analyses_name_target = 'prevtarget_preverrors_errors_prevbelief'
elif control_type == 'belief':
    analyses_name_feedback = 'belief_errors_nexterrors'
    analyses_name_target = 'prevbelief_preverrors_errors'

if is_short:
    time_lockeds = [['feedback', analyses_name_feedback,
                     np.linspace(-0.2, 3, 321)]]
    save_folder = 'Exp2_td_short_%s_%s_%s' % (hpass, control_type, ICA)
    output_folder = 'td_long2_%s_short_%s' % (hpass, ICA)
else:
    time_lockeds = [['feedback', analyses_name_feedback,
                     np.linspace(-2, 5, 701)],
                    ['target', analyses_name_target,
                     np.linspace(-5, 2, 701)]]
    save_folder = 'Exp2_td_long_%s_%s_%s' % (hpass, control_type, ICA)
    output_folder = 'td_long2_%s_%s' % (hpass, ICA)

if not op.exists( op.join(path_fig, save_folder) ):
    os.mkdir(op.join(path_fig, save_folder) )

# Plot errors in each environment together
controls = ['classic', 'b2b']

regression_types = ['Ridge', 'xgboost']

for time_locked in time_lockeds:
    times = time_locked[2]
    for control in controls:
        all_scores_all = list()
        all_scores_stable = list()
        all_scores_random = list()
        for subject in subjects:
            results_folder = output_folder
            if control == 'classic':
                fname_all = 'allscores_%s_%s.npy' % ( time_locked[0],
                                                        time_locked[1])
                fname_stable = 'stablescores_%s_%s.npy' % ( time_locked[0],
                                                              time_locked[1])
                fname_random = 'randomscores_%s_%s.npy' % ( time_locked[0],
                                                              time_locked[1])
            elif control == 'b2b':
                fname_all = 'allpartial_scores_%s_%s.npy' % ( time_locked[0],
                                                                time_locked[1])
                fname_stable = 'stablepartial_scores_%s_%s.npy' % ( time_locked[0],
                                                                   time_locked[1])
                fname_random = 'randompartial_scores_%s_%s.npy' % ( time_locked[0],
                                                                      time_locked[1])
            sc_all = np.load(op.join(path_data, subject, 'results',
                                     results_folder, fname_all))
            sc_stable = np.load(op.join(path_data, subject, 'results',
                                        results_folder, fname_stable))
            sc_random = np.load(op.join(path_data, subject, 'results',
                                        results_folder, fname_random))
            all_scores_all.append(sc_all)
            all_scores_stable.append(sc_stable)
            all_scores_random.append(sc_random)
        all_scores_all = np.array(all_scores_all)
        all_scores_stable = np.array(all_scores_stable)
        all_scores_random = np.array(all_scores_random)
        all_scores_all = np.moveaxis(all_scores_all, 1, 0)
        all_scores_stable = np.moveaxis(all_scores_stable, 1, 0)
        all_scores_random = np.moveaxis(all_scores_random, 1, 0)
        # Plot
        # Plot errors
        plt.title('%s decoding error' % control, fontsize=14)
        # Plot errors during all environment
        # plt.plot(times, all_scores_all[1].mean(0), color=colors[3], label='All trials', linewidth=0.5)
        # sig = decod_stats(all_scores_all[1]) < 0.05
        # plt.fill_between(times, all_scores_all[1].mean(0), where=sig, color=colors[3], alpha=0.3)
        # Plot errors during stable environment
        plt.plot(times, all_scores_stable[1].mean(0), color=colors[0], label='Stable environment', linewidth=0.5)
        sig = decod_stats(all_scores_stable[1]) < save_folder, 0.05
        plt.fill_between(times, all_scores_stable[1].mean(0), where=sig, color=colors[0], alpha=0.3)
        # Plot errors during random environment
        plt.plot(times, all_scores_random[1].mean(0), color=colors[1], label='Random environment', linewidth=0.5)
        sig = decod_stats(all_scores_random[1]) < 0.05
        plt.fill_between(times, all_scores_random[1].mean(0), where=sig, color=colors[1], alpha=0.3)
        plt.legend()
        fname_fig = op.join(path_fig,save_folder,'Exp2_%s_%s_errors' % (time_locked[0], control) )
        plt.savefig(fname_fig, dpi=400)
        plt.close()


# Plot errors difference in stable and random
controls = ['classic', 'b2b']
for time_locked in time_lockeds:
    times = time_locked[2]
    for control in controls:
        all_scores_stable = list()
        all_scores_random = list()
        for subject in subjects:
            results_folder = output_folder
            if control == 'classic':
                fname_stable = '%s_stablescores_%s_%s.npy' % (subject,
                                                              time_locked[0],
                                                              time_locked[1])
                fname_random = '%s_randomscores_%s_%s.npy' % (subject,
                                                              time_locked[0],
                                                              time_locked[1])
            elif control == 'b2b':
                fname_stable = '%s_stablepartial_scores_%s_%s.npy' % (subject,
                                                                   time_locked[0],
                                                                   time_locked[1])
                fname_random = '%s_randompartial_scores_%s_%s.npy' % (subject,
                                                                      time_locked[0],
                                                                      time_locked[1])
            sc_stable = np.load(op.join(path_data, subject, 'results/',
                                        results_folder, fname_stable))
            sc_random = np.load(op.join(path_data, subject, 'results/',
                                        results_folder, fname_random))
            all_scores_stable.append(sc_stable)
            all_scores_random.append(sc_random)
        all_scores_stable = np.array(all_scores_stable)
        all_scores_random = np.array(all_scores_random)
        all_scores_stable = np.moveaxis(all_scores_stable, 1, 0)
        all_scores_random = np.moveaxis(all_scores_random, 1, 0)
        diff = all_scores_stable[1] - all_scores_random[1]
        # Plot
        # Plot errors
        plt.title('%s decoding error differences' % control, fontsize=14)
        # Plot errors during stable environment
        plt.plot(times, diff.mean(0), color=colors[0], linewidth=0.5)
        sig = decod_stats(diff) < 0.05
        plt.fill_between(times, diff.mean(0), where=sig, color=colors[0], alpha=0.3)
        fname_fig = op.join( path_fig, save_folder, 'Exp2_%s_%s_errors_diff' % (time_locked[0], control) )
        plt.savefig(fname_fig,
                    dpi=400)
        plt.close()


# # Plot all analyses together separately for each environment
# envs = ['all', 'stable', 'random']
# controls = ['classic', 'b2b']
# for env in envs:
#     for time_locked in time_lockeds:
#         times = time_locked[2]
#         for control in controls:
#             all_scores = list()
#             for subject in subjects:
#                 results_folder = output_folder
#                 if control == 'classic':
#                     fname = '%s_%sscores_%s_%s.npy' % (subject, env,
#                                                        time_locked[0],
#                                                        time_locked[1])
#                 elif control == 'b2b':
#                     fname = '%s_%spartial_scores_%s_%s.npy' % (subject, env,
#                                                                time_locked[0],
#                                                                time_locked[1])
#                 sc = np.load(op.join(path_data, subject, 'results/',
#                                      results_folder, fname))
#                 all_scores.append(sc)
#             all_scores = np.array(all_scores)
#             all_scores = np.moveaxis(all_scores, 1, 0)
#             # Plot
#             colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C6']
#             for ii, scores in enumerate(all_scores):
#                 plt.title('%s %s %s' % (env, control, time_locked[1]))
#                 plt.plot(times, scores.mean(0), color=colors[ii],
#                          label=str(ii))
#                 sig = decod_stats(scores) < 0.05
#                 plt.fill_between(times, scores.mean(0),
#                                  where=sig, color=colors[ii], alpha=0.3)
#                 plt.legend()
#             plt.savefig('/Users/quentinra/Desktop/figs_memerror/%s/Exp2_%s_%s_%s_%s' % (save_folder, env, control,
#                                                                                         time_locked[0], time_locked[1]),
#                         dpi=400)
#             plt.close()

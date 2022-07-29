import os
import os.path as op
import numpy as np
from scipy.stats import linregress
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
from config2 import subjects, path_fig, path_data
from scipy.stats import ttest_1samp, spearmanr
from pingouin import partial_corr

sns.set_palette('colorblind')
plt.style.use('seaborn')
color_palette = sns.color_palette("colorblind", 8).as_hex()
colors = [color_palette[1], color_palette[7]]
my_pal = {'Stable': colors[0], 'Random': colors[1]}

target_angs = (np.array([157.5, 112.5, 67.5, 22.5]) + 90) * \
              (np.pi/180)
target_coords = calc_target_coordinates_centered(target_angs)

ICA = 'with_ICA'
#hpass = 'no_filter'
#save_folder = 'corr_es_SPoC_decod_slide_%s_%s' % (hpass, ICA)
save_folder   = 'corr_spoc_es_sliding2_%s' % (hpass)
#/p/project/icei-hbp-2020-0012/lyon/memerr/data2/sub08_TXVPROYY/results/spoc_sliding_no_filte
if not op.exists( op.join(path_fig, save_folder) ):
    os.mkdir(op.join(path_fig, save_folder) )

#fname_full = genFnSliding(results_folder, env,
#                regression_type,time_locked,
#                analysis_name,freq_name,tmin_cur,tmax_cur)
envs = ['stable', 'random']

DEBUG = 0
if DEBUG:
    subjects = subjects[:1]
    envs = ['stable']
    print("--------------   DEBUG MODE  --------------")
    print("--------------   DEBUG MODE  --------------")
    print("--------------   DEBUG MODE  --------------")

fnames_failed = []

#all_scores_stable = list()
#all_diff_stable = list()
#all_es_stable = list()
#all_corr_stable = list()
#all_scores_random = list()
#all_diff_random = list()
#all_es_random = list()
#all_corr_random = list()
#mean_es_stable = list()
#mean_es_random = list()
#mean_corr_stable = list()
#mean_corr_random = list()
#all_preverror_stable = list()
#all_preverror_random = list()

#tpls = []
df_persubj=[]
from postproc import collectResults
for subject in subjects:
    print(f'Starting collecting subject {subject}')
    #df_collect = collectResults(subject,hpass)
    df_collect = collectResults(subject,output_folder,freq_name,
                        keys_to_extract = ['par','scores_es','diff','es', 'prev_error', 'corr'] )
    df_persubj += [df_collect]
    #tmins = df_persubj[df_persubj[env] == 'stable' ]   ['tmin']
df = pd.concat( df_persubj )

#df = df.astype(

# they are still strings here
tmins_srt = sorted( set( df['tmin'] ), key=lambda x: float(x) )
if DEBUG:
    tmins_srt = tmins_srt[:1]

scores_cur_env = []
fnames_cur = []

df['pp:diff_es:r']       = df.apply(lambda x: spearmanr(x['diff'], x['es'])[0], axis=1 )
df['pp:diff_abscorr:r']  = df.apply(lambda x: spearmanr(x['diff'], np.abs(x['corr']) )[0], axis=1 )
df['pp:abspe_es:r']      = df.apply(lambda x: spearmanr(np.abs(x['prev_error']), x['es'])[0], axis=1 )

# we don't use p anyway so far
#df['pp:diff_es:p']       = df.apply(lambda x: spearmanr(x['diff'], x['es'])[1], axis=1 )
#df['pp:diff_abscorr:p']  = df.apply(lambda x: spearmanr(x['diff'], np.abs(x['corr']) )[1], axis=1 )
#df['pp:abspe_es:p']      = df.apply(lambda x: spearmanr(np.abs(x['pe']), x['es'])[1], axis=1 )


def pcorr_r(diff,es,pe):
    #print( type(diff), type(es), type(pe) )
    #print(diff.shape, es.shape, pe.shape)
    data = np.vstack( [diff, es, np.abs(pe)]).T
    data = pd.DataFrame(data=data, columns=['diff', 'es', 'abspe'])
    results = partial_corr(data=data, x='diff', y='es', covar=['abspe'],
                            method='spearman')
    #all_r_stable.append(results.values[0, 1])
    return results.values[0,1]

def pcorr_p(diff,es,pe):
    data = np.vstack( [diff, es, np.abs(pe)]).T
    data = pd.DataFrame(data=data, columns=['diff', 'es', 'abspe'])
    results = partial_corr(data=data, x='diff', y='es', covar=['abspe'],
                            method='spearman')
    #all_r_stable.append(results.values[0, 1])
    #all_p_stable.append(results.values[0, 3])
    return results.values[0,3]

df['pp:diff_pe_es:r']    = df.apply(lambda x: pcorr_r(x['diff'], x['es'], x['prev_error']), axis=1 )
#df['pp:diff_pe_es:p']    = df.apply(lambda x: pcorr_p(x['diff'], x['es'], ['prev_error']), axis=1 )
ppnames = [ 'pp:diff_pe_es:r', 'pp:diff_es:r',    'pp:diff_abscorr:r', 'pp:abspe_es:r']

for env in envs:
    df_curenv = df[ df['env'] == env ]
    df_curenv_srt = df_curenv.sort_values(by=['tmin'], key=lambda x: list(map(float,x) ) )
    scores = []
    scores_std = []
    for tmin in tmins_srt:
        df_curwnd = df_curenv_srt[ df_curenv_srt['tmin'] == tmin ]
        #scores.append( df_curwnd.mean['scores'] )
        #diffs.append( df_curwnd.mean['diff'] )
        #err_senss.append( df_curwnd.mean['es'] )

        for ppname in ppnames:
            res = ttest_1samp(df_curwnd[ ppname ] ,0)
            print(env,tmin,ppname, res)

        # mean over subjects
        sc = np.array ( list( df_curwnd['scores_es'].to_numpy() ) )
        scm = sc.mean(axis=1) # mean over splits
        scores += [ scm.mean() ]

        scores_std += [ scm.std() ]

        if plot_diff_vs_es:
            nr = 10; nc =2
            ww = 5; hh = 3
            fig, axs = plt.subplots(nr,nc,figsize=(nc*ww,nr*hh) , sharex='col', sharey='row' )
            axs = axs.ravel()
            assert len(axs) >= len(subjects)
            #for ii in range(len(all_diff)):
            for ii,subject in enumerate(subjects):
                df_curwnd_cursubj = df_curwnd[ df_curwnd['subject'] == subject ]
                diff = df_curwnd_cursubj['diff']
                es   = df_curwnd_cursubj['es']
                if len(diff) and len(es):
                    diff = diff.to_numpy()[0]
                    es   = es.to_numpy()[0]
                    ax = axs[ii]
                    ax.plot(diff, es, 'o', alpha=0.1, color=colors[1])
                    #ax.set_xticks([])
                    #ax.set_yticks([])
                    ax.set_title(f'{subject},{env} tmin={tmin} diff vs es')

            time_locked = 'target'
            fname_fig = op.join( path_fig, save_folder,
                f'diff_vs_es__{tmin}_{env}_{regression_type}_{freq_name}_{time_locked}.png' )
            plt.savefig(fname_fig, dpi=300)
            print(f'Fig saved to {fname_fig}')
            plt.close()


    fig = plt.figure(figsize = (10,4) )
    ax = plt.gca()
    tmins_srt_f = list(map(float,tmins_srt) )
    scores = np.array(scores)
    scores_std = np.array(scores_std)
    ax.plot(tmins_srt_f  ,scores)
    ax.plot(tmins_srt_f  ,scores - scores_std, ls=':')
    ax.plot(tmins_srt_f  ,scores + scores_std, ls=':')
    ax.set_title(f'{env}: scores decoding erros sens {regression_type}')

    fname_fig = op.join( path_fig, save_folder,
        f'tmin_vs_scores_es__{env}_{regression_type}_{freq_name}_{time_locked}.png' )
    plt.savefig(fname_fig, dpi=300)
    print(f'Fig saved to {fname_fig}')
    plt.close()


        #all_r_stable = list()
        #all_p_stable = list()
        ## per subj
        #for ii in range(len(all_diff_stable)):
        #    diff = all_diff_stable[ii]
        #    es =   all_es_stable[ii]
        #    r, p = spearmanr(diff, es)
        #    all_r_stable.append(r)
        #    all_p_stable.append(p)
        #all_r_stable = np.array(all_r_stable)
        #all_p_stable = np.array(all_p_stable)
        #ttest_1samp(all_r_stable, 0)
    #    df_cur = df_curenv[df_curenv['tmin'] == tmin ]
                #results_folder = output_folder
                #fname = op.join(results_folder,
                #                f'{rt}_{env}_scores_{freq_name}.npy' )
                #sc = np.load(op.join(path_data, subject, 'results',
                #                        fname))
                #fname = op.join(results_folder,
                #                f'{rt}_{env}_diff_{freq_name}.npy')
                #diff = np.load(op.join(path_data, subject, 'results',
                #                        fname))
                #fname = op.join(results_folder,
                #                f'{rt}_{env}_corr_{freq_name}.npy')
                #corr = np.load(op.join(path_data, subject, 'results',
                #                        fname))
                #fname = op.join(results_folder,
                #                f'{rt}_{env}_preverror_{freq_name}.npy')
                #preverror = np.load(op.join(path_data, subject, 'results',
                #                    fname))

                #fname = op.join(path_data, subject, 'behavdata', f'err_sens_{task}.npz')
                #f = np.load(fname, allow_pickle=True)['arr_0'][()]
                #env2err_sens      = f['env2err_sens'][()]
                #es = env2err_sens[env]

                #fname = op.join(results_folder,
                #                '%ses_%s.npy' % (env, freq_name))
                #es = np.load(op.join(path_data, subject, 'results/',
                #                     fname))

#                if env == 'stable':
#                    all_scores_stable.append(sc)
#                    all_diff_stable.append(diff)
#                    all_es_stable.append(es)
#                    all_corr_stable.append(corr)
#                    all_preverror_stable.append(preverror)
#                    mean_es_stable.append(es.mean())
#                    mean_corr_stable.append(corr.mean())
#                elif env == 'random':
#                    all_scores_random.append(sc)
#                    all_diff_random.append(diff)
#                    all_es_random.append(es)
#                    all_corr_random.append(corr)
#                    all_preverror_random.append(preverror)
#                    mean_es_random.append(es.mean())
#                    mean_corr_random.append(corr.mean())
#all_scores_stable = np.array(all_scores_stable)
#all_diff_stable = np.array(all_diff_stable)
#all_es_stable = np.array(all_es_stable)
#all_corr_stable = np.array(all_corr_stable)
#all_scores_random = np.array(all_scores_random)
#all_diff_random = np.array(all_diff_random)
#all_es_random = np.array(all_es_random)
#all_corr_random = np.array(all_corr_random)
#all_preverror_stable = np.array(all_preverror_stable)
#all_preverror_random = np.array(all_preverror_random)
#
#mean_es_stable = np.array(mean_es_stable)
#mean_es_random = np.array(mean_es_random)
#mean_corr_stable = np.array(mean_corr_stable)
#mean_corr_random = np.array(mean_corr_random)

sys.exit(0)

######################################################
################## some other stuff
######################################################

# Get the error sensitivity measure
ind_es_stable = list()
ind_es_random = list()
all_es_stable = list()
all_es_random = list()
mean_es_stable = list()
mean_es_random = list()
ind_errors_stable = list()
ind_errors_random = list()
mean_errors_stable = list()
mean_errors_random = list()
nb_sub = len(subjects)
for subject in subjects:
    task = 'VisuoMotor'  # 'VisuoMotor_' or 'LocaError_'

    fname = op.join(path_data, subject, 'behavdata',
                    f'behav_{task}_df.pkl' )
    behav_df_full = pd.read_pickle(fname)
    behav_df = pd.read_pickle(fname)

    # Perturbations
    perturbations = np.array(behav_df['perturbation'])
    # Environment
    environment = np.array(behav_df['environment']).astype(int)
    # Targets position
    targets = np.array(behav_df['target_inds'])
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
                      prev_errors, environment, perturbations, errors]

    Y = np.array(analyses_value)
    non_hit = np.array(non_hit)
    # remove trials following hit (because no previous error)
    # non_hit = ~(~non_hit | ~np.insert(non_hit, 0, 1)[:-1])
    non_hit = np.insert(non_hit, 0, 0)[:-1]
    # remove first trials (because no previous error)
    first_trials = np.where(behav_df['trials'] == 0)
    non_hit[first_trials] = False
    Y = Y[:, non_hit]
    # Error positions
    errors_stable = Y[7][np.where(Y[5] == 0)]
    errors_random = Y[7][np.where(Y[5] == 1)]
    prev_errors_stable = Y[4][np.where(Y[5] == 0)]
    prev_errors_random = Y[4][np.where(Y[5] == 1)]
    ind_errors_stable.append(errors_stable)
    ind_errors_random.append(errors_random)
    mean_errors_stable.append(errors_stable.mean())
    mean_errors_random.append(errors_random.mean())
    # Compute error sensitivity
    es = ((Y[0]-Y[2]) - (Y[1]-Y[3]))/Y[4]
    es_stable = es[np.where(Y[5] == 0)]
    es_random = es[np.where(Y[5] == 1)]
    all_es_stable.extend(es_stable)
    all_es_random.extend(es_random)
    ind_es_stable.append(es_stable)
    ind_es_random.append(es_random)
    mean_es_stable.append(es_stable.mean())
    mean_es_random.append(es_random.mean())
all_es_stable = np.array(all_es_stable)
all_es_random = np.array(all_es_random)
mean_es_stable = np.array(mean_es_stable)
mean_es_random = np.array(mean_es_random)
mean_errors_stable = np.array(mean_errors_stable)
mean_errors_random = np.array(mean_errors_random)

# Get the SPoC decoding measure
#analysis_name = 'prevmovement_preverrors_errors_prevbelief'
analyses = ['Prev_movement', 'Prev_errors', 'Errors', 'Prev_belief']

time_locked = 'target'
freqs = ['broad', 'theta', 'alpha', 'beta', 'gamma']
environment = ['stable', 'random']
control = 'b2b'  # 'classic'
for freq_name in freqs:
    all_scores_stable = list()
    all_scores_random = list()
    for subject in subjects:
        # results_folder = 'decoding_no_hpass_no_bsl'
        results_folder = output_folder
        for env in environment:
            if control == 'classic':
                fname = '%s_%sscores_%s_%s.npy' % (subject,
                                                   env,
                                                   analysis_name,
                                                   freq_name)
            if control == 'b2b':
                fname = '%s_%spartial_scores_%s_%s.npy' % (subject,
                                                           env,
                                                           analysis_name,
                                                           freq_name)
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
    type = 2 * ([analyses[0]] * nb_sub + [analyses[1]] * nb_sub + [analyses[2]] *
                nb_sub + [analyses[3]] * nb_sub)
    cond = ['Stable'] * 4 * nb_sub + ['Random'] * 4 * nb_sub
    data = pd.DataFrame({'Decoding Performance': scores,
                        'Condition': cond, 'Type': type})
    my_pal = {'Stable': colors[0], 'Random': colors[1]}

    prev_error_data = data[data['Type'] == 'Prev_errors']
    mean_preverror_stable = prev_error_data['Decoding Performance'][prev_error_data['Condition']=='Stable']
    mean_preverror_random = prev_error_data['Decoding Performance'][prev_error_data['Condition']=='Random']

    # Plot correlation
    all_preverror = np.concatenate((np.array(mean_preverror_stable),
                                   np.array(mean_preverror_random)))
    all_es = np.concatenate((mean_es_stable, mean_es_random))
    # Compute linear regression
    slope_all, intercept_all, r_all, p_all, _ = linregress(all_preverror,
                                                           all_es)
    line_all = slope_all*all_preverror+intercept_all
    slope_stable, intercept_stable, r_stable, p_stable, _ = linregress(mean_preverror_stable,
                                                                       mean_es_stable)
    line_stable = slope_stable*mean_preverror_stable+intercept_stable
    slope_random, intercept_random, r_random, p_random, _ = linregress(mean_preverror_random,
                                                                       mean_es_random)
    line_random = slope_random*mean_preverror_random+intercept_random
    # Plot dot for stable and random
    plt.plot(mean_preverror_stable, mean_es_stable, 'o', color=colors[0])
    plt.plot(mean_preverror_random, mean_es_random, 'o', color=colors[1])
    # Plot line for stable and random
    plt.plot(mean_preverror_stable, line_stable, color=colors[0])
    plt.plot(mean_preverror_random, line_random, color=colors[1])
    # Plot line for all
    min = np.min(all_preverror)
    max = np.max(all_preverror)
    aa = np.arange(min, max, (max-min)/10.)
    np.ravel([colors]*5)
    for aa1, aa2, color in zip(aa[:-1], aa[1:], np.ravel([colors]*5)):
        wh = np.where((all_preverror > aa1) & (all_preverror < aa2))
        plt.plot(all_preverror[wh], line_all[wh], color=color)

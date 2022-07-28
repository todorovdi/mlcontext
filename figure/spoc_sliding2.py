import os, sys
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from config2 import stage2time_bounds,genFnSliding
from config2 import subjects, path_fig, path_data
from base2 import decod_stats
import seaborn as sns
from config2 import analysis_name2var_ord

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
plt.rc('font', size=SMALL_SIZE)          # decoding_types default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



#path_data = '/Volumes/data/MemErrors/data2/'
#hpass = '0.1'  # '0.1', 'detrend', no_hpass
decoding_types = ['classic', 'b2b']
ICA = 'with_ICA'
control_type = 'movement'
#is_short = True

def getAnalysisName(time_locked,control_type):
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
    r = {}
    r['target']  = analyses_name_target
    r['feedback']= analyses_name_feedback
    return r[time_locked]

#if is_short:
#    time_lockeds = [['feedback', analyses_name_feedback,
#                     np.linspace(-0.2, 3, 321)]]
#    save_folder = 'Exp2_td_short_%s_%s_%s' % (hpass, control_type, ICA)
#    output_folder = 'td_long2_%s_short_%s' % (hpass, ICA)
#else:
#    time_lockeds = [['feedback', analyses_name_feedback,
#                     np.linspace(-2, 5, 701)],
#                    ['target', analyses_name_target,
#                     np.linspace(-5, 2, 701)]]
#    save_folder = 'Exp2_td_long_%s_%s_%s' % (hpass, control_type, ICA)
#    output_folder = 'td_long2_%s_%s' % (hpass, ICA)

save_folder = 'Exp2_sliding_%s_%s_%s' % (hpass, control_type, ICA)

if not op.exists( op.join(path_fig, save_folder) ):
    os.mkdir(op.join(path_fig, save_folder) )

# Plot errors in each environment together
decoding_types = ['classic', 'b2b']

files_missing = []

time_lockeds = ['target', 'feedback']
envs = ['stable','random']
#envs = ['stable','random','all']  # so far I did not compute for 'all'
env2color = dict( zip(envs,colors) )


#shift = 0.25
#dur   = 0.464
#shift = 0.116
#dur   = 0.464

def collectResults(subject,hpass):
    output_folder = f'spoc_sliding_{hpass}'
    from config2 import freq_name2freq
    import re
    freqstr = '|'.join( freq_name2freq.keys() )

    dir_full = op.join(path_data, subject, 'results', output_folder)
    fns = os.listdir ( dir_full )
    tuples = []
    regex = re.compile(r'(stable|random)_(Ridge|xgboost)_(feedback|target)_(.*)_(' +freqstr+ ')_t=(.*),(.*)\.npz')
    for fn in fns:
        r = re.match( regex, fn)
        #print(r )
        if r is None:
            print(f'wrong fn = {fn}')
        grps = r.groups(); #print(grps)
        env_cur,rt_cur,time_locked_cur,analysis_name_cur,freq_name_cur,tmin,tmax = grps

        if (regression_type,freq_name) != (rt_cur,freq_name_cur):
            continue

        fn_full = op.join(dir_full,fn)
        f = np.load( fn_full, allow_pickle=1)
        tuples += [ (os.stat(fn_full).st_mtime, fn, fn_full, f['par'][()], *grps )  ]
        del f

    srt = sorted(tuples, key=lambda x: x[0])

    par = pars[-1]
    if par['slide_windows_type'] == 'auto':
        from config2 import stage2time_bounds
        start, end = stage2time_bounds[time_locked]
        start, end = eval( par.get(f'time_bounds_slide_{time_locked}', (start,end) ) )
        shift = par.get('slide_window_shift',None)
        dur = par.get('slide_window_dur', None)
        tmins = np.arange(start,end,shift)
        tmaxs = dur + tmins

        tminmax = zip(tmins,tmaxs)

        #tuples_a.shape
    import pandas as pd
    df_collect = pd.DataFrame(tuples,
        columns=['mtime', 'fn', 'fn_full', 'par','env','rt','time_locked','analysis_name','freq_name','tmin','tmax' ] )

    return df_collect


fnames_failed = []
#ss = []

tpls = []

for time_locked in time_lockeds:
    #start, end = stage2time_bounds[time_locked]
    #tmins = np.arange(start,end,shift)
    #tmaxs = dur + tmins
    #tminmaxs = list(zip(tmins,tmaxs) )
    #analysis_name = getAnalysisName(time_locked,control_type)


    for decoding_type in decoding_types:
        #all_scores_all = list()
        #all_scores_stable = list()
        #all_scores_random = list()
        env2all_scores = dict(   zip(envs, [[]]*len(envs) ) )
        for env in envs:
            #sub_df['fn_full']
            fnames_cur =  []
            #env2all_scores[env] = list with len = len(subjects)
            scores_cur_env = []
            for subject in subjects:
                df_collect = collectResults(subject,hpass)
                sub_df = df_collect[ (df_collect['time_locked'] == time_locked) &\
                                    (df_collect['env'] == env) ]
                tmins = sub_df['tmin']
                tmaxs = sub_df['tmax']


                scores_cur_env_cur_subj = []
                for  fname_full, tmin in sub_df[ ['fn_full','tmin'] ].to_numpy():
                #for ti,(tmin_cur,tmax_cur) in enumerate(tminmaxs):
                    #s =  f'{env}_{subject}_{ti}'
                    #ss += [s]
                    #results_folder = op.join(path_data,subject,'results',output_folder)
                    #fname_full = genFnSliding(results_folder, env,regression_type,
                    #    time_locked,analysis_name,freq_name,tmin_cur,tmax_cur)

                    if not os.path.exists(fname_full):
                        print(f'ERROR: {fname_full} does not exist, skipping')
                        fnames_failed.append(fname_full)
                        continue
                    else:
                        f = np.load(fname_full)
                        print(f'INFO: loaded {fname_full}')
                        fnames_cur.append(fname_full)
                    if decoding_type == 'classic':
                        sc = f['scores']
                    elif decoding_type == 'b2b':
                        sc = f['partial_scores']

                    sc_aug = np.empty( len(sc) + 1)
                    sc_aug[:len(sc)] = sc
                    sc_aug[len(sc)] = float(tmin)
                    scores_cur_env_cur_subj.append( sc_aug  )
                #    ############ end of cycle over time
                #for ti,(tmin_cur,tmax_cur) in enumerate(tminmaxs):
                #    s =  f'{env}_{subject}_{ti}'
                #    ss += [s]
                #    results_folder = op.join(path_data,subject,'results',output_folder)
                #    fname_full = genFnSliding(results_folder, env,regression_type,
                #        time_locked,analysis_name,freq_name,tmin_cur,tmax_cur)

                #    if not os.path.exists(fname_full):
                #        print(f'ERROR: {fname_full} does not exist, skipping')
                #        fnames_failed.append(fname_full)
                #        continue
                #    else:
                #        f = np.load(fname_full)
                #        print(f'INFO: loaded {fname_full}')
                #        fnames_cur.append(fname_full)
                #    if decoding_type == 'classic':
                #        sc = f['scores']
                #    elif decoding_type == 'b2b':
                #        sc = f['partial_scores']

                #    scores_cur_env_cur_subj.append(sc)
                #    ############ end of cycle over time
                scores_cur_env.append(np.array(scores_cur_env_cur_subj) )
                ############ end of cycle over subjects
            #shape = (20, 61, 5)#
            a = np.array( scores_cur_env )
            # to get vars x subj x times
            aa = a.transpose( (2,0,1))
            env2all_scores[env] = aa
            assert  env2all_scores[env].shape[1] == len(subjects)
        assert  env2all_scores['stable'].shape[1] == len(subjects)

        tpls += [ (time_locked,decoding_type,env2all_scores) ]

                #if decoding_type == 'classic':
                #    fname_all = f'all_{regression_type}_scores_%s_%s.npy' % ( time_locked[0],
                #                                            time_locked[1])
                #    fname_stable = f'stable_{regression_type}_scores_%s_%s.npy' % ( time_locked[0],
                #                                                  time_locked[1])
                #    fname_random = f'random_{regression_type}_scores_%s_%s.npy' % ( time_locked[0],
                #                                                  time_locked[1])
                #elif decoding_type == 'b2b':
                #    fname_all = f'all_{regression_type}_partial_scores_%s_%s.npy' % ( time_locked[0],
                #                                                    time_locked[1])
                #    fname_stable = f'stable_{regression_type}_partial_scores_%s_%s.npy' % ( time_locked[0],
                #                                                       time_locked[1])
                #    fname_random = f'random_{regression_type}_partial_scores_%s_%s.npy' % ( time_locked[0],
                #                                                          time_locked[1])
                #fn = op.join(path_data, subject, 'results',
                #                         results_folder, fname_all)
                #if not os.path.exists( fn):
                #    print(f'WARNING missing {fn}')
                #    files_missing += [fn]
                #    continue
                #sc_all = np.load(op.join(path_data, subject, 'results',
                #                         results_folder, fname_all))
                #sc_stable = np.load(op.join(path_data, subject, 'results',
                #                            results_folder, fname_stable))
                #sc_random = np.load(op.join(path_data, subject, 'results',
                #                            results_folder, fname_random))
                #all_scores_all.append(sc_all)
                #all_scores_stable.append(sc_stable)
                #all_scores_random.append(sc_random)

        if not len(env2all_scores['stable'] ) :
            print('ERROR: no scores found')
            sys.exit(1)


        analysis_name = sub_df['analysis_name'].to_list()[0]
        var_order = analysis_name2var_ord[analysis_name]

        for vari,varn in enumerate(var_order):
            plt.title(f'{decoding_type} decoding {varn}', fontsize=14)
            for env,scs in env2all_scores.items():
                scs_all = np.array(scs)
                tmins_all = scs[-1,:,:]
                tmins_ = tmins_all[0]  # take of the first subject
                scs = scs_all[:-1,:,:]

                vals = scs[vari,...]
                ys_ = vals.mean(axis=0)  # mean over subjects

                # tmins are not sorted
                a = sorted( zip(tmins_,ys_), key=lambda x: x[0] )
                tmins,ys = zip(*a)

                plt.plot(tmins, ys, color=env2color[env], label=f'{env} environment', linewidth=0.5)
                sig = decod_stats(vals) < 0.05
                plt.fill_between(tmins, ys, where=sig, color=env2color[env], alpha=0.3)
                # Plot errors during random environment
                #plt.plot(times, all_scores_random[1].mean(0), color=colors[1], label='Random environment', linewidth=0.5)
                #sig = decod_stats(all_scores_random[1]) < 0.05
                #plt.fill_between(times, all_scores_random[1].mean(0), where=sig, color=colors[1], alpha=0.3)

            plt.legend()
            fname_fig = op.join(path_fig,save_folder,
                f'Exp2_{regression_type}_{freq_name}_{time_locked}_{decoding_type}_{varn}' )
            plt.savefig(fname_fig, dpi=400)
            plt.close()

print('First fig finsihed')
############################################################################
############################################################################

#sys.exit(0)

            #all_scores_all = np.array(all_scores_all)
            #all_scores_stable = np.array(all_scores_stable)
            #all_scores_random = np.array(all_scores_random)
            #all_scores_all = np.moveaxis(all_scores_all, 1, 0)
            #all_scores_stable = np.moveaxis(all_scores_stable, 1, 0)
            #all_scores_random = np.moveaxis(all_scores_random, 1, 0)

            # Plot
            # Plot errors
            # Plot errors during all environment
            # plt.plot(times, all_scores_all[1].mean(0), color=colors[3], label='All trials', linewidth=0.5)
            # sig = decod_stats(all_scores_all[1]) < 0.05
            # plt.fill_between(times, all_scores_all[1].mean(0), where=sig, color=colors[3], alpha=0.3)

            ## Plot errors during stable environment
            #plt.plot(times, all_scores_stable[1].mean(0), color=colors[0], label='Stable environment', linewidth=0.5)
            #sig = decod_stats(all_scores_stable[1]) < 0.05
            #plt.fill_between(times, all_scores_stable[1].mean(0), where=sig, color=colors[0], alpha=0.3)
            ## Plot errors during random environment
            #plt.plot(times, all_scores_random[1].mean(0), color=colors[1], label='Random environment', linewidth=0.5)
            #sig = decod_stats(all_scores_random[1]) < 0.05
            #plt.fill_between(times, all_scores_random[1].mean(0), where=sig, color=colors[1], alpha=0.3)
            #plt.legend()
            #fname_fig = op.join(path_fig,save_folder,f'Exp2_{regression_type}_%s_%s_errors' % (time_locked[0], decoding_type) )
            #plt.savefig(fname_fig, dpi=400)
            #plt.close()


np.savez( pjoin(path_fig,save_folder, f'{regression_type}_{freq_name}_gathered.npz'), time_locked_decoding_type_env2scores=tpls)

for (time_locked,decoding_type,env2all_scores) in tpls:

    analysis_name = getAnalysisName(time_locked,control_type)
    var_order = analysis_name2var_ord[analysis_name]

    for vari,varn in enumerate(var_order):
        env2scs = {}
        for env,scs in env2all_scores.items():
            scs_all = np.array(scs)
            tmins_all = scs[-1,:,:]
            tmins_ = tmins_all[0]  # take of the first subject
            scs = scs_all[:-1,:,:]

            vals_ = scs[vari,...]
            #ys_ = vals.mean(axis=0)  # mean over subjects

            # tmins are not sorted
            a = sorted( zip(tmins_,vals_.T), key=lambda x: x[0] )
            tmins,vals = zip(*a)
            vals = np.array(vals).T

            env2scs[env] = np.array(vals)

        # TODO: use argsort to solve
        diff = env2scs['stable'] - env2scs['random']

        plt.title(f'{decoding_type} decoding {varn} differences', fontsize=14)
        # Plot errors during stable environment
        plt.plot(tmins, diff.mean(0), color=colors[0], linewidth=0.5)
        sig = decod_stats(diff) < 0.05
        plt.fill_between(tmins, diff.mean(0), where=sig, color=colors[0], alpha=0.3)
        fname_fig = op.join( path_fig, save_folder,
            f'Exp2{varn}_{regression_type}_{freq_name}_{time_locked}_{decoding_type}' )
        plt.savefig(fname_fig, dpi=400)
        print(f'Fig saved to {fname_fig}')
        plt.close()

print('Second fig finsihed')

#for time_locked in time_lockeds:
#    start, end = stage2time_bounds[time_locked]
#    tmins = np.arange(start,end,shift)
#    tmaxs = dur + tmins
#
#    for decoding_type in decoding_types:
#        #envs = ['stable','random','all']  # so far I did not compute for 'all'
#        envs = ['stable','random']
#        env2all_scores = dict(   zip(envs, [[]]*len(envs) ) )
#        for subject in subjects:
#            for env in ['stable','random']:
#
#                results_folder = op.join(path_data,subject,'results',output_folder)
#                fname_full = genFnSliding(results_folder, env,regression_type,
#                    time_locked,analysis_name,freq_name,tmin_cur,tmax_cur)
#                f = np.load(fname_full)
#                if decoding_type == 'classic':
#                    sc = f['scores']
#                elif decoding_type == 'b2b':
#                    sc = f['partial_scores']
#
#                env2all_scores[env].append(sc)
#
#            for env,scs in env2all_scores.items():
#                scs = np.array(scs)
#                scs = np.moveaxis(scs, 1, 0)
#                env2all_scores[env ] = scs
#            #all_scores_stable = np.moveaxis(all_scores_stable, 1, 0)
#            #all_scores_random = np.moveaxis(all_scores_random, 1, 0)
#            diff = env2all_scores['stable'][1] - env2all_scores['random'][1]
#            # Plot
#            # Plot errors
#            plt.title('%s decoding error differences' % decoding_type, fontsize=14)
#            # Plot errors during stable environment
#            plt.plot(tmins, diff.mean(0), color=colors[0], linewidth=0.5)
#            sig = decod_stats(diff) < 0.05
#            plt.fill_between(tmins, diff.mean(0), where=sig, color=colors[0], alpha=0.3)
#            fname_fig = op.join( path_fig, save_folder, f'Exp2_{regression_type}_%s_%s_errors_diff' % (time_locked[0], decoding_type) )
#            plt.savefig(fname_fig, dpi=400)
#            print(f'Fig saved to {fname_fig}')
#            plt.close()




# Plot errors difference in stable and random
#decoding_types = ['classic', 'b2b']
#for time_locked in time_lockeds:
#    times = time_locked[2]
#    for decoding_type in decoding_types:
#        all_scores_stable = list()
#        all_scores_random = list()
#        for subject in subjects:
#            results_folder = output_folder
#            if decoding_type == 'classic':
#                fname_stable = f'stable_{regression_type}_scores_%s_%s.npy' % ( time_locked[0],
#                                                              time_locked[1])
#                fname_random = f'random_{regression_type}_scores_%s_%s.npy' % ( time_locked[0],
#                                                              time_locked[1])
#            elif decoding_type == 'b2b':
#                fname_stable = f'stable_{regression_type}_partial_scores_%s_%s.npy' % ( time_locked[0],
#                                                                   time_locked[1])
#                fname_random = f'random_{regression_type}_partial_scores_%s_%s.npy' % ( time_locked[0],
#                                                                      time_locked[1])
#            sc_stable = np.load(op.join(path_data, subject, 'results/',
#                                        results_folder, fname_stable))
#            sc_random = np.load(op.join(path_data, subject, 'results/',
#                                        results_folder, fname_random))
#            all_scores_stable.append(sc_stable)
#            all_scores_random.append(sc_random)
#        all_scores_stable = np.array(all_scores_stable)
#        all_scores_random = np.array(all_scores_random)
#        all_scores_stable = np.moveaxis(all_scores_stable, 1, 0)
#        all_scores_random = np.moveaxis(all_scores_random, 1, 0)
#        diff = all_scores_stable[1] - all_scores_random[1]
#        # Plot
#        # Plot errors
#        plt.title('%s decoding error differences' % decoding_type, fontsize=14)
#        # Plot errors during stable environment
#        plt.plot(times, diff.mean(0), color=colors[0], linewidth=0.5)
#        sig = decod_stats(diff) < 0.05
#        plt.fill_between(times, diff.mean(0), where=sig, color=colors[0], alpha=0.3)
#        fname_fig = op.join( path_fig, save_folder, f'Exp2_{regression_type}_%s_%s_errors_diff' % (time_locked[0], decoding_type) )
#        plt.savefig(fname_fig, dpi=400)
#        print(f'Fig saved to {fname_fig}')
#        plt.close()


# # Plot all analyses together separately for each environment
# envs = ['all', 'stable', 'random']
# decoding_types = ['classic', 'b2b']
# for env in envs:
#     for time_locked in time_lockeds:
#         times = time_locked[2]
#         for decoding_type in decoding_types:
#             all_scores = list()
#             for subject in subjects:
#                 results_folder = output_folder
#                 if decoding_type == 'classic':
#                     fname = '%s_%sscores_%s_%s.npy' % (subject, env,
#                                                        time_locked[0],
#                                                files_missing        time_locked[1])
#                 elif decoding_type == 'b2b':
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
#                 plt.title('%s %s %s' % (env, decoding_type, time_locked[1]))
#                 plt.plot(times, scores.mean(0), color=colors[ii],
#                          label=str(ii))
#                 sig = decod_stats(scores) < 0.05
#                 plt.fill_between(times, scores.mean(0),
#                                  where=sig, color=colors[ii], alpha=0.3)
#                 plt.legend()
#             plt.savefig('/Users/quentinra/Desktop/figs_memerror/%s/Exp2_%s_%s_%s_%s' % (save_folder, env, decoding_type,
#                                                                                         time_locked[0], time_locked[1]),
#                         dpi=400)
#             plt.close()

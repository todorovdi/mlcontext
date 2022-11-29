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
from base2 import (calc_target_coordinates_centered, radius, radius_target,
                   radius_cursor,   decod_stats)
from config2 import subjects, path_fig, path_data
from scipy.stats import ttest_1samp, spearmanr
from pingouin import partial_corr

# plots is a file in the same dir. When I run this script from jupyter using
# run magic (beforehand adding PARENT dir in sys.path) it works
from plots import *

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
save_folder   = f'corr_spoc_es_sliding2_{hpass}'
#/p/project/icei-hbp-2020-0012/lyon/memerr/data2/sub08_TXVPROYY/results/spoc_sliding_no_filte
if not op.exists( op.join(path_fig, save_folder) ):
    os.mkdir(op.join(path_fig, save_folder) )

#fname_full = genFnSliding(results_folder, env,
#                regression_type,time_locked,
#                analysis_name,freq_name,tmin_cur,tmax_cur)
if DEBUG:
    #subjects = subjects[:1]
    print("--------------   DEBUG MODE  --------------")
    print("--------------   DEBUG MODE  --------------")
    print("--------------   DEBUG MODE  --------------")

fnames_failed = []

#subject_inds
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

if isinstance(env,str):
    envstr = env
else:
    envstr = ','.join(env)

fname_df_full = pjoin(path_fig, save_folder, f'df_{envstr}')

print('varnames = ',varnames)
varnames_elem = list(set(varnames) - set(['err_sens','correction']) )

df_mode = 'preloaded'
if not use_preload_df:
    if load_df and os.path.exists(fname_df_full):
        df = pd.read_pickle( fname_df_full )

        df_mode = 'loaded'
    else:
        for subject in np.array(subjects)[subject_inds]:
            print(f'Starting collecting subject {subject}')
            #df_collect = collectResults(subject,hpass)
            #df_collect = collectResults(subject,output_folder,freq_name,
            #        keys_to_extract = ['par','scores_err_sens','diff_err_sens_pred',
            #                           'err_sens', 'prev_error', 'correction'] )
            # , 'non_hit' is not per dec var
            df_collect1 = collectResults(subject,output_folder,freq_name,
                    keys_to_extract = ['par'], env=env,
                    time_locked = time_locked,
                    time_start=time_start, time_end=time_end )
            if df_collect1 is None or len(df_collect1) == 0:
                print(f'Found nothing for subject {subject}')
                continue
            fns = list( df_collect1['fn'] )

            df_collect2 = collectResults(subject, output_folder, freq_name,
                    keys_to_extract = varnames,
                                        parent_key = 'decoding_per_var',
                                        df = df_collect1.copy(),
                                        time_start=time_start, time_end=time_end)
            df_collect2['dec_type'] = 'classic'

            df_collect3 = collectResults(subject, output_folder, freq_name,
                    keys_to_extract = varnames_elem,
                                        parent_key = 'decoding_per_var_b2b',
                                        df = df_collect1,
                                        time_start=time_start, time_end=time_end)
            df_collect3['dec_type'] = 'b2b'

            vnpostfix = '_varnames'
            for ci in np.where(df_collect3.columns.str.endswith(vnpostfix))[0]:
                bn = df_collect3.columns[ci][:-len(vnpostfix)]
                #TODO: problem here: if I have same varname decoded
                # in two different sets of vars for b2b, it will overwrite
                # so maybe I have to duplicate df_collect3 for each set
                order = df_collect3[f'{bn}{vnpostfix}']._values[0]
                order_sc = [f'{s}_scores' for s in order]
                def f(x):
                    vns = x[f'{bn}_varnames']
                    scores = x[f'{bn}_scores']
                    #correction,error_varnames
                    r = []
                    for vn,sc in zip(vns,scores):
                        #x[ f'{vn}_scores'] = [sc]  # one-el list
                        r += [sc]
                    return tuple(r)
                df_collect3[order_sc ] = df_collect3.apply(f,1, result_type = 'expand')

                order_vals = [f'{s}_vals' for s in order]
                def f(x):
                    vns = x[f'{bn}_varnames']
                    vals = x[f'{bn}_vals']
                    #correction,error_varnames
                    r = []
                    for vi in range( vals.shape[1]):
                        #x[ f'{vn}_scores'] = [sc]  # one-el list
                        r += [ vals[:,vi] ]
                    return tuple(r)
                df_collect3[order_vals ] = df_collect3.apply(f,1, result_type = 'expand')
            # TODO: some info from params
            # trim_outliers
            # trial_group_col_calc
            # freq_name
            # time_locked

            #df_collect2['perturbation'] = np.nan
            #df_collect3 = collectResults(subject, output_folder, freq_name,
            #        keys_to_extract = varnames,
            #                            parent_key = 'decoding_per_var_and_pert',
            #                            df = df_collect2,
            #                            cokey = 'perturbation',
            #                            time_start=time_start, time_end=time_end)

            #for kte in set(df_collect2.columns) - set(df_collect1.columns):
            #    df_collect1[kte] = df_collect2[kte]

            #df_persubj += [df_collect2] #, df_collect3]
            df_persubj += [df_collect2, df_collect3] #, df_collect3]
            if not DEBUG:
                del df_collect1
                del df_collect2
                del df_collect3
            #tmins = df_persubj[df_persubj[env] == 'stable' ]   ['tmin']
        df = pd.concat( df_persubj, ignore_index=True )
        #df.reset_index(inplace=True)
        #df.drop('index',1,inplace=True)

        df_mode = 'new'

assert set( df['subject'] ) == set(np.array(subjects)[subject_inds])

row  =df.iloc[0]
for kte in varnames:
    if  len( row[f'{kte}_vals'] ) > len(row[f'{kte}_diff']):
        df[f'{kte}_vals'] = df.apply(lambda x: x[f'{kte}_vals'][x['{kte}_non_hit']],1)

#df = df.astype(

# they are still strings here
tmins_srt = sorted( set( df['tmin'] ), key=lambda x: float(x) )
if DEBUG:
    tmins_srt = tmins_srt[:1]

scores_cur_env = []
fnames_cur = []

compute_aux = 0
##################################   Add columns to the database
# compute correlation between  differences (between prediction and reality) and error sens
if compute_aux:
    def f(x):
        if x['err_sens_vals'] is not None:
            r = x['err_sens_vals']#[0]
        else:
            r = np.nan
        if len(x['prev_error_vals'])  == len(r): #, (len(x['prev_error_vals']), len(r))
            res = spearmanr(    np.abs(x['prev_error_vals']), r )
        else:
            res = np.nan
        return res
    df['pp:abspe_es:r']      = df.apply(f, axis=1 )


    df['pp:diff_es:r']       = df.apply(lambda x: spearmanr(
        x['err_sens_diff'], x['err_sens_vals'])[0] if x['err_sens_vals'] is not None else np.nan, axis=1 )
    # compute correlation between  differences (between prediction and reality) and error sens
    df['pp:diff_abscorr:r']  = df.apply(lambda x: spearmanr(
        x['correction_diff'], np.abs(x['correction_vals']) )[0] if x['err_sens_vals'] is not None else np.nan, axis=1 )
    #df['pp:abspe_es:r']      = df.apply(lambda x: spearmanr(
    #    np.abs(x['prev_error_vals']), x['err_sens_vals'])[0] if x['err_sens_vals'] is not None else np.nan, axis=1 )

    df['abspe'] = df['prev_error_vals'].apply(lambda x: np.abs(x) if x is not None else None )

    # Partial correlation [1] measures the degree of association between x and y,
    # after removing the effect of one or more controlling variables (covar, or Z).
    # Practically, this is achieved by calculating the correlation coefficient
    # between the residuals of two linear regressions:
    # x∼Z, y∼Z

    # Returns statspandas.DataFrame
    # 'n': Sample size (after removal of missing values)
    # 'r': Partial correlation coefficient
    # 'CI95': 95% parametric confidence intervals around r
    # 'p-val': p-value

    def pcorr(row):
        colnames = [ 'err_sens_diff','err_sens_vals', 'abspe' ]
        vals = [ row[coln] for coln in colnames ]
        if vals[0] is None:
            return np.nan, np.nan
        data = pd.DataFrame( dict( zip(colnames,vals) ) )
        results = partial_corr(data=data, x=colnames[0], y=colnames[1], covar=[colnames[2]],
                                method='spearman')

        #x['err_sens_diff'], x['err_sens_vals'], x['prev_error_vals']
        #data = np.vstack( [diff, err_sens, np.abs(pe)]).T
        #data = pd.DataFrame(data=data, columns=['diff_err_sens_pred', 'err_sens', 'abspe'])
        #results = partial_corr(data=data, x='diff_err_sens_pred', y='err_sens', covar=['abspe'],
        #                        method='spearman')
        #all_r_stable.append(results.values[0, 1])
        #all_p_stable.append(results.values[0, 3])
        return results.values[0,1], results.values[0,3]



    df[ ['pp:diff_es_pe:r','pp:diff_es_pe:p'] ]    = df.apply(pcorr,1,
                                        result_type='expand')

#df['pp:diff_pe_es:r']    = df.apply(lambda x: pcorr_r(x['diff_err_sens_pred'], x['err_sens'], x['prev_error']), axis=1 )
#df['pp:diff_pe_es:p']    = df.apply(lambda x: pcorr_p(x['diff_err_sens_pred'], x['err_sens'], ['prev_error']), axis=1 )
ppnames = [ 'pp:diff_es_pe:r', 'pp:diff_es:r',    'pp:diff_abscorr:r', 'pp:abspe_es:r']
#########################################

# The one-sample t-test is a statistical hypothesis test used to determine
# whether an unknown population mean is different from a specific value.
# it assumes normality

# The Spearman rank-order correlation coefficient is a nonparametric measure of
# the monotonicity of the relationship between two datasets. Unlike the Pearson
# correlation, the Spearman correlation does not assume that both datasets are
# normally distributed. Like other correlation coefficients, this one varies
# between -1 and +1 with 0 implying no correlation. Correlations of -1 or +1
# imply an exact monotonic relationship. Positive correlations imply that as x
# increases, so does y. Negative correlations imply that as x increases, y
# decreases.

for kte in varnames:
    df[f'{kte}_scores_mean_over_folds'] =  df[f'{kte}_scores'].apply(lambda x: np.mean(x) if x is not None else np.nan)
    df[f'{kte}_scores_std_over_folds']  =  df[f'{kte}_scores'].apply(lambda x: np.std(x) if x is not None else np.nan)

df.to_pickle( fname_df_full )
print(f'df saved to {fname_df_full}')

if exit_after == 'collect':
    sys.exit(0)

tl = list( set(df['time_locked']) )
assert len(tl) == 1
#time_locked = tl[0]



#pertvals = df['perturbation'].unique()
from config2 import pertvals



from behav_proc import getSubjPertSeqCode
df['pert_seq_code'] = df.apply(lambda x: getSubjPertSeqCode(x['subject']), 1 )
#pscvals = [ (0,), (1,), (0,1) ]

suffix_vals = df['custom_suffix'].unique()

for kte in varnames:
    # plot mean across subj
    #nr = len(pertvals)
    #nc = len(pscvals)
    nr = 1
    nc = len(env)
    if plot_diff:
        raise ValueError('need to correct row numbers')
        nr *= 2

    #if isinstance(env,list):
    #    nc *= len(env)
    ww = 5; hh = 3
    fig, axs = plt.subplots(nr,nc,figsize=(nc*ww,nr*hh) , sharex='col',
                            sharey='row' )
    axs = axs.reshape((nr,nc))
    #for psci,psc in enumerate( pscvals ):
    for sufi,suffix in enumerate( suffix_vals ):
        for envi,env_s in enumerate(env):
            #if len(psc ) == 1:
            #    df_curenv = df[ df['pert_seq_code'] == psc[0] ]
            #else:
            #    df_curenv = df
            #df_curenv = df_curenv[ df_curenv['env'] == env_s ]
            df_curenv = df.query(f'env == "{env_s}" and custom_suffix == "{suffix}"' )
            assert len(df)
            df_curenv_srt = df_curenv.sort_values(by=['tmin'],
                            key=lambda x: list(map(float,x) ) )

            #for pvi,pertv in enumerate(pertvals):
            #scores_all = []  # without mean across fold
            scores = []
            diffs = []
            scores_std = []
            diffs_std = []

            #if np.isnan(pertv):
            #    maskp = df_curenv_srt['perturbation'].isnull()
            #else:
            #    maskp = df_curenv_srt['perturbation'] == pertv
            #grp = df_curenv_srt[maskp].groupby( ['tmin'])
            grp = df_curenv_srt.groupby( ['tmin'])

            #display(grp.size(), grp.size())
            gsz = grp.size()[0]
            #if len(psc) == 2:
            #    assert gsz == len(subject_inds), (gsz, len(subject_inds) )
            me  = grp.mean()
            std = grp.std()

            me = me.reset_index().sort_values(by=['tmin'],
                                            key=lambda x: list(map(float,x) ) )
            std = std.reset_index().sort_values(by=['tmin'],
                                            key=lambda x: list(map(float,x) ) )

            #tmins_srt_f = list(map(float,tmins_srt) )
            #tmins_srt_f = list(map(float,tmins_srt) )
            #tmins_srt_f = list( map(float, me['tmin']) )
            #tmins_srt_f = list( map(float, me['tmin']) )
            tmins_srt_f =  me['tmin']

            # these are scores_es from archive, which are spearmanr between y_preds and y
            scores     = np.array(me [f'{kte}_scores_mean_over_folds'] )
            scores_std = np.array(std[f'{kte}_scores_mean_over_folds'] )

            #print(scores)

            pvi = 0
            #ax =axs[pvi, psci * len(env) + envi]
            ax =axs[pvi, envi]
            p = ax.plot(tmins_srt_f  ,scores, label=f'{suffix}, N={gsz}')
            c = p[-1].get_color()
            ax.plot(tmins_srt_f  ,scores - scores_std, ls=':', c= c)
            ax.plot(tmins_srt_f  ,scores + scores_std, ls=':', c=c)
            #ttl   = f'{env_s}, suffix={suffix} N={gsz}:\n {kte} scores {regression_type}'
            ttl   = f'{env_s}:\n {kte} scores {regression_type}'
            ax.set_title(ttl)
            ax.set_xlabel('tmin')

            mask = ( ( scores - scores_std ) >  0 ) |  ( ( scores + scores_std ) < 0 )
            #ax.plot(tmins_srt_f,scores)
            ax.fill_between(tmins_srt_f, scores, where=mask, color=colors[0], alpha=0.3)

            ax.axhline(0, ls='--', c='brown')
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))

            ax.set_ylabel('spearmanr')
            ax.legend()

            if plot_diff:
                # these are diff from archive, which are abs diff y_preds and y
                ax =axs[pvi, envi]
                diffs = np.array(diffs)
                diffs_std = np.array(diffs_std)
                ax.plot(tmins_srt_f  ,diffs)
                ax.plot(tmins_srt_f  ,diffs - diffs_std, ls=':')
                ax.plot(tmins_srt_f  ,diffs + diffs_std, ls=':')
                ax.set_title(f'{env_s}: diffs decoding {kte} {regression_type}')


fname_fig = op.join( path_fig, save_folder,
    f'tmin_vs_scores_{kte}__{envstr}_{regression_type}_{freq_name}_{time_locked}.pdf' )
plt.suptitle( f'{kte} freq={freq_name} {time_locked}' )
plt.savefig(fname_fig, dpi=300)
print(f'Fig saved to {fname_fig}')
if not DEBUG:
    plt.close()


if plot_scores_per_subj:
    for kte in varnames:
        axs = plotScoresPerSubj(df, subjects, env, kte = kte,
                            ww =4 ,hh = 2, ylim=( -0.3,0.3) )


        fname_fig = op.join( path_fig, save_folder,
            f'tmin_vs_scoresPS_{kte}__{regression_type}_{freq_name}_{time_locked}.pdf' )
        plt.savefig(fname_fig, dpi=300)
        print(f'Fig saved to {fname_fig}')
        plt.close()




sys.exit(0)



# TODO: needs overhaul
for kte in varnames:
    for pvi,pertv in enumerate(pertvals):
        for tmin in tmins_srt:
            # it's important that here tmin is a string (comparing floats is a pain)
            df_curwnd = df_curenv_srt[ df_curenv_srt['tmin'] == tmin ]

            #scores.append( df_curwnd.mean['scores'] )
            #diffs.append( df_curwnd.mean['diff_err_sens_pred'] )
            #err_senss.append( df_curwnd.mean['err_sens'] )

            # test whether partial correlations and spearman correlations are nonzero
            # only print, but don't plot
            # Q: so this is across subjects right? whereas the correlation itself
            # is across trials?
            if compute_aux:
                for ppname in ppnames:
                    res = ttest_1samp(df_curwnd[ ppname ] ,0)
                    print(f'{env:6},{tmin:6},{ppname:14}, stat={res.statistic:.3f}, pvalue={res.pvalue:.5f}')

            # mean over subjects
            sc = np.array ( list( df_curwnd[f'{kte}_scores'].to_numpy() ) )

            #scores_all += [sc]

            scm = sc.mean(axis=1) # mean over splits
            scores += [ scm.mean() ]
            scores_std += [ scm.std() ]

            #diff_err_sens_pred = np.array ( list( df_curwnd['diff_err_sens_pred'].to_numpy() ) )
            #diffm = diff_err_sens_pred.mean(axis=1) # mean over time within window
            diffm = df_curwnd[f'{kte}_diff'].apply(lambda x: np.mean(x))
            diffs += [ diffm.mean() ]
            diffs_std += [ diffm.std() ]

            if plot_diff_vs_es:
                nr = 10; nc =2
                ww = 5;  hh = 3
                fig, axs = plt.subplots(nr,nc,figsize=(nc*ww,nr*hh) , sharex='col', sharey='row' )
                axs = axs.ravel()
                assert len(axs) >= len(subjects)
                #for ii in range(len(all_diff)):
                for ii,subject in enumerate(subjects):
                    df_curwnd_cursubj = df_curwnd[ df_curwnd['subject'] == subject ]
                    diff = df_curwnd_cursubj[f'{kte}_diff']
                    err_sens   = df_curwnd_cursubj[f'{kte}_vals']
                    if len(diff) and len(err_sens):
                        assert len(diff) == 1
                        diff = diff.to_numpy()[0]
                        err_sens   = err_sens.to_numpy()[0]
                        ax = axs[ii]
                        ax.plot(diff, err_sens, 'o', alpha=0.1, color=colors[1])
                        #ax.set_xticks([])
                        #ax.set_yticks([])
                        ax.set_title(f'{subject},{env} tmin={tmin} diff vs {kte}')


                fname_fig = op.join( path_fig, save_folder,
                    f'diff_vs_{kte}__{tmin}_{env}_{regression_type}_{freq_name}_{time_locked}.png' )
                plt.savefig(fname_fig, dpi=300)
                print(f'Fig saved to {fname_fig}')
                plt.close()

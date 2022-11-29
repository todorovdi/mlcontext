import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config2 import *
import seaborn as sns
from behav_proc import getTrialGroupName

trial_group_names = {'trialwe':'we', 'trialwb':'wb', 'trials':'no',
        'trialwtgt':'tgt',  'trialwpert_we':'pertwe',
        'trialwpert_wb':'pertwb',
        'trialwtgt_wpert_we':'wtgt_wpert_we',
        'trialwtgt_wpert_wb':'wtgt_wpert_wb',
         'trialwtgt_wb':'tgt_wb',
         'trialwtgt_we':'tgt_we',
         'trialwtgt_wpert':'tgt_wpert',
         'trialwpert':'wpert',
        }



def subjStat(df, coln, trialcol_calc, trialcol_av,
        preserve_trailcol_names = False, res_col_prefix = 'err_sens_',
        preserve_multi_trial_av = False, q=0.05,
        verbose=0):
    #if coln.endswith('envAll'):
    #    assert trialcol == 'trials', (coln, trialcol)
    '''
    coln is what was used for calculation,
    trialcol is what will be used for averaging
    for each trial across subjects (some trialcol give several time per subj)
    '''
    npsc = df['pert_seq_code'].nunique()
    if npsc > 1:
        print(f'subjStat WARNING: pert_seq_code is not unique'
                f'for {coln,trialcol_calc, trialcol_av}')

    filter_cols = ['environment','target_inds',
            'pert_seq_code','block_name','perturbation']


    def f(dfc):
        '''
        input -- database with fixed trials but many subjects
        '''
        a = np.array( dfc.loc[ ~dfc[coln].isnull(), coln ] )
        suba = a[~(np.isnan(a) | np.isinf(a))]

        if npsc == 1:
            p = dfc['perturbation'].values[0]
        else:
            p = np.abs( dfc['perturbation'].values[0] )
#         if dfc['trials'].values[0] == 8:
#             display(dfc,suba)
        #print(df[coln])
        me = np.nan
        std = np.nan
        sem = np.nan
        absme, absstd,abssem = np.nan,np.nan ,np.nan
        if len(suba):
            suba  = np.array(suba)

            me = np.mean(suba)
            std = np.std(suba)
            sem = std / np.sqrt(len(suba))

            asuba = np.abs(suba)
            absme  = np.mean(asuba)
            absstd = np.std(asuba)
            abssem = absstd / np.sqrt(len(asuba))
        #return pd.DataFrame( {coln: [r]} )

        if preserve_trailcol_names:
            calc_col_name = trialcol_calc
            av_col_name = trialcol_av
        else:
            calc_col_name = 'trial_calc'
            av_col_name   = 'trial_av'

        if preserve_multi_trial_av:
            colval_trial_av = [ list(dfc[trialcol_av]) ]
        else:
            colval_trial_av = list(dfc[trialcol_av])[0]
        d = { calc_col_name: list(dfc[trialcol_calc])[0] ,
                  av_col_name: colval_trial_av  ,
                  'perturbation':[p] }
        for fc in filter_cols:
            if fc == 'perturbation':
                continue
            d[fc] = [ dfc[fc].values[0] ]


        d.update({ res_col_prefix + 'mean': [me],
                  res_col_prefix + 'std': [std],
                  res_col_prefix + 'sem':[sem],
                  res_col_prefix + 'absmean': [absme],
                  res_col_prefix + 'asbstd': [absstd],
                  res_col_prefix + 'abssem':[abssem],
                   'nav':[len(suba)] } )
        return pd.DataFrame( d  ) #.mean()


    assert coln in df.columns, coln
    assert np.any(~np.isnan(df[coln])), coln

    cols = [trialcol_calc]
    if trialcol_calc != trialcol_av:
        cols += [trialcol_av]
    cols += [coln] + filter_cols

    if q is not None:
        df = truncateDf(df,coln,q, verbose=verbose)


    grp = df[cols].groupby(trialcol_av)

    if verbose:
        display(grp.count())
    #grpme = grp.apply(lambda g: g.mean(skipna=True))
    grpstat = grp.apply(lambda g: f(g) )
    #for ti,subdf in grp:
    #grpme = grpme.drop('trials',1).reset_index(drop=True)
    grpstat = grpstat.reset_index(drop=True)
    grpstat['N'] = np.array( grp.size() )
    grpstat['nav'] = grpstat['nav'].astype('int')
    grpstat['trialcol_calc'] =  trialcol_calc
    grpstat['trialcol_av'] =  trialcol_av

    return grpstat  #trial_inds_nh, err_sens_nh


def subjStat_old(df, coln, verbose=0):
    assert coln in df.columns, coln
    assert np.any(~np.isnan(df[coln])), coln
    grp = df[['trials',coln]].groupby('trials')

    if verbose:
        display(grp.count())

    #grpme = grp.apply(lambda g: g.mean(skipna=True))
    grpme = grp.apply(lambda g: av(g,coln=coln))
    #for ti,subdf in grp:

    #grpme = grpme.drop('trials',1).reset_index(drop=True)
    grpme = grpme.reset_index(drop=True)
    grpme['N'] = np.array( grp.size() )
    grpme['nav'] = grpme['nav'].astype('int')
    #grpme = grpme.reset_index()
    err_sens_nh   = grpme[coln]
    trial_inds_nh = grpme['trials']
    return trial_inds_nh, err_sens_nh


def cleanErrSensCols(df_all):
    todrop = []
    for col in df_all.columns:
        if col.startswith('err_sens'):
            print(col)
            todrop += [col]
    df_all.drop(labels=todrop,axis=1,inplace=True)

#tinh, err_sens_nh = subjStat(df, coln)
#err_sens_nh

    #{'trials':list(df['trials'])[0],

#av( grp.get_group(5), coln)
#grp.apply(lambda g: av(g,coln))#.reset_index()


#def getColn(pertv, tgti, env, trialgrp, pert_seq_code):
def getColn0(pertv, tgti, env, trialgrp ) :
    # this ver was when I was using same getColn for calc and for av
    coln = 'err_sens'
    if pertv is not None:
        coln += f'_pert{int(pertv)}'
    else:
        coln += '_pertAll'
    if tgti is not None:
        coln += f'_tgt{tgti}'
    else:
        coln += '_tgtAll'
    if env in env2envcode:
        coln += '_envSep'
    else:
        coln += '_envAll'

    #if pert_seq_code is None or \
    #        (isinstance(pert_seq_code, tuple) and len(pert_seq_code) == 2):
    #    coln += f'_pscAll'
    #else:
    #    coln += f'_psc{pert_seq_code}'

    #trial_group_names = {'within_env':'we', 'within_block':'wb', 'no':'no'}
    if trialgrp is not None:
        assert trialgrp in trial_group_names, trialgrp
        coln += '_tg' + trial_group_names[trialgrp]

    return coln


#def getColn(pertv, tgti, env, trialgrp ) :
def getColn(pertv, tgti, env, block, trialgrp ) :
    tgn = getTrialGroupName(pertv, tgti, env, block)
    coln = 'err_sens_' + trial_group_names[tgn]

    if trialgrp is not None:
        assert trialgrp in trial_group_names, trialgrp
        coln += '_tgav=' + trial_group_names[trialgrp]

    return coln

def getMeanErrSensSubDf(dfme,env,block_name,pertv,gseqc,tgti,
        trial_group_col_calc, trial_group_col_av, verbose=0):

    if verbose:
        tpl = env,block_name,pertv,gseqc,tgti,trial_group_col_calc,trial_group_col_av
        print_tpl_statcalc(tpl)
    assert trial_group_col_calc is not None
    if gseqc != (0,1):
        pert_seq_code = str( gseqc[0] )
    else:
        pert_seq_code = '0,1'
    df = getSubDf(dfme, None, pertv,tgti,env, block_name,
            pert_seq_code = pert_seq_code,
            non_hit = False, verbose=verbose, nonenan=True)
    m = df['trial_group_col_calc'] == trial_group_col_calc
    if sum(m) == 0 and verbose:
        print('empty after trial_group_col_calc = ', trial_group_col_calc,
                set(df['trial_group_col_calc'] ))
        display(df)
    df = df[m]

    if trial_group_col_av is not None:
        df = df[df['trial_group_col_av'] == trial_group_col_av]
    return df


def plotErrSens(ax, dfme,env,block_name,pertv,gseqc,tgti,
        trial_group_col_calc, trial_group_col_av,  df=None,
       span_type = 'sem', ylim=(-7,7), respect_trial_inds=True,
       data_type = 'mean'):
    subj = 'mean';
    print(env,pertv,gseqc,tgti,trial_group_col_calc,trial_group_col_av)

    tgtc = ['b','g','r','brown']

    if not( dfme is None and df is not None):
        df = getMeanErrSensSubDf(dfme,env,pertv,block_name,gseqc,tgti,
            trial_group_col_calc, trial_group_col_av)

    #for coln in colns:
    # TODO cycle over pert_seq_code
    #trial_inds = trial_inds[nhna]
    #err_sens   = err_sens[nhna]
    #df = dfme[ (dfme['coln'] == coln) & (dfme['env'] == env) ]

    print(f'  found {len(df)} rows')
    if len(df) == 0:
        print('skipping due to empty df')
        return
#                 if not ( (tgti is None) or (pertv )):
#                     assert len(df) == 1, len(df)
    assert len(df) == 1, len(df)
    for ind,row in df.iterrows():
#                     if tgti is None and np.isnan(row['target_inds'] ):
#                         continue
        if data_type == 'mean':
            err_sens_me_nh   = np.array(row['err_sens_me_nh'])
            err_sens_std_nh  = np.array(row['err_sens_std_nh'])
            err_sens_sem_nh  = np.array(row['err_sens_sem_nh'])
        elif data_type == 'absmean':
            err_sens_me_nh   = np.array(row['err_sens_abs_me_nh'])
            err_sens_std_nh  = np.array(row['err_sens_abs_std_nh'])
            err_sens_sem_nh  = np.array(row['err_sens_abs_sem_nh'])
        trial_inds_nh    = np.array(row['trial_inds_calc_nh'])
        pert_nh          = np.array(row['pert_nh'])
        nav              = np.array(row['nav_nh'])
        coln = row['coln']
        psc_cur = row['pert_seq_code']
        tgti_true = row['target_inds']
        if tgti_true is None:
            col = 'b'
        elif not np.isnan(tgti_true):
            tgti_true = int(tgti_true)
            col = tgtc[tgti_true]
        else:
            col = 'b'
#                     if tgti is not None:
#                         col = tgtc[tgti]
#                     else:
#                         col = 'b'

        if span_type == 'std':
            span = err_sens_std_nh
        else:
            span = err_sens_sem_nh

        #invsqrtnav = 1 / np.sqrt( nav )
        up     = err_sens_me_nh + span #* invsqrtnav
        bottom = err_sens_me_nh - span #* invsqrtnav

        menav = np.nanmean(nav)

        coef     = 0.8 * max(ylim) / 30  #0.21
        coef_nav = 0.9 * max(ylim) / np.max(nav)
        navlbl = f'nav*{coef_nav:.1f} me={menav:.0f}'
        alphanav=0.3
        navpar = dict(lw=0, alpha=alphanav, marker='.',
                    label=navlbl)
        if respect_trial_inds:
            ax.plot(trial_inds_nh, err_sens_me_nh , label=f'tgti={tgti_true}',
                    ls='', marker = 'o' , c= col)
            ax.set_xlabel(trial_group_col_calc)

            ax.fill_between(trial_inds_nh, bottom, up ,
                     color= col, alpha=0.3, )
            ax.plot(trial_inds_nh, pert_nh * coef, c='grey', ls=':', lw=3)
            ax.plot(trial_inds_nh, nav * coef_nav, **navpar)
        else:
            ax.plot(err_sens_me_nh , label=f'tgti={tgti_true}', c= col)
            ax.set_xlabel('trial ind loc')
            ax.fill_between(np.arange(len(err_sens_me_nh)), bottom, up, color= col, alpha=0.3, )

            ax.plot(pert_nh * coef, c='grey', ls=':', lw=3)
            ax.plot(nav * coef_nav, **navpar)

        me = np.nanmean(err_sens_me_nh)
        std = np.nanstd(err_sens_me_nh)
        ma = np.nanmax(err_sens_me_nh)
        ax.axhline( me, ls='--',    c= col, alpha=0.5,
                   label = f'mean={me:.1f}, max={ma:.1f}')
        #ax.axhline( me-std, ls=':', c= col, alpha=0.5, label = f'std = {std:.1f}' )  # too large
        #ax.axhline( me+std, ls=':', c= col, alpha=0.5 )

        ax.axhline(0, ls=':')

        if env in env2envcode and trial_group_col_calc == 'trialwe':
            ax.axvline(192,ls=':')


        ax.set_title(f'{subj[:5]} env={env} pert={pertv} tgti={tgti} psc={psc_cur}'
                f'\n{coln}')

#                     if pertv is not None:
#                         ax.axhline(pertv / 10., ls=':', c='red', label='pertv / 10')
    ax.set_ylim(ylim)
    ax.legend()

    return df


def computeStat(df_all,env,bn,pertv,gseqc,tgti,
        dist_rad_from_prevtgt,dist_trial_from_prevtgt,
        trial_group_col_calc,
        trial_group_col_av,
        colns_set = None, verbose=0):
    tpl = env,bn,pertv,gseqc,tgti,\
            dist_rad_from_prevtgt,dist_trial_from_prevtgt,\
            trial_group_col_calc,trial_group_col_av
    print( sprintf_tpl_statcalc(tpl) )
    pert_seq_code = None
    if gseqc != (0,1):
        pert_seq_code = gseqc[0]
    df = getSubDf(df_all, 'mean', pertv,tgti,env, bn,
            pert_seq_code,
            dist_rad_from_prevtgt,dist_trial_from_prevtgt,
            non_hit = False, verbose=verbose)
    if (len(df) == 0):
        #rowi += 1
        return None
    trial_group_col_calc = getTrialGroupName(pertv, tgti, env, bn)
    coln_calc = getColn(pertv, tgti, env, bn, None)
    if colns_set is not None:
        assert coln_calc in colns_set
    coln_av = getColn(pertv, tgti, env, bn, trial_group_col_av)
    grpstat = subjStat(df, coln_calc,
                       trial_group_col_calc, trial_group_col_av)
    #pscstr = ','.join( list( map(str, gseqc ) ) )
    ## some things are for filtering, some for actual usage,
    ## that's why pertrubation is pertv and pert_nh is actual vals
    #r = { 'environment':env, 'block_name':bn,
    #     'perturbation':pertv, 'pert_seq_code':pscstr,
    #     'pert_nh':grpstat['perturbation'],
    #     'target_inds':tgti,  'nav_nh': grpstat['nav'],
    #     'trial_inds_calc_nh':grpstat[trial_group_col_av],
    #     'trial_inds_av_nh':grpstat[trial_group_col_av],
    #     'err_sens_me_nh':grpstat['mean'],
    #     'err_sens_std_nh':grpstat['std'],
    #     'err_sens_sem_nh':grpstat['sem'],
    #     'err_sens_abs_me_nh': grpstat['absmean'],
    #     'err_sens_abs_std_nh':grpstat['absstd'],
    #     'err_sens_abs_sem_nh':grpstat['abssem'],
    #     'coln':coln_av, 'trial_group_col_calc':trial_group_col_calc,
    #     'trial_group_col_av':trial_group_col_av }
    return grpstat, coln_calc, coln_av


def computeStat_old(df_all,env,bn,pertv,gseqc,tgti,trial_group_col_calc,trial_group_col_av,
        colns_set = None):
    from figure.plot_behav2 import print_tpl_statcalc
    tpl = env,bn,pertv,gseqc,tgti,trial_group_col_calc,trial_group_col_av
    print_tpl_statcalc(tpl)
    pert_seq_code = None
    if gseqc != (0,1):
        pert_seq_code = gseqc[0]
    df = getSubDf(df_all, 'mean', pertv,tgti,env, non_hit = False,
                 pert_seq_code = pert_seq_code)
    trial_group_col_calc = getTrialGroupName(pertv, tgti, env, bn)
    coln_calc = getColn(pertv, tgti, env, bn, None)
    if colns_set is not None:
        assert coln_calc in colns_set
    if (len(df) == 0):
        #rowi += 1
        return None
    coln_av = getColn(pertv, tgti, env, bn, trial_group_col_av)
    grpstat = subjStat(df, coln_calc,
                       trial_group_col_calc, trial_group_col_av)
    #pscstr = ','.join( list( map(str, list( sorted(df['pert_seq_code'].unique() ) ) ) ) )
    pscstr = ','.join( list( map(str, gseqc ) ) )
    # some things are for filtering, some for actual usage,
    # that's why pertrubation is pertv and pert_nh is actual vals
    r = { 'environment':env, 'block_name':bn,
         'perturbation':pertv, 'pert_seq_code':pscstr,
         'pert_nh':grpstat['perturbation'],
         'target_inds':tgti,  'nav_nh': grpstat['nav'],
         'trial_inds_calc_nh':grpstat[trial_group_col_av],
         'trial_inds_av_nh':grpstat[trial_group_col_av],
         'err_sens_me_nh':grpstat['mean'],
         'err_sens_std_nh':grpstat['std'],
         'err_sens_sem_nh':grpstat['sem'],
         'err_sens_abs_me_nh': grpstat['absmean'],
         'err_sens_abs_std_nh':grpstat['absstd'],
         'err_sens_abs_sem_nh':grpstat['abssem'],
         'coln':coln_av, 'trial_group_col_calc':trial_group_col_calc,
         'trial_group_col_av':trial_group_col_av }
    return r




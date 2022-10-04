import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config2 import *
import seaborn as sns


def getSubjPertSeqCode(subj, task = 'VisuoMotor'):
    fname = op.join(path_data, subj, 'behavdata',
                f'behav_{task}_df.pkl' )
    behav_df_full = pd.read_pickle(fname)
    dfc = behav_df_full

    test_triali = pert_seq_code_test_trial
    r = dfc.loc[dfc['trials'] == test_triali,'perturbation']
    assert len(r) == 1
    if r.values[0] > 5.:
        pert_seq_code = 0
    else:
        pert_seq_code = 1

    return pert_seq_code

def addCols(df_all):
    '''
    inplace
    '''
    subjects     = df_all['subject'].unique()
    tgt_inds_all = df_all['target_inds'].unique()
    pertvals     = df_all['perturbation'].unique()

    subj = subjects[0]


    test_triali = pert_seq_code_test_trial
    subj2pert_seq_code = {}
    for subj in subjects:
        mask = df_all['subject'] == subj
        dfc = df_all[mask]
        r = dfc.loc[dfc['trials'] == test_triali,'perturbation']
        assert len(r) == 1
        if r.values[0] > 5.:
            pert_seq_code = 0
        else:
            pert_seq_code = 1
        subj2pert_seq_code[subj] = pert_seq_code

    def f(row):
        return subj2pert_seq_code[row['subject']]

    df_all['pert_seq_code'] = df_all.apply(f,1)

    #########################   index within block (second block same numbers)

    def f(row):
        env = envcode2env[ row['environment']]
        triali = row['trials']
        if env == 'stable' and triali < 200:
            block_name = env + '1'
        elif env == 'stable' and triali > 300:
            block_name = env + '2'
        elif env == 'random' and triali < 450:
            block_name = env + '1'
        elif env == 'random' and triali > 500:
            block_name = env + '2'
        else:
            display(row)
            raise ValueError(f'wrong combin {env}, {triali}')
        return block_name
    df_all['block_name'] = df_all.apply(f,1)

    #df_all['trialwb'] = None  # within respective block
    #df_all['trialwe'] = None  # within respective env (inc both blocks)
    # important to do it for a fixed subject
    df_all['trialwb'] = -1
    df_all['trialwe'] = -1

    mask = df_all['subject'] == subj
    dfc = df_all[mask]

    assert np.min( np.diff( dfc['trials'] ) ) > 0

    trials_starts = {}
    for bn in block_names:
        fvi = dfc[dfc['block_name'] == bn].first_valid_index()
        assert fvi is not None
        trials_starts[bn] = dfc.loc[fvi,'trials']
    assert np.max( list(trials_starts.values() ) ) <= 767

    def f(row):
        bn = row['block_name']
        start = trials_starts[bn]
        return row['trials'] - start

    df_all['trialwb'] = -1
    for subj in subjects:
        mask = df_all['subject'] == subj
        df_all.loc[mask, 'trialwb'] = df_all[mask].apply(f,1)
    assert np.all( df_all['trialwb'] >= 0)

    ########################   index within env (second block -- diff numbers)

    # within single subject
    envchanges  = dfc.loc[dfc['environment'].diff() != 0,'trials'].values
    envchanges = list(envchanges) + [len(dfc)]

    envinterval = []
    for envi,env in enumerate(block_names):
        envinterval += [ (env, (envchanges[envi], envchanges[envi+1])) ]
    block_trial_bounds = dict( ( envinterval ) )

    #block_trial_bounds = {'stable1': [0,192],
    #'random1': [192,384],
    #'stable2': [384,576],
    #'random2': [576,768]}
    def f(row):
        bn = row['block_name']
        tbs = block_trial_bounds[bn]
        start_cur = tbs[0]
        bnbase = bn[:-1]
        tbs0 = block_trial_bounds[bnbase + '1']
        end_first_rel = tbs0[-1] - tbs0[0]
        add = 0
        if bn.endswith('2'):
            add = end_first_rel
        #return row['trials'] - start_cur + add
        r = row['trials'] - start_cur + add
    #     if bn == 'random2':
    #         import pdb; pdb.set_trace()
        return r

    df_all['trialwe'] = df_all.apply(f,1)
    assert np.all( df_all['trialwe'] >= 0)


    ##########################   index within pertrubation (within block)

    bn2trial_st = {}  # block name 2 trial start
    for bn in ['stable1', 'stable2']:
        dfc_oneb = dfc[dfc['block_name'] == bn]
        df_starts = dfc_oneb.loc[dfc_oneb['perturbation'].diff() != 0]
        trial_st = df_starts['trials'].values

        last =  dfc_oneb.loc[dfc_oneb.last_valid_index(), 'trials']
        trial_st = list(trial_st) +  [last + 1]

        bn2trial_st[bn] = trial_st
        assert len(trial_st) == 6 # - 1

    def f(row):
        t = row['trials']
        bn = row['block_name']
        if bn not in bn2trial_st:
            return None
        trial_st = bn2trial_st[bn]
        for tsi,ts in enumerate(trial_st[:-1]):
            ts_next = trial_st[tsi+1]
            if t >= ts and t < ts_next:
                r = tsi
                break
        return r

    df_all['pert_stage_wb'] = df_all.apply(f,1)

    def f(row):
        bn = row['block_name']
        ps = row['pert_stage_wb']
        if bn not in bn2trial_st:
            return None
        start = bn2trial_st[bn][int(ps)]
        return row['trials'] - start

    df_all['trialwpert_wb'] = df_all.apply(f,1)

    ######################## index within pert within env

    # we use the same but add the end of last trial of stable1.
    # note that this way we distinguish (kind of ) zero pert in the end of
    # first part and zero pert in the beg of second part
    def f(row):
        bn = row['block_name']
        ps = row['pert_stage_wb']
        if bn not in bn2trial_st:
            return None
        start = bn2trial_st[bn][int(ps)]
        add = 0
        if bn == 'stable2':
            add = bn2trial_st['stable1'][int(ps) + 1]
        #start0 = bn2trial_st[bn][int(ps)]
        return row['trials'] - start + add

    df_all['trialwpert_we'] = df_all.apply(f,1)

    ############################# index within target (assuming sorted over trials)

    df_all['trialwtgt'] = -1
    for subj in subjects:
        for tgti in tgt_inds_all:
            mask = (df_all['target_inds'] == tgti) & (df_all['subject'] == subj)
            trials = df_all.loc[mask, 'trials']
            assert np.all(np.diff(trials.values) > 0)
            df_all.loc[mask, 'trialwtgt'] = np.arange(len(trials) )
    #df_all['trialwtgt'] = df_all['trialwtgt'].astype(int)

    ########################### (assuming sorted over trials)

    df_all['trialwtgt_wpert_wb'] = -1
    df_all['trialwtgt_wpert_we'] = -1
    for subj in subjects:
        for tgti in tgt_inds_all:
            #for pertv in df_all['perturbation'].unique()
            mask = (df_all['target_inds'] == tgti) & (df_all['subject'] == subj)
            for pertv in pertvals:
                mask_pert = mask & (df_all['perturbation'] == pertv)
                for bn in block_names:
                    mask_bn = mask_pert & (df_all['block_name'] == bn)
                    trials = df_all.loc[mask_bn, 'trials']
                    df_all.loc[mask_bn, 'trialwtgt_wpert_wb'] = np.arange(len(trials) )
                for envc in envcode2env:
                    mask_env = mask_pert & (df_all['environment'] == envc)
                    trials = df_all.loc[mask_env, 'trials']
                    df_all.loc[mask_env, 'trialwtgt_wpert_we'] = np.arange(len(trials) )
    #df_all['trialwtgt_wpert_wb'] = df_all['trialwtgt_wpert_wb'].astype(int)

    # tcolnames = [s for s in df_all.columns if s.find('trial') >= 0]
    tmax = df_all['trials'].max()
    tcolnames = ['trialwb',
     'trialwe',
     'trialwpert_wb',
     'trialwpert_we',
     'trialwtgt',
     'trialwtgt_wpert_wb',
     'trialwtgt_wpert_we']
    for tcn in tcolnames:
        assert df_all[tcn].max() <= tmax
        assert df_all[tcn].max() >= 0

    addNonHitCol(df_all)

def addNonHitCol(df):
    # in place
    from base2 import point_in_circle_single, radius_cursor, radius_target
    from error_sensitivity import target_coords

    def f(row):
        #print(row.keys())
        target_ind = row['target_inds']
        feedbackX  = row['feedbackX']
        feedbackY  = row['feedbackY']
        nh = point_in_circle_single(target_ind, target_coords, feedbackX,
                        feedbackY, radius_target + radius_cursor)
        return nh

    df['non_hit_not_adj'] = df.apply(f,1)


def getSubDf(df, subj, pertv,tgti,env, non_hit=False,
        pert_seq_code=None, block_name=None, verbose=0, nonenan=False ):
    assert env in ['stable','random','all'], env
    if pertv is not None:
        pvm = np.abs( np.array(df['perturbation'], dtype=float) - pertv ) < 1e-10
        df = df[pvm]
    elif nonenan:
        pvm = df['perturbation'].isna()
        df = df[pvm]
    if len(df) == 0 and verbose:
        print('empty after perturbation')

    if tgti is not None:
        df = df[df['target_inds'] == float(tgti) ]
    elif nonenan:
        pvm = df['target_inds'].isna()
        df = df[pvm]
    if len(df) == 0 and verbose:
        print('empty after target_ind')

    if (subj is not None):
        if isinstance(subj,list):
            df = df[df['subject'].isin(subj) ]
        elif subj != 'mean':
            df = df[df['subject'] == subj]
    if len(df) == 0 and verbose:
        print('empty after subject')

    # not env == 'all'
    if env is not None:
        if env in env2envcode:  # make trial inds (within subject)
            # could be str or int codes
            if isinstance(df['environment'].values[0], str):
                df = df[df['environment'] == env]
            else:
                envcode = env2envcode[env]
                #db_inds = np.where(df['environment'] == envcode)[0]
                df = df[df['environment'] == envcode]
        elif isinstance(env,str) and env == 'all':
            if isinstance(df['environment'].values[0], str):
                df = df[df['environment'] == env]
        else:
            raise ValueError(f'wrong env = {env}')
    if len(df) == 0 and verbose:
        print('empty after environment')

    if non_hit:
        df = df[df['non_hit'] ]

    # this is a subject parameter
    if pert_seq_code is not None:
        df = df[df['pert_seq_code'] == pert_seq_code]
    if len(df) == 0 and verbose:
        print('empty after pert_seq_code')

    if block_name is not None:
        df = df[df['block_name'] == block_name]
    if len(df) == 0 and verbose:
        print('empty after block_name')

    return df

#df = getSubDf(df_all, subj, pertv,tgti,env)

def subjStat(df, coln, trialcol, verbose=0):
    if coln.endswith('envAll'):
        assert trialcol == 'trials', (coln, trialcol)
    '''
    coln is what was used for calculation,
    trialcol is what will be used for averaging
    for each trial across subjects (some trialcol give several time per subj)
    '''
    global n
    n = 0
    npsc = df['pert_seq_code'].nunique()
    if npsc > 1:
        print(f'subjStat WARNING: pert_seq_code is not unique for {coln}')

    def f(dfc,coln, trialcol):
        '''
        input -- database with fixed trials but many subjects
        '''
        #global n
        a = np.array( dfc.loc[ ~dfc[coln].isnull(), coln ] )
        suba = a[~(np.isnan(a) | np.isinf(a))]

        if npsc == 1:
            p = dfc['perturbation'].values[0]
        else:
            p = np.abs( dfc['perturbation'].values[0] )
#         if dfc['trials'].values[0] == 8:
#             display(dfc,suba)
#             n += 1
        #print(df[coln])
        if len(suba):
            suba  = np.array(suba)
            me = np.mean(suba)
            std = np.std(suba)
            sem = std / np.sqrt(len(suba))
        else:
            me = np.nan
            std = np.nan
            sem = np.nan
        #return pd.DataFrame( {coln: [r]} )
        return pd.DataFrame( {trialcol:list(dfc[trialcol])[0] ,
                              'mean': [me],'std': [std],
                              'sem':[sem],
                              'perturbation':p, 'nav':len(suba)}  ) #.mean()

    assert coln in df.columns
    assert np.any(~np.isnan(df[coln])), coln
    grp = df[[trialcol,coln,'perturbation']].groupby(trialcol)

    if verbose:
        display(grp.count())
    #grpme = grp.apply(lambda g: g.mean(skipna=True))
    grpstat = grp.apply(lambda g: f(g,coln=coln,trialcol=trialcol))
    #for ti,subdf in grp:
    #grpme = grpme.drop('trials',1).reset_index(drop=True)
    grpstat = grpstat.reset_index(drop=True)
    grpstat['N'] = np.array( grp.size() )
    grpstat['nav'] = grpstat['nav'].astype('int')

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
    trial_group_names = {'trialwe':'we', 'trialwb':'wb', 'trials':'no',
            'trialwtgt':'tgt',  'trialwpert_we':'pertwe',
            'trialwpert_wb':'pertwb',
            'trialwtgt_wpert_we':'wtgt_wpert_we',
            'trialwtgt_wpert_wb':'wtgt_wpert_wb'}
    if trialgrp is not None:
        assert trialgrp in trial_group_names, trialgrp
        coln += '_tg' + trial_group_names[trialgrp]

    return coln

def getColn(pertv, tgti, env, trialgrp ) :
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
    trial_group_names = {'trialwe':'we', 'trialwb':'wb', 'trials':'no',
            'trialwtgt':'tgt',  'trialwpert_we':'pertwe',
            'trialwpert_wb':'pertwb',
            'trialwtgt_wpert_we':'wtgt_wpert_we',
            'trialwtgt_wpert_wb':'wtgt_wpert_wb'}
    if trialgrp is not None:
        assert trialgrp in trial_group_names, trialgrp
        coln += '_tg' + trial_group_names[trialgrp]

    return coln


def plotErrSens(ax, dfme,env,pertv,gseqc,tgti,trial_group_col,
       span_type = 'sem', ylim=(-7,7), respect_trial_inds=True  ):
    subj = 'mean';
    print(env,pertv,gseqc,tgti,trial_group_col)

    tgtc = ['b','g','r','brown']

    #for coln in colns:
    # TODO cycle over pert_seq_code
    #trial_inds = trial_inds[nhna]
    #err_sens   = err_sens[nhna]
    #df = dfme[ (dfme['coln'] == coln) & (dfme['env'] == env) ]
    if gseqc != (0,1):
        pert_seq_code = str( gseqc[0] )
    else:
        pert_seq_code = '0,1'
    df = getSubDf(dfme, None, pertv,tgti,env, non_hit = False,
                 pert_seq_code = pert_seq_code, verbose=1, nonenan=True)
    df = df[df['trial_group_col'] == trial_group_col]

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
        err_sens_me_nh   = np.array(row['err_sens_me_nh'])
        err_sens_std_nh  = np.array(row['err_sens_std_nh'])
        err_sens_sem_nh  = np.array(row['err_sens_sem_nh'])
        trial_inds_nh    = np.array(row['trial_inds_nh'])
        pert_nh          = np.array(row['pert_nh'])
        nav              = np.array(row['nav_nh'])
        coln = row['coln']
        psc_cur = row['pert_seq_code']
        tgti_true = row['target_inds']
        if not np.isnan(tgti_true):
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
            ax.set_xlabel(trial_group_col)

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

        if env in env2envcode and trial_group_col == 'trialwe':
            ax.axvline(192,ls=':')


        ax.set_title(f'{subj[:5]} env={env} pert={pertv} tgti={tgti} psc={psc_cur}'
                f'\n{coln}')

#                     if pertv is not None:
#                         ax.axhline(pertv / 10., ls=':', c='red', label='pertv / 10')
    ax.set_ylim(ylim)
    ax.legend()

    return df

#import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config2 import *

trial_group_cols_all = ['trialwb',
 'trialwe',
 'trialwpert_wb',
 'trialwpert_we',
 'trialwtgt',
 'trialwpert',
 'trialwtgt_we',
 'trialwtgt_wb',
 'trialwtgt_wpert_wb',
 'trialwtgt_wpert_we',
 'trialwpertstage_wb',
 'trialwtgt_wpertstage_wb',
 'trialwtgt_wpertstage_we' ]

# dfcc queries
qs_notspec_not_afterpause = ('trial_index > @numtrain and trial_type != "error_clamp"'
       ' and prev_trial_type != "pause"')

qs_inside_sandwich = ('trial_index > @numtrain and trial_type != "error_clamp" '
                     ' and special_block_type == "error_clamp_sandwich"')

qs_notspec_not_sandwich_not_afterpause = ('trial_index > @numtrain and trial_type != "error_clamp"'
        ' and special_block_type != "error_clamp_sandwich" and prev_trial_type != "pause"')

spectrials = ["error_clamp", "pause", "break"]

qs_notspec = ('trial_index > @numtrain '
            ' and ~trial_type.isin(@spectrials)'
        ' and special_block_type != "error_clamp_sandwich"')

qs_notEC = ('trial_index > @numtrain and trial_type != "error_clamp"')


qs_notspec_not_afterspec = ('trial_index > @numtrain and trial_type != "error_clamp"'
                   ' and prev_trial_type not in @spectrials')

canonical_context_pair_listnames = ['both_close','some_close','tgt_close',
    'pert_close','pert_same','tgt_same']

 #vfts = dfcc['vis_feedback_type'].unique()
from collections.abc import Iterable

def genCtxPairLists(dfcc):
    tgtis = dfcc['tgti_to_show'].unique()
    perts = dfcc['perturbation'].unique()

    from itertools import product
    all_ctx = list(product(perts,tgtis))
    all_ctx_pairs = list( product(all_ctx,all_ctx) )
    ctx_pairs_both_close = []
    ctx_pairs_some_close = []
    ctx_pairs_tgt_close  = []
    ctx_pairs_pert_close = []
    ctx_pairs_pert_same = []
    ctx_pairs_tgt_same = []
    for pair in all_ctx_pairs:
        (pert1,tgti1),(pert2,tgti2) = pair
        pc = abs(pert1 - pert2) <= 20
        tc = abs(tgti1 - tgti2) <= 1
        if tgti1 == tgti2:
            ctx_pairs_tgt_same += [pair]
        if pert1 == pert2:
            ctx_pairs_pert_same += [pair]
        if pc:
            ctx_pairs_pert_close += [pair]
        if tc:
            ctx_pairs_tgt_close += [pair]
        if tc and pc:
            ctx_pairs_both_close += [pair]
        if tc or pc:
            ctx_pairs_some_close += [pair]
        
    canonical_context_pair_lists = dict(zip(canonical_context_pair_listnames,
    [ctx_pairs_both_close, ctx_pairs_some_close, ctx_pairs_tgt_close ,
    ctx_pairs_pert_close, ctx_pairs_pert_same, ctx_pairs_tgt_same] ))
    return all_ctx, canonical_context_pair_lists

#list( all_ctx_pairs )
def getCtxPairProps(ctxpair, canonical_context_pair_lists):
    if isinstance(ctxpair, tuple):
        r = {}
        for ln in canonical_context_pair_listnames:
            if ctxpair is None:
                r[ln] = None
            else:
                r[ln] = ctxpair in canonical_context_pair_lists[ln]            
        return r
    elif isinstance(ctxpair, Iterable):
        r = []
        for rr in ctxpair:
            assert isinstance(rr,tuple)
            r += [ getCtxPairProps(rr, canonical_context_pair_lists) ]
        return pd.DataFrame(r)

# otherwise we get too many. We need reset_index having been already applied
def setContextSimilarityCols(dftmp, canonical_context_pair_lists):
    assert np.sum( dftmp.index.duplicated() )  == 0
    mx = dftmp.groupby(['subject','trial_index']).size().max() 
    assert mx == 1, mx
    
    subjects = dftmp['subject'].unique()

    pref = 'prev_ctx_'
    for ln in canonical_context_pair_listnames:
        dftmp[pref + ln] = None
    for subj in subjects:
        dfcc_tmp = dftmp.query(f'subject == "{subj}"')
        assert len(dfcc_tmp) > 0
        inds = dfcc_tmp.index

        # maybe I could take 1: to skip first invalid after shift
        tgtis = dfcc_tmp['tgti_to_show'].values[1:].astype(int)
        perts = dfcc_tmp['perturbation'].values[1:].astype(int)
        #bis   = dfcc_tmp['block_ind'].values[1:].astype(int)

        prev_tgtis = dfcc_tmp['tgti_to_show'].shift(1).values[1:].astype(int)
        prev_perts = dfcc_tmp['perturbation'].shift(1).values[1:].astype(int)
        #prev_bis   = dfcc_tmp['block_ind'].shift(1).values[1:].astype(int)

        ctx_pairs = list(zip( list(zip(prev_perts,prev_tgtis)) , list(zip(perts,tgtis)) ))
        ctx_pairs = [((None,None),(None,None))] + ctx_pairs
        #print(len(dfcc_tmp), len(ctx_pairs) )

        pair_props = getCtxPairProps(ctx_pairs, canonical_context_pair_lists)
        for ln in canonical_context_pair_listnames:
            dftmp.loc[inds, pref + ln] = pair_props[ln].to_numpy() 
            
        #bs = [False]  + list(bis == prev_bis)
        dftmp.loc[inds, 'prev_block_ind_diff'] = dfcc_tmp['block_ind'] - dfcc_tmp['block_ind'].shift(1)
        



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

def addBehavCols(df_all, inplace=True, skip_existing = False,
                 dset = 'Romain_Exp2_Cohen'):
    '''
    This is for NIH experiment
    inplace, does not change database lengths (num of rows)
    '''
    if not inplace:
        df_all = df_all.copy()
    subjects     = df_all['subject'].unique()
    tgt_inds_all = df_all['target_inds'].unique()
    pertvals     = df_all['perturbation'].unique()

    subj = subjects[0]

    df_all['dist_trial_from_prevtgt'] = np.nan
    for subj in subjects:
        for tgti in tgt_inds_all:
            if tgti is None:
                continue
            dfc = df_all[(df_all['subject'] == subj) & (df_all['target_inds'] == tgti)]
            df_all.loc[dfc.index,'dist_trial_from_prevtgt'] =\
                df_all.loc[dfc.index, 'trials'].diff()

    #dist_deg_from_prevtgt
    #dist_trial_from_prevtgt
    # better use strings otherwise its difficult to group later
    lbd = lambda x : f'{x:.2f}'
    df_all['dist_rad_from_prevtgt'] = None
    for subj in subjects:
        dfc = df_all[df_all['subject'] == subj]
        df_all.loc[dfc.index,'dist_rad_from_prevtgt'] =\
            df_all.loc[dfc.index, 'target_locs'].diff().abs().apply(lbd,1)

    # signed distance
    df_all['distsgn_rad_from_prevtgt'] = None
    for subj in subjects:
        dfc = df_all[df_all['subject'] == subj]
        df_all.loc[dfc.index,'distsgn_rad_from_prevtgt'] =\
            df_all.loc[dfc.index, 'target_locs'].diff().apply(lbd,1)


    dts = np.arange(1,6)
    for subj in subjects:
        for dt in dts:
            dfc = df_all[df_all['subject'] == subj]
            df_all.loc[dfc.index,f'dist_rad_from_tgt-{dt}'] =\
                df_all.loc[dfc.index, 'target_locs'].diff(periods=dt).abs().apply(lbd,1)


    if dset == 'Romain_Exp2_Cohen':
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

        if not (skip_existing and ('block_name' not in df_all.columns) ):
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


    from collections import OrderedDict
    dfc = df_all[df_all['subject'] == subj]
    block_names = list(OrderedDict.fromkeys(dfc['block_name'] ))

    #block_names = list(sorted( df_all['block_name'].unique() ))


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
    if dset == 'Romain_Exp2_Cohen':
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
            if r < 0:
                raise ValueError('r < 0 ')
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
        print(bn2trial_st)

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
            if np.isnan(ps):
                return None
            if bn.endswith('2'):
                ps = int(ps) + 5

            return ps
        df_all['pert_stage'] = df_all.apply(f,1)

        ############################

        def f(row):
            bn = row['block_name']
            ps = row['pert_stage_wb']
            if bn not in bn2trial_st:
                return None
            start = bn2trial_st[bn][int(ps)]
            return row['trials'] - start

        df_all['trialwpertstage_wb'] = df_all.apply(f,1)

        #############################################

        def f(row):
            bn = row['block_name']
            ps = row['pert_stage_wb']
            if np.isnan(ps):
                return None
            else:
                ps = int(ps)
            if bn not in bn2trial_st:
                return None
            ps = int(ps)
            start = bn2trial_st[bn][ps]
            #2,4
            r =  row['trials'] - start

            bnrebase = bn[:-1] + '1'
            l0 = bn2trial_st[bnrebase][1] - bn2trial_st[bnrebase][0]
            l1 = bn2trial_st[bnrebase][3] - bn2trial_st[bnrebase][2]
            l15 = bn2trial_st[bnrebase][2] - bn2trial_st[bnrebase][1]
            if ps == 2:
                r += l0
            elif ps == 4:
                r += l0 + l1
            elif ps == 3:
                r += l15

            return int(r)

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
            elif bn == 'random2':
                add = bn2trial_st['random1'][int(ps) + 1]
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

    if dset == 'Romain_Exp2_Cohen':
        df_all['trialwtgt_wpert_wb'] = -1
        df_all['trialwtgt_wpertstage_wb'] = -1
        df_all['trialwtgt_wpertstage_we'] = -1
        df_all['trialwtgt_wpert_we'] = -1
        df_all['trialwtgt_we'] = -1
        df_all['trialwtgt_wb'] = -1
    df_all['trialwpert']   = -1
    df_all['trialwtgt_wpert']   = -1
    for subj in subjects:
        mask0 = (df_all['subject'] == subj)
        for pertv in pertvals:
            mask_pert0 = mask0 & (df_all['perturbation'] == pertv)
            df_all.loc[mask_pert0, 'trialwpert'] = np.arange(sum(mask_pert0 ) )

        for tgti in tgt_inds_all:
            #for pertv in df_all['perturbation'].unique()
            mask = (df_all['target_inds'] == tgti) & mask0
            for pertv in pertvals:
                mask_pert = mask & (df_all['perturbation'] == pertv)
                df_all.loc[mask_pert, 'trialwtgt_wpert'] = np.arange(sum(mask_pert) )

                if dset == 'Romain_Exp2_Cohen':
                    for bn in block_names:
                        mask_bn = mask_pert & (df_all['block_name'] == bn)
                        trials = df_all.loc[mask_bn, 'trials']
                        df_all.loc[mask_bn, 'trialwtgt_wpert_wb'] = np.arange(len(trials) )
                    for envc in envcode2env:
                        mask_env = mask_pert & (df_all['environment'] == envc)
                        trials = df_all.loc[mask_env, 'trials']
                        df_all.loc[mask_env, 'trialwtgt_wpert_we'] = np.arange(len(trials) )

            if dset == 'Romain_Exp2_Cohen':
                for pert_stage in range(5):
                    for bn in block_names:
                        mask_ps = mask & (df_all['pert_stage_wb'] == float(pert_stage) ) &\
                                ( df_all['block_name'] == bn )
                        trials = df_all.loc[mask_ps, 'trials']
                        df_all.loc[mask_ps, 'trialwtgt_wpertstage_wb'] = np.arange( len(trials) )
                    for envc in envcode2env:
                        mask_ps = mask & (df_all['pert_stage_wb'] == float(pert_stage) ) &\
                                (df_all['environment'] == envc)
                        trials = df_all.loc[mask_ps, 'trials']
                        df_all.loc[mask_ps, 'trialwtgt_wpertstage_we'] = np.arange( len(trials) )



            if dset == 'Romain_Exp2_Cohen':
                for bn in block_names:
                    mask_bn = mask & (df_all['block_name'] == bn)
                    trials = df_all.loc[mask_bn, 'trials']
                    df_all.loc[mask_bn, 'trialwtgt_wb'] = np.arange(len(trials) )
                for envc in envcode2env:
                    mask_env = mask & (df_all['environment'] == envc)
                    trials = df_all.loc[mask_env, 'trials']
                    df_all.loc[mask_env, 'trialwtgt_we'] = np.arange(len(trials) )
    #df_all['trialwtgt_wpert_wb'] = df_all['trialwtgt_wpert_wb'].astype(int)

    # trial_group_cols_all = [s for s in df_all.columns if s.find('trial') >= 0]
    tmax = df_all['trials'].max()
    for tcn in trial_group_cols_all:
        if dset == 'Romain_Exp2_Cohen':
            assert df_all[tcn].max() <= tmax, tcn
            assert df_all[tcn].max() >= 0,    tcn
        else:
            if tcn not in df_all:
                continue
            if df_all[tcn].max() <= tmax or df_all[tcn].max() >= 0:
                print(f'problem with {tcn}')

    if dset == 'Romain_Exp2_Cohen':
        vars_to_pscadj = [ 'error']
        # 'prev_error' ?
        for varn in vars_to_pscadj:
            df_all[f'{varn}_pscadj'] = df_all[varn]
            df_all.loc[df_all['pert_seq_code'] == 1, f'{varn}_pscadj']= -df_all[varn]

        vars_to_pscadj = [ 'org_feedback']
        for varn in vars_to_pscadj:
            df_all[f'{varn}_pscadj'] = df_all[varn] - np.pi
            cond = df_all['pert_seq_code'] == 1
            df_all.loc[cond, f'{varn}_pscadj']=  - ( df_all.loc[cond,varn]  -np.pi)

        df_all['error_pscadj_pertstageadj'] = df_all['error_pscadj']
        c = (df_all['pert_stage_wb'] == 3) & (df_all['block_name'] == 'stable1')
        df_all.loc[c, 'error_pscadj_pertstageadj'] = -df_all.loc[c, 'error_pscadj']
        c = (df_all['pert_stage_wb'] == 1) & (df_all['block_name'] == 'stable2')
        df_all.loc[c, 'error_pscadj_pertstageadj'] = -df_all.loc[c, 'error_pscadj']

        c = (df_all['pert_stage_wb'] == 4) & (df_all['block_name'] == 'stable1')
        df_all.loc[c, 'error_pscadj_pertstageadj'] = -df_all.loc[c, 'error_pscadj']
        c = (df_all['pert_stage_wb'] == 2) & (df_all['block_name'] == 'stable2')
        df_all.loc[c, 'error_pscadj_pertstageadj'] = -df_all.loc[c, 'error_pscadj']

        addNonHitCol(df_all)
    return df_all

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


def getSubDf(df, subj, pertv, tgti, env, block_name=None, pert_seq_code=None,
        dist_rad_from_prevtgt=None, dist_trial_from_prevtgt=None,
        non_hit=False, verbose=0, nonenan=False ):
    '''
    if nonenan is True, then NaN in numeric columns are treated as None
    inputs should be NOT lists
    '''
    assert not isinstance(pertv,list)
    assert not isinstance(tgti,list)
    assert not isinstance(block_name,list)
    assert not isinstance(subj,list)
    # and so on

    assert env in ['stable','random','all'], env
    if pertv is not None:
        assert isinstance(pertv, float) or isinstance(pertv,int), pertv
        pvm = np.abs( np.array(df['perturbation'], dtype=float) - pertv ) < 1e-10
        df = df[pvm]
    elif nonenan:
        pvm = df['perturbation'].isna()
        df = df[pvm]
    if verbose:
        if len(df) == 0 and verbose:
            print('empty after perturbation')
        else:
            print('after pert len = ',len(df ))

    if tgti is not None:
        # this is a bit ugly but since int != np.int64 it should be easier
        if str(int(tgti) ) in ['0','1','2','3']:
            df = df[df['target_inds'] == float(tgti) ]
        elif isinstance(tgti, str) and tgti == 'all':
            pvm = ~df['target_inds'].isna()
            df  = df[pvm]
    elif nonenan:
        pvm = df['target_inds'].isna()
        df = df[pvm]
    if verbose:
        if len(df) == 0:
            print('empty after target_ind')
        else:
            print('after target_ind len = ',len(df ))



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
    if verbose:
        if len(df) == 0 and verbose:
            print('empty after env')
        else:
            print('after env len = ',len(df ))

    if non_hit:
        df = df[df['non_hit'] ]

    # this is a subject parameter
    if pert_seq_code is not None:
        df = df[df['pert_seq_code'] == pert_seq_code]
    if verbose:
        if len(df) == 0 and verbose:
            print('empty after pert_seq_code')
        else:
            print('after pert_seq_code len = ',len(df ))

    if (block_name is not None):
        if block_name not in ['all', 'only_all']:
            df = df[df['block_name'] == block_name]
        elif block_name == 'only_all':
            df = df[df['block_name'] == 'all']
    if verbose:
        if len(df) == 0 and verbose:
            print('empty after block_name')
        else:
            print('after block_name len = ',len(df ))

    if dist_rad_from_prevtgt is not None:
        assert type(df['dist_rad_from_prevtgt']._values[0]) == type(dist_rad_from_prevtgt)
        df = df[df['dist_rad_from_prevtgt'] == dist_rad_from_prevtgt]
    if verbose:
        if len(df) == 0 and verbose:
            print('empty after dist_rad_from_prevtgt')
        else:
            print('after dist_rad_from_prevtgt len = ',len(df ))

    if dist_trial_from_prevtgt is not None:
        df = df[df['dist_trial_from_prevtgt'] == dist_trial_from_prevtgt]
    if verbose:
        if len(df) == 0 and verbose:
            print('empty after dist_trial_from_prevtgt')
        else:
            print('after dist_trial_from_prevtgt len = ',len(df ))

    if verbose:
        print('final len(df) == ',len(df) )
    return df

def plot(df, coln):
    subjects =df['subject'].unique()
    dfc = df.query('subject == @subjects[0]')

#df = getSubDf(df_all, subj, pertv,tgti,env)
def calcQuantilesPerESCI(df_all, grp, coln, q = 0.05, infnan_handle = 'skip_calc' ):
    #dfs = []
    res = {}
    for gk,ginds in grp.groups:
        df = df_all.loc[ginds]
        assert len(df) <= ntrials_per_subject, len(df)
        if infnan_handle in ['skip_calc', 'discard']:
            # mask of good
            mask =  ~ ( df[coln].isna() | np.isinf( df[coln] ) )
            df_calc = df[mask]
        else:
            mask = np.ones(len(df), dtype=bool)
            df_calc = df

        q_low = df_calc[coln].quantile(q)
        q_hi  = df_calc[coln].quantile(1-q)
        res[gk] = q_low, q_hi
    return res

   #     mask_comp = (df[coln] < q_hi) & (df[coln] > q_low)
   #     # we don't want save comparisons with inf
   #     df_flt = df[mask & mask_comp]
   #     dfs += [df_flt]

   # if infnan_handle == 'discard':
   #     res = pd.concat(dfs, ignore_index=True)

    ## per err sens calc instance, within subject
    #subj2qts = {}
    #subjects     = df_all['subject'].unique()
    #for subj in subjects:
    #    mask_subj = df_all['subject'] == subj
    #    dat = df_all.loc[mask_subj,coln]
    #    dat = dat[~np.isinf(dat)]
    #    qts = np.nanquantile(dat,[q,1-q])
    #    subj2qts[subj] = qts
    #return subj2qts


def truncateDf(df, coln, q=0.05, verbose=0, infnan_handling='keepnan', inplace=False,
               return_mask = False, trialcol = 'trials',
               cols_uniqify = ['trial_shift_size',
                  'trial_group_col_calc'] ):
    if not inplace:
        df = df.copy()

    ntrials_per_subject = df[trialcol].nunique()

    grp0 = df.groupby([trialcol] + cols_uniqify)
    mx=  max(grp0.size() )
    assert mx <= df['subject'].nunique()

    #mask = np.ones(len(df), dtype=bool)

    # good
    mask =  ~ ( df[coln].isna() | np.isinf( df[coln] ) )

    #print('fff')
    #if clean_infnan:
    #    # mask of good
    #    mask =  ~ ( df[coln].isna() | np.isinf( df[coln] ) )
    #    df =  df[ mask ]
    #    mask = np.array(mask)

    if (q is not None) and (q > 1e-10):
        #gk2qts = calcQuantilesPerESCI(df, coln, q=q)
        df.loc[~mask, coln] = np.nan
        grp = df.groupby(['subject'] + cols_uniqify)

        mgsz = np.max(grp.size() )
        print(  f' np.max(grp.size() )  == {mgsz}')
        assert mgsz <= ntrials_per_subject

        low = grp[coln].quantile(q=q)
        hi  = grp[coln].quantile(q=1-q)
        assert not hi.reset_index()[coln].isna().any(), hi

        dfs = []
        for gk,ginds in grp.groups.items():

            dftmp = df.loc[ginds]

            assert len(ginds) == len(dftmp) , 'perhaps index was not reset after concat'
            #print(len(ginds) )
            # DEBUG
            # if len(dftmp) > ntrials_per_subject:
            #    return gk, ginds, dftmp, grp
            assert len(dftmp) <= ntrials_per_subject,(len(dftmp), ntrials_per_subject)

            lowc = low[gk]
            hic  = hi[gk]

            mask_good  =  ~ ( dftmp[coln].isna() | np.isinf( dftmp[coln] ) )
            mask_trunc = (dftmp[coln] < hic)  & (dftmp[coln] > lowc)
            if infnan_handling in ['keepnan', 'discard']:
                mask_bad = (~mask_good) | (~mask_trunc)
            elif infnan_handling == 'do_nothing':
                mask_bad = (~mask_trunc)
            else:
                raise ValueError(f'Wrong {infnan_handling}')
            dftmp.loc[mask_bad  , coln ] = np.nan

            if np.all( (np.isnan(dftmp[coln]) | np.isinf(dftmp[coln]))  ):
                display(dftmp[ ['subject','trials'] + cols_uniqify + ['err_sens']] )
                print(gk,len(ginds), sum(mask_bad), sum(mask_good), sum(mask_trunc) )
                raise ValueError(gk)

            if infnan_handling == 'discard':
                dftmp = dftmp[~mask_bad]

            dfs += [dftmp]
        df = pd.concat(dfs, ignore_index = 1)
    elif infnan_handling == 'discard':
        df = df[mask]

        #subj2qts = calcQuantilesPerESCI(df, coln, q=q)
        ##df = df.copy()
        #for subj,qts in subj2qts.items():
        #    mask_cur = (df['subject'] == subj) & \
        #        ( ( df[coln] < qts[0]) | (df[coln] > qts[1]) )
        #    df.loc[mask_cur,coln] = np.nan
        #    if verbose:
        #        print(f'Setting {sum(mask_cur)} / {len(mask_cur)} points to NaN')
        # assert np.any( ~(np.isnan(dftmp[coln]) | np.isinf(dftmp[coln]))  ), len(ginds)

    #if clean_infnan:
    #    mask2 = ~ ( df[coln].isna() | np.isinf( df[coln] ) )
    #    df =  df[ mask2]
    #    mask2 = np.array(mask2)
    #    indsgood = np.where(mask)[0]
    #    mask[ indsgood[~mask2] ] = False
    #    #mask[ indsgood[mask2] ] = True
    #    #mask[~mask2] = False

    r = df
    if return_mask:
        r = df, mask
    return r


def getTrialGroupName(pertv, tgti, env, block):
    assert isinstance(env,str)
    coln = 'trial'
    if tgti is not None:
        coln += 'wtgt'
    if not (pertv is None or (isinstance(pertv,str) and pertv == 'all' ) ):
        if coln != 'trial':
            coln += '_'
        coln += 'wpert'
    if env in env2envcode and ( (block is None) or (block == 'all') ) :
        if coln != 'trial':
            coln += '_'
        coln += 'we'
    if block is not None and (block in block_names):
        if coln != 'trial':
            coln += '_'
        coln += 'wb'
    assert not ( ( coln.find('wb') >= 0) and (coln.find('we') >= 0  ) ), coln

    if coln == 'trial': # if use all
        coln = 'trials' # just historically so

    return coln

def computeErrSensVersions(df_all, envs_cur,block_names_cur,
        pertvals_cur,gseqcs_cur,tgt_inds_cur,
        dists_rad_from_prevtgt_cur,dists_trial_from_prevtgt_cur,
        subj_list=None, error_type='MPE',
        coln_nh = 'non_hit',
        coln_nh_out = 'non_hit_shifted',
        trial_shift_sizes = [1],
        DEBUG=0, allow_duplicating=True, time_locked = 'target',
        addvars = None, target_info_type = 'inds',
        coln_correction_calc = None, coln_error = 'error',
        computation_ver = 'computeErrSens2',
        df_fulltraj = None,  trajPair2corr = None,
        verbose=0 ):
    '''
        if allow_duplicating is False we don't allow creating copies
        of subsets of indices within subject (this can be useful for decoding)
    '''
    from config2 import block_names
    assert isinstance(dists_trial_from_prevtgt_cur, list)
    assert isinstance(dists_rad_from_prevtgt_cur, list)

    assert ('index', 'level_0') not in df_all.columns

    if not allow_duplicating:
        assert not ( (None and tgt_inds_cur) and\
                np.isin(tgt_inds_cur,np.arange(4,dtype=int) ).any() )
        assert not ( ( (None and envs_cur) or ('all' in envs_cur) ) and\
                np.isin(envs_cur, env2envcode.keys() ).any() )
        assert not ( ( (None and block_names_cur) or ('all' in block_names_cur) ) and\
                np.isin(block_names_cur, block_names ).any() )
        k = 'perturbation'
        pv = df_all.loc[~df_all[k].isnull(), k].unique()
        assert not ( ( (None and pertvals_cur) or ('all' in pertvals_cur) ) and\
                np.isin(pertvals_cur, pv).any() )
        assert not ( ( (None and pertvals_cur) or ('all' in pertvals_cur) ) and\
                np.isin(pertvals_cur, pv).any() )
        assert len(trial_shift_sizes) == 1

    dfme = []
    from itertools import product as itprod
    if computation_ver == 'computeErrSens2':
        from error_sensitivity import computeErrSens2 as computeES
        addargs = {}
    elif computation_ver == 'computeErrSens3':
        from error_sensitivity import computeErrSens3 as computeES
        addargs = None # will be added later per subj


    p = itprod(envs_cur,block_names_cur,pertvals_cur,gseqcs_cur,tgt_inds_cur,
               dists_rad_from_prevtgt_cur,dists_trial_from_prevtgt_cur)
    p = list(p)
    print('len(prod ) = ',len(p))
    print('prod = ',p)

    if subj_list is None:
        subj_list = df_all['subject'].unique()

    #colns_set  = []; colns_skip = [];
    debug_break = 0
    dfs = []; #df_inds = []
    for subj in subj_list: #[:1]:
        for tpl in p:
            #print(len(tpl), tpl)
            (env,block_name,pertv,gseqc,tgti,drptgt,dtptgt) = tpl

            #print(tpl)
            tpl = env,block_name,pertv,gseqc,tgti,\
                drptgt,dtptgt,\
                None,None
            print(f'subj = {subj}, prod tuple contents = ', sprintf_tpl_statcalc(tpl) )
            #df = df_all
            # if we take only non-hit then, since we'll compute err sens sequentially
            # we'll get wrong
            #if trial_group_col in ['trialwb']:
            #    raise ValueError('not implemented')
            df = getSubDf(df_all, subj, pertv,tgti,env,block_name,
                          non_hit = False, verbose=verbose)
            db_inds = df.index
            #df_inds += [db_inds]

            tgn = getTrialGroupName(pertv, tgti, env, block_name)
            print(f'  selected {len(df)} inds out of {len(df_all) }; tgn = {tgn}')
            #coln = getColn(pertv, tgti, env, block_name, None) #, trial_group_col)
            #print('  ',tgn, coln,len(df))
            if (len(df) == 0) or (len(db_inds) == 0):
                #rowi += 1
                print('skip',tgn,subj)
                #colns_skip += [coln]
                if DEBUG:
                    debug_break = 1
                    break
                continue

            if computation_ver == 'computeErrSens3':
                if df_fulltraj is not None:
                    addargs = {'df_fulltraj': \
                               df_fulltraj.query(f'subject == "{subj}"'),
                               'trajPair2corr':trajPair2corr}
                else:
                    addargs = {}


            for tsz in trial_shift_sizes:
                escoln = 'err_sens'
                #if tsz == 0:
                #    escoln = 'err_sens':
                #else:
                #    escoln = 'err_sens_-{tsz}t':
                # resetting index is important
                dfri = df.reset_index()
                r = computeES(dfri, df_inds=None,
                    error_type=error_type,
                    colname_nh = coln_nh,
                    correct_hit = 'inf', shiftsz = tsz,
                    err_sens_coln=escoln,
                    time_locked = time_locked, addvars=addvars,
                    target_info_type = target_info_type,
                    coln_correction_calc = coln_correction_calc,
                    coln_error = coln_error,
                    recalc_non_hit = False, **addargs)

                if computation_ver == 'computeErrSens2':
                    nhna, df_esv, ndf2vn = r
                elif computation_ver == 'computeErrSens3':
                    ndf2vn = None
                    nhna, df_esv = r

                # if I don't convert to array then there is an indexing problem
                # even though I try to work wtih db_inds it assigns elsewhere
                # (or does not assigne at all)
                es_vals = np.array( df_esv[escoln] )
                assert np.any(~np.isnan(es_vals)), tgn  # at least one is not None
                assert np.any(~np.isinf(es_vals)), tgn  # at least one is not None

                #colns_set += [coln]

                dfcur = df.copy()
                dfcur['trial_shift_size'] = tsz  # NOT _nh, otherwise different number
                dfcur['time_locked'] = time_locked  # NOT _nh, otherwise different number
                dfcur[escoln] = es_vals  # NOT _nh, otherwise different number
                dfcur['trial_group_col_calc'] = tgn
                dfcur['error_type'] = error_type
                # here it means shfited by 1 within subset
                dfcur[coln_nh_out] = np.array( df_esv[coln_nh_out] )

                dfcur['correction'] = np.array( df_esv['correction'] )
                for cn in ['trial_inds_glob_prevlike_error', 'trial_inds_glob_nextlike_error',
                           f'prev_{escoln}', 'prev_error' ]:
                    dfcur[cn] = df_esv[cn].to_numpy()

                if computation_ver == 'computeErrSens2':
                    errn = ndf2vn['prev_error']
                    dfcur['dist_rad_from_prevtgt2'] = dfcur['target_locs'].values -\
                        df_esv['prev_target'].values
                    dfcur[errn] = np.array( df_esv[errn] )
                else:
                    dfcur['dist_rad_from_prevtgt2'] =\
                        df_esv['target_loc'].values -\
                        df_esv['prev_target_loc'].values

                for avn in addvars:
                    if avn in dfcur.columns:
                        continue
                    dfcur[avn] = np.array(df_esv[avn])

                #lbd(0.5)
                #print(dfcur['target_locs'].values, df_esv['prev_target'].values )
                #raise ValueError('f')

                # convert to string
                lbd = lambda x : f'{x:.2f}'
                dfcur['dist_rad_from_prevtgt2'] =\
                    dfcur['dist_rad_from_prevtgt2'].abs().apply(lbd, 1)

                dfs += [ dfcur.reset_index(drop=True)  ]

                if DEBUG and tgn == 'trialwtgt_we':
                    debug_break = 1; print('brk')
                    db_inds_save = db_inds
                    df_save = df
                    break
            if debug_break:
                break
        if debug_break:
            break
        print('Subj = ',subj, 'computation finished successfully')
    print('computeErrSensVersions: Main calc finished successfully')

    df_all2 = pd.concat(dfs)
    df_all2.reset_index(inplace=True, drop=True)
    df_all2.drop(['feedbackX','feedbackY'],axis=1,inplace=True)
    if 'trajectoryX' in df_all2.columns:
        df_all2.drop(['trajectoryX','trajectoryY'],axis=1,inplace=True)


    df_all2.loc[df_all2['trials'] == 0, coln_nh_out] = False
    #df_all2.loc[df_all2['trials'] == 0, 'non_hit_not_adj'] = False
    df_all2.loc[df_all2['trials'] == 0, 'err_sens'] = np.inf
    return df_all2, ndf2vn


def sprintf_tpl_statcalc(tpl):
    ''' used to pring tuples of specific format, for debug'''
    env,bn,pertv,gseqc,tgti,\
        dist_rad_from_prevtgt,dist_trial_from_prevtgt,\
        trial_group_col_calc,trial_group_col_av = tpl
    locs = locals().copy()
    s = ''
    for ln,lv in locs.items():
        if ln == 'tpl':
            continue
        lneff = ln
        st = 'trial_group_col_'
        st2 = 'dist_rad_from_prevtgt'
        st3 = 'dist_trial_from_prevtgt'
        if ln.startswith(st):
            lneff = 'tgc' + ln[len(st):]
        elif ln.startswith(st2):
            lneff = 'drpt' + ln[len(st2):]
        elif ln.startswith(st3):
            lneff = 'dtpt' + ln[len(st3):]
        s += f'{lneff}={lv}; '
    return s[:-2]

#############################   Context cahgne experimeng




#def area2(xs,ys):

def area(xs,ys, start_ideal = None, end_ideal = None, verbose=0,
         allow_middle_stops = True, stops_found_action = 'warning',
         ax = None, plotargs=None, xshift = 0, intersections = 'allow' ):
    from base2 import rot
    from shapely.geometry import LineString, Point, Polygon
    import sys

    if end_ideal is None:
        end_ideal = (0, float(params['dist_tgt_from_home'] ) )
    if start_ideal is None:
        start_ideal = (0, float( params['radius_home'])  )

    allxs = [start_ideal[0]] + list(xs) + [end_ideal[0]]
    allys = [start_ideal[1]] + list(ys) + [end_ideal[1]]
    ssq = np.diff(allxs)**2 + np.diff(allys)**2

    maxy = np.max(allys)

    if verbose:
        print('ST: '  ,allxs,allys)

    stops = np.where(ssq <= sys.float_info.epsilon)[0]
    if len(stops) == 0:
        c = True
    else:
        c = (np.min(stops) > 1) and (np.max(stops) < len(ys) ) and allow_middle_stops

    if (not c) and np.min( ssq ) > 1e-10:
        if stops_found_action == 'warning':
            print( f'WARNING: complete stop present at {stops}' )
        elif stops_found_action == 'exception':
            raise ValueError( f'complete stop present at {stops}' )

    if verbose:
        plt.figure()
        print(f'xs = {xs},  ys = {ys}, start_ideal={start_ideal}, end_ideal={end_ideal}, ')
        plt.scatter( allxs, allys)

    # where signs diffesr in conseq trials?
    xs_signtest = np.array(xs) + xshift
    xsnext = xs_signtest[1:]
    sconst = np.sign(xs_signtest[:-1]) * np.sign(xsnext)
    if verbose:
        print('sconst=',sconst)
    if len(sconst) == 0:
        sct = 0
    else:
        sct = np.min(sconst)

    sgn2col = {-1:'red',1:'green' }

    if sct >= 0:
        inp = [start_ideal] + list(zip(xs, ys))  +\
               [end_ideal]
        inp = tuple(inp)
        if verbose:
            print('inp = ', inp)

        sgnxs = np.sign( np.sum(xs_signtest) )


        if ax is not None:
            inpa = np.array(inp)
            ax.fill( inpa[:,0], inpa[:,1]  , **plotargs,
                    color =sgn2col[ int(sgnxs)  ]  )

        poly = Polygon(inp)
        resa = poly.area * sgnxs
        if verbose:
            display(poly)
    else:
        if intersections != 'allow':
            raise ValueError('Found intersections!')

        resa = 0.
        chis = np.where(sconst < 0)[0]
        last_chi = None
        last_chcoords = None
        for chi in chis:
            #print(chis)
            # it is not always enough for some reason to use just these local
            # intersection points. So I am taking a global one
            #A,B = (0,ys[chi]), (0,ys[chi+1])
            A,B = (-xshift,-maxy/2), (-xshift,maxy)

            C,D = (xs[chi],ys[chi]), (xs[chi+1],ys[chi+1])
            line1 = LineString([A, B])
            line2 = LineString([C, D])
            isec = line1.intersection(line2)
            assert hasattr(isec, 'x'), 'No intersection found'

            if verbose:
                print('chi  = ',chi, 'ABCD = ',A,B,C,D)

            if last_chi is None:
                st = start_ideal
                inds = slice(0, chi + 1)
            else:
                st = last_chcoords
                inds = slice(last_chi + 1, chi + 1)
            #print(isec)
            inp = [st] + list(zip(xs[inds], ys[inds])) + [(isec.x, isec.y)]
            if verbose:
                print('inp = ',inp)
            assert len(xs[inds] )

            sgnxs = np.sign( np.sum(xs_signtest[inds]) )
            if ax is not None:
                inpa = np.array(inp)
                ax.fill( inpa[:,0], inpa[:,1]  , **plotargs,
                    color =sgn2col[ int(sgnxs)  ]  )

            poly = Polygon(inp)
            a = poly.area * sgnxs
            resa += a

            if verbose:
                display(poly)

            last_chi = chi
            last_chcoords = isec.x, isec.y

            if verbose:
                print(f'chi={chi}, last_chcoords={last_chcoords}, a= {a} ' )

        st = last_chcoords
        inds = slice(last_chi + 1, len(xs))
        inp = [st] + list(zip(xs[inds], ys[inds])) + [end_ideal]

        sgnxs = np.sign( np.sum(xs_signtest[inds]) )
        if ax is not None:
            inpa = np.array(inp)
            ax.fill( inpa[:,0], inpa[:,1]  , **plotargs,
                    color =sgn2col[ int(sgnxs)  ]  )

        poly = Polygon(inp)
        a = poly.area * sgnxs
        resa += a

        if verbose:
            print(f'inp = {inp}, a= {a} ')
            display(poly)
        # add last segment, from last intersection to the end

    return resa



def readParamFiles(fnp, inpdir, phase_to_collect = 'TARGET_AND_FEEDBACK'):
    with open(pjoin(inpdir,fnp), 'r') as f:
        lines = f.readlines()

    triggerdict_start_line = '# phase2trigger'
    stl = -1
    for linei,line in enumerate(lines):
        if line.startswith(triggerdict_start_line):
            stl = linei
            break

    if stl < 0 :
        phase2trigger = {'JOYSTICK_CALIBRATION_LEFT':3,
                         'JOYSTICK_CALIBRATION_RIGHT':4,
                         'JOYSTICK_CALIBRATION_UP':5,
                         'JOYSTICK_CALIBRATION_DOWN':6,
                         'JOYSTICK_CALIBRATION_CENTER':7,
                         'TRAINING_START':8,
                         'TRAINING_END':9,
                          'REST':10, 'RETURN':15,
                          'GO_CUE_WAIT_AND_SHOW':25,
                          'TARGET':20, 'FEEDBACK': 35,
                          'TARGET_AND_FEEDBACK':30,
                       'ITI':40, 'BREAK':50, 'PAUSE':60 }
    else:
        endi = -1
        for i in range(linei + 1, len(lines)):
            if lines[i].find('}') >= 0:
                endi = i
                break
        phase2trigger = eval(''.join( lines[linei+1:endi+1] ).replace('\n',''))

    trigger2phase = dict( zip(phase2trigger.values(),phase2trigger.keys()) )

    ##################   process param file
    triggerdict_start_line = '# trial param and phase 2 trigger values'
    stl = -1
    for linei,line in enumerate(lines):
        if line.startswith(triggerdict_start_line):
            stl = linei
            break

    ##########  params
    params = {}
    for line in lines[:stl]:
        if line.startswith('#'):
            continue
        r = line.replace(' ','').replace('\n','').split('=')
        if len(r) != 2:
            print(line, r )
            lhs = r[0] 
            rhs = '='.join(r[1:])
        else:
            lhs,rhs = r
        params[lhs] = rhs

    early_reach_end_event = params.get('early_reach_end_event')
    if early_reach_end_event is None:
        early_reach_end_event = params.get('reach_end_event')
    print('early_reach_end_event = ', early_reach_end_event)


    stage2pars = {}


    phase2trigs = {}
    phase2trigs[phase_to_collect] = []
    for line in lines[stl + 3:]:
        if line.startswith('}'):
            break
        k,v = line.split(':')
        v = int(v.replace(',','' ) )
        k = k.replace('"','').replace(' ','')
        tt,vft,tgti,phase = k.split(',')
        if len(tgti) > 0:
            tgti = int(tgti)
        else:
            tgti = None
        stage2pars[ v ] = tt,vft,tgti,phase
        if phase == phase_to_collect:
            phase2trigs[phase_to_collect] += [v]
        #print(line)

    intpars = ['width','height', 'num_training', 
               'n_context_appearences']
    for pn in intpars:
        params[pn] = int(params[pn])

    fpars = ['radius_home', 'radius_cursor', 'dist_tgt_from_home', 
             'radius_target', 'FPS', 'time_feedback']
    for pn in fpars:
        params[pn] = float(params[pn])

    return params, phase2trigger, trigger2phase, stage2pars

    #print(stage2pars)

def getGeomInfo(params):
    from exper_protocol.utils import (get_target_angles,
        calc_target_positions, calc_err_eucl, coords2anglesRad, screen2homec,
                                     homec2screen)

    #targetAngs = get_target_angles(int(params['num_targets']),
    #            'fan', float(params['target_location_spread']))

    targetAngs = list(map(float, eval( params['target_angles'] ) ) )

    home_position = (int(round(params['width']/2.0)),
                    int(round(params['height']/2.0)))

    print(targetAngs)
    # list of 2-ples
    target_coords = calc_target_positions(targetAngs, home_position,
                                          params['dist_tgt_from_home'])
    # first positive x, then 0, then negative x
    target_coords_homec = screen2homec( *tuple(zip(*target_coords)), home_position  )
    print('target_coords =', target_coords)

    return home_position, target_coords



def row2multierr(row, dfc, grp_perti, home_position, target_coords,
                 params, revert_pert = True,
                 ax = None, axo = None,
                 force_entire_traj=False, addinfo=None, titlecols=[],
                 xlim=(-140,140),
                 vertline = 'tgt_ta' ):
    '''
    it reverts answer for neg pert
    row['perturbation'] is only used to decide if we revert or no
    '''
    from base2 import rot
    from exper_protocol.utils import screen2homec
    from shapely.geometry import LineString
    ti = row['trial_index']
    #print(f'trial_index = {ti}')
    idx = grp_perti.groups[ti]
    dfcurtr = dfc.loc[idx[1:]]
    pert = row['perturbation']
    le = len(dfcurtr)

    if axo is None and ax is not None:
        axo = ax

    assert dfcurtr['current_phase_trigger'].min() == dfcurtr['current_phase_trigger'].max()

    target_coords_homec = screen2homec( *tuple(zip(*target_coords)), home_position  )
    txc,tyc = target_coords_homec
    if np.diff(txc).min() < 0:
        print('inverting x of targets')
        txc = -txc

    #print(ti)

    #dfcurtr#[['feedbackX', 'feedbackY', 'unpert_feedbackY']]
    txc,tyc = target_coords_homec
    tgti = dfcurtr['tgti_to_show']._values[0]
    tgtcur = np.array(  [ txc[tgti], tyc[tgti] ] )


    fbXY = dfcurtr[['feedbackX', 'feedbackY']].to_numpy()
    #fbXY.shape
    fbXhc, fbYhc = screen2homec(fbXY[:,0], fbXY[:,1], home_position  )
    fbXYhc = np.array( [fbXhc, fbYhc], dtype=float )

    fb0pt = np.array( [fbXhc[0], fbYhc[0] ] , dtype=float)
    tgtcur_adj = tgtcur - fb0pt
    fbXYhc_adj = fbXYhc - fb0pt[:,None]

    ang_tgt = np.math.atan2(*tuple(tgtcur_adj) )

    fbXYhc_ta = rot( *fbXYhc, ang_tgt, fb0pt )
    fbXhc_ta, fbYhc_ta = fbXYhc_ta

    tgtcur_ta = rot( *tgtcur, ang_tgt, fb0pt )

    rh = float( params['radius_home'])
    dirx = rh * np.sin(ang_tgt) + fb0pt[0]
    diry = rh * np.cos(ang_tgt) + fb0pt[1]
    dirtgt0 = np.array( [dirx,diry] )

    # rotation of target direction (home-origned) around fb0pt
    # and then setting it to the crossing of the home circle
    dirtgt_ta = rot(*dirtgt0, ang_tgt, fb0pt) #- fb0pt[0],0
    dirtgt_ta[1] = np.sqrt( rh**2 - dirtgt_ta[0]**2 )

    ds = np.sqrt( fbXhc**2 + fbYhc**2 )
    leave_home_coef = 1.
    inds_leavehome = np.where(ds > rh * leave_home_coef )[0]


    if len(inds_leavehome) > 1:
        ind  = inds_leavehome[0]  # first index when fb traj is outside home
        times = dfcurtr['time'].to_numpy()
        time = times[ind]
        td = time - times[0]

        from scipy.interpolate import interp1d
        #print('tgti = ', tgti)

        lentrunc = le - ind

        #tgtcur = -txc[tgti], tyc[tgti]

        #from scipy.integrate import quad
        ptinds = np.arange(lentrunc)
        # ang of tgt, counted from vertical , pos X give positive angles

        rh2 = np.sqrt( dirx**2 + diry**2 )
        coef = rh / rh2
        dirtgt = (dirtgt0 - fb0pt) * coef + fb0pt

        fxint = interp1d([0, lentrunc-1 ], [dirtgt[0], txc[tgti]], 'linear' )
        tgtpathX_interp = fxint(ptinds)
        fyint = interp1d([0, lentrunc-1 ], [dirtgt[1], tyc[tgti]], 'linear' )
        tgtpathY_interp = fyint(ptinds)

        #curveX, curveY = rot(fbXhc_ta, fbYhc_ta, ang_tgt)
        curveX, curveY = fbXhc_ta, fbYhc_ta

        ofbXY = dfcurtr[['unpert_feedbackX', 'unpert_feedbackY']].to_numpy()
        ofbXhc, ofbYhc = screen2homec(ofbXY[:,0], ofbXY[:,1], home_position  )
        curveX0, curveY0 = rot(ofbXhc, ofbYhc, ang_tgt)

        fbXYlh_ta = fbXYhc_ta[:,ind]


        ideal_lh = dirtgt_ta

        if vertline == 'tgt_ta':
            xshift = -tgtcur_ta[0]
            intersections = 'allow'
        elif vertline == 'veryleft':
            xshift = -float( params['width'] )
            intersections = 'prohibit'


        # negative when feedback on the left of the ideal target
        traja = area(curveX[ind:], curveY[ind:], ideal_lh,
             tgtcur_ta, xshift = xshift,
                     verbose=0, ax=ax, plotargs = {'alpha':0.2},
                     intersections = intersections)

        from base2 import areaOne
        traja2 = areaOne(curveX[ind:], curveY[ind:], ideal_lh, tgtcur_ta)

        trajoa = area(curveX0[ind:], curveY0[ind:], ideal_lh,
             tgtcur_ta, xshift = xshift,
                    verbose=0, ax=axo, plotargs = {'alpha':0.2},
                     intersections = intersections)

        #try:
        #    traja = area(fbXhc_ta[ind:], fbYhc_ta[ind:], (0,rh),
        #         (0,float(params['dist_tgt_from_home'])), verbose=0)
        #except (ValueError,AttributeError) as e:
        #    traja = np.nan
        #    print(f'area: Error for {ti}: {e}')


        tr = LineString(list(zip(curveX[ind:], curveY[ind:])) )
        length = tr.length

        # yes, it should be > 0 to reverse, not < 0.
        if (pert > 0) and revert_pert:
            traja = -traja
        # not that we DO NOT want to revert trajoa

        # dist at leave home moment
        start_dist = np.sqrt( (tgtpathX_interp[0] - fbXhc[ind+0] )**2 +\
                         (tgtpathY_interp[0] - fbYhc[ind+0] )**2 )

        fb_lh_pt0adj = fbXhc[ind+0] - fb0pt[0], fbYhc[ind+0] - fb0pt[1]
        ang_fb_lh = np.math.atan2( *fb_lh_pt0adj )
        # positive when fb is on the right of ideal
        #print('ang_fb_lh' , ang_fb_lh * 180 / np.pi, 'ang_tgt', ang_tgt  * 180 / np.pi)
        error_lh_ang = ( ang_fb_lh - ang_tgt  ) * 180 / np.pi
        #error_lh_ang = -error_lh_ang

        ofb_lh_pt0adj = ofbXhc[ind+0] - fb0pt[0], ofbYhc[ind+0] - fb0pt[1]
        ang_ofb_lh = np.math.atan2( *ofb_lh_pt0adj )
        error_unpert_lh_ang = ( ang_ofb_lh - ang_tgt  ) * 180 / np.pi
        #plt.plot(ptinds, fxint(ptinds))

        ds2 = np.sqrt( (tgtpathX_interp - fbXhc[ind:] )**2 +\
                      (tgtpathY_interp - fbYhc[ind:] )**2 )
        err = np.sum(ds2)
        #if normalize:
        #    err /= len(ds2)

        #print('tgt vs tgtpath last X', tgtcur[0], tgtpathX_interp[-1] )
        #print('tgt vs tgtpath last Y', tgtcur[1], tgtpathY_interp[-1] )

        #enddist = np.sqrt( (txc[tgti] - fbXhc[-1])**2 + (tyc[tgti] - fbYhc[-1])**2 )
        enddist = np.sqrt( (tgtcur[0] - fbXhc[-1])**2 +\
                (tgtcur[1] - fbYhc[-1])**2 )
        #enddist2 = ds2[-1]

        #print(txc, fbXhc)
        #print( dfcurtr['error_distance']._values[-1], enddist, enddist2 )
    else:
        err = np.nan
        start_dist = np.nan
        td = np.nan
        traja = np.nan
        traja2 = np.nan
        trajoa = np.nan
        length = np.nan
        enddist = np.nan
        #enddist2 = np.nan
        error_lh_ang = np.nan
        error_unpert_lh_ang = np.nan
        ind = 0


    if ax is not None:
        rt = float(params['radius_target'] )
        for tgti_ in range(len(txc) ):
            if tgti_ == tgti:
                continue
            crc = plt.Circle((txc[tgti_], tyc[tgti_]), rt, color='blue', lw=2, fill=False,
                             alpha=0.3, ls='--')
            ax.add_patch(crc)
        crc = plt.Circle((txc[tgti], tyc[tgti]), rt, color='blue', lw=2, fill=False,
                         alpha=0.6)
        ax.add_patch(crc)
        ax.scatter( [tgtcur_ta[0]] , [tgtcur_ta[1] ]  ,
                   s=290, c='blue', alpha=0.5)

        ##################
        # nonaligned fb
        if abs(ang_tgt) > 1e-10:
            ax.scatter(fbXhc , fbYhc , alpha=0.3, label='fb (homec)',
                       marker='+', s = 60, c='r')

        # aligned fb
        ax.scatter(fbXhc_ta[:ind] , fbYhc_ta[:ind] , alpha=0.3, c='r')
        ax.scatter(fbXhc_ta[ind:] , fbYhc_ta[ind:] , alpha=0.4, c='r', label='fb ta (homec)')
        # mark black first exit point
        if ind > 0:
            ax.scatter( [fbXYlh_ta[0]], [fbXYlh_ta[1]] , alpha=0.8, c='k', s= 10)


            # nonaligned ofb
            axo.scatter(ofbXhc , ofbYhc , alpha=0.4, label='ofb (homec)',
                       marker='x', s = 30, c='magenta')

            # aligned ofb
            if abs(ang_tgt) > 1e-10:
                axo.scatter(curveX0[ind:] , curveY0[ind:] , alpha=0.4,
                           label='ofb ta (homec)',
                           marker='*', s = 30, c='magenta')

            ns = 6
            ptinds_s = np.arange(ns)
            fxint = interp1d([0, ns -1], [fb0pt[0], dirtgt[0]], 'linear' )
            tgtpathX_interp_s = fxint(ptinds_s)
            fyint = interp1d([0, ns-1 ], [fb0pt[1], dirtgt[1]], 'linear' )
            tgtpathY_interp_s = fyint(ptinds_s)

            ax.scatter(tgtpathX_interp_s , tgtpathY_interp_s , alpha=0.2,
                       c='cyan', s = 15)

        #################

        ax.scatter( *list(zip(dirtgt0)) , alpha=0.8, c='k', s= 10,
                   marker = 'x', label='dirtgt0')

        if ind > 0:
            ax.scatter( *list(zip(dirtgt) ) , alpha=0.8, c='k', s= 24,
                       marker = '+', label='dirtgt shiftscaled')

            ax.scatter( *list(zip(dirtgt_ta)) , alpha=0.8, c='k', s= 10,
                       marker = '*', label='dirtgt_ta')

            ############
            if len(tgtpathX_interp) > 20:
                skip = 3
            else:
                skip = 1
            tgtpathX_interp_vis = [tgtpathX_interp[0]] + list(tgtpathX_interp[1:-1:skip]) + [ tgtpathX_interp[-1] ]
            tgtpathY_interp_vis = [tgtpathY_interp[0]] + list(tgtpathY_interp[1:-1:skip]) + [ tgtpathY_interp[-1] ]
            ax.scatter(tgtpathX_interp_vis , tgtpathY_interp_vis , alpha=0.2, label='fb ideal', c='cyan')
            crc = plt.Circle((0, 0), rh, color='r', lw=2, fill=False,
                             alpha=0.6)
            ax.add_patch(crc)

        #################

        vft = dfcurtr['vis_feedback_type'].to_numpy()[0]
        #td = time_lh - dfcurtr["time"].to_numpy()[0]

        s = '\n'
        if addinfo is not None:
            r = addinfo
            for cols_ in titlecols:
                for col in cols_:
                    if col is None:
                        continue
                    s += f'{col}='
                    colv = r[col]
                    if isinstance(colv,float):
                        s += f'{colv:.2f}'
                    else:
                        s += f'{colv}'
                    s+='; '
                s += '\n'
            #s = f'\nerror={r["error_endpoint_ang"]:.1f}; trialwb={r["trialwb"]}; tt={r["trial_type"]}'

        ax.set_title(f'ti={ti}; vft={vft}; ' + s)
        ax.legend(loc='lower left')
        ax.set_xlim(xlim)

    return err, error_lh_ang, error_unpert_lh_ang, start_dist, traja, traja2, trajoa, td, length,  enddist

def row2multierr_test(home_position, target_coords, params,
                      test_type = 'ideal_traj',
                      pert=0, tgti=0, nsteps = 30,
                     ind_leavehome = 5,
                      ang_traj_rot = 12):
    from scipy.interpolate import interp1d
    from exper_protocol.utils import screen2homec, homec2screen
    rh = float( params['radius_home'])

    print(f'test_type = {test_type}; pert = {pert}; ang_traj_rot = {ang_traj_rot}')

    trial_index = -1
    row = {'trial_index':trial_index, 'perturbation':pert }

    time = 1000
    dt = 1./120.
    dfc = [  ]
    vft = f'rot{int(pert)}' # for plotTraj
    d0 = {'trial_index':trial_index,  'current_phase_trigger':-1,
          'tgti_to_show':tgti, 'vis_feedback_type':vft}

    pert *= np.pi / 180
    ang_traj_rot *= np.pi / 180

    rng = np.arange(nsteps)
    target_coords_homec = screen2homec( *tuple(zip(*target_coords)), home_position  )
    txc,tyc = target_coords_homec
    tgtcur = txc[tgti], tyc[tgti]
    ang = np.math.atan2(*tgtcur)  # ang of tgt
    dirx = rh * np.sin(ang)  # point on
    diry = rh * np.cos(ang)

    lentrunc = len(rng) - ind_leavehome
    ptinds = np.arange(lentrunc)

    fxint = interp1d([0, lentrunc ], [dirx, tgtcur[0] ], 'linear' )
    tgtpathX_interp = fxint(ptinds)
    fyint = interp1d([0, lentrunc ], [diry, tgtcur[1] ], 'linear' )
    tgtpathY_interp = fyint(ptinds)

    # rotation of ideal traj by specified angle (unrelated to pert)
    tgtpathX_interp_r, tgtpathY_interp_r =\
            rot(tgtpathX_interp, tgtpathY_interp, ang_traj_rot)

    tgtpathX_interp_pert, tgtpathY_interp_pert =\
            rot(tgtpathX_interp, tgtpathY_interp, pert)

    tgtpathX_interp_r_pert, tgtpathY_interp_r_pert =\
            rot(tgtpathX_interp, tgtpathY_interp, pert + ang_traj_rot)

    # to screen coords

    tgtpathX_interp_s, tgtpathY_interp_s =\
            homec2screen( tgtpathX_interp, tgtpathY_interp,
                         home_position)
    tgtpathX_interp_rs, tgtpathY_interp_rs =\
            homec2screen( tgtpathX_interp_r, tgtpathY_interp_r,
                         home_position)
    tgtpathX_interp_pert_s, tgtpathY_interp_pert_s =\
            homec2screen(tgtpathX_interp_pert,
                         tgtpathY_interp_pert, home_position)
    tgtpathX_interp_r_pert_s, tgtpathY_interp_r_pert_s =\
            homec2screen(tgtpathX_interp_r_pert,
                         tgtpathY_interp_r_pert, home_position)


    zigx = 30
    if test_type == 'unpert_traj_ideal':
        curveX0, curveY0 = tgtpathX_interp_s, tgtpathY_interp_s
        curveX,  curveY  = tgtpathX_interp_pert_s, tgtpathY_interp_pert_s
    elif test_type == 'unpert_traj_rot':
        curveX0, curveY0 = tgtpathX_interp_rs, tgtpathY_interp_rs
        curveX,  curveY  = tgtpathX_interp_r_pert_s, tgtpathY_interp_r_pert_s
    elif test_type == 'unpert_zig_left':
        tmpX = dirx,np.mean([dirx,tgtcur[0]])-zigx,tgtcur[0]
        tmpY = diry,np.mean([diry,tgtcur[1]]) ,tgtcur[1]
        curveX0, curveY0 = homec2screen(tmpX,tmpY,home_position)
        curveX, curveY = homec2screen(*rot(tmpX,tmpY,pert), home_position)
    elif test_type == 'unpert_zig_right':
        tmpX = dirx,np.mean([dirx,tgtcur[0]])+zigx,tgtcur[0]
        tmpY = diry,np.mean([diry,tgtcur[1]]) ,tgtcur[1]
        curveX0, curveY0 = homec2screen(tmpX,tmpY,home_position)
        curveX, curveY = homec2screen(*rot(tmpX,tmpY,pert), home_position)
    elif test_type == 'unpert_zigzag_left':
        tmpX = dirx,np.mean([dirx,tgtcur[0]])-zigx,np.mean([dirx,tgtcur[0]])+zigx
        tmpY = diry,np.mean([diry,tgtcur[1]]) ,tgtcur[1]
        curveX0, curveY0 = homec2screen(tmpX,tmpY,home_position)
        curveX, curveY = homec2screen(*rot(tmpX,tmpY,pert), home_position)
    elif test_type == 'pert_zig_left':
        tmpX = dirx,np.mean([dirx,tgtcur[0]])-zigx,tgtcur[0]
        tmpY = diry,np.mean([diry,tgtcur[1]]) ,tgtcur[1]
        curveX, curveY = homec2screen(tmpX,tmpY,home_position)
        curveX0, curveY0 = homec2screen(*rot(tmpX,tmpY,-pert), home_position)
    #elif test_type == 'pert_zig_rigth':
    elif test_type == 'pert_traj_ideal':
        curveX, curveY = tgtpathX_interp_s, tgtpathY_interp_s
        curveX0, curveY0 = rot(tgtpathX_interp_s, tgtpathY_interp_s, -pert)

    #print(tmpX, tmpY)
    #print(curveX0, curveY0)
    #print(curveX, curveY)

    org = home_position
    for i in range(ind_leavehome):
        d = d0.copy()
        d['feedbackX'],d['feedbackY'] = home_position
        d['unpert_feedbackX'],d['unpert_feedbackY'] = home_position
        time += dt
        d['time'] = time
        dfc += [d  ]

    # ideal reach, equal to ideal traj
    for i in range(ind_leavehome, len(rng) ):
        ii = i - ind_leavehome
        d = d0.copy()
        X,Y = curveX0[ii], curveY0[ii]
        #print(ii,X - home_position[0],Y)
        d['unpert_feedbackX'],d['unpert_feedbackY'] = X,Y
        X2,Y2 = curveX[ii], curveY[ii]
        d['feedbackX'],d['feedbackY'] = X2,Y2
        time += dt
        d['time'] = time
        dfc += [d  ]

    dfc = pd.DataFrame(dfc)
    dfc['error_distance' ]  = np.nan # for plotTraj
    grp_perti = dfc.groupby('trial_index')


    dfcurtr = dfc
    ax = plt.gca()
    plotTraj(ax, dfcurtr, home_position, target_coords_homec, params,
            calc_area = False, show_dist_guides = False, verbose=0,
                 force_entire_traj=False, addinfo=None)

    r = row2multierr(row, dfc, grp_perti, home_position, target_coords,
                 params, revert_pert = False )
    z = zip( ['err', 'error_lh_ang', 'error_unpert_lh_ang', 'start_dist',
              'traja', 'traja2', 'trajoa', 'td',
        'length',  'enddist'], r)
    d = dict(z)
    for k,v in d.items():
        print(f'{k} = {v:.2f}' )
    #err, start_dist, traja, trajoa, td, length,  enddist, enddist2 = r
    #print(r)

def calcAdvErrors(dfcc, dfc, grp_perti, target_coords,
                  home_position, params, revert_pert= False):
    from scipy.interpolate import interp1d
    from shapely.geometry import LineString
    from exper_protocol.utils import screen2homec
    from behav_proc import area
    from base2 import rot

    #target_coords_homec = screen2homec( *tuple(zip(*target_coords)), home_position  )

    #tx,ty = target_coords_homec
    #if np.diff(tx).min() < 0:
    #    print('inverting x of targets')
    #    tx = -tx


    def f(row):
        return row2multierr(row, dfc, grp_perti, home_position,
            target_coords, params, revert_pert = revert_pert)
    dfcc[['error_intdist2_nn', 'error_lh_ang', 'error_unpert_lh_ang',
          'error_distance_lh',
      'error_area_signed_nn','error_area2_signed_nn', 'error_area_ofb_signed_nn',
          'time_lh', 'traj_length',
     'enddist']] = dfcc.apply(lambda x: f(x) ,1, result_type='expand')

    from base2 import calcNormCoefSectorArea
    norm_coef =  calcNormCoefSectorArea(params)

    dfcc['error_area_signed_nn' ]      *= norm_coef
    dfcc['error_area2_signed_nn' ]      *= norm_coef
    dfcc['error_area_ofb_signed_nn' ]  *= norm_coef

    #def f(row, normalize):
    #    return row2multierr(row, dfc, grp_perti, home_position,
    #                 target_coords, params, normalize, curve_for_area_calc='org_feedback')
    #dfcc['error_area_ofb_signed_nn'] = dfcc.apply(lambda x: f(x, True)[2] ,1)

    ####################################

    des_time = 30 * 1e-3
    frame = 1 / float( params['FPS'] )
    nframes_stat = int(np.ceil(des_time / frame))
    print('nframes_stat = ',nframes_stat)

    hitr = float(params['radius_target']) + float(params['radius_cursor']) / 2
    dfcc['time_mvt'] = np.nan
    #dfcc.loc[dfcc['nonhit'], 'time_mvt']  = float(params['time_feedback'] ) - dfcc.loc[dfcc['nonhit'], 'time_lh']
    for rowi, row in dfcc.iterrows():
        if row['nonhit']:
            time_mvt = float(params['time_feedback']) - row['time_lh']
        else:
        #dfr = dfc_all.query('subject == @subjects[0] and trial_index == @ti')
            ti = row['trial_index']
            dfr = dfc.query('trial_index == @ti')

            dfsum = (dfr['error_distance'] <= hitr).astype(int).\
                rolling(nframes_stat).sum()
            inds = np.where(dfsum > nframes_stat - 1e-10)[0]
            if len(inds) == 0:
                lastind = len(dfr) - 1
            else:
                inds_leave_tgt = np.where( np.diff(inds) > 1 )[0]
                if len(inds_leave_tgt):
                    lastind = inds_leave_tgt[-1]
                else:
                    lastind = inds[0]

            time_mvt = lastind / float( params['FPS'] ) - row['time_lh']
        #row['time_mvt'] = time_mvt
        dfcc.loc[rowi, 'time_mvt'] = time_mvt

    ####################################


    dfcc['error_intdist2'] = dfcc['error_intdist2_nn'] / dfcc['time_mvt']
    dfcc['error_area_signed'] = dfcc['error_area_signed_nn'] / dfcc['time_mvt']
    dfcc['error_area2_signed'] = dfcc['error_area2_signed_nn'] / dfcc['time_mvt']
    dfcc['error_area_ofb_signed'] = dfcc['error_area_ofb_signed_nn'] / dfcc['time_mvt']

    coef_endpt_err  = 0.5
    dfcc['error_aug2'] = dfcc['error_intdist2'] + coef_endpt_err * dfcc['error_distance']

    trajlen_ideal = params['dist_tgt_from_home']  - \
            float(params['radius_home'] ) - float(params['radius_target'] )
    dfcc['traj_length_adj'] = dfcc['traj_length'] - trajlen_ideal

def loadTriggerLog(fnf, CONTEXT_TRIGGER_DICT):
    '''
    fnf includes extension
    '''
    dftriglog = pd.read_csv(fnf, delimiter=';', names = ['trigger', 'time','addinfo'])

    def f(row):
        r = row['addinfo']
        tind = -100
        #print(r, type(r))
        if (r is not None) and (not isinstance(r,float)):
            r = eval(r)
            tind = r.get('trial_index')
        return tind
    dftriglog['trial_index'] = dftriglog.apply(f,1)
    dftriglog = dftriglog.query('trigger != 0').reset_index().drop(labels='index',axis=1)

    CONTEXT_TRIGGER_DICT_inv = dict(zip(CONTEXT_TRIGGER_DICT.values(), CONTEXT_TRIGGER_DICT.keys()))

    def f2(row):
        r = None,None,None,None
        ai = row.get('addinfo',None)
        if ai is not None:
            #print('ai',ai)
            try:
                ai = eval(ai)
            except TypeError as e:
                return r
            tpl = ai.get('tpl',None)
            #print(tpl)
            if tpl is not None:
                r = tpl
                #CONTEXT_TRIGGER_DICT_inv[tpl]
        return r
    dftriglog[['trial_type', 'vis_feedback_type', 'tgti_to_show', 'phase']] = \
        dftriglog.apply(f2,1, result_type='expand')
    return dftriglog

def printPretrialMistakesDiag(df, params):
    # on which trials had participant leaving home during motor prep? And how many times?
    df_notmid = df.query('subphase_relation != "middle"')
    dfsz= df_notmid.groupby(['trial_index','phase','subphase_relation']).size()
    print('Num leave home motor prep')
    print(dfsz[dfsz > 1])

    ph = 'REST'
    szrest = df.query('phase == @ph').groupby('trial_index').size()
    nframes_normal = np.min(szrest) + 1
    print(f'for {ph} nframes_normal = ',nframes_normal)

    # on which trials participant left home during REST or had to adjust position 
    # in the beginning of the trial to return to rest? 
    print( szrest[szrest> nframes_normal] / float(params['FPS']) )

    return dfsz

def set_streaks(df, inds = None, inplace=True):
    '''
    takes behav dataset for single subject and detects phase changes (including
    those happening within trial including when same phase repeats more than once)
    sets for every timeframe whether it is a beginning of the end of the phase
    '''
    if not inplace:
        df = df.copy()
    df['subphase_relation'] = 'middle'
    if inds is not None:
        df_ = df.loc[inds]
    else:
        df_ = df
    
    #df_['subphase_relation'] = 'middle'
    ph = df_['phase'] 
    #assert np.sum(rest_mask) > 0

    csr = (ph != ph.shift()).cumsum()
    df_['csr'] = csr
    #print(csr.max())
    number_of_rest_streaks = len( df_.groupby(csr).first() )#.query('phase == "REST"') )
    #print(number_of_rest_streaks)

    firstis = csr.to_frame().reset_index().groupby('phase').first()['index'].values  # first ind of 
    lastis = csr.to_frame().reset_index().groupby('phase').last()['index'].values  # first ind of 

    df.loc[firstis,'subphase_relation'] = 'first'
    df.loc[lastis, 'subphase_relation'] = 'last'
    #display(df_.loc[firstis, ['phase','csr','subphase_relation']])
    
    # useful to choose last streak of REST or GO_CUE
    df['csr_neg_within_trial'] = df['csr'] - df.groupby('trial_index')['csr'].transform('max')
    #df_.drop(labels=['csr'],axis=1)
    return df, list(zip(firstis,lastis))

def aggRows(df, coln_time, operation, grp = None, coltake='corresp', 
            colgrp = 'trial_index' ):
    # coln_time = 'time'
    assert operation in ['min','max']
    assert df[coln_time].diff().min() >= 0  # need monotnicity
    assert coltake is not None

    if grp is None:
        grp = df.groupby(colgrp)

    if coltake != 'corresp':
        cns = [cn for cn in df.columns if cn != coln_time]
        if operation == 'min':
            coltake = 'first'
        elif operation == 'max':
            coltake = 'last'
        agg_d = dict(zip(cns, len(cns) * [coltake]))
        agg_d[coln_time] = operation
        dfr = grp.agg(agg_d)
    else:
        if operation == 'min':
            idx = grp[coln_time].idxmin()
        elif operation == 'max':
            idx = grp[coln_time].idxmax()
        dfr = df.loc[idx]
    return dfr

def compareTriggers(df, dfev, dftriglog):
    print('df contents')
    #print(len( df_trst.query('phase == "REST"') ))

    phase2best_neg_csr = {'REST':-3, 'GO_CUE_WAIT_AND_SHOW':-2 }
    phases = df['phase'].unique()
    #df.dtypes.values == np.dtype('O')
    #phase_take_max = []
    #cns = [cn for cn in df.columns if cn != 'time']
    #agg_d = dict(zip(cns, len(cns) * ['first']))
    #agg_d['time'] = 'min'
    ##if phase in phase_take_max:
    ##    agg_d['time'] = 'max'
    ##else:
    ##    agg_d['time'] = 'min'
    ##{'time': 'min', **df.select_dtypes(include=['object']).apply(lambda x: x.iloc[0])}
    grp_perti = df.groupby(['trial_index'])
    #df_trst = grp_perti.agg(agg_d)
    
    operation = 'min'
    coln_time = 'time'
    df_trst = aggRows(df, coln_time, operation, grp = grp_perti )
    print(len( df_trst ))

    phase2df = {}
    phase2dfev = {}
    phase2dftriglog = {}
    for phase in phases:    
        if phase in phase2best_neg_csr.keys():
            crs_neg_v = phase2best_neg_csr[phase]
            # sel only last streak
            #df_trst2 = df.query('phase == @phase and csr_neg_within_trial == @crs_neg_v '
            #                         ' and subphase_relation == "first"')#[['csr_within_trial','phase']]
            # sel all streaks
            df_trst2 = df.query('phase == @phase '
                                     ' and subphase_relation == "first"')#[['csr_within_trial','phase']]
        else:
            grp_perti2 = df.query('phase == @phase').groupby(['trial_index'])
            df_trst2 = aggRows(df, coln_time, operation, grp = grp_perti2 )
            #df_trst2 = grp_perti.agg(agg_d)


        phase2df[phase] = df_trst2
        phase2dfev[phase] = dfev.query('phase == @phase')
        l2 = len(  phase2dfev[phase])

        phase2dftriglog[phase] =  dftriglog.query('phase == @phase')
        l3 = len(phase2dftriglog[phase])
        print(f'{phase:20}, df len = {len( df_trst2 ):3}, dfev len = {l2:3} triglog len = ',l3)

    ########

    print('')
    print('##############')
    print('Differences of timing per phase between df and trigger timing')
    for phase in phases: 
        if phase == 'BREAK':
            continue
        dftmps = [phase2df[phase],phase2dfev[phase] ]#,phase2dftriglog[phase]]
        vss = []
        for dftmp in dftmps:
            vs = dftmp['time_since_first_REST'].values
            vss+=[vs]
        dif = vss[0] - vss[1]
        if len(dif):
            print('{:20}, diff time log - meg min={:.4f}, max={:.4f}'.format( phase, np.min( dif), np.max( dif) ) )
        else:
            print(f'zero dif len for {phase}')


def addBasicInfo(df, phase2trigger, params,
                home_position, target_coords,
                phase_to_collect = 'TARGET_AND_FEEDBACK', training_end_sep_trial = False):
    #### 
    # This is for context change experiment
    ############################################################

    assert df['subject'].nunique() == 1

    trigger2phase = dict( zip(phase2trigger.values(),phase2trigger.keys()) )
    print(trigger2phase)
    df['phase'] = df.apply(lambda row: trigger2phase[row['current_phase_trigger']], 1)

    c = df['phase'] == phase_to_collect
    #subj = 'romain'
    #subj = 'dima2'
    #subj = 'coumarane'
    #subj = 'romain2'

    #subj = '2023-SE1-pilot3'
    #c &= df['subject'] == subj; print(f"RESTRICT TO ONE SUBJECT {subj}")

    # it DOES NOT include pauses
    dfc = df[c].copy().reset_index()
    assert len(dfc)


    df, flpairs = set_streaks(df,inds = None) 

    #assert dfc['current_phase_trigger'].nunique() == 1

    from exper_protocol.utils import (calc_err_eucl, coords2anglesRad,
                                      screen2homec, homec2screen)
    # targetAngs = get_target_angles(self.params['num_targets'],
    #             params['target_location_pattern'],
    #             params['target_location_spread'])



    ########################  set target_locs and extract perturbation


    def f(row):
        #fb = (row['unpert_feedbackX'],  row['unpert_feedbackY'])
        x,y = target_coords[row['tgti_to_show']]
        ang = coords2anglesRad(x,y, home_position)
        #ang -= np.pi / 2 + np.pi
        #print(ang)
        return ang / np.pi * 180

        #eturn ang
    dfc['target_locs'] = dfc.apply(f,1)

    pertn2pertv = dict( zip(['veridical', 'rot15', 'rot30', 'rot-15',
                            'rot-20','rot20'],[0,15,30,-15,-20,20]) )

    dfc['perturbation'] = dfc.apply(lambda row: pertn2pertv[row['vis_feedback_type']],1 )


    # add time since right phase of the trial start

    grp_perti = dfc.groupby(['trial_index'])
    mi = grp_perti['time'].min().reset_index()
    #display(mi)
    ti2min = mi.set_index('trial_index').\
        to_dict('index')

    def f(row, normalize=1):
        ti = row['trial_index']
    #     idx = grp_perti.groups[ti]
    #     dfcurtr = dfc.loc[idx[1:]]
    #     mint = dfcurtr['time'].min()
        mint = ti2min[ti]['time']
        return row['time'] - mint

    dfc['time_since_trial_start'] = dfc.apply(f,1)


    ####################  get pauses

    lbd = lambda x : f'{x:.2f}'
    dfc['dist_rad_from_prevtgt'] = dfc['target_locs'].diff().abs().apply(lbd,1)

    trg = phase2trigger['PAUSE']
    dfcp = df.query('current_phase_trigger == @trg')
    grp = dfcp.groupby(['trial_index'])
    idx = grp['time'].transform(max) == dfcp['time'] #.size()
    dfcpc = dfcp.loc[idx]

    ##########################

    # selmax
    # TODO: use agg instead of == time
    grp = dfc.groupby(['trial_index'])
    #idx = grp['time'].transform(max) == dfc['time'] #.size()
    #dfcc = dfc.loc[idx]
    # this should work better 
    dfcc = aggRows(dfc, 'time', 'max', grp, coltake = 'corresp') 

    # it DOES NOT include pauses
    dfcc = dfcc.reset_index(drop=True)


    all_ctx, cpls = genCtxPairLists(dfcc)
    setContextSimilarityCols(dfcc, cpls)
    assert (~dfcc['prev_ctx_some_close'].isnull()).sum() > 0

    numtrain = params['num_training']
    ##################################   set ctx id and count appearances

    def f(row):
        try:
            tgti = int( row['tgti_to_show'] )
            pert = int( row['perturbation'] )
            ctx = (pert, tgti)
            ctxi = all_ctx.index(ctx)
        except ValueError as e:
            print(e,tgti,pert)
            ctxi = -1000
            
        return ctxi

    dfcc['ctxid'] = dfcc.apply(f,1)
    assert not dfcc['ctxid'].isna().any()
    assert np.sum(dfcc['ctxid'] < 0) == 0

    dfcc['Nctx_app'] = -1000
    #grp = dfcc_all_.query(qs_notspec).groupby(['subject','ctxid'])
    grp = dfcc.query(qs_notspec).groupby(['ctxid'])
    for g in grp.groups:
        dfcc_tmp = dfcc.loc[grp.groups[g] ]
        assert dfcc_tmp['time'].diff().max() > 0
        # dont need +1 because I want to skip training block
        r = (dfcc_tmp['block_ind'] != dfcc_tmp['block_ind'].shift(1)).cumsum() #+ 1
        dfcc.loc[grp.groups[g], 'Nctx_app'] =  r
        #print(g, r.max(), dfcc_tmp['block_ind'].unique())
    assert np.max(dfcc['Nctx_app'] ) == params['n_context_appearences']


    ##################################





    #dfcc['feedbackY']          = params['height'] - dfcc['feedbackY']
    #dfcc['cursorY']            = params['height'] - dfcc['cursorY']
    #dfcc['unpert_feedbackY']   = params['height'] - dfcc['unpert_feedbackY']

    grp = dfcc.groupby('block_ind')
    block_starts = grp['trial_index'].min()

    block_starts = block_starts.to_dict()

    def f(row):
        block_ind = row['block_ind']
        ts = block_starts[block_ind]
        return row['trial_index'] - ts
    dfcc['trialwb'] = dfcc.apply(f,1)


    hitr = float(params['radius_target']) + float(params['radius_cursor']) / 2
    dfcc['nonhit'] = dfcc['error_distance'] > hitr


    ############################## compute angular info


    def f(row):
        fb = (row['feedbackX'], row['feedbackY'])
        r = calc_err_eucl(fb, target_coords, row['tgti_to_show'])
        return r
    dfcc['error_eucl'] = dfcc.apply(f,1)  # takes last


    def f(row):
        fb = (row['feedbackX'],  row['feedbackY'])
        try:
            ang = coords2anglesRad(*fb, home_position)
        except ZeroDivisionError as e:
            ang = np.nan
        #ang -= np.pi / 2
        return ang / np.pi * 180
        #eturn ang
    dfcc['feedback_abs'] = dfcc.apply(f,1)

    def f(row):
        fb = (row['unpert_feedbackX'],  row['unpert_feedbackY'])
        try:
            ang = coords2anglesRad(*fb, home_position)
        except ZeroDivisionError as e:
            ang = np.nan
        #ang -= np.pi / 2
        return ang / np.pi * 180
        #eturn ang
    dfcc['org_feedback_abs'] = dfcc.apply(f,1)


    # feedback is relative to the target (including pert)
    dfcc['feedback']     = (dfcc['feedback_abs'] - 90)    - (dfcc['target_locs'] - 90)
    # org feedback is relative to the target (NOT including pert)
    dfcc['org_feedback'] = dfcc['org_feedback_abs'] - dfcc['target_locs']
    dfcc['error_endpoint_ang'] = dfcc['feedback_abs'] - dfcc['target_locs']

    #################################  compound error info


    #print("AAAAAAAAAAAAAAAAAA")
    #assert dfc['current_phase_trigger'].nunique() == 1
    #assert dfcc['current_phase_trigger'].nunique() == 1
    calcAdvErrors(dfcc, dfc, grp_perti, target_coords,
                  home_position, params, revert_pert = False)

    dfcc.loc[dfcc['time_lh'].isna(), 'error_endpoint_ang'] = np.nan
    dfcc.loc[dfcc['time_lh'].isna(), 'error_pert_adj'] = np.nan


    colns_adj180 = ['error_endpoint_ang', 'error_lh_ang', 'error_unpert_lh_ang']
    for coln in colns_adj180:
        c = dfcc[coln] > 180
        dfcc.loc[c, coln] =  dfcc.loc[c, coln] - 360
        c = dfcc[coln] < -180
        dfcc.loc[c, coln] =  dfcc.loc[c, coln] + 360


    c = dfcc['perturbation'] < 0
    dfcc['error_endpoint_ang_pert_adj'] = dfcc['error_endpoint_ang']
    dfcc.loc[c, 'error_endpoint_ang_pert_adj'] = -dfcc.loc[c,'error_endpoint_ang_pert_adj']


    col = 'error_area_signed'
    col2 = 'error_pert_adj'
    col3 = 'error_distance'
    if col in dfcc.columns:
        coef = dfcc[col].quantile(0.75) / dfcc[col2].quantile(0.75)
        dfcc['error_aug3'] = dfcc[col] + coef * dfcc[col2]

        #sc = np.sqrt( dfcc[col3] / hitr )
        ## bigger error -- stronger scale
        #dfcc['error_area_signed_scaled_ed'] = dfcc[col] * sc
        #dfcc['error_area_ofb_signed_scaled_ed'] = dfcc['error_area_ofb_signed'] * sc

        dfcc['error_area_signed_tln']     = dfcc['error_area_signed_nn'] / dfcc['traj_length']
        dfcc['error_area_ofb_signed_tln'] = dfcc['error_area_ofb_signed_nn'] / dfcc['traj_length']


        sc = np.power( np.maximum(dfcc['error_distance'] / hitr, 1.)   , 1./8. )
        dfcc['error_area_signed_tln_scaled_ed'] = dfcc['error_area_signed_tln'] * sc
        dfcc['error_area_ofb_signed_tln_scaled_ed'] = dfcc['error_area_ofb_signed_tln'] * sc

        dfcc['error_area_signed_scaled_ed'] = dfcc['error_area_signed'] * sc
        dfcc['error_area_ofb_signed_scaled_ed'] = dfcc['error_area_ofb_signed'] * sc

        dfcc['error_area_signed_nn_scaled_ed']     = dfcc['error_area_signed_nn'] * sc
        dfcc['error_area2_signed_nn_scaled_ed']     = dfcc['error_area2_signed_nn'] * sc
        dfcc['error_area_ofb_signed_nn_scaled_ed'] = dfcc['error_area_ofb_signed_nn'] * sc
    else:
        print(f'WARNING: {col} not in dfcc.columns')

    # sanity check (that we have continuous trial increase)
    dfccdiff = dfcc['trial_index'].diff()
    dfccdiff = dfccdiff[~dfccdiff.isna()]
    if training_end_sep_trial:
        checkval = 3
    else:
        checkval = 2
    assert (dfccdiff <= checkval).all()  # can have pauses where we get 2
    #dfcc.loc[dfccdiff > 2]

    ###############

    tgtichange = dfc['tgti_to_show'].diff() > 0

    #$dfct = dfc.query('dist_rad_from_prevtgt= "0.00"')
    dfct = dfc
    grp = dfct.groupby(['tgti_to_show','vis_feedback_type','dist_rad_from_prevtgt'])

    idx = grp['time'].transform(min) == dfct['time'] #.size()
    dfctc = dfct.loc[idx]

    dftmp = df[['trial_index','time','trial_type'] ].groupby('trial_index').min(numeric_only=0).reset_index().set_index('trial_index')
    dftmp['prev_trial_type'] = None
    dftmp['prev_trial_type'] =dftmp['trial_type'].shift(1)

    dfcc = dfcc.set_index('trial_index')
    dfcc['prev_trial_type'] = dftmp.loc[dfcc.index,'prev_trial_type']

    dfcc = dfcc.reset_index()


    # df  -- row per screen update, with streak relations, all phases
    # dfc -- row per screen update, only TARGET_AND_FEEDBACK
    # dfcc -- row per trial
    # dfcpc -- row per pause
    # dfctc -- row per target change
    return df, dfc, dfcc, dfcp, dfctc

def plotTraj2(ax, dfcurtr, home_position, target_coords_homec, params,
        calc_area = False, show_dist_guides = False, verbose=0,
             force_entire_traj=False, addinfo=None, titlecols=[],
             xlim=(-140,140) ):
    from exper_protocol.utils import (get_target_angles,
        calc_target_positions, calc_err_eucl, coords2anglesRad, screen2homec,
                                     homec2screen)
    from scipy.interpolate import interp1d
    ll = len(dfcurtr)

    ti = dfcurtr['trial_index']._values[0]

    #dfcurtr#[['feedbackX', 'feedbackY', 'unpert_feedbackY']]

    fbXY = dfcurtr[['feedbackX', 'feedbackY']].to_numpy()
    #fbXY.shape

    if verbose:
        print(fbXY[:,0], fbXY[:,1] )
    fbXhc, fbYhc = screen2homec(fbXY[:,0], fbXY[:,1], home_position  )
    fbXYhc = np.array( [fbXhc, fbYhc], dtype=float )

    rh = float( params['radius_home'])
    ds = np.sqrt( fbXhc**2 + fbYhc**2 )
    leave_home_coef = 1.
    inds_leavehome = np.where(ds > rh * leave_home_coef )[0]
    txc,tyc = target_coords_homec
    tgti = dfcurtr['tgti_to_show']._values[0]
    tgtcur = np.array( [txc[tgti], tyc[tgti] ], dtype=float )


    fb0pt = np.array( [fbXhc[0], fbYhc[0] ] , dtype=float)
    # shift target in frame with zero at start of the reach
    tgtcur_adj = tgtcur - fb0pt

    ang = np.math.atan2(*tuple(tgtcur_adj) )
    if verbose:
        print('tgtcur_adj', tgtcur_adj, 'ang', ang)

    fbXYhc_adj = fbXYhc - fb0pt[:,None]

    fbXYhc_ta = rot( *fbXYhc, ang, fb0pt )
    fbXhc_ta, fbYhc_ta = fbXYhc_ta

    tgtcur_ta = rot( *tgtcur, ang, fb0pt )

    pert=  addinfo['perturbation']
    #ofbXhc0, ofbYhc0 = rot(fbXhc, fbYhc, pert / 180 * np.pi  )

    ofbXY = dfcurtr[['unpert_feedbackX', 'unpert_feedbackY']].to_numpy()
    ofbXhc, ofbYhc = screen2homec(ofbXY[:,0], ofbXY[:,1], home_position  )

    #assert( np.linalg.norm(ofbXhc0- ofbXhc)  < 1e-10 )

    if len(inds_leavehome) and (not force_entire_traj):
        ind  = inds_leavehome[0]
    else:
        ind = 0

    time_lh = dfcurtr['time'].to_numpy()[ind]

    if verbose:
        print(f'ti = {ti}; tgti = {tgti}; ang={ang}; ind = {ind}; '
              f'inds_leavehome = {inds_leavehome}')

    lentrunc = ll - ind

    #from scipy.integrate import quad
    ptinds = np.arange(lentrunc)
    # for some reason order of tgts gets inverted. Or maybe only x coords?
    dirx = rh * np.sin(ang) + fb0pt[0]
    diry = rh * np.cos(ang) + fb0pt[1]
    dirlh = np.array( [dirx,diry] )

    ax.scatter( *list(zip(dirlh)) , alpha=0.8, c='k', s= 10,
               marker = 'x', label='dirlh')
    if verbose:
        print( 'dirlh norm', np.linalg.norm(dirlh ) )

    #################################################

    rh2 = np.sqrt( dirx**2 + diry**2 )
    coef = rh / rh2
    if verbose:
        print('coef dir ' ,coef)
    dirlh = (dirlh - fb0pt) * coef + fb0pt
    if verbose:
        print( 'dir shiftsc norm', np.linalg.norm(dirlh ) )

    ax.scatter( *list(zip(dirlh) ) , alpha=0.8, c='k', s= 24,
               marker = '+', label='dirlh shiftscaled')


    #################################################


    if verbose:
        print('fb0pt ', fb0pt)
    dirlh_ta = rot(*dirlh, ang, fb0pt) #- fb0pt[0],0
    if verbose:
        print('dirlh_ta ',dirlh_ta)
    # cathet from hyptohenuse and cathet
    dirlh_ta[1] = np.sqrt( rh**2 - dirlh_ta[0]**2 )
    if verbose:
        print('dirlh_ta impr',dirlh_ta)
        print('tgtcur_ta',tgtcur_ta)

    ax.scatter( *list(zip(dirlh_ta)) , alpha=0.8, c='k', s= 10,
               marker = '*', label='dirlh_ta')


    fxint = interp1d([0, lentrunc-1 ], [dirlh[0], txc[tgti]], 'linear' )
    tgtpathX_interp = fxint(ptinds)
    fyint = interp1d([0, lentrunc-1 ], [dirlh[1], tyc[tgti]], 'linear' )
    tgtpathY_interp = fyint(ptinds)

    ns = 6
    ptinds_s = np.arange(ns)
    fxint = interp1d([0, ns -1], [fb0pt[0], dirlh[0]], 'linear' )
    tgtpathX_interp_s = fxint(ptinds_s)
    fyint = interp1d([0, ns-1 ], [fb0pt[1], dirlh[1]], 'linear' )
    tgtpathY_interp_s = fyint(ptinds_s)

    lns = []
    for i in range(0,lentrunc,7):
        ln = plt.Line2D( [tgtpathX_interp[i], fbXhc[ind+i] ],
                [tgtpathY_interp[i] , fbYhc[ind+i] ]  )
        #plt.Line2D( [tgtpathX_interp[i], fbXhc[ind+i] ],  [tgtpathY_interp[i] , fbYhc[ind+i] ]  )
        lns.append(ln)

    #plt.plot(ptinds, fxint(ptinds))

    ds2 = np.sqrt( (tgtpathX_interp - fbXhc[ind:] )**2 + (tgtpathY_interp - fbYhc[ind:] )**2 )
    err = np.sum(ds2) / len(ds2)

    fbXYlh_ta = fbXYhc_ta[:,ind]

    if calc_area and len(inds_leavehome):
        ideal_lh = dirlh_ta
        #traja = area(fbXhc_ta[ind:], fbYhc_ta[ind:], (0,rh),
        #             (0,float(params['dist_tgt_from_home'])), verbose=0)
        traja = area(fbXhc_ta[ind:], fbYhc_ta[ind:], ideal_lh,
                     tgtcur_ta, verbose=0, ax=ax,
                     plotargs = {'alpha':0.2})

    else:
        traja = np.nan


    #dfcurtr['tgti_to_show']._values[0]


    if verbose:
        print(f'ds[-1] = {ds[-1]}, error_distance= {dfcurtr["error_distance"].values[-1]}')
    #print(f'err={err}, errAug={errAug}')


    rt = float(params['radius_target'] )
    for tgti_ in range(len(txc) ):
        if tgti_ == tgti:
            continue
        crc = plt.Circle((txc[tgti_], tyc[tgti_]), rt, color='blue', lw=2, fill=False,
                         alpha=0.3, ls='--')
        ax.add_patch(crc)
    crc = plt.Circle((txc[tgti], tyc[tgti]), rt, color='blue', lw=2, fill=False,
                     alpha=0.6)
    ax.add_patch(crc)

    #ax.scatter( [tgtcur_adj[0]] , [tgtcur_adj[1] ]  ,
    #           s=290, c='cyan', alpha=0.5)
    ax.scatter( [tgtcur_ta[0]] , [tgtcur_ta[1] ]  ,
               s=290, c='magenta', alpha=0.5)


    if abs(ang) > 1e-10:
        ax.scatter(fbXhc , fbYhc , alpha=0.3, label='fb (homec)', marker='+', s = 60, c='orange')

    ax.scatter(ofbXhc , ofbYhc , alpha=0.4, label='ofb (homec)', marker='x', s = 30, c='magenta')
    #ax.scatter(ofbXhc0 , ofbYhc0 , alpha=0.4, label='ofb0 (homec)', marker='+', s = 30, c='magenta')


    ax.scatter( [fbXYlh_ta[0]], [fbXYlh_ta[1]] , alpha=0.8, c='k', s= 10)

    ax.scatter(fbXhc_ta[:ind] , fbYhc_ta[:ind] , alpha=0.3, c='r')
    ax.scatter(fbXhc_ta[ind:] , fbYhc_ta[ind:] , alpha=0.4, c='r', label='fb tgtalign (homec)')

    ax.scatter(tgtpathX_interp_s , tgtpathY_interp_s , alpha=0.2, c='cyan', s = 15)


    if len(tgtpathX_interp) > 20:
        skip = 3
    else:
        skip = 1
    tgtpathX_interp_vis = [tgtpathX_interp[0]] + list(tgtpathX_interp[1:-1:skip]) + [ tgtpathX_interp[-1] ]
    tgtpathY_interp_vis = [tgtpathY_interp[0]] + list(tgtpathY_interp[1:-1:skip]) + [ tgtpathY_interp[-1] ]
    ax.scatter(tgtpathX_interp_vis , tgtpathY_interp_vis , alpha=0.2, label='fb ideal', c='cyan')
    crc = plt.Circle((0, 0), rh, color='r', lw=2, fill=False,
                     alpha=0.6)
    ax.add_patch(crc)
    if show_dist_guides:
        for ln in lns:
            #plt.gca().add_patch(ln)
            ax.add_line(ln)

    vft = dfcurtr['vis_feedback_type'].to_numpy()[0]
    td = time_lh - dfcurtr["time"].to_numpy()[0]

    s = '\n'
    if addinfo is not None:
        r = addinfo
        for cols_ in titlecols:
            for col in cols_:
                s += f'{col}='
                colv = r[col]
                if isinstance(colv,float):
                    s += f'{colv:.2f}'
                else:
                    s += f'{colv}'
                s+='; '
            s += '\n'
        #s = f'\nerror={r["error_endpoint_ang"]:.1f}; trialwb={r["trialwb"]}; tt={r["trial_type"]}'

    ax.set_title(f'ti={ti}; {vft}; time_lh={td:.3f}'
                 f'\n int2norm = {err:.2f}; area= {traja:.2f}' + s)
    ax.legend(loc='lower left')
    ax.set_xlim(xlim)


def plotTraj(ax, dfcurtr, home_position, target_coords_homec, params,
        calc_area = False, show_dist_guides = False, verbose=0,
             force_entire_traj=False, addinfo=None, titlecols=[]):
    from exper_protocol.utils import (get_target_angles,
        calc_target_positions, calc_err_eucl, coords2anglesRad, screen2homec,
                                     homec2screen)
    from scipy.interpolate import interp1d
    ll = len(dfcurtr)

    ti = dfcurtr['trial_index']._values[0]

    #dfcurtr#[['feedbackX', 'feedbackY', 'unpert_feedbackY']]

    fbXY = dfcurtr[['feedbackX', 'feedbackY']].to_numpy()
    #fbXY.shape

    if verbose:
        print(fbXY[:,0], fbXY[:,1] )
    fbXhc, fbYhc = screen2homec(fbXY[:,0], fbXY[:,1], home_position  )


    rh = float( params['radius_home'])
    ds = np.sqrt( fbXhc**2 + fbYhc**2 )
    leave_home_coef = 1.
    inds_leavehome = np.where(ds > rh * leave_home_coef )[0]
    txc,tyc = target_coords_homec
    tgti = dfcurtr['tgti_to_show']._values[0]
    ang = np.math.atan2(txc[tgti], tyc[tgti])

    tx_adj, ty_adj = txc[tgti] - fbXhc[0], tyc[tgti] - fbYhc[0]
    ang = np.math.atan2(tx_adj, ty_adj)

    fbXhc_ta, fbYhc_ta = rot(fbXhc, fbYhc, ang)

    pert=  addinfo['perturbation']
    #ofbXhc0, ofbYhc0 = rot(fbXhc, fbYhc, pert / 180 * np.pi  )

    ofbXY = dfcurtr[['unpert_feedbackX', 'unpert_feedbackY']].to_numpy()
    ofbXhc, ofbYhc = screen2homec(ofbXY[:,0], ofbXY[:,1], home_position  )

    #assert( np.linalg.norm(ofbXhc0- ofbXhc)  < 1e-10 )

    if len(inds_leavehome) and (not force_entire_traj):
        ind  = inds_leavehome[0]
    else:
        ind = 0

    time_lh = dfcurtr['time'].to_numpy()[ind]

    print(f'ti = {ti}; tgti = {tgti}; ang={ang}; ind = {ind}; inds_leavehome = {inds_leavehome}')

    lentrunc = ll - ind

    #from scipy.integrate import quad
    ptinds = np.arange(lentrunc)
    # for some reason order of tgts gets inverted. Or maybe only x coords?
    dirx = rh * np.sin(ang) + fbXhc[0]
    diry = rh * np.cos(ang) + fbYhc[0]
    rh2 = np.sqrt( dirx**2 + diry**2 )
    dirx *= rh / rh2
    diry *= rh / rh2

    fxint = interp1d([0, lentrunc-1 ], [dirx, txc[tgti]], 'linear' )
    tgtpathX_interp = fxint(ptinds)
    fyint = interp1d([0, lentrunc-1 ], [diry, tyc[tgti]], 'linear' )
    tgtpathY_interp = fyint(ptinds)

    ns = 6
    ptinds_s = np.arange(ns)
    fxint = interp1d([0, ns -1], [fbXhc[0]  , dirx], 'linear' )
    tgtpathX_interp_s = fxint(ptinds_s)
    fyint = interp1d([0, ns-1 ], [fbYhc[0], diry], 'linear' )
    tgtpathY_interp_s = fyint(ptinds_s)

    lns = []
    for i in range(0,lentrunc,7):
        ln = plt.Line2D( [tgtpathX_interp[i], fbXhc[ind+i] ],
                [tgtpathY_interp[i] , fbYhc[ind+i] ]  )
        #plt.Line2D( [tgtpathX_interp[i], fbXhc[ind+i] ],  [tgtpathY_interp[i] , fbYhc[ind+i] ]  )
        lns.append(ln)

    #plt.plot(ptinds, fxint(ptinds))

    ds2 = np.sqrt( (tgtpathX_interp - fbXhc[ind:] )**2 + (tgtpathY_interp - fbYhc[ind:] )**2 )
    err = np.sum(ds2) / len(ds2)


    if calc_area and len(inds_leavehome):
        traja = area(fbXhc_ta[ind:], fbYhc_ta[ind:], (0,rh),
                     (0,float(params['dist_tgt_from_home'])), verbose=0)
    else:
        traja = np.nan


    #dfcurtr['tgti_to_show']._values[0]


    print(f'ds[-1] = {ds[-1]}, error_distance= {dfcurtr["error_distance"].values[-1]}')
    #print(f'err={err}, errAug={errAug}')


    txx = [0] + [ txc[i] for i in range(len(txc)) if i != tgti ]
    tyy = [0] + [ tyc[i] for i in range(len(tyc)) if i != tgti ]
    ax.scatter(txx, tyy  ,
               s=133, label='targets', c='blue', alpha=0.2)
    ax.scatter( [txc[tgti]] , [tyc[tgti] ]  ,
               s=290, c='blue', alpha=0.5)

    if abs(ang) > 1e-10:
        ax.scatter(fbXhc , fbYhc , alpha=0.3, label='fb (homec)', marker='+', s = 60, c='orange')

    ax.scatter(ofbXhc , ofbYhc , alpha=0.4, label='ofb (homec)', marker='x', s = 30, c='magenta')
    #ax.scatter(ofbXhc0 , ofbYhc0 , alpha=0.4, label='ofb0 (homec)', marker='+', s = 30, c='magenta')

    ax.scatter(fbXhc_ta[:ind] , fbYhc_ta[:ind] , alpha=0.3, c='r')
    ax.scatter(fbXhc_ta[ind:] , fbYhc_ta[ind:] , alpha=0.4, c='r', label='fb tgtalign (homec)')

    ax.scatter(tgtpathX_interp_s , tgtpathY_interp_s , alpha=0.2, c='cyan', s = 15)
    ax.scatter(tgtpathX_interp , tgtpathY_interp , alpha=0.2, label='fb ideal', c='cyan')
    crc = plt.Circle((0, 0), rh, color='r', lw=2, fill=False,
                     alpha=0.6)
    ax.add_patch(crc)
    if show_dist_guides:
        for ln in lns:
            #plt.gca().add_patch(ln)
            ax.add_line(ln)

    vft = dfcurtr['vis_feedback_type'].to_numpy()[0]
    td = time_lh - dfcurtr["time"].to_numpy()[0]

    s = '\n'
    if addinfo is not None:
        r = addinfo
        for cols_ in titlecols:
            for col in cols_:
                s += f'{col}='
                colv = r[col]
                if isinstance(colv,float):
                    s += f'{colv:.2f}'
                else:
                    s += f'{colv}'
                s+='; '
            s += '\n'
        #s = f'\nerror={r["error_endpoint_ang"]:.1f}; trialwb={r["trialwb"]}; tt={r["trial_type"]}'

    ax.set_title(f'ti={ti}; {vft}; time_lh={td:.3f}'
                 f'\n int2norm = {err:.2f}; area= {traja:.2f}' + s)
    ax.legend(loc='lower left')
    ax.set_xlim(-140,140)



def multiplot(dfccos, dfcpcos, dfcos, ynames,
              start_ind = 12, ww = 20, hh=3, xlim=None,
              colnx = 'trials', show_titles = False):
    import seaborn as sns


    #pause_inds = dfcpcos['trial_index'].to_numpy()
    pause_inds = dfcpcos['trial_index'].unique()
    block_ends = dfcos.groupby(['block_ind'])['trial_index'].max()

    grp = dfccos.query('trial_type != "error_clamp"').groupby('block_ind')
    block_ends = grp['trial_index'].max()
    # error -- angular signed
    #'error_eucl',
    #for yname in ['error_aug2', 'error_intdist2', 'error_intdist2nn','error_intdist','error_distance',
    #             'error_distance_lh']:
    df_ = dfccos.query('trial_index >= @start_ind')
    #ynames = ['error','error_pert_adj']

    nr = len(ynames)
    nc = 1
    fig,axs =  plt.subplots(nr,nc,figsize=(nc * ww, nr*hh),
                            sharex='col')
    axs = axs.flatten()
    for axi, yname in enumerate(ynames):
    #for yname in ['error_intdist2nn']:
        #yname = 'error_intdist2'
        #dfcc['tmp']
        #plt.figure(figsize=(20,3))
        ax = axs[axi]
        #sns.lineplot(dfcc, x='trial_index', y ='perturbation', alpha=0.4, c='r')

        #mult = df_[yname].quantile(0.95) / 40
        dfgood = df_.loc[ ~np.isinf( df_[yname]  ), yname ]
        vals = np.abs(dfgood )

        q = 0.05
        qmi,qma = dfgood.quantile([q, 1-q] )

        #mult = np.quantile(vals, q=0.95)  / 40
        q = vals.quantile(0.95)
        mult = q / 40
        #mult = vals.max() / 40

        print(yname, ' mult =', mult)
        #if yname == 'error_intdist2nn':
        #    mult = dfcc[yname].max()
        #else:
        #    mult = 1.
        sns.lineplot(df_, x=colnx, y =yname, ax=ax)


        ax.plot(dfccos[colnx], dfccos['perturbation']._values * mult,
                alpha=0.4, c='r', label='pert')
        ax.vlines(pause_inds, -20 * mult, 20 * mult, color='green', ls = ':',
                label='pause')

        ax.vlines(block_ends, -20 * mult, 20 * mult, color='grey', ls = '--',
                label='block')

        ax.plot(dfccos[colnx], dfccos['tgti_to_show'] * 7 * mult,
                color='cyan', alpha=0.5,
            label='target', ls=':')
        ax.grid(False)
        ax.axhline(y=0, ls=':', c='red')

        #plt.savefig( pjoin(path_fig, f'{subj}_{yname}.pdf' ) )
        #plt.close()
        if axi == nr - 1:
            ax.legend(loc='lower right')

        if xlim is not None:
            ax.set_xlim(xlim)

        me = abs(  np.mean([qma, qmi]) )
        ax.set_ylim( min(-20*mult,qmi - me*0.2 ),
                    max(20*mult, qma + me*0.2 ) )

        if axi != len(ynames) - 1:
            ax.set_xlabel('')

        if show_titles:
            ax.set_title(yname)

def alignDfs(df1,cols1, df2,cols2, suff2 = '_2', verbose=0,
             check_agreement = True):
    print(len(df1), len(df2))
    assert set(cols1) < set(df1.columns)
    assert set(cols2) < set(df2.columns)
    #s = set( df2.index.names ) & set(cols2)
    names = list( df2.index.names )
    if (set(names ) & set( df2.columns ) ) > set([]):
        #print('fff')
        df2mod = df2[cols2 + names ].copy().\
            drop(columns=list( names )).reset_index().drop(columns= set(names) - set(cols2)  )
        print(df2mod.columns)
    else:
        df2mod = df2[cols2].copy().reset_index()
    rd = {}
    for coln in df2.columns:
        rd[coln] = coln + suff2
    df2mod = df2mod.rename(columns=rd)
    dfr = pd.concat([df1[cols1].reset_index(),df2mod],axis=1)
    #cols = [x for pair in zip( cols1,df2mod.columns ) for x in pair]
    cols_int = list( set(cols1) & set(cols2) )
    cols_int_mod = [rd[col] for col in cols_int]

    z = [['index']] + list(zip( cols_int,
                 cols_int_mod ) )
    #print(z)
    #print(list(map(list,z ) ) ) 
    cols = sum( list(map(list,z ) ), [] )
    cols += list( set(cols1) - set(cols2) ) 
    cols += [rd[col] for col in set(cols2) - set(cols1)  ]

    print(cols)
    dfr = dfr[cols ]
    
    if check_agreement:
        for colnc in (set(cols1) & set(cols2)):
            print(colnc)
            inds = dfr[dfr[colnc] != dfr[colnc + suff2]].index
            print(colnc, 'disagree at inds ',list(inds) )
    return dfr

def getStartEndMultiSubj(df_, ynames = ['err_sens']):
    grp = df_.groupby(['subject', 'block_ind'])

    # note that we take here highly cleaned data mostly. So there can be sometimes only a couple of valid trials
    # e.g. if particiapnt did not manage to leave the home during most of the block
    df1 = grp['trial_index'].min().to_frame().rename(columns={'trial_index':'block_start_trial_index'})
    df2 = grp['trial_index'].max().to_frame().rename(columns={'trial_index':'block_end_trial_index'})

    #sts = grp['trial_index'].idxmin().to_frame().rename(columns={'trial_index':'block_start_trial_index'}) 

    block_bounds = pd.concat( [df1, df2], axis=1)

    # some dirty hacks. I shoudl be using  idxmin
    sts = block_bounds.query('block_ind >= 0')['block_start_trial_index']._values
    df_block_start = df_.loc[df_['trial_index'].isin(sts)]

    ends = block_bounds.query('block_ind >= 0')['block_end_trial_index']._values
    df_block_end = df_.loc[df_['trial_index'].isin(ends)]

    df_block_start = df_block_start[['subject','block_ind', 'trial_type'] + ynames].set_index(['subject','block_ind'])
    #ynames_mod = [yn + '_start' for yn in ynames]
    #rd = dict(zip(ynames,ynames_mod))
    #df_block_start = df_block_start.rename(columns=rd)
    df_block_start['type'] = 'start'

    df_block_end =   df_block_end[['subject','block_ind', 'trial_type'] + ynames].\
        set_index(['subject','block_ind'])
    df_block_end['type'] = 'end'

    df_startend = pd.concat([df_block_start, df_block_end])
    return df_startend

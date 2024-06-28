# import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config2 import *
from collections.abc import Iterable

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
qs_notspec_not_afterpause = ('( trial_index > @numtrain and trial_type != "error_clamp"'
       ' and prev_trial_type != "pause" )')

qs_inside_sandwich = ('( trial_index > @numtrain and trial_type != "error_clamp" '
                     ' and special_block_type == "error_clamp_sandwich" )')

qs_notspec_not_sandwich_not_afterpause = ('( trial_index > @numtrain and trial_type != "error_clamp"'
        ' and special_block_type != "error_clamp_sandwich" and prev_trial_type != "pause" )')

spectrials = ["error_clamp", "pause", "break"]

qs_notspec = ('( trial_index > @numtrain '
            ' and ~trial_type.isin(@spectrials)'
        ' and special_block_type != "error_clamp_sandwich" )')

qs_notEC = ('( trial_index > @numtrain and trial_type != "error_clamp" )')

qs_notspec_not_afterspec = ('( trial_index > @numtrain and trial_type != "error_clamp"'
                   ' and prev_trial_type not in @spectrials )')

qs_easy0 = '(vis_feedback_type == "veridical" and tgti_to_show == 1)'
qs_easy1 = '(tgti_to_show == 2 and vis_feedback_type == "rot20")'
qs_easy2 = '(tgti_to_show == 0 and vis_feedback_type == "rot-20")'
qs_easy_wide = '(' + qs_easy0 + ' or ' + qs_easy1 + ' or ' + qs_easy2 + ')'


qs_not_easy0 = '( not (vis_feedback_type == "veridical" and tgti_to_show == 1) )'
qs_not_easy_wide =  '( not ' + qs_easy_wide + ' )'

# to estimate motor variability -- veridical not easy
# trialwb limit is rather arbitrary here. Just wanted not beginning and not end
qs_motor_var = ' ( ' + qs_not_easy_wide + ' and ' + qs_notspec +\
               (' and vis_feedback_type == "veridical"'
              ' and trialwb > 4'
              ' and ~time_lh.isna() and trial_index >= @numtrain )' )

# when to discard err_sens
thr_lh_ang_deg = 2
thr_lh_ang_rad = thr_lh_ang_deg * ( np.pi / 180 )

thr_signed_area2_nn_scaled_ed = 0.028 # this was just 5% quantile of abs

canonical_context_pair_listnames = ['both_close','some_close','tgt_close',
    'pert_close','pert_same','tgt_same']



coln_ctx_closeness = ['prev_ctx_both_close',
       'prev_ctx_some_close', 'prev_ctx_tgt_close', 'prev_ctx_pert_close',
       'prev_ctx_pert_same', 'prev_ctx_tgt_same']

coln2pubname = {'error_lh2_ang':'Starting angle',
                'error_lh2_ang_valid':'Starting angle',
                'error_area2_signed_nn':'Signed area' ,
                'error_area2_signed_nn_valid':'Signed area' }

 #vfts = dfcc['vis_feedback_type'].unique()

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

def getAllValidDsnames( d ):
    from glob import glob
    from pathlib import Path
    exclude = ['SE1-009_', 'SE1-010_']
    exclude += ['2023-SE1-020_context_change_20230502_141402'] # just was stopped after training
    fnbs = []
    for fnf in sorted( glob(d + '2023-SE1-0*[0-9].log') ):
        name = Path(fnf).name
        skip = 0
        for excl in exclude:
            if name.find(excl) >= 0:
                skip = 1
                print('Skip name',name,excl)
        if skip:
            continue
        fnb = name.replace('.log','')
        #print( f'"{fnb}",' )
        fnbs += [fnb]

    fnbs += ['2023-SE1-pilot3_context_change_20230317_105210']

    # check no repeats
    import re
    snums=list(sorted( re.findall(r'SE1-0(\d+)_context', '_'.join(fnbs)) ) )
    assert len(snums) == len(set( snums ) )

    return fnbs

def addTrueMvtTime(dfcc_all, dfc_all, home_position,radius_home,
                   thr1 = 1e-3, thr2 = 1e-3, inplace=True,
                  calc_std = 0  ):
    '''
    works inplace
    '''
    print('Starting addTrueMvtTime')
    if not inplace:
        dfcc_all = dfcc_all.copy()
        #dfc_all = dfc_all.copy()
    # leave home like

    if calc_std:
        dfc_all['jax1_std'] = dfc_all.groupby(['subject','trial_index'])['jax1'].\
            expanding().std().reset_index(drop=True)
        dfc_all['jax2_std'] = dfc_all.groupby(['subject','trial_index'])['jax2'].\
            expanding().std().reset_index(drop=True)

    import time
    t0 = time.time()
    grp = dfc_all.groupby(['subject','trial_index'])


    # jax2 vertical, jax1 horizontal

    def f(row):
        tind = row['trial_index']
        subj = row['subject']
        #qs = 'subject == @subj and trial_index == @tind'
        #dftr = dfc_all.query(qs)

        # this is faster than making many individ queries
        dftr = dfc_all.loc[ grp.groups[(subj,tind)] ]

        jax12 = dftr[['jax1','jax2']].values
        jax12_0 = jax12[0]

        jax12_dev =  np.abs( jax12 - jax12_0[None, :]  )
        #print(jax12_dev[::10,:] )

        if calc_std:
            jax1_std,jax2_std = dftr[['jax1_std','jax2_std']].values.T
            tmin1 = np.where(jax1_std > thr1)[0]
            if len( tmin1 ):
                tmin1 = tmin1[0]
                tmin1 = dftr.iloc[tmin1]['time_since_trial_start']
            else:
                tmin1 = np.nan

            tmin2 = np.where(jax2_std > thr2)[0]
            if len( tmin2 ):
                tmin2 = tmin2[0]
                tmin2 = dftr.iloc[tmin2]['time_since_trial_start']
            else:
                tmin2 = np.nan
        else:
            tmin1 = np.where(jax12_dev[:,0] > thr1)[0]
            if len( tmin1 ):
                tmin1 = tmin1[0]
                tmin1 = dftr.iloc[tmin1]['time_since_trial_start']
            else:
                tmin1 = np.nan

            tmin2 = np.where(jax12_dev[:,1] > thr2)[0]
            #print(tmin2)
            if len( tmin2 ):
                tmin2 = tmin2[0]
                tmin2 = dftr.iloc[tmin2]['time_since_trial_start']
            else:
                tmin2 = np.nan

        # jax1 and jax2 are always valid but feedback values are only since 1st frame (i.e. 0th is bad)
        dftr_valid = dftr.iloc[1:]
        fbXY = dftr_valid[['feedbackX','feedbackY']].values
        indlh = calcIndLh(fbXY, home_position, radius_home )
        if indlh is not None:
            time_lh = dftr.iloc[indlh + 1]['time_since_trial_start']
        else:
            time_lh = None


        indlh = calcIndLh(fbXY, home_position, radius_home, leave_home_coef=3. )
        if indlh is not None:
            time_l3h = dftr.iloc[indlh + 1]['time_since_trial_start']
        else:
            time_l3h = None

        #print(time_lh)
        # slower
        #tmin2 = dftr.query('jax2_std > @thr')['time_since_trial_start'].min()
        return tmin1, tmin2, time_lh, time_l3h

    #####  caring about possible joystick bugs (when it gets mechanically locked in some direction) #############
    # tm == true movement (assuming before they just
    # follow easy joystick traj is not true move)
    # this time counts since trial start
    # rather slow, takes 15 sec
    dfcc_all[['time_tmstart_jax1', 'time_tmstart_jax2', 'time_lh','time_l3h' ] ] = dfcc_all.apply(f,1, result_type='expand')
    t1 = time.time()
    print('addTrueMvtTime main part took ', t1-t0)

    # time true mvt start 1 minus 2
    dfcc_all['ttms1m2'] = dfcc_all['time_tmstart_jax1'] - dfcc_all['time_tmstart_jax2']

    dfcc_all['time_tmstart'] = dfcc_all[['time_tmstart_jax1', 'time_tmstart_jax2'] ].max(axis=1)

    dfcc_all['time_tmstart2'] = dfcc_all['time_lh']
    # for easy I may never leave actually
    easy = dfcc_all['is_easy_wide']
    #dfcc_all.loc[~easy, 'time_tmstart2'] = dfcc_all.loc[~easy, ['time_lh', 'time_tmstart'] ].max(axis=1)


    #r = dfcc_all.loc[easy, ['time_tmstart_jax1', 'time_tmstart_jax2'] ].min(axis=1)
    #dfcc_all.loc[easy, ['time_tmstart_jax1', 'time_tmstart_jax2'] ].min(axis=1)

    ea = dfcc_all.loc[easy]
    def f(row):
        ts = np.array( [ row['time_tmstart_jax1'], row['time_tmstart_jax2' ] ] )
        if ~np.all(np.isnan(ts) ):
            mi = np.nanmin(ts )
            if mi > row['time_l3h']:
                r = row['time_lh']
            else:
                r = np.nanmax( [mi, row['time_lh'] ] )
        else:
            r = row['time_lh']
        return r
    #dfcc_all.loc[easy, 'time_tmstart2'] = ea.apply(f,1)
    dfcc_all['time_tmstart2'] = dfcc_all.apply(f,1)


    return dfcc_all, dfc_all


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
        assert ( dfcc_tmp['trial_index'].diff().iloc[1:] > 0) .all()

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

def correctPertCol_NIH(df_all, use_sub_angles = 1):
    c = df_all['environment'] == 0
    df_all['perturbation_'] = -100000.
    df_all.loc[c, 'perturbation_'] = df_all.loc[c,'perturbation']#.copy()
    from base2 import subAngles
    if use_sub_angles:
        df_all.loc[~c, 'perturbation_'] = subAngles(df_all.loc[~c,'feedback'], 
                df_all.loc[~c,'org_feedback'])  / np.pi * 180
    else:
        df_all.loc[~c, 'perturbation_'] = (df_all.loc[~c,'feedback'] - df_all.loc[~c,'org_feedback']) / np.pi * 180
    df_all['perturbation'] = df_all['perturbation_']
    df_all = df_all.drop(columns=['perturbation_'])

def addBehavCols(df_all, inplace=True, skip_existing = False,
                 dset = 'Romain_Exp2_Cohen', fn_events_full = None, trial_col0 = 'trials'):
    '''
    This is for NIH experiment (and for Bonaiuto data as well)
    inplace, does not change database lengths (num of rows)
    '''
    assert df_all.index.is_unique
    if not inplace:
        df_all = df_all.copy()
    subjects     = df_all['subject'].unique()
    tgt_inds_all = df_all['target_inds'].unique()
    pertvals     = df_all['perturbation'].unique()
    if len(pertvals) > 10:
        print(f'WARNING: too many pertvals! len={len(pertvals)}')
        if dset == 'Romain_Exp2_Cohen':
            # maybe we extended perturbations to nonzero vals in random. Then we only take stable
            pertvals_eff = df_all.query('environment == 0')['perturbation'].unique() 
    else:
        pertvals_eff = pertvals

    subj = subjects[0]

    # by default perturbation in NIH data is == 0 for random, which is confusing
    correctPertCol_NIH(df_all)


    if fn_events_full is not None:
        from meg_proc import addTrigPresentCol_NIH
        #fn_events = f'{task}_{hpass}_{ICAstr}_eve.txt'
        #fn_events_full = op.join(path_data, subject, fn_events )
        # these are all events, including non-target triggers
        events0 = mne.read_events(fn_events_full)
        event_ids_all_for_EC = stage2evn2event_ids['target']['all']
        mask   = np.isin(events[:,2], event_ids_all_for_EC)
        events = events0[mask]
        df_all = addTrigPresentCol_NIH(df_all, events)

    df_all['dist_trial_from_prevtgt'] = np.nan
    for subj in subjects:
        for tgti in tgt_inds_all:
            if tgti is None:
                continue
            dfc = df_all[(df_all['subject'] == subj) & (df_all['target_inds'] == tgti)]
            df_all.loc[dfc.index,'dist_trial_from_prevtgt'] =\
                df_all.loc[dfc.index, trial_col0].diff()

    #dist_deg_from_prevtgt
    #dist_trial_from_prevtgt
    # better use strings otherwise its difficult to group later
    lbd = lambda x : f'{x:.2f}'
    df_all['dist_rad_from_prevtgt'] = None
    for subj in subjects:
        dfc = df_all[df_all['subject'] == subj]
        df_all.loc[dfc.index,'dist_rad_from_prevtgt'] =\
            df_all.loc[dfc.index, 'target_locs'].diff().abs()
    df_all['dist_rad_from_prevtgt'] = df_all['dist_rad_from_prevtgt'].apply(lbd)

    # signed distance
    df_all['distsgn_rad_from_prevtgt'] = None
    for subj in subjects:
        dfc = df_all[df_all['subject'] == subj]
        df_all.loc[dfc.index,'distsgn_rad_from_prevtgt'] =\
            df_all.loc[dfc.index, 'target_locs'].diff()#.apply(lbd,1)
    df_all['distsgn_rad_from_prevtgt'] = df_all['distsgn_rad_from_prevtgt'].apply(lbd)


    dts = np.arange(1,6)
    for subj in subjects:
        for dt in dts:
            dfc = df_all[df_all['subject'] == subj]
            df_all.loc[dfc.index,f'dist_rad_from_tgt-{dt}'] =\
                df_all.loc[dfc.index, 'target_locs'].diff(periods=dt).abs()
    for dt in dts:
        df_all[f'dist_rad_from_tgt-{dt}'] = df_all[f'dist_rad_from_tgt-{dt}'].apply(lbd)


    if dset == 'Romain_Exp2_Cohen':
        df_all['subject_ind'] = df_all['subject'].str[3:5].astype(int)

        test_triali = pert_seq_code_test_trial
        subj2pert_seq_code = {}
        for subj in subjects:
            mask = df_all['subject'] == subj
            dfc = df_all[mask]
            r = dfc.loc[dfc[trial_col0] == test_triali,'perturbation']
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
                triali = row[trial_col0]
                if env == 'stable' and triali < 200:
                    block_name = env + '1'
                elif env == 'stable' and triali > 300:
                    block_name = env + '2'
                elif env == 'random' and triali < 450:
                    block_name = env + '1'
                elif env == 'random' and triali > 500:
                    block_name = env + '2'
                else:
                    print(row)
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

    assert np.min( np.diff( dfc[trial_col0] ) ) > 0

    trials_starts = {}
    for bn in block_names:
        fvi = dfc[dfc['block_name'] == bn].first_valid_index()
        assert fvi is not None
        trials_starts[bn] = dfc.loc[fvi,trial_col0]
    assert np.max( list(trials_starts.values() ) ) <= 767

    def f(row):
        bn = row['block_name']
        start = trials_starts[bn]
        return row[trial_col0] - start

    df_all['trialwb'] = -1
    for subj in subjects:
        mask = df_all['subject'] == subj
        df_all.loc[mask, 'trialwb'] = df_all[mask].apply(f,1)
    assert np.all( df_all['trialwb'] >= 0)

    ########################   index within env (second block -- diff numbers)

    # within single subject
    if dset == 'Romain_Exp2_Cohen':
        envchanges  = dfc.loc[dfc['environment'].diff() != 0,trial_col0].values
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
            #return row[trial_col0] - start_cur + add
            r = row[trial_col0] - start_cur + add
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
            trial_st = df_starts[trial_col0].values

            last =  dfc_oneb.loc[dfc_oneb.last_valid_index(), trial_col0]
            trial_st = list(trial_st) +  [last + 1]

            bn2trial_st[bn] = trial_st
            assert len(trial_st) == 6, len(trial_st) # - 1

        #bn2trial_st = df_all.groupby('block_name')[trial_col0].min().to_dict()
        print(bn2trial_st)

        def f(row):
            t = row[trial_col0]
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
            return row[trial_col0] - start

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
            r =  row[trial_col0] - start

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
            return row[trial_col0] - start + add

        df_all['trialwpert_we'] = df_all.apply(f,1)

        ############################# index within target (assuming sorted over trials)

    df_all['trialwtgt'] = -1
    for subj in subjects:
        for tgti in tgt_inds_all:
            mask = (df_all['target_inds'] == tgti) & (df_all['subject'] == subj)
            trials = df_all.loc[mask, trial_col0]
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
        if len(pertvals) > 10:
            mask_pert0 = mask0 & (df_all['environment'] == 0)
        else:
            mask_pert0 = mask0
                  
        for pertv in pertvals_eff:
            mask_pert = mask_pert0 & (df_all['perturbation'] == pertv)
            df_all.loc[mask_pert, 'trialwpert'] = np.arange(sum(mask_pert ) )

        for tgti in tgt_inds_all:
            #for pertv in df_all['perturbation'].unique()
            mask = (df_all['target_inds'] == tgti) & mask0
            for pertv in pertvals_eff:
                mask_pert = mask_pert0 & mask & (df_all['perturbation'] == pertv)
                df_all.loc[mask_pert, 'trialwtgt_wpert'] = np.arange(sum(mask_pert) )

                if dset == 'Romain_Exp2_Cohen':
                    for bn in block_names:
                        mask_bn = mask_pert & (df_all['block_name'] == bn)
                        trials = df_all.loc[mask_bn, trial_col0]
                        df_all.loc[mask_bn, 'trialwtgt_wpert_wb'] = np.arange(len(trials) )
                    for envc in envcode2env:
                        mask_env = mask_pert & (df_all['environment'] == envc)
                        trials = df_all.loc[mask_env, trial_col0]
                        df_all.loc[mask_env, 'trialwtgt_wpert_we'] = np.arange(len(trials) )

            if dset == 'Romain_Exp2_Cohen':
                for pert_stage in range(5):
                    for bn in block_names:
                        mask_ps = mask & (df_all['pert_stage_wb'] == float(pert_stage) ) &\
                                ( df_all['block_name'] == bn )
                        trials = df_all.loc[mask_ps, trial_col0]
                        df_all.loc[mask_ps, 'trialwtgt_wpertstage_wb'] = np.arange( len(trials) )
                    for envc in envcode2env:
                        mask_ps = mask & (df_all['pert_stage_wb'] == float(pert_stage) ) &\
                                (df_all['environment'] == envc)
                        trials = df_all.loc[mask_ps, trial_col0]
                        df_all.loc[mask_ps, 'trialwtgt_wpertstage_we'] = np.arange( len(trials) )



            if dset == 'Romain_Exp2_Cohen':
                for bn in block_names:
                    mask_bn = mask & (df_all['block_name'] == bn)
                    trials = df_all.loc[mask_bn, trial_col0]
                    df_all.loc[mask_bn, 'trialwtgt_wb'] = np.arange(len(trials) )
                for envc in envcode2env:
                    mask_env = mask & (df_all['environment'] == envc)
                    trials = df_all.loc[mask_env, trial_col0]
                    df_all.loc[mask_env, 'trialwtgt_we'] = np.arange(len(trials) )
    #df_all['trialwtgt_wpert_wb'] = df_all['trialwtgt_wpert_wb'].astype(int)

    # trial_group_cols_all = [s for s in df_all.columns if s.find('trial') >= 0]
    tmax = df_all[trial_col0].max()
    for tcn in trial_group_cols_all:
        if dset == 'Romain_Exp2_Cohen':
            assert df_all[tcn].max() <= tmax, tcn
            if ('wpert' in tcn) and (len(pertvals) > 10):
                mx = df_all.query('environment == 0')[tcn].max()
                assert  mx >= 0,    (tcn, mx)
            else:
                assert df_all[tcn].max() >= 0,    tcn
        else:
            if tcn not in df_all:
                continue
            if df_all[tcn].max() <= tmax or df_all[tcn].max() >= 0:
                print(f'problem with {tcn}')

    if dset == 'Romain_Exp2_Cohen':
        #pscAdj_NIH(df_all, ['error',  ] ) 
        #pscAdj_NIH(df_all, [ 'org_feedback', 'feedback' ], subpi = np.pi ) 
        #def f(x):    
        #    if x > np.pi / 2.:
        #        x -= 2*np.pi
        #    elif x < -np.pi / 2.:
        #        x += 2*np.pi
        #    return x
        #df_all['error'] = df_all['error'].apply(f)
        adjustErrBoundsPi(df_all, ['error'])

        print('ddd')
        badcols =  checkErrBounds(df_all)
        if len(badcols):
            print('bad cols 1 ', badcols)

        df_all['error_deg'] = (df_all['error'] / np.pi) * 180 


        df_all['vals_for_corr'] = df_all['target_locs'] - df_all['org_feedback'] # movement 

        vars_to_pscadj = [ 'error', 'perturbation', 'vals_for_corr']
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


    badcols =  checkErrBounds(df_all)
    if len(badcols):
        print('bad cols ', badcols)

    return df_all

def addBehavCols2(df):
    '''this is not a replacement of addBehavCols, it just adds more stuff

        args: 
            df -- one row one trial
    '''
    assert df.groupby(['subject'])['trial_index'].diff().max() == 1
    assert not df.duplicated(['subject','trials']).any()
    #del df_all_multi_tsz
    #df['env'] = df['environment'].apply(lambda x: envcode2env[x])
    assert 'env' in df
    df['feedback_deg'] = df['feedback'] / np.pi * 180
    df['error_deg'] = df['error'] / np.pi * 180 


    checkErrBounds(df,['error','prev_error','error_deg'])


    df['env'] = df['env'].astype('category')
    df['subject'] = df['subject'].astype('category')
    df['trial_index'] = df['trials']

    df['error_abs'] = df['error'].abs()
    df['prev_error_abs'] = df['prev_error'].abs()

    df['prev_error_pscadj'] = df.groupby(['subject','block_name'],
            observed=True)['error_pscadj'].shift(1, fill_value=0)
    df['prev_error_pscadj_abs'] = df['prev_error_pscadj'].abs()


    # remove NaN for random
    df['pert_stage_wb'] = df['pert_stage_wb'].where(
        df['env'] =="stable", -1 )
    df['pert_stage_wb'] = df['pert_stage_wb'].astype(int)

    df['pert_stage'] = df['pert_stage'].where(
        df['env'] =="stable", -1 )
    df['pert_stage'] = df['pert_stage'].astype(int)

    print( df['pert_stage'].unique(),  df['pert_stage_wb'].unique() )

    df['ps_'] = 'rnd'
    c = df['pert_stage_wb'].isin([1,3])
    df.loc[c,'ps_'] = 'pert' 
    c = df['pert_stage_wb'].isin([2,4])
    df.loc[c,'ps_'] = 'washout' 
    c = df['pert_stage_wb'].isin([0])
    df.loc[c,'ps_'] = 'pre' 
    print( df['ps_'].unique() )

    # solve problem with first trial in pert having high ES because it was not preceived yet
    # so I shift everything
    df['ps2_'] = None
    #c = df['trialwpertstage_wb'] == 0
    #df.loc[c,'ps2_']  = df['ps_'].shift(1)

    df['ps2_']  = df.groupby(['subject','block_name'], observed=True).shift(1)['ps_']
    df.loc[ (df['env'] == 'stable') & (df['trialwb'] == 0), 'ps2_' ]  = 'pre'   # otherwise it is None
    df.loc[ (df['env'] == 'random') & (df['trialwb'] == 0), 'ps2_' ]  = 'rnd'   # otherwise it is None
    #print( df['ps2_'].unique() )
    #print( df.loc[df['ps2_'].isnull(),['trials','ps2_']])
    nu = df['ps2_'].isnull()
    #display(df[nu])
    assert not nu.any(), np.sum(nu)
    dfneq =  (df['ps_'] != df['ps2_'])
    #print( df.loc[dfneq, ['subject','trials','trialwb','ps2_','ps_','pert_stage']].iloc[:20])
    _mx  = df.loc[dfneq].groupby(['subject','pert_stage'], observed=True).size().max() 
    assert _mx == 1, _mx

    #print( df['ps2_'].unique() )
    #dfneq =  (df['ps_'] != df['ps2_'])
    #assert df.loc[dfneq].groupby(['subject','pert_stage'], observed=True).size().max() == 1

    # add contra
    df['ps3_'] = 'rnd'
    c = df['pert_stage_wb'].isin([1])
    df.loc[c,'ps3_'] = 'pert_pro' 
    c = df['pert_stage_wb'].isin([3])
    df.loc[c,'ps3_'] = 'pert_contra' 
    c = df['pert_stage_wb'].isin([2,4])
    df.loc[c,'ps3_'] = 'washout' 
    c = df['pert_stage_wb'].isin([0])
    df.loc[c,'ps3_'] = 'pre' 
    print( df['ps3_'].unique() )

    # separate randoms
    df['ps4_'] = 'rnd'
    c = df['pert_stage_wb'].isin([1,3])
    df.loc[c,'ps4_'] = 'pert' 
    c = df['pert_stage_wb'].isin([2,4])
    df.loc[c,'ps4_'] = 'washout' 
    c = df['block_name'] == 'random1'
    df.loc[c,'ps4_'] = 'rnd1' 
    c = df['block_name'] == 'random2'
    df.loc[c,'ps4_'] = 'rnd2' 
    print( df['ps4_'].unique() )

    df['trialwpertstage_wb'] = df['trialwpertstage_wb'].where(df['env'] =="stable", 
                                        df['trialwb'])
    df['trialwpertstage_wb'] = df['trialwpertstage_wb'].astype(int)

    assert not df.duplicated(['subject','trials']).any()

    df['thr'] = "mestd*0" # for compat
    df = addErrorThr(df)

    ###################################

    #dfc = df_wthr # NOT COPY here, we really want to add it to 
    dfc = df # NOT COPY here, we really want to add it to 
    # df_wthr (to filter TAN by consistency later)
    dfc['err_sens_change'] = dfc['err_sens'] - dfc['prev_err_sens']

    # without throwing away small errors
    dfc['subj'] = dfc['subject'].str[3:5]

    dfc['prevprev_error'] = dfc.groupby(['subject'],
                    observed=True)['prev_error'].shift(1, fill_value=0)
    dfc['err_sign_same'] =  np.sign( dfc['prev_error'] ) *\
        np.sign( dfc['prevprev_error'] )  

    dfc['err_sens_change'] = dfc['err_sens'] - dfc['prev_err_sens']

    dfc['dist_rad_from_prevprevtgt'] = \
        dfc.groupby('subject', observed=True)['target_locs'].diff(2).abs()

    # if one of the errors is small
    m = (dfc['prevprev_error'].abs() * 180 / np.pi < dfc['error_deg_initstd'] ) | \
        (dfc['prev_error'].abs() * 180 / np.pi <  dfc['error_deg_initstd']   ) 
    # if both of the errors are small
    m2 = (dfc['prevprev_error'].abs() * 180 / np.pi < dfc['error_deg_initstd'] ) & \
        (dfc['prev_error'].abs() * 180 / np.pi <  dfc['error_deg_initstd']   ) 

    # set sign to one when unclear
    dfc['err_sign_same2'] = dfc['err_sign_same']
    dfc['err_sign_same2'] = dfc['err_sign_same2'].where( m, 1)#.astype(int)

    # set sign to zero when unclear
    dfc['err_sign_same3'] = dfc['err_sign_same']
    dfc['err_sign_same3'] = dfc['err_sign_same3'].where( m, 0)#.astype(int)

    # set sign to minus one when unclear
    dfc['err_sign_same4'] = dfc['err_sign_same']
    dfc['err_sign_same4'] = dfc['err_sign_same4'].where( m, -1)#.astype(int)

    # set sign to zero when unclear, more strict
    dfc['err_sign_same5'] = dfc['err_sign_same']
    dfc['err_sign_same5'] = dfc['err_sign_same5'].where( m2, 0)#.astype(int)

    # set NaNs to 0
    mnan = ~(dfc['prevprev_error'].isna() | dfc['prev_error'].isna())
    for coln_suffi in range(1,6):
        if coln_suffi == 1:
            s = ''
        else:
            s = str(coln_suffi)
        coln_cur = 'err_sign_same' + s
        dfc[coln_cur] = dfc[coln_cur].where( mnan, 0)
        dfc[coln_cur] = dfc[coln_cur].astype(int)

    dfc['dist_rad_from_prevprevtgt'] = dfc['dist_rad_from_prevprevtgt'].\
        apply(lambda x: f'{x:.2f}' )
    dfc['dist_rad_from_prevprevtgt'] 
    #assert not dfc['err_sign_same'].isna().any()
    #dfc['err_sign_same'] = dfc['err_sign_same'].astype(int)

    dfc['err_sign_pattern'] = np.sign( dfc['prevprev_error'] ).apply(str) + \
        np.sign( dfc['prev_error'] ).apply(str)    
    print('N bads =',sum(dfc['err_sign_pattern'].str.contains('nan')))
    dfc.loc[dfc['err_sign_pattern'].str.contains('nan'),'err_sign_pattern'] = ''
    dfc['err_sign_pattern'] = dfc['err_sign_pattern'].astype(str)
    dfc['err_sign_pattern'] = dfc['err_sign_pattern'].str.replace('1.0','1')

    # for stats
    dfc['error_pscadj_abs'] = dfc['error_pscadj'].abs()
    dfc['trialwpertstage_wb'] = dfc['trialwpertstage_wb'].\
        where(dfc['environment'] == 0, dfc['trialwb'])
    dfc['trialwpertstage_wb'] = dfc['trialwpertstage_wb'].astype(int)

    dfc['error_change'] = dfc['error'] - dfc['error'].shift(1)
    dfc['error_pscadj_change'] = dfc['error_pscadj'] - dfc['error_pscadj'].shift(1)
    def f(x):    
        if x > np.pi:
            x -= 2*np.pi
        elif x < -np.pi:
            x += 2*np.pi
        return x
    dfc['error_change'] = dfc['error_change'].apply(f)
    dfc['error_pscadj_change'] = dfc['error_pscadj_change'].apply(f)
    dfc.loc[dfc['trialwb'] == 0, 'err_sens'] = np.nan

    ##########################

    dfni = df[~np.isinf(df['err_sens'])]
    ES_thr = calcESthr(dfni, 5.)
    dfall = truncateNIHDfFromES(df, mult=5., ES_thr=ES_thr)


    ###################################
    # just get perturabtion and env scheme from one subject
    # will be needed for plotting time resovled
    #dfall = dfall.reset_index()
    dfc_p = df.query(f'subject == "{subjects[0]}"')
    dfc_p = dfc_p.sort_values('trials')
    pert = dfc_p['perturbation'].values[:192*4]
    tr   = dfc_p['trials'].values[:192*4]
    envv = dfc_p['environment'].values[:192*4].astype(float)
    envv[envv == 0] = np.nan
    pert[envv == 1] = np.nan

    ##############################

    return df,dfall,ES_thr,envv,pert

def addWindowStatCols(dfc, ES_thr, varn0s = ['error_pscadj', 'error_pscadj_abs'],
                     histlens_min = 3, histlens_max = 40,
                      mav_d__make_abs = False, min_periods = 2, cleanTan = True   ):
    from pandas.errors import PerformanceWarning
    import warnings
    print( 'dfc.trial_group_col_calc.nunique() = ', dfc.trial_group_col_calc.nunique() )

    dfcs = dfc.sort_values(
        ['pert_seq_code', 'subject', 'trial_group_col_calc','trials']).copy()

    assert dfcs.trials.diff().max() == 1
    # good to add block name because we make a pause between so supposedly we loose memory about last errors
    grp = dfcs.\
        groupby(['pert_seq_code', 'subject', 'trial_group_col_calc','block_name'],
               observed=True)

    #varn0s = ['err_sens','error', 'org_feedback']
    #varn0s = ['err_sens','error_pscadj', 'error_change','error_pscadj_abs'] #, 'org_feedback_pscadj']
    #varn0s = ['error_pscadj', 'error_pscadj_abs'] #, 'org_feedback_pscadj']

    
    histlens = np.arange(histlens_min, histlens_max)
    ddof = 1 # pandas uses 1 by def for std calc
    for std_mavsz_ in histlens:
        for varn in varn0s:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',category=PerformanceWarning)

                for g,gi in grp.groups.items():
                    dfcs.loc[gi,f'{varn}_std{std_mavsz_}'] = dfcs.loc[gi,varn].shift(1).\
                        rolling(std_mavsz_, min_periods = min_periods).std(ddof=ddof)   
                    dfcs.loc[gi,f'{varn}_mav{std_mavsz_}'] = dfcs.loc[gi,varn].shift(1).\
                        rolling(std_mavsz_, min_periods = min_periods).mean()   

                dfcs[f'{varn}_invstd{std_mavsz_}'] = 1/dfcs[f'{varn}_std{std_mavsz_}']
                dfcs[f'{varn}_var{std_mavsz_}']    = dfcs[f'{varn}_std{std_mavsz_}'] ** 2
                dfcs[f'{varn}_mavsq{std_mavsz_}']  = dfcs[f'{varn}_mav{std_mavsz_}'] ** 2
                # shoud I change? so far I took abs of mav for mav d std and mav d var
                if mav_d__make_abs:
                    dfcs[f'{varn}_mav_d_std{std_mavsz_}']  = dfcs[f'{varn}_mav{std_mavsz_}'].abs() / dfcs[f'{varn}_std{std_mavsz_}']
                    dfcs[f'{varn}_mav_d_var{std_mavsz_}']  = dfcs[f'{varn}_mav{std_mavsz_}'].abs() / dfcs[f'{varn}_var{std_mavsz_}']
                else:
                    dfcs[f'{varn}_mav_d_std{std_mavsz_}']  = dfcs[f'{varn}_mav{std_mavsz_}'] / dfcs[f'{varn}_std{std_mavsz_}']
                    dfcs[f'{varn}_mav_d_var{std_mavsz_}']  = dfcs[f'{varn}_mav{std_mavsz_}'] / dfcs[f'{varn}_var{std_mavsz_}']
                dfcs[f'{varn}_Tan{std_mavsz_}']    = dfcs[f'{varn}_mavsq{std_mavsz_}'] / dfcs[f'{varn}_var{std_mavsz_}']
                dfcs[f'{varn}_invmavsq{std_mavsz_}'] = 1 / dfcs[f'{varn}_mavsq{std_mavsz_}']
                dfcs[f'{varn}_invmav{std_mavsz_}']   = 1 / dfcs[f'{varn}_mav{std_mavsz_}']
                dfcs[f'{varn}_std_d_mav{std_mavsz_}']   = dfcs[f'{varn}_std{std_mavsz_}'] / dfcs[f'{varn}_mav{std_mavsz_}']
                dfcs[f'{varn}_invTan{std_mavsz_}']   = dfcs[f'{varn}_std{std_mavsz_}']**2 / dfcs[f'{varn}_mav{std_mavsz_}']**2

    if cleanTan:
        print('Cleaning Tan')
        for std_mavsz_ in histlens:
            for varn in varn0s:
                c = dfcs['trialwb'] < std_mavsz_
                dfcs.loc[c,f'{varn}_Tan{std_mavsz_}'] = np.nan

    dfcs_fixhistlen_untrunc = dfcs.copy()
                
    # remove too big ES
    dfcs1 = dfcs.query('err_sens.abs() <= @ES_thr')
    dfcs_fixhistlen  = truncateDf(dfcs1, 'err_sens', q=0.0, infnan_handling='discard',  cols_uniqify = ['subject'],
                                  verbose=True) #,'env'
    dfcs_fixhistlen['environment'] = dfcs_fixhistlen['environment'].astype(int)
    #dfcs_fixhistlen_untrunc = dfcs_fixhistlen.copy()
    import gc; gc.collect()
    print('addWindowStatCols: Finished')
            
    return dfcs, dfcs_fixhistlen, dfcs_fixhistlen_untrunc, histlens
    #dfall_notclean_ = pd.concat(dfs)
    # ttrs = pd.concat(ttrs)
    # ttrs = ttrs.rename(columns={'p-val':'pval'})

def getQueryPct(df,qs,verbose=True):
    szprop = df.query(qs).groupby(['subject'],observed=True).size() / df.groupby(['subject'],observed=True).size()
    szprop *= 100
    me,std = szprop.mean(), szprop.std()
    if verbose:
        print(f'{qs} prpopration mean = {me:.3f} %, std = {std:.3f} %')
    return me,std

def truncLargeStats(dfcs_fixhistlen_untrunc, histlens, std_mult, 
    varns0 = [ 'error_pscadj_abs', 'error_pscadj'], suffixes = None):
    '''
    it NaNifies outliers but does not remove them (because if I were to remove rows for all, the dataset will become super small)
    '''
    # remove too large entries
    maxhl = np.max(histlens) ; print(maxhl)

    # NaN-ify too big stat values
    std_mult = 5.
    
    if suffixes is None:
        suffixes = 'mav,std,invstd,mavsq,mav_d_std,mav_d_var,Tan,invmavsq,invmav,std_d_mav,invTan'.split(',')

    varnames_all = []
    for varn0 in varns0: #'error_change']:
        for std_mavsz_ in histlens:#[1::10]:
            #varnames_toshow0_ = []
            for suffix in suffixes:
                varn = f'{varn0}_{suffix}{std_mavsz_}'
            #    varn = 
            #for varn in  ['{varn0}_std{std_mavsz_}',
            #              '{varn0}_invstd{std_mavsz_}',                     
            #             '{varn0}_mavsq{std_mavsz_}','{varn0}_invmavsq{std_mavsz_}',
            #              '{varn0}_mav_d_std{std_mavsz_}','{varn0}_std_d_mav{std_mavsz_}',
            #              '{varn0}_mav_d_var{std_mavsz_}',
            #             '{varn0}_Tan{std_mavsz_}','{varn0}_invTan{std_mavsz_}',
            #             '{varn0}_std{std_mavsz_}',
            #             '{varn0}_mav{std_mavsz_}','{varn0}_invmav{std_mavsz_}']:        
                varnames_all += [varn.format(varn0=varn0,std_mavsz_=std_mavsz_)]
    varnames_all            

    # here untrunc meaning without removing big stat vals (but big ES vals were removed already)
    dfcs_fixhistlen_ = dfcs_fixhistlen_untrunc.copy()
    #dfcs_fixhistlen_ = dfcs_fixhistlen
    #cs = np.ones(len(dfcs_fixhistlen), dtype=bool)
    #for varnames_toshow in varnames_toshow0:
    me_pct_excl = []
    for varn in varnames_all:
        std = dfcs_fixhistlen_untrunc[varn].std()
        c = dfcs_fixhistlen_untrunc[varn].abs() > std*  std_mult
        me_,std_ = getQueryPct(dfcs_fixhistlen_untrunc, f'{varn} > {std* std_mult}', False)
        print('Num excl: {:30}, mean={:.3} %, std={:.3f} len={:7},  stdthr={:.4f}'.format(varn,me_, std_, len(c), std*std_mult) )
        dfcs_fixhistlen_.loc[c,varn] = np.nan
        me_pct_excl += [{'varn':varn, 'mean_excl':me_, 'std_excl':std_,
                         'std_thr':std*std_mult}]
    me_pct_excl = pd.DataFrame(me_pct_excl)

    kill_Tan_2nd = False
    if kill_Tan_2nd:
        varnames_all_Tanlike = []
        for varn0 in varns0: #'error_change']:
            for std_mavsz_ in range(2,maxhl+1):#[1::10]:
                #varnames_toshow0_ = []
                for varn in  ['{varn0}_mav_d_std{std_mavsz_}',
                              '{varn0}_mav_d_var{std_mavsz_}',
                             '{varn0}_Tan{std_mavsz_}']:        
                    varnames_all_Tanlike += [varn.format(varn0=varn0,std_mavsz_=std_mavsz_)]

        for varn in varnames_all_Tanlike:
            dfcs_fixhistlen_.loc[dfcs_fixhistlen_['trialwb'] == 2, varn] = np.nan
    return dfcs_fixhistlen_, me_pct_excl

def _addErrorThr(df, stds):
    # estimate error at second halfs of init stage
    df_wthr = df.merge(stds, on='subject')

    df_wthr['error_initstd'] = df_wthr.error_deg_initstd /  180 * np.pi 
    #df_wthr
    return df_wthr

def _calcStds(df):
    qs_initstage = 'pert_stage_wb.abs() < 1e-10'
    df_init = df.query(qs_initstage + ' and trialwb >= 10')
    grp = df_init.groupby(['subject','pert_stage'],observed=True)
    #display(grp.size())
    #df_init['feedback_deg'].min(), df_init['feedback_deg'].max()

    #grp['error_deg'].std()

    stds = df_init.groupby(['subject'],observed=True)['error_deg'].std()#.std()
    return stds

def addErrorThr(df):
    # def df_thr thing
    stds = _calcStds(df)
    mestd = stds.mean()

    print('mestd = {:.4f}, stds.std() = {:.4f} '.format(mestd, stds.std() ) )

    stds = stds.to_frame().reset_index().rename(columns={'error_deg':'error_deg_initstd'})


    df_wthr = _addErrorThr(df, stds)
    return df_wthr 


def pscAdj_NIH(df_all, cols, subpi = False, inplace=True):
    if not inplace:
        df_all = df_all.copy()
    if subpi:
        sub = np.pi
    else:
        sub = 0
    vars_to_pscadj = cols
    for varn in vars_to_pscadj:
        df_all[f'{varn}_pscadj'] = df_all[varn] - np.pi
        cond = df_all['pert_seq_code'] == 1
        df_all.loc[cond, f'{varn}_pscadj']=  - ( df_all.loc[cond,varn]  - sub)
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
        raise ValueError(f'Nothing for subject {subj}')

    # not env == 'all'
    if (env is not None) and ('environment' in df.columns):
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

def getIdealPath(start, end, lentrunc):
    dirx, diry = start
    from scipy.interpolate import interp1d
    ptinds = np.arange(lentrunc)
    rh2 = np.sqrt( dirx**2 + diry**2 )
    coef = rh / rh2
    dirtgt = (dirtgt0 - fb0pt) * coef + fb0pt

    fxint = interp1d([0, lentrunc-1 ], [dirtgt[0], txc[tgti]], 'linear' )
    tgtpathX_interp = fxint(ptinds)
    fyint = interp1d([0, lentrunc-1 ], [dirtgt[1], tyc[tgti]], 'linear' )
    tgtpathY_interp = fyint(ptinds)

    return np.array(tgtpathX_interp, tgtpathY_interp).T

def calcIndLh( XY, home_position, radius_home, leave_home_coef=1. ):
    from exper_protocol.utils import screen2homec
    Xhc, Yhc = screen2homec(XY[:,0], XY[:,1], home_position  )
    ds = np.sqrt( Xhc**2 + Yhc**2 )
    inds_leavehome = np.where(ds > radius_home * leave_home_coef )[0]
    if len(inds_leavehome):
        indlh  = inds_leavehome[0]  # first index when  traj is outside home
    else:
        indlh = None

    return indlh

def analyzeTraj( XY, tgtXY, home_position, radius_home, indlh = None,
        nframes_offset = 10        ):
    # 10 frames = 82 ms
    # XS.shape = npts x 2
    from exper_protocol.utils import screen2homec
    from base2 import rot
    # often the very first is fully zero
    skipped0 = False
    if tuple(XY[0,:] ) == (0,0):
        print('skip 0 ')
        XY = XY[1:, :]
        skipped0 = True

    target_coords_homec = screen2homec( *tgtXY, home_position  )
    tgtcur = np.array(  list(target_coords_homec), dtype=float)


    Xhc, Yhc = screen2homec(XY[:,0], XY[:,1], home_position  )
    XYhc = np.array( [Xhc, Yhc], dtype=float ).T
    pt0 = XYhc[0]
    XYhc_adj = XYhc - pt0[None,:]

    # angle of target wrt first point of the traj
    tgtcur_adj = tgtcur - pt0
    ang_tgt = np.math.atan2(*tuple(tgtcur_adj) )

    # as if target vas pure vertical
    XYhc_ta = rot( *(XYhc.T), ang_tgt, pt0 )
    Xhc_ta, Yhc_ta = XYhc_ta

    tgtcur_ta     = rot( *tgtcur, ang_tgt, pt0 )

    rh = radius_home
    dirx = rh * np.sin(ang_tgt) + pt0[0]
    diry = rh * np.cos(ang_tgt) + pt0[1]
    dirtgt0 = np.array( [dirx,diry] )

    dirtgt_ta = rot(*dirtgt0, ang_tgt, pt0) #- pt0[0],0
    dirtgt_ta[1] = np.sqrt( rh**2 - dirtgt_ta[0]**2 )

    #ds = np.sqrt( Xhc**2 + Yhc**2 )
    #leave_home_coef = 1.
    #inds_leavehome = np.where(ds > rh * leave_home_coef )[0]
    #if len(inds_leavehome) and (indlh is None):
    #    indlh  = inds_leavehome[0]  # first index when  traj is outside home


    if indlh is not None:
        # point where trajectory exited home
        ptlh = XYhc[indlh]
        lh_pt0adj = ptlh - pt0
        # angle from vertical starting at 0th pt
        ang_lh = np.math.atan2( *lh_pt0adj )
        error_lh_ang = ( ang_lh - ang_tgt  )

        # some number of frames after leaving home when we'll measure 
        nframes_offset = min( nframes_offset, len(XYhc ) - indlh - 1 )

        ptpostlh = XYhc[indlh + nframes_offset]
        # direction after leaving home
        vec = ptpostlh - ptlh
        ang_lh_to_postlh = np.math.atan2( *vec )

        tgtcur_adj_lh = tgtcur - ptlh
        # angle of target wrt lh point
        ang_tgt_lh = np.math.atan2(*tuple(tgtcur_adj_lh) )

        error_lh2_ang = ( ang_lh_to_postlh - ang_tgt_lh  )
    else:
        error_lh_ang = None
        ang_lh = None

        ang_lh_to_postlh = None
        error_lh2_ang = None



    #return fbXYhc_ta
    return locals()

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

def getMaskNotNanInf(vals, axis = None):
    if (vals.ndim > 1) and (axis is not None):
        #r = ~np.any( np.isinf(y), axis=1)
        r = ~ np.any ( np.isnan(vals) | np.isinf(vals), axis=axis )
    else:
        r = ~ ( np.isnan(vals) | np.isinf(vals) )
    return r

def truncateDf(df, coln, q=0.05, infnan_handling='keepnan', inplace=False,
    return_mask = False, trialcol = 'trials',
    cols_uniqify = ['trial_shift_size',
                                'trial_group_col_calc', 'retention_factor_s'] , 
    verbose=False ,
    hi = None, low=None,
    trunc_hi = True, trunc_low = True, abs=False, retloc=False):

    if not inplace:
        df = df.copy()

    ntrials_per_subject = df[trialcol].nunique()

    grp0 = df.groupby([trialcol] + cols_uniqify, observed=True)
    mx=  max(grp0.size() )
    assert mx <= df['subject'].nunique(), mx

    #df = df.set_index( ['subject', trialcol] + cols_uniqify )
    #mask = np.ones(len(df), dtype=bool)

    # good
    #mask =  ~ ( df[coln].isna() | np.isinf( df[coln] ) )
    mask = getMaskNotNanInf(df[coln] )

    #print('fff')
    #if clean_infnan:
    #    # mask of good
    #    mask =  ~ ( df[coln].isna() | np.isinf( df[coln] ) )
    #    df =  df[ mask ]
    #    mask = np.array(mask)

    if  ( (q is not None) and (q > 1e-10) ) or (hi is not None) or (low is not None):
        #gk2qts = calcQuantilesPerESCI(df, coln, q=q)
        # before some of them could be inf, not nan
        df.loc[~mask, coln] = np.nan
        if abs:
            df_ = df[['subject'] + cols_uniqify + [coln] ].copy()
            df_[coln] = df_[coln].abs()
            grp = df_.groupby(['subject'] + cols_uniqify)
            #grp = grp[coln].apply(lambda x, gk: x.abs(), group_keys=True )
            #display(grp)
        else:
            grp = df.groupby(['subject'] + cols_uniqify)

        mgsz = np.max(grp.size() )
        print(  f' np.max(grp.size() )  == {mgsz}')
        assert mgsz <= ntrials_per_subject

        if low is None:
            qlow = grp[coln].quantile(q=q)
        if hi is None:
            qhi  = grp[coln].quantile(q=1-q)

            assert not qhi.reset_index()[coln].isna().any(), qhi

        dfs = []
        # for each group separately
        for gk,ginds in grp.groups.items():

            dftmp = df.loc[ginds]

            if abs:
                dftmp_ = df_.loc[ginds]
                assert dftmp_[coln].min() >= 0
            else:
                dftmp_ = dftmp

            assert len(ginds) == len(dftmp) , 'perhaps index was not reset after concat'
            #print(len(ginds) )
            # DEBUG
            # if len(dftmp) > ntrials_per_subject:
            #    return gk, ginds, dftmp, grp
            assert len(dftmp) <= ntrials_per_subject,(len(dftmp), ntrials_per_subject)

            if low is None:
                lowc = qlow[gk]
            else:
                lowc = low
            if hi is None:
                hic  = qhi[gk]
            else:
                hic = hi

            if verbose:
                print( 'gk={}, lowc={:.6f}, hic={:.6f}'.format( gk, lowc,hic))

            mask_good  =  ~ ( dftmp[coln].isna() | np.isinf( dftmp[coln] ) )
            #mask_trunc =
            #mask_trunc = pd.DataFrame(index=dftmp.index, columns=[0], dtype=bool).fillna(True)
            mask_trunc = pd.Series(index=dftmp.index, dtype=bool)
            mask_trunc[:] = True

            # what to keep
            if trunc_hi:
                mask_trunc &= (dftmp_[coln] < hic)
            if trunc_low:
                mask_trunc &= (dftmp_[coln] > lowc)

            # what to remove
            # if keepnan, then mark as nan, but don't remove
            if infnan_handling in ['keepnan', 'discard']:
                mask_bad = (~mask_good) | (~mask_trunc)
            elif infnan_handling == 'do_nothing':
                mask_bad = (~mask_trunc)
            else:
                raise ValueError(f'Wrong {infnan_handling}')
            #display(mask_good)
            #return
            #display(mask_bad)
            #display(dftmp)
            dftmp.loc[mask_bad  , coln ] = np.nan

            mask = mask & (~mask_bad)

            if np.all( (np.isnan(dftmp[coln]) | np.isinf(dftmp[coln]))  ):
                display(dftmp[ ['subject','trials'] + cols_uniqify + ['err_sens']] )
                print(gk,len(ginds), sum(mask_bad), sum(mask_good), sum(mask_trunc) )
                raise ValueError(gk)

            # if keepnan, then mark as nan, but don't remove
            if infnan_handling == 'discard':
                dftmp = dftmp[~mask_bad]

            dfs += [dftmp]
        df = pd.concat(dfs, ignore_index = 0)  #TODO should I really ignore index here??
        print('dubplicate check ')
        assert not df.duplicated().any()
    elif infnan_handling == 'discard':
        if verbose:
            sz = df[~mask].groupby('subject').size() / df.groupby('subject').size() * 100 
            print( f'Discarded percentage {sz.mean():.3f}, (std={sz.std():.3f} )' )
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
    if retloc:
        r = df, locals()
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
        drop_feedback = False,
        verbose=0, use_sub_angles = 0, retention_factor = 1.,
          reref_target_locs = True,
          long_shift_numerator = False, err_sens_coln = 'err_sens'  ):
    '''
        if allow_duplicating is False we don't allow creating copies
        of subsets of indices within subject (this can be useful for decoding)
    '''
    from config2 import block_names

    if not isinstance(retention_factor, list):
        retention_factor = [retention_factor]
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
        addargs0 = {}
    elif computation_ver == 'computeErrSens3':
        from error_sensitivity import computeErrSens3 as computeES
        #addargs = None # will be added later per subj
        addargs0 = {'use_sub_angles': use_sub_angles}


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
                       'trajPair2corr':trajPair2corr }
                else:
                    addargs = addargs0
                    addargs['reref_target_locs'] = reref_target_locs
                addargs['long_shift_numerator'] = long_shift_numerator
            else:
                addargs = addargs0


            
            for tsz in trial_shift_sizes:
                for rf in retention_factor:
                    #if tsz == 0:
                    #    escoln = 'err_sens':
                    #else:
                    #    escoln = 'err_sens_-{tsz}t':
                    # resetting index is important
                    if 'level_0' in df.columns:
                        df = df.drop(columns=['level_0'])
                    if 'index' in df.columns:
                        df = df.drop(columns=['index'])
                    dfri = df.reset_index()
                    r = computeES(dfri, df_inds=None,
                        error_type=error_type,
                        colname_nh = coln_nh,
                        correct_hit = 'inf', shiftsz = tsz,
                        err_sens_coln=err_sens_coln,
                        coln_nh_out = coln_nh_out,
                        time_locked = time_locked, addvars=addvars,
                        target_info_type = target_info_type,
                        coln_correction_calc = coln_correction_calc,
                        coln_error = coln_error,
                        recalc_non_hit = False, 
                        retention_factor = float(rf),
                        **addargs)

                    print(f'computation_ver = {computation_ver}. retention_factor={rf} tsz = {tsz}')
                    if computation_ver == 'computeErrSens2':
                        nhna, df_esv, ndf2vn = r
                    elif computation_ver == 'computeErrSens3':
                        ndf2vn = None
                        nhna, df_esv = r

                    # if I don't convert to array then there is an indexing problem
                    # even though I try to work wtih db_inds it assigns elsewhere
                    # (or does not assigne at all)
                    es_vals = np.array( df_esv[err_sens_coln] )
                    assert np.any(~np.isnan(es_vals)), tgn  # at least one is not None
                    assert np.any(~np.isinf(es_vals)), tgn  # at least one is not None

                    #colns_set += [coln]

                    dfcur = df.copy()
                    dfcur['trial_shift_size'] = tsz  # NOT _nh, otherwise different number
                    dfcur['time_locked'] = time_locked  # NOT _nh, otherwise different number
                    dfcur[err_sens_coln] = es_vals  # NOT _nh, otherwise different number
                    dfcur['trial_group_col_calc'] = tgn
                    dfcur['error_type'] = error_type
                    # here it means shfited by 1 within subset
                    dfcur[coln_nh_out] = np.array( df_esv[coln_nh_out] )

                    dfcur['correction'] = np.array( df_esv['correction'] )
                    if 'belief_' in df_esv.columns:
                        dfcur['belief_'] = np.array( df_esv['belief_'] )

                    # copy columns including prev_err_sens
                    for cn in ['trial_inds_glob_prevlike_error', 'trial_inds_glob_nextlike_error',
                               f'prev_{err_sens_coln}', 'prev_error', 
                               'retention_factor', 'retention_factor_s',
                               'vals_for_corr1','vals_for_corr2',
                              'prevlike_error' ]:
                        if cn not in df_esv.columns:
                            print(f'WARNING: {cn} not in df_esv.columns')
                        else:
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
                        dfcur['dist_rad_from_prevtgt_shiftrespect'] =\
                            df_esv['target_loc'].values -\
                            df_esv['prev_target_loc_shiftrespect'].values

                    for avn in addvars:
                        if avn in dfcur.columns:
                            continue
                        dfcur[avn] = np.array(df_esv[avn])

                    #lbd(0.5)
                    #print(dfcur['target_locs'].values, df_esv['prev_target'].values )
                    #raise ValueError('f')


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
    if drop_feedback and ( 'feedbackX' in df_all2.columns ):
        df_all2.drop(['feedbackX','feedbackY'],axis=1,inplace=True)

    if 'trajectoryX' in df_all2.columns:
        df_all2.drop(['trajectoryX','trajectoryY'],axis=1,inplace=True)

    # convert to string
    lbd = lambda x : f'{x:.2f}'
    df_all2['dist_rad_from_prevtgt2'] =\
        df_all2['dist_rad_from_prevtgt2'].abs().apply(lbd)


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
    if stl == -1:
        print(f'WRONG pram file format {fnp}, exiting')
        raise ValueError(f'WRONG pram file format {fnp}')

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
    prev_line = None
    for line in lines[stl + 3:]:
        if line.startswith('}'):
            break
        if ':' not in line:
            print(f'readParamFiles: {fnp} PROBLEM: line=   ',line)
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
        prev_line  = line
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

def getGeomInfo(params, exp = 'beh_only', verbose = 1):
    from exper_protocol.utils import (get_target_angles,
        calc_target_positions, calc_err_eucl, coords2anglesRad, screen2homec,
                                     homec2screen)

    #targetAngs = get_target_angles(int(params['num_targets']),
    #            'fan', float(params['target_location_spread']))

    targetAngs = list(map(float, eval( params['target_angles'] ) ) )

    if exp == 'beh_only':
        home_position = (int(round(params['width']/2.0)),
                        int(round(params['height']/2.0)))
    elif exp == 'beh_meg':
        # recall it is different in pilots (me and Maelys) and non-pilot
        raise ValueError('Not yet impl, look at home defined in protocol code')

    if verbose:
        print(targetAngs)
    # list of 2-ples
    target_coords = calc_target_positions(targetAngs, home_position,
                                          params['dist_tgt_from_home'])
    # first positive x, then 0, then negative x
    target_coords_homec = screen2homec( *tuple(zip(*target_coords)), home_position  )
    if verbose:
        print('getGeomInfo: target_coords =', target_coords)

    return home_position, target_coords


def row2multierr(row, dfc, grp_perti, home_position, target_coords,
                 params, revert_pert = True,
                 ax = None, axo = None,
                 force_entire_traj=False, addinfo=None, titlecols=[],
                 xlim=(-140,140),
                 vertline = 'tgt_ta', calc_area = True,
                exitpt_col = 'time_lh' ):
    ''' row is a row of dfcc '''

    if revert_pert:
        raise ValueError('in this version -- not impl')
    if ax is not None:
        raise ValueError('in this version -- not impl')

    from base2 import rot
    from exper_protocol.utils import screen2homec
    ti = row['trial_index']
    #print(f'trial_index = {ti}')
    idx = grp_perti.groups[ti]
    dfcurtr = dfc.loc[idx[1:]]; skipped0 = 1

    # here we want to include zeroth
    times = dfc.loc[idx, 'time'].to_numpy()

    pert = row['perturbation']
    le = len(dfcurtr)

    if axo is None and ax is not None:
        axo = ax

    assert dfcurtr['current_phase_trigger'].min() == dfcurtr['current_phase_trigger'].max()

    target_coords_homec = screen2homec( *tuple(zip(*target_coords)), home_position  )
    #txc,tyc = target_coords_homec
    #if np.diff(txc).min() < 0:
    #    print('inverting x of targets')
    #    txc = -txc
    radius_home = float( params['radius_home'])
    tgtXY = np.array([ row['target_coordX'],row['target_coordY'] ] )

    fbXY = dfcurtr[['feedbackX','feedbackY']].values

    t = row[exitpt_col]
    if not np.isnan(t):
        indlh =  dfcurtr.query('time_since_trial_start <= @t').nlargest(1, 'time_since_trial_start').index[0]
        indlh =  dfcurtr.index.get_loc( indlh ) # get iloc
    else:
        indlh = None
    fb = analyzeTraj( fbXY, tgtXY, home_position,
                       radius_home, indlh = indlh )
    indlh = fb['indlh']
    ofbXY = dfcurtr[['unpert_feedbackX','unpert_feedbackY']].values
    ofb = analyzeTraj( ofbXY, tgtXY, home_position,
                       radius_home, indlh = indlh )

    #curveX, curveY = fbXhc_ta, fbYhc_ta
    traja2 = None
    traja2o = None
    length = None
    #time_lh = None  # wrong for some cols, better use what was preset
    if indlh is not None:
        if fb['skipped0'] or skipped0:
            indlh_eff = indlh + 1
        else:
            indlh_eff = indlh
        #time_lh = times[indlh_eff]
        #time_lh = time_lh - times[0]

        from base2 import areaOne
        traja2 = areaOne( *fb['XYhc_ta'] ,
                fb['dirtgt_ta'], fb['tgtcur_ta'])

        traja2o = areaOne( *ofb['XYhc_ta'] ,
                ofb['dirtgt_ta'], ofb['tgtcur_ta'])
        try:
            from shapely.geometry import LineString
            tr = LineString(list( fb['XYhc_ta'].T ))
            length = tr.length
        except ImportError as e:
            length = None

        #fb_ang_lh        = fb['ang_lh']
        #ofb_ang_lh       = ofb['ang_lh']
        #fb_error_lh_ang  = fb['error_lh_ang']
        #ofb_error_lh_ang = ofb['error_lh_ang']

        # num of frames of mvt (frames after leave home)
        lenmvt =  le - indlh
        # ideal path for feedback
        #tgtpathXY_interp = getIdealPath( (dirx,diry), fb['dirtgt0'],
        #                                lenmvt )

    return locals()


def row2multierr_old(row, dfc, grp_perti, home_position, target_coords,
                 params, revert_pert = True,
                 ax = None, axo = None,
                 force_entire_traj=False, addinfo=None, titlecols=[],
                 xlim=(-140,140),
                 vertline = 'tgt_ta', calc_area = True ):
    '''
    it reverts answer for neg pert
    row['perturbation'] is only used to decide if we revert or no
    '''
    from base2 import rot
    from exper_protocol.utils import screen2homec
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
    # shift target by the first point of fb
    fbXYhc_adj = fbXYhc - fb0pt[:,None]

    ofbXY = dfcurtr[['unpert_feedbackX', 'unpert_feedbackY']].to_numpy()
    ofbXhc, ofbYhc = screen2homec(ofbXY[:,0], ofbXY[:,1], home_position  )
    ofbXYhc = np.array( [ofbXhc, ofbYhc], dtype=float )
    ofb0pt = np.array( [ofbXhc[0], ofbYhc[0] ] , dtype=float)
    ofbXYhc_adj = ofbXYhc - ofb0pt[:,None]

    # angle of target wrt first point of fb
    tgtcur_adj = tgtcur - fb0pt
    ang_tgt = np.math.atan2(*tuple(tgtcur_adj) )

    # angle of target wrt first point of ofb
    tgtcur_adj_ofb = tgtcur - ofb0pt
    ang_tgt_ofb = np.math.atan2(*tuple(tgtcur_adj_ofb) )

    # as if target vas pure vertical
    fbXYhc_ta = rot( *fbXYhc, ang_tgt, fb0pt )
    fbXhc_ta, fbYhc_ta = fbXYhc_ta

    # as if target vas pure vertical
    ofbXYhc_ta = rot( *ofbXYhc, ang_tgt_ofb, ofb0pt )
    ofbXhc_ta, ofbYhc_ta = ofbXYhc_ta

    tgtcur_ta     = rot( *tgtcur, ang_tgt, fb0pt )
    tgtcur_ta_ofb = rot( *tgtcur, ang_tgt_ofb, ofb0pt )

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

        curveX, curveY = fbXhc_ta, fbYhc_ta
        #curveX, curveY = rot(fbXhc_ta, fbYhc_ta, ang_tgt)

        curveX0, curveY0 = rot(ofbXhc, ofbYhc, ang_tgt, ofb0pt)  # ofb ta

        fbXYlh_ta  = fbXYhc_ta[:,ind]
        ofbXYlh_ta = ofbXYhc_ta[:,ind]

        ideal_lh = dirtgt_ta

        if vertline == 'tgt_ta':
            xshift = -tgtcur_ta[0]
            intersections = 'allow'
        elif vertline == 'veryleft':
            xshift = -float( params['width'] )
            intersections = 'prohibit'


        traja, traja2, trajoa = None,None,None
        if calc_area:
            # negative when feedback on the left of the ideal target
            traja = area(curveX[ind:], curveY[ind:], ideal_lh,
                tgtcur_ta, xshift = xshift,
                        verbose=0, ax=ax, plotargs = {'alpha':0.2},
                        intersections = intersections)

            from base2 import areaOne
            # area of the single curve
            traja2 = areaOne(curveX[ind:], curveY[ind:], ideal_lh, tgtcur_ta)

            trajoa = area(curveX0[ind:], curveY0[ind:], ideal_lh,
                tgtcur_ta_ofb, xshift = xshift,
                        verbose=0, ax=axo, plotargs = {'alpha':0.2},
                        intersections = intersections)

        #try:
        #    traja = area(fbXhc_ta[ind:], fbYhc_ta[ind:], (0,rh),
        #         (0,float(params['dist_tgt_from_home'])), verbose=0)
        #except (ValueError,AttributeError) as e:
        #    traja = np.nan
        #    print(f'area: Error for {ti}: {e}')


        try:
            from shapely.geometry import LineString
            tr = LineString(list(zip(curveX[ind:], curveY[ind:])) )
            length = tr.length
        except ImportError as e:
            length = None

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

        # calc angle wrt actual target
        ofb_lh_pt0adj = ofbXYhc[:,ind] - ofb0pt
        ang_ofb_lh = np.math.atan2( *ofb_lh_pt0adj )
        error_unpert_lh_ang = ( ang_ofb_lh - ang_tgt  ) * 180 / np.pi
        #plt.plot(ptinds, fxint(ptinds))

        # FINISHED HERE ON FRIDAY
        # was trying to print all info to make sure I compute the right thing
        # for lh angle
        # doubts on should I rotate (and compute angles)
        # wrt home center or wrt start position
        # if I look at angles for ofb_ta they will be already errors (because
        # rot to true target so that it's vertical. Vertica wrt starting point
        # of ofb)

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
        ang_fb_lh, ang_tgt, ang_ofb_lh = np.nan,np.nan,np.nan
        ind = 0


    if ax is not None:
        plot_ta = 1
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
        if plot_ta:
            ax.scatter( [tgtcur_ta[0]] , [tgtcur_ta[1] ]  ,
                    s=290, c='blue', alpha=0.5)

        ##################
        # nonaligned fb
        if abs(ang_tgt) > 1e-10:
            ax.scatter(fbXhc , fbYhc , alpha=0.3, label='fb (homec)',
                       marker='+', s = 60, c='r')

        # aligned fb
        if plot_ta:
            ax.scatter(fbXhc_ta[:ind] , fbYhc_ta[:ind] , alpha=0.3, c='r')
            ax.scatter(fbXhc_ta[ind:] , fbYhc_ta[ind:] , alpha=0.4, c='r', label='fb ta (homec)')
        # mark black first exit point
        if ind > 0:
            ang_fb_ta_lh = np.math.atan2( *(fbXYlh_ta - fbXYhc_ta[:,0]) )
            #pt_ = np.array(curveX0[ind], curveY0[ind]) - ofb0pt
            ang_ofb_ta_lh = np.math.atan2( *(ofbXYlh_ta - ofbXYhc_ta[:,0] ) )
            #
            c = (1 / np.pi) * 180
            if plot_ta:
                ax.scatter( [fbXYlh_ta[0]], [fbXYlh_ta[1]] , alpha=0.8,
                           c='k', s= 10, label=f'dir fb ta = {ang_fb_ta_lh*c:.1f}')

                ax.scatter( [curveX0[ind]], [curveY0[ind]] , alpha=0.8,
                        c='k', s= 10, label=f'dir ofb ta = {ang_ofb_ta_lh*c:.1f}')

            print(f'{ti}, pert={pert} ind={ind}--> tgt={ang_tgt * c:.1f} '
                f' fblh={ang_fb_lh * c :.1f}, ofblh{ang_ofb_lh * c:.1f}'
                f' fblhta={ang_fb_ta_lh * c :.1f}, ofblhta={ang_ofb_ta_lh * c:.1f}'
                f'\n error_lh_ang = {error_lh_ang:.1f}, error_unpert_lh_ang={error_unpert_lh_ang:.1f}' )
            print(f'   ofbXYhc_ta[0] = {ofbXYhc_ta[:,0]}, fb= {fbXYhc_ta[:,0]}')


            # nonaligned ofb
            axo.scatter(ofbXhc , ofbYhc , alpha=0.4, label='ofb (homec)',
                       marker='x', s = 30, c='magenta')

            # aligned ofb
            if (abs(ang_tgt) > 1e-10) and (plot_ta):
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

    return (err, error_lh_ang, error_unpert_lh_ang,
            start_dist, traja, traja2, trajoa, td, length,
            enddist, ang_fb_lh, ang_tgt, ang_ofb_lh)

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

    r = row2multierr_old(row, dfc, grp_perti, home_position, target_coords,
                 params, revert_pert = False )
    z = zip( ['err', 'error_lh_ang', 'error_unpert_lh_ang', 'start_dist',
              'traja', 'traja2', 'trajoa', 'td',
        'length',  'enddist'], r)

    angvars = ['ang_lh', 'error_lh_ang' ]
    c = 180 / np.pi

    d = dict(z)
    for k,v in d.items():
        print(f'{k} = {v:.2f}' )
    #err, start_dist, traja, trajoa, td, length,  enddist, enddist2 = r
    #print(r)

def calcAdvErrors(dfcc, dfc, grp_perti, target_coords,
                  home_position, params, revert_pert= False,
                  norm_coef = None):
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


    #def f(row):
    #    locs = row2multierr(row, dfc, grp_perti, home_position,
    #        target_coords, params, revert_pert = revert_pert,
    #                        exitpt_col = 'time_lh')
    #    fb  = locs['fb']
    #    ofb = locs['ofb']
    #    r = (fb['error_lh_ang'], ofb['error_lh_ang'],
    #        fb['ang_lh'], ofb['ang_lh'],
    #        locs['traja2'],
    #         locs['length'], locs['time_lh'],
    #         fb['ang_tgt'],
    #         fb['ang_lh_to_tgt'],
    #         fb['error_lh2_ang'], ofb['error_lh2_ang']   )
    #    del locs

    #    return r

    #dfcc[['error_lh_ang', 'error_unpert_lh_ang',
    #      'ang_fb_lh', 'ang_ofb_lh',
    #  'error_area2_signed_nn',
    #      'traj_length', 'time_lh',
    #      'ang_tgt',
    #       'ang_lh_to_tgt', 'error_lh2_ang',
    #            'error_unpert_lh2_ang' ]] =\
    #    dfcc.apply(lambda x: f(x) ,1, result_type='expand')


    def f(row):
        locs = row2multierr(row, dfc, grp_perti, home_position,
            target_coords, params, revert_pert = revert_pert,
                            exitpt_col = 'time_tmstart2')
        fb  = locs['fb']
        ofb = locs['ofb']
        r = (fb['error_lh_ang'], ofb['error_lh_ang'],
            fb['ang_lh'], ofb['ang_lh'],
            locs['traja2'],
            locs['traja2o'],
             locs['length'],
             fb['ang_tgt'],
             fb['ang_lh_to_postlh'], fb['error_lh2_ang'],
             ofb['error_lh2_ang']   )
        del locs

        return r
    dfcc[['error_lh_ang', 'error_unpert_lh_ang',
          'ang_fb_lh', 'ang_ofb_lh',
      'error_area2_signed_nn',
      'error_unpert_area2_signed_nn',
          'traj_length',
          'ang_tgt',
           'ang_fb_lh_to_postlh', 'error_lh2_ang',
                'error_unpert_lh2_ang' ]] =\
        dfcc.apply(lambda x: f(x) ,1, result_type='expand')


    dfcc['error_lh_ang_deg'] = dfcc['error_lh_ang'] * 180 / np.pi
    dfcc['error_lh_ang_deg_abs'] = dfcc['error_lh_ang_deg'].abs
    dfcc['error_lh2_ang_deg'] = dfcc['error_lh2_ang'] * 180 / np.pi
    dfcc['error_lh2_ang_deg_abs'] = dfcc['error_lh2_ang_deg'].abs()


    #def f(row):
    #    r = row2multierr_old(row, dfc, grp_perti, home_position,
    #        target_coords, params, revert_pert = revert_pert)
    #    return r
    #dfcc[['error_intdist2_nn', 'error_lh_ang', 'error_unpert_lh_ang',
    #      'error_distance_lh',
    #  'error_area_signed_nn','error_area2_signed_nn', 'error_area_ofb_signed_nn',
    #      'time_lh', 'traj_length', 'enddist','ang_fb_lh',
    #      'ang_tgt', 'ang_ofb_lh']] =\
    #    dfcc.apply(lambda x: f(x) ,1, result_type='expand')

    if norm_coef is None:
        from base2 import calcNormCoefSectorArea
        norm_coef =  calcNormCoefSectorArea(params)

    #dfcc['error_area_signed_nn' ]      *= norm_coef
    #dfcc['error_area_ofb_signed_nn' ]  *= norm_coef
    dfcc['error_area2_signed_nn' ]     *= norm_coef

    #def f(row, normalize):
    #    return row2multierr(row, dfc, grp_perti, home_position,
    #                 target_coords, params, normalize, curve_for_area_calc='org_feedback')
    #dfcc['error_area_ofb_signed_nn'] = dfcc.apply(lambda x: f(x, True)[2] ,1)

    ############### calc movement time, not counting time after reaching target

    des_time = 30 * 1e-3
    frame = 1 / float( params['FPS'] )
    nframes_stat = int(np.ceil(des_time / frame))
    # print('nframes_stat = ',nframes_stat)  # for des_frame = 30 * 1e-3 it is = 4

    hitr = float(params['radius_target']) + float(params['radius_cursor']) / 2
    dfcc['time_mvt'] = np.nan
    #dfcc.loc[dfcc['nonhit'], 'time_mvt']  = float(params['time_feedback'] ) - dfcc.loc[dfcc['nonhit'], 'time_lh']
    for rowi, row in dfcc.iterrows():
        if row['nonhit']:
            # TODO: has to be updated for MEG task, since
            # time of mvt is controlled differently there
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


    #dfcc['error_intdist2'] = dfcc['error_intdist2_nn'] / dfcc['time_mvt']
    #dfcc['error_area_signed'] = dfcc['error_area_signed_nn'] / dfcc['time_mvt']
    dfcc['error_area2_signed'] = dfcc['error_area2_signed_nn'] / dfcc['time_mvt']
    #dfcc['error_area_ofb_signed'] = dfcc['error_area_ofb_signed_nn'] / dfcc['time_mvt']

    #coef_endpt_err  = 0.5
    #dfcc['error_aug2'] = dfcc['error_intdist2'] + coef_endpt_err * dfcc['error_distance']

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
    '''
    Take row with highest/lowest value of coln_time
    coln_time is the column on value of which one wants to aggregate
    '''
    assert coln_time in df.columns
    if grp is None:
        assert colgrp in df.columns
    # coln_time = 'time'
    assert operation in ['min','max']
    from datetime import timedelta
    if coln_time == 'time':
        diffmin = df[coln_time].diff().min()
        if isinstance(diffmin, timedelta):
            assert diffmin.total_seconds() >= 0, diffmin  # need monotnicity
        else:
            assert diffmin >= 0, diffmin  # need monotnicity
    assert coltake is not None

    if grp is None:
        grp = df.groupby(colgrp)
    else:
        if colgrp is not None:
            print('aggRows WARNING: Column ', colgrp, ' is not used due to grp != None')

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
    return dfr.sort_values([coln_time])

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

    phase2df   = {}
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
                phase_to_collect = 'TARGET_AND_FEEDBACK', training_end_sep_trial = False,
                reshifted = 0, def_subject_ind = 1, check_num_context_appearances = 1 ):
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
                            'rot-20','rot20', 
                             'rot45', 'rot90','rot135', 'error_clamp' ],[0,15,30,-15,-20,20, 45, 90,135, 0]) )

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

    # just for convenience to have an interger index
    # since numbers start from 0 and we had a pilot, assign 0 to pilot
    if def_subject_ind:
        dfcc['subject_ind'] = dfcc['subject'].str[-3:]
        dfcc.loc[dfcc['subject'].str.contains('pilot'), 'subject_ind'] = 0
        dfcc['subject_ind'] = dfcc['subject_ind'].astype(int)


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

    if not reshifted and check_num_context_appearances:
        assert np.max(dfcc['Nctx_app'] ) == params['n_context_appearences']


    ##################################

    dfcc['is_easy'] = False
    dfcc.loc[dfcc.query(qs_easy0).index, 'is_easy'] = True
    dfcc['is_easy_wide'] = False
    dfcc.loc[dfcc.query(qs_easy_wide).index, 'is_easy_wide'] = True


    radius_home = params['radius_home']
    # same thr for both axes
    # 1e-3
    thr_jax = 7e-4
    #thr_jax = 9e-4
    addTrueMvtTime(dfcc, dfc, home_position, radius_home, thr_jax, thr_jax)

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
    # TODO: _abs here is misleading, it is not absolute value, just not related to target
    dfcc['feedback_tgt_unrel'] = dfcc.apply(f,1)

    def f(row):
        fb = (row['unpert_feedbackX'],  row['unpert_feedbackY'])
        try:
            ang = coords2anglesRad(*fb, home_position)
        except ZeroDivisionError as e:
            ang = np.nan
        #ang -= np.pi / 2
        return ang / np.pi * 180
        #eturn ang
    dfcc['org_feedback_tgt_unrel'] = dfcc.apply(f,1)


    # feedback is relative to the target (including pert)
    dfcc['feedback']     = (dfcc['feedback_tgt_unrel'] - 90)    - (dfcc['target_locs'] - 90)
    # org feedback is relative to the target (NOT including pert)
    dfcc['org_feedback'] = dfcc['org_feedback_tgt_unrel'] - dfcc['target_locs']
    dfcc['error_endpoint_ang'] = dfcc['feedback_tgt_unrel'] - dfcc['target_locs']

    #################################  compound error info



    #print("calcAdvErrors start")
    #assert dfc['current_phase_trigger'].nunique() == 1
    #assert dfcc['current_phase_trigger'].nunique() == 1
    calcAdvErrors(dfcc, dfc, grp_perti, target_coords,
                  home_position, params, revert_pert = False)

    # I cannot do it before because I don't have time_lh before that

    dfcc.loc[dfcc['time_lh'].isna(), 'error_endpoint_ang'] = np.nan
    #dfcc.loc[dfcc['time_lh'].isna(), 'error_pert_adj'] = np.nan



    colns_adj180 = ['error_endpoint_ang', 'error_lh_ang', 'error_unpert_lh_ang']
    for coln in colns_adj180:
        c = dfcc[coln] > 180
        dfcc.loc[c, coln] =  dfcc.loc[c, coln] - 360
        c = dfcc[coln] < -180
        dfcc.loc[c, coln] =  dfcc.loc[c, coln] + 360


    c = dfcc['perturbation'] < 0
    dfcc['error_endpoint_ang_pert_adj'] = dfcc['error_endpoint_ang']
    dfcc.loc[c, 'error_endpoint_ang_pert_adj'] = -dfcc.loc[c,'error_endpoint_ang_pert_adj']


    col = 'error_area2_signed'
    #col = 'error_area_signed'
    #col2 = 'error_pert_adj'
    col3 = 'error_distance'
    if col in dfcc.columns:
        #coef = dfcc[col].quantile(0.75) / dfcc[col2].quantile(0.75)
        #dfcc['error_aug3'] = dfcc[col] + coef * dfcc[col2]

        #sc = np.sqrt( dfcc[col3] / hitr )
        ## bigger error -- stronger scale
        #dfcc['error_area_signed_scaled_ed'] = dfcc[col] * sc
        #dfcc['error_area_ofb_signed_scaled_ed'] = dfcc['error_area_ofb_signed'] * sc

        #dfcc['error_area_signed_tln']     = dfcc['error_area_signed_nn'] / dfcc['traj_length']
        #dfcc['error_area_ofb_signed_tln'] = dfcc['error_area_ofb_signed_nn'] / dfcc['traj_length']

        dfcc['error_area2_signed_tln']     = dfcc['error_area2_signed_nn'] / dfcc['traj_length']

        sc = np.power( np.maximum(dfcc['error_distance'] / hitr, 1.)   , 1./8. )
        #dfcc['error_area_signed_tln_scaled_ed'] = dfcc['error_area_signed_tln'] * sc
        #dfcc['error_area_ofb_signed_tln_scaled_ed'] = dfcc['error_area_ofb_signed_tln'] * sc

        #dfcc['error_area_signed_scaled_ed'] = dfcc['error_area_signed'] * sc
        #dfcc['error_area_ofb_signed_scaled_ed'] = dfcc['error_area_ofb_signed'] * sc

        #dfcc['error_area_signed_nn_scaled_ed']     = dfcc['error_area_signed_nn'] * sc
        dfcc['error_area2_signed_nn_scaled_ed']     = dfcc['error_area2_signed_nn'] * sc
        #dfcc['error_area_ofb_signed_nn_scaled_ed'] = dfcc['error_area_ofb_signed_nn'] * sc
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

    # here we want to set prev_trial_type. We want it to incluse pauses.
    # But we still don't want pauses to be part of dfcc, so we use intermediate df
    # TODO: use agg here instead
    dftmp = df[['trial_index','time','trial_type'] ].groupby('trial_index').\
        min(numeric_only=0).reset_index().set_index('trial_index')
    dftmp['prev_trial_type'] = None
    dftmp['prev_trial_type'] =dftmp['trial_type'].shift(1)

    dfcc = dfcc.set_index('trial_index')
    dfcc['prev_trial_type'] = dftmp.loc[dfcc.index,'prev_trial_type']


    dfcc['prev_time_lh'] = dfcc['time_lh'].shift(1)

    # very literal, has problems when we have EC before.
    # Pauses are not a problem because they are excluded from dfcc
    # it should include switch from pretraining
    dfcc['prev_perturbation']     = dfcc['perturbation'].shift(1)
    dfcc['perturbation_diff']     = dfcc['perturbation'].diff()
    dfcc['perturbation_diff_abs'] = dfcc['perturbation'].diff().abs()


    # pertrubation_valid is when we
    # we define pertrubation as if we did not have error clamps
    # so basically we set error_clamp trial pertrubation to the block perturbation value. Maybe 'valid' is a bit confusing here
    dfcc_tmp = dfcc.copy()
    c = dfcc_tmp['trial_type'] == 'error_clamp'
    dfcc_tmp['perturbation_tmp'] = np.nan
    dfcc_tmp.loc[~c,'perturbation_tmp'] = dfcc_tmp.loc[~c,'perturbation']
    dfcc_tmp['perturbation_tmp'] = dfcc_tmp['perturbation_tmp'].\
        fillna(None, method='ffill')#.astype(int)

    dfcc['prev_perturbation_valid']     = dfcc_tmp['perturbation_tmp'].shift(1)
    dfcc['perturbation_valid_diff']     = dfcc_tmp['perturbation_tmp'].diff()
    dfcc['perturbation_valid_diff_abs'] = dfcc_tmp['perturbation_tmp'].diff().abs()
    #dfcc['prev_perturbation_valid'] = dfcc['perturbation'].shift(1)


    dfcc = dfcc.reset_index()

    # this has to be after reset index, otherwise trial index is index and not part of the columns
    coln_errs_to_propag = ['error_lh2_ang', 'error_unpert_lh2_ang', 'error_area2_signed_nn' ]
    coln_errs_to_propag_shift = ['trial_index' ]
    propagToValidErrCols(dfcc, coln_errs_to_propag, coln_errs_to_propag_shift)

    dfcc['prev_block_pert'] = dfcc['prev_perturbation_valid']
    c = dfcc['trialwb'] == 0
    dfcc['prev_block_pert'] = dfcc['prev_block_pert'].where(c, np.nan)
    dfcc['prev_block_pert'] = dfcc['prev_block_pert'].fillna(None, method='ffill')

    dfcc['error_lh2_ang_deg'] = dfcc['error_lh2_ang'] * 180 / np.pi

    # df  -- row per screen update, with streak relations, all phases
    # dfc -- row per screen update, only TARGET_AND_FEEDBACK
    # dfcc -- row per trial
    # dfcpc -- row per pause
    # dfctc -- row per target change
    return df, dfc, dfcc, dfcp, dfctc

def propagToValidErrCols(dfcc, colns, colns_shift):
    # inplace
    # assumes sorted by trial_index
    # propagate skipping not exiting home circle and error clamps (pauses should already be thrown away)
    # still keeps nan at the very beginning sometimes
    assert dfcc['subject'].nunique() == 1
    dfcc_tmp = dfcc.copy()
    c = ( dfcc_tmp['trial_type'] == 'error_clamp'  ) | ( dfcc_tmp['time_lh'].isna() )
    for coln in  colns:
        dfcc_tmp['tmpcol__'] = np.nan
        dfcc_tmp.loc[~c,'tmpcol__'] = dfcc_tmp.loc[~c, coln]
        dfcc[coln + '_valid' ] = dfcc_tmp['tmpcol__'].fillna(None, method='ffill')

    assert 'trial_index' in dfcc_tmp
    for coln in  colns_shift:
        dfcc_tmp['tmpcol__'] = np.nan
        dfcc_tmp.loc[~c,'tmpcol__'] = dfcc_tmp.loc[~c, coln]
        dfcc[ 'prev_' + coln + '_valid' ] = dfcc_tmp['tmpcol__'].fillna(None, method='ffill').shift(1)

def plotTraj3(ax, row, dfc_all, df_es, colscols, params, verbose = 0 ,
             traj_to_plot = ['feedback'],  exitpt_col = 'time_lh',
              traj_alpha = 0.5, color_fb = 'b', color_ofb='orange', markersize = 2 ):
    ''' row of dfcc'''
    ti = row['trial_index']
    subj = row['subject']
    print(f'Plotting S{subj[-3:] } trial {ti} ({row["trialwb"]}) pert={row["perturbation"]} tgt={row["target_locs"]:.0f} ptt = {row["prev_trial_type"]}',
          f'ppe = {row["prev_perturbation_valid"]}')
    home_position, target_coords  = getGeomInfo(params, verbose=0)

    tind1 = ti
    #tind2 = tind1 -1
    from behav_proc import analyzeTraj

    tmp = {}
    tmpo = {}
    #for tind in [tind1, tind2]:
    for tind in [tind1]:
        dftmp = dfc_all.query('subject == @subj and trial_index == @tind').iloc[1:]
        fbXY = dftmp[['feedbackX','feedbackY']]
        tgt = dftmp[['target_coordX','target_coordY']].iloc[10].values

        t = row[exitpt_col]
        if not np.isnan(t):
            indlh =  dftmp.query('time_since_trial_start <= @t').nlargest(1, 'time_since_trial_start').index[0]
            indlh =  dftmp.index.get_loc( indlh ) # get iloc
        else:
            indlh = None
        rd = analyzeTraj(fbXY.values, tgt, home_position,
                    params['radius_home'],  indlh = indlh)
        tmp[tind] = rd

        ofbXY = dftmp[['unpert_feedbackX','unpert_feedbackY']]
        rd = analyzeTraj(ofbXY.values, tgt, home_position,
                    params['radius_home'], indlh=indlh)
        tmpo[tind] = rd


    rt = params['radius_target']
    rh = params['radius_home']
    pp_per_tind = [dict(ls='-', c=color_fb, marker='o', markersize=markersize,
                        alpha=traj_alpha),
                   dict(ls=':', c=color_fb, marker='o', markersize=markersize)][:1 ]
    ppo_per_tind = [dict(ls='-', c=color_ofb, marker='o', markersize=markersize,
                        alpha=traj_alpha ),
                    dict(ls=':', c=color_ofb, marker='o', markersize=markersize)] [:1 ]
    #for tind,pp,ppo in zip([tind1, tind2],pp_per_tind,ppo_per_tind):
    # take first only as zip works this way
    for tind,pp,ppo in zip([tind1],pp_per_tind,ppo_per_tind):
        tgtcur = tmp[tind]['tgtcur']
        rd = tmp[tind]
        crc = plt.Circle(tgtcur, rt, color='blue', lw=markersize, fill=False,
                         alpha=0.6, ls=pp['ls'])
        ax.add_patch(crc)

        c = (1 / np.pi) * 180

        if 'feedback' in traj_to_plot:
            # main plot
            ax.plot( *rd['XYhc'].T, **pp)

            indlh = rd['indlh']
            if indlh is not None:
                XYlh = rd['XYhc'][indlh]
                ang_lh = rd['ang_lh']
                ang_lh_to_postlh = rd['ang_lh_to_postlh']

                error_lh_ang = rd['error_lh_ang']
                error_lh2_ang = rd['error_lh2_ang']

                if verbose:
                    #print('0pt={}, lhpt={}'.format(rd['pt0'], XYlh ))

                    print('lh={:5.6f}, tgt={:5.2f}, err={:5.2f}  err2={:5.2f}'.format(ang_lh*c, rd['ang_tgt'] * c,
                          error_lh_ang * c, error_lh2_ang * c) )

                # just one pt
                ax.scatter( *XYlh , alpha=1,
                    c='k', s= 15,
                    label=f'{tind} ang_lh2 = {ang_lh_to_postlh*c:.1f}')

        if 'org_feedback' in traj_to_plot:
            rdo = tmpo[tind]
            ax.plot( *rdo['XYhc'].T, **ppo)

            indlh = rdo['indlh']
            XYlh = rdo['XYhc'][indlh]
            ang_lh = rdo['ang_lh']
            ang_lh_to_postlh = rdo['ang_lh_to_postlh']
            # just one pt
            if ang_lh is not None:
                ax.scatter( *XYlh , alpha=1,
                    c='k', s= 15,
                    label=f'{tind} ang_lh2 = {ang_lh_to_postlh*c:.1f}')
                    #label=f'{tind} ang_lh_0pt = {ang_lh_to_postlh*c:.1f}')


    ax.scatter( 0,0, alpha=0.8,
            c='k', s= 25 )
    crc = plt.Circle((0, 0), rh, color='r', lw=2, fill=False,
                     alpha=0.3)
    ax.add_patch(crc)
    ax.legend()

    s = ''
    if (df_es is not None) and (colscols is not None):
        s += '\n'
        r = df_es.query('subject == @subj and trial_index == @tind')
        assert len(r) == 1
        r = r.to_dict('records')[0]
        for cols_ in colscols:
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

    ax.set_title(f'S{subj[-3:]} ti={ti} ({row["trialwb"]}); pert={row["perturbation"]}; ' + s)


def plotTraj2(ax, dfcurtr, home_position, target_coords_homec, params,
        calc_area = False, show_dist_guides = False, verbose=0,
             force_entire_traj=False, addinfo=None, titlecols=[],
             xlim=(-140,140) ):
    from exper_protocol.utils import (get_target_angles,
        calc_target_positions, calc_err_eucl, coords2anglesRad, screen2homec,
                                     homec2screen)
    from scipy.interpolate import interp1d
    from base2 import rot
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
    '''time traces for multiple errors'''
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
    # get block starts and ends
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

def markBlockStarts(df, ax, plot_argv):
    '''vert lines at the start of first trial in block
    input is row=trial'''
    qs = 'block_ind >= 0'
    block_starts = df.query(qs).groupby('block_ind')['trial_index'].min()
    if ax is not None:
        ax.vlines(block_starts, **plot_argv)
    return block_starts

def printBadTrialInfo(df, df_big, coln='err_sens'):
    # Total percentage of bad trials for a subset
    rinf  = np.isinf(df[coln]).sum() / len(df_big)
    rnan  = np.isnan(df[coln]).sum() / len(df_big)
    rtot = len(df) / len(df_big)
    print('inf = {:.2f}%,  nan = {:.2f}%  (total={:.2f}%)'.format( rinf * 100, rnan * 100, rtot *100) )


def calcErrorDrops(dfcc_all, cols, inds = [0,1,2,4], numtrain = 12, inc_veridical = 0,
                  pert_change_handling = 'do_nothing', trialwbcol = 'trialwb_notfail' ):
    '''  trialwbcol = 'trialwb_notfail' # or trialwb 
    calc drops of errror between the start of the block and the end
    excludes veridical
    '''
    assert pert_change_handling in ['invalidate', 'do_nothing']
    # inds should be increasing
#    from behav_proc import qs_notspec
    assert inds[0] == 0
    qs = qs_notspec + f' and trialwb <= {np.max(inds) }'
    if not inc_veridical:
        #qs +=' and vis_feedback_type != "veridical"'
        qs +=' and prev_block_pert != perturbation'
    df_to_use = dfcc_all.query(qs)

    dfdrops = []
    v1 = 0
    if v1:
        # loop across error columns
        for coln in cols:
            assert inds[0] == 0

            def f(dfcur):
                #print(dfcur[coln])
                dfs = []
                # takes slices with fixed index within trial
                for ind in inds:
                    dfs += [dfcur.query('trialwb == @ind')]

                #print(df1[coln])
                #print(df2[coln])
                vss = []
                # subtract abs value of error (largest) at first ind - given
                # we look at absolut values only
                # here we should only get one value per group
                for df in dfs[1:]:
                    #vs = np.abs(dfs[0][coln].values - df[coln].values)
                    vs = np.abs(dfs[0][coln].values) - np.abs(df[coln].values)
                    vss += [vs]
                if len(vs) == 0:
                    #display(dfcur)
                    return None
                else:
                    assert len(vs) ==1, len(vs)
                    if dfs[0]['prev_ctx_pert_same'].values[0]:
                        # Q: how to handle failed reaches?
                        #print('Skipping due to same tgt')
                        return None
                    else:
                        #v = (df1[coln].values - df3[coln].values)[0]
                        return [ vs[0] if len(vs) else np.nan  for vs in vss]
                        #return [abs( vs[0] ), abs(v)]
            df_notspec_notver_difflearn = df_to_use.\
                groupby(['subject','block_ind','Nctx_app', 'ctxid']).apply(f).reset_index()
            df_notspec_notver_difflearn= df_notspec_notver_difflearn.rename(columns={0:'absdrop'})
            df_notspec_notver_difflearn['coln'] = coln
            dfdrops += [df_notspec_notver_difflearn]

        dfdrop = pd.concat(dfdrops, ignore_index=1)

        dfdrops2 = []  # this is duplication but easier to plot later
        rs = []
        # loop across drop indices, except first (which is zero normally)
        for i,ind in enumerate(inds[1:] ):
            dfdrop[f'absdrop_{ind}'] = dfdrop['absdrop'].apply(lambda x: x[i] if x is not None else None)

            dfdrop_ = dfdrop.copy()
            dfdrop_['absdrop'] = dfdrop_[f'absdrop_{ind}']
            dfdrop_.drop( columns=[f'absdrop_{ind}'] )
            dfdrop_['droptoind'] = ind
            dfdrops2 += [ dfdrop_ ]
        dfdrop2 = pd.concat(dfdrops2)

        # compute difference between drop on first and on last appearances
        # has some NaNs :(
        rs = []
        # loop across drop indices, except first (which is zero normally)
        for ind in inds[1:]:
            coln = f'absdrop_{ind}'
            # drop at last context apperence - drop at first one
            lbd = lambda x: np.abs( x.loc[x['Nctx_app'].idxmax(), coln] ) -\
                np.abs (x.loc[x['Nctx_app'].idxmin(), coln] )
            r = dfdrop.groupby(['coln', 'subject','ctxid']).apply(lbd)
            r = r.to_frame().rename(columns={0:coln})#.reset_index()
            rs += [r]

        dfdrop_last = pd.concat(rs, axis=1, ignore_index=0)
    else:
        df = df_to_use.copy()

        # first trial when we experience new pert might be weird. Definitely for area, not so much so for angle
        if pert_change_handling == 'invalidate':
            pert_changed = df['prev_ctx_pert_same'] == False  # chagne of pert
            # invalidate first after pert
            df.loc[pert_changed, 'time_lh'] = np.nan
        # we don't want to include failed trials in indexing of trial within block
        df['nan_count'] = df.groupby(['subject','block_ind'], group_keys=False )['time_lh'].apply(lambda x: x.isna().cumsum())
        df['trialwb_notfail'] = df['trialwb'].sub(df['nan_count'])
        df['trialwb_notfail'] = df['trialwb_notfail'].where( df['time_lh'].notna(), np.nan )
        df = df.drop('nan_count', axis=1)

        dfs3 = []
        for coln in cols:
            #cols_out0 = ['trial_index', 'block_ind', 'Nctx_app', 'ctxid', 'prev_ctx_pert_same']  + [coln]
            cols_out = ['trial_index', coln ] 
            cols_out0 = ['trial_index', 'prev_ctx_pert_same', coln]

            dfs = []
            # choose trials with indices I request
            for ind in inds:
                if ind == 0:
                    cols_cur = cols_out0 
                else:
                    cols_cur = cols_out 

                #def f(dftmp, ind):
                #    dftmp = dftmp.query(f'{trialwbcol} == {ind}')
                #    return dftmp[cols_cur]
                    #if len(dftmp) == 0:
                    #    return np.nan
                    #else:
                    #    assert len(dftmp) == 1, len(dftmp)
                    #    #return [ dftmp[ coln ].values[0], dftmp[ 'trial_index' ].values[0]  ]
                    #    return dftmp[cols_cur]

                #def f2(dftmp, ind):
                #    dftmp = dftmp.query('trialwb == @ind')
                #    if len(dftmp) == 0:
                #        return np.nan
                #    else:
                #        assert len(dftmp) == 1, len(dftmp)
                #        return dftmp[ 'trial_index' ].values[0]

                # I want to have all these columns in the output
                # trialwbcol is set in the arg and determines whether we skip failed reaches or not
                dftmp = df.query(f'{trialwbcol} == {ind}').set_index(['subject','block_ind'] ).copy()

                #dftmp = df.groupby(['subject','block_ind','Nctx_app', 'ctxid']).apply( lambda x: f(x, ind) )

                #for coli, col in enumerate(cols_cur ):
                #    dftmp[ '{}_{}'.format(col,ind) ] = dftmp[ col]

                dftmp = dftmp.add_suffix(f'_{ind}')

                #dftmp = dftmp.drop(columns=cols_cur)
                    #dftmp[0].apply(lambda x: x[coli] if x is not None else None)

                #dftmp = df.groupby(['subject','block_ind']).apply(lambda x: f(x, ind) )
                #dftmp = dftmp.to_frame().rename(columns={0:f'coln_{ind}'} )

                #dftmp2 = df.groupby(['subject','block_ind']).apply(lambda x: f2(x, ind) )
                #dftmp2 = dftmp2.to_frame().rename(columns={0:f'trial_index_{ind}'} )
                #dfs += [dftmp, dftmp2]

                #df[[f'coln_{ind}', f'trial_index{ind}'] ] = df.groupby(['subject','block_ind']).apply(f, result_type='expand')

                #dftmp = df.query('trialwb == @ind').set_index(['subject','block_ind'])
                #if ind == 0:
                #    cols_cur = cols_out0 
                #else:
                #    cols_cur = cols_out 
                #dftmp = dftmp[cols_cur]
                #dftmp = dftmp.rename(columns={coln: f'coln_{ind}', 'trial_index':f'trial_index{ind}' } )
                dfs += [dftmp]

            df2 = pd.concat(dfs, join='outer', axis=1).reset_index()


            dfs2 = []
            # we won't need absdrop from inds[0] but we may need colnval
            for ind in inds:  # [1:]:
                #dftmp = df.query('prev_ctx_pert_same0 == False').copy()
                dftmp = df2.copy()
                dftmp['absdrop'] = dftmp[f'{coln}_0'].abs() - dftmp[f'{coln}_{ind}'].abs()
                dftmp['colnval'] = dftmp[f'{coln}_{ind}'].abs() 
                dftmp['droptoind'] = ind
                dfs2 += [dftmp]

            #dfs2 = []
            #for ind in inds[1:]:
            #    dftmp = df2.query('prev_ctx_pert_same0 == False').copy()
            #    dftmp['absdrop'] = dftmp[f'coln_{ind}'].abs() - dftmp['coln_0'].abs()
            #    dftmp['droptoind'] = ind
            #    dfs2 += [dftmp]


            df3 = pd.concat(dfs2 , axis=0).reset_index(drop=True)   # copy to defragment
            #df3 = df3.T.drop_duplicates().T
            #df3 = df3.set_index(['subject','block_ind','droptoind'] )
            df3['coln'] = coln
            dfs3 += [ df3   ]

        dfdrop2 = pd.concat(dfs3, axis=0, ignore_index=1)
        dfdrop = None

    columns = list(dfcc_all.columns) + ['trialwb_notfail']
    rd = dict( zip(  [ c + '_0' for c in columns ], columns ) )
    dfdrop2 = dfdrop2.rename( columns=rd)
    dfdrop2['pert_change_handling'] = pert_change_handling
    #return dfdrop, dfdrop2, dfdrop_last
    return dfdrop2
    #return locals()

# make polynomial fits
def plotPolys(ax, dftmp, fitcol, degs=range(2,6), mean=1):
    if mean:
        me = dftmp.groupby(fitcol).median().reset_index()
        dftmp = me
    dftmp[fitcol] = pd.to_numeric(dftmp[fitcol] )
    esv, dv = dftmp[['err_sens',fitcol]]._values.T
    print(np.min(dv),dv,dv-np.min(dv),esv)
    #pr = np.polyfit(esv,dv,2)
    from numpy.linalg import LinAlgError
    dvu = np.unique(dv)
    dvu = np.array( list(sorted(dvu)) )
    print(dvu)
    for deg in degs:
        try:
            pr = np.polyfit(dv-np.min(dv),esv-np.min(esv),deg)
        except (SystemError,LinAlgError):
            print(f'Failed deg={deg}')
            print(dv,esv, np.std(dv))
            continue

        poly = np.poly1d(pr)
        #if len(degs) > 1:
        if mean:
            lbl = f'polynomial fit of means deg={deg}'
        else:
            lbl = f'polynomial fit deg={deg}'
        #else:
        #    lbl = None
        esv2 = poly(dvu-np.min(dvu)) + np.min(esv)
        print(dvu-np.min(dvu), esv)
        ax.plot(range(len(dvu)) , poly(dvu-np.min(dvu)) + np.min(esv),
                label=lbl, c='grey', lw=0.85 )
    return pr
    #ax.legend(loc='lower right')

def getPvals(dftmp, fitcol, pairs, alternative='greater'):
    from scipy.stats import ttest_ind
    pvalues = []
    pair2pv = {}
    for drp in pairs:
        if isinstance(drp[0],str):
            vs1 = dftmp.query(f'{fitcol} == "{drp[0]}"')['err_sens']
            vs2 = dftmp.query(f'{fitcol} == "{drp[1]}"')['err_sens']
        else:
            vs1 = dftmp.query(f'{fitcol} == {drp[0]}')['err_sens']
            vs2 = dftmp.query(f'{fitcol} == {drp[1]}')['err_sens']
        #ttr = ttest_ind(vs1,vs2)

        assert not np.any( np.isnan(vs1) )
        assert not np.any( np.isnan(vs2) )

        ttr = ttest_ind(vs1,vs2, alternative=alternative)
        pvalues += [ttr.pvalue]

        pair2pv[drp] = ttr.pvalue

        print(drp, len(vs1),len(vs2), ttr.pvalue)
    # pvalues = [
    #     sci_stats.mannwhitneyu(robots, flight, alternative="two-sided").pvalue,
    #     sci_stats.mannwhitneyu(flight, sound, alternative="two-sided").pvalue,
    #     sci_stats.mannwhitneyu(robots, sound, alternative="two-sided").pvalue
    # ]

    # pvalues
    # [0.00013485140468088997, 0.2557331102364572, 0.00022985464929005115]

    # Transform each p-value to "p=" in scientific notation
    formatted_pvalues = [f'p={pvalue:.2e}' for pvalue in pvalues]
    return pvalues, formatted_pvalues, pair2pv

#plotPolys(ax,dftmp,fitcol,mean=0)
#ax = plt.gca()
#plotPolys(ax,dftmp,fitcol, degs=rng, mean=meanfit)


def extractCtxChangeRelevantRows(df_all_multi_tsz, edfc, time_locked,
                                 first_trial_to_take = 'adaptive' ):
    '''
    get trials important for ctx change (and not change in the of a block)
    '''
    assert first_trial_to_take in ['first', 'second', 'adaptive']
    qs_es = ('error_data_for_calc == @edfc'
              ' and trial_shift_size == 1 '
              ' and trial_group_col_calc == "trials"'
              ' and time_locked == @time_locked'
             )
    # allow sandwiches ' and special_block_type != "error_clamp_sandwich"'
    # if we prohibit prev EC here we won't get all first trials
    # after context switch

    #qs_es += (' and ~trial_type.isin(@spectrials) ')

    print(qs_es)
    #' and prev_trial_type != "error_clamp"'

    dfsw_pre = df_all_multi_tsz.query(qs_es).copy()
    assert not dfsw_pre.duplicated().any()
    print(len(dfsw_pre))


    c = dfsw_pre['trial_type'] == 'error_clamp'
    dfsw_pre['perturbation_tmp'] = np.nan
    dfsw_pre.loc[~c,'perturbation_tmp'] = dfsw_pre.loc[~c,'perturbation']
    dfsw_pre['perturbation_tmp'] = dfsw_pre['perturbation_tmp'].\
        fillna(None, method='ffill').astype(int)

    # c2 = c | c.shift(-1) | c.shift(-2) | c.shift(2)  | c.shift(1)
    # dfsw_pre.loc[c2,['trialwb','trial_type','perturbation','perturbation_tmp']].iloc[:40]

    grp_subj = dfsw_pre.groupby(['subject'])

    dfsw_pre['perturbation_diff']     = grp_subj['perturbation_tmp'].diff()
    dfsw_pre['perturbation_diff_abs'] = grp_subj['perturbation_tmp'].diff().abs()
    # it is important to do it after computing diff,
    # otherwise 1st block has all pert_diff = NaN
    dfsw_pre = dfsw_pre.query('block_ind >= 0')
    grp_subj_bi = dfsw_pre.groupby(['subject','block_ind'],
                                   group_keys=True)

    assert not dfsw_pre.query('block_ind == 0 and trialwb == 0')['perturbation_diff'].isna().any()

    # trial order dependent
    cols_dif = ['perturbation_diff',  'perturbation_diff_abs',
                    'dist_rad_from_prevtgt2', 'prev_block_ind_diff']
    cols_ctx_change_tiodep = cols_dif + coln_ctx_closeness

    #grp_subj_bi.size()


    ##################   Set end blocks
    def f(subdf):
        # here we take the first row of block, NOT the first non-nan row
        row0 = subdf.iloc[0]
        assert row0['trialwb'] == 0
        # to select es value trial
        qs = '~time_lh.isna() and ~trial_type.isin(@spectrials)'
        subj,bi = subdf.name
        # TODO: do pauses in the middle receive block_ind?

        #if bi % 8 == 4:
        subdf15 = subdf.query('~trial_type.isin(@spectrials)')
        cjmp = subdf15['trial_index'].diff() > 1
        ti_jumps = subdf15.loc[cjmp, 'trial_index']
        if len(ti_jumps):
            ti_jump = ti_jumps.values[0]
            #print(subdf15.loc[cjmp, 'trialwb'].iloc[0])
            #print('ti_jump = ',ti_jump)
        else:
            ti_jump = 1e6
        qs += f' and trial_index < {ti_jump}'

        subdf2 = subdf.query(qs)
        #print(len(subdf), len(subdf.query('~time_lh.isna()' )),
        #                      len(subdf2))
        ind = subdf2['trial_index'].nlargest(3).index[0]
        if np.isnan( subdf2.loc[ind,'prev_error'] ):
            ind = subdf2['trial_index'].nlargest(3).index[1]

        #print(ind)
        res = subdf2.loc[ind]
        for col in coln_ctx_closeness:
            res[col] = True
        for col in cols_dif:
            res[col] = 0

        return res

    df_end_block = grp_subj_bi.apply(f)

    print ( df_end_block.groupby('trialwb').size() )
    df_end_block = df_end_block.drop(labels=['subject','block_ind'], axis=1)

    infs_eb = np.isinf(df_end_block['err_sens'])
    assert infs_eb.sum() < 16, infs_eb.sum()

    # we accept zero errors either if had super
    # easy or if prev error is NaN
    infrows = df_end_block.loc[infs_eb]
    easy = (infrows['tgti_to_show'] == 1) & (infrows['perturbation'] == 0 )
    prevnan =  infrows['prev_error'].isna()
    print('neasy = {},  nprev_err_nan = {}'.format(easy.sum(),  prevnan.sum()) )
    assert (easy | prevnan).all()

    print(f'removing {infs_eb.sum()} infs')
    df_end_block_clean = df_end_block.loc[~infs_eb]
    #df_end_block.loc[infs_eb, cols]

    ##################################################
    ##########  Set beg blocks
    ##################################################
    print('Starting getting beginning blocks')
    # if it is change of pert I'd use second
    def f(subdf):
        # here we take the first row of block, NOT the first non-nan row
        row0 = subdf.iloc[0]
        pertdif = row0['perturbation_diff_abs']
        assert row0['trialwb'] == 0
        assert not np.isnan(pertdif)

        # to select es value trial
        qs = '~time_lh.isna() and ~trial_type.isin(@spectrials)'
        subdf2 = subdf.query(qs)
        pert_changed = pertdif > 1e-4

        # ideally I'd do it like that but since for the first
        # trial of block quite often the last before that was
        # an error clamp, they will have inf err sens
    #     if pert_changed:
    #         ind = subdf2['trial_index'].nsmallest(2).index[1]
    #     else:
    #         ind = subdf2['trial_index'].nsmallest(1).index[0]

        # always take err sens calc based on what happens between
        # second and first trial
        sm = subdf2['trial_index'].nsmallest(2)
        if first_trial_to_take == 'first':
            ind = sm.index[0]
        elif first_trial_to_take == 'second':
            ind = sm.index[1]
        if first_trial_to_take == 'adaptive':
            # if tgt is same, we'll notice it right away
            if row0['prev_ctx_tgt_same']:
                ind = sm.index[0]
            else:
                ind = sm.index[1]

        #print(ind)
        res = subdf2.loc[ind]
        for col in cols_ctx_change_tiodep:
            res[col] = row0[col]
        return res

    #     res = np.zeros(len(x))
    #     display(x)
    #     print(len(x), len(x.query(qs_pertchange)), len(x.query(qs_pertstay)))
    #     inds_pc = x.query(qs_pertchange)['trial_index'].nsmallest(2).index[[1]]
    #     #res[inds]
    #     inds_ps = x.query(qs_pertstay)['trial_index'].nsmallest(1).index[[0]]
    #     res = pd.concat( x[inds_pc], x[inds_ps] )
    #     #df['Low'].nsmallest(2).index[1]
    #     return res#['time_lh']
    dfsw_care = grp_subj_bi.apply(f)

    dfsw_care = dfsw_care.drop(labels=['subject','block_ind'], axis=1)
    print( dfsw_care.groupby(by=['subject'],axis='index').size().describe() )

    assert not dfsw_care['err_sens'].isna().any()

    assert dfsw_care.groupby(['subject']).size().max() == 63
    assert dfsw_care.groupby(['subject','trial_index']).size().max() == 1

    dfsw_care['perturbation_diff_abs'] = dfsw_care['perturbation_diff_abs'].astype(int)

    infs = np.isinf(dfsw_care['err_sens'])
    infrows = dfsw_care.loc[infs]
    easy = (infrows['tgti_to_show'] == 1) & (infrows['perturbation'] == 0 )
    prevnan =  infrows['prev_error'].isna()
    print('neasy = {},  nprev_err_nan = {}'.format(easy.sum(),  prevnan.sum()) )
    easypnan = (easy | prevnan)
    if easypnan.sum() > 0:
        assert easypnan.all()

    pct_inf = 100 * infs.sum() / len(dfsw_care['err_sens'])
    print('pct_inf = {:.2f}% = {} entries'.format(pct_inf, infs.sum()) )
    # check they they are really invalid and we shoud discard them
    #assert dfsw_care.loc[infs,'prev_error'].isna().all()
    dfsw_care_clean = dfsw_care[~infs]

    ########################  Concat

    print( len(dfsw_care), len(df_end_block_clean) )

    dfes_sw = pd.concat( [df_end_block_clean, dfsw_care_clean] )
    assert not dfes_sw.duplicated().any()

    dfes_sw['dist_rad_from_prevtgt2'] = \
        dfes_sw['dist_rad_from_prevtgt2'].astype(float).astype(int)


    # look at infs
    #cols = ['trial_index', 'trialwb','tgti_to_show',
    #        'perturbation','perturbation_diff',
    #       'prev_error','prev_trial_type','err_sens',
    #        'time_lh',coln_error]
    #dfsw_care.loc[infs ,cols]

    ## look at pre infs
    #dfsw_pre.query('subject == "2023-SE1-001" and trial_index in [189,190]')

    return dfes_sw


def shiftTrials(dfcc_all_sub, dfcc_all, shift=1):
    if isinstance(shift, int):
        shift = [shift]

    rows = []
    for i,row in dfcc_all_sub.iterrows():
        tind = row['trial_index']
        subj = row['subject']

        row0 = dfcc_all.query('subject == @subj and trial_index == @tind')
        
        #row0['_sh'] = 0
        #rows += [row0]
        for sh in shift:
            tind1 = tind + sh
            df_ = dfcc_all.query('subject == @subj and trial_index == @tind1')
            if len(df_):
                row1 = dict( df_.iloc[0] )
            else:
                continue

            row1['_sh'] = sh
            rows += [row1]
            # only mvt phase
            #dfpretraj = dfc_all.query('subject == @subj')
            #grp_perti = dfpretraj.groupby(['trial_index'])
    return pd.DataFrame(rows).sort_values(['trial_index','_sh'])

def plotAllTrajs(dfcc_all_sub, dfc_all, params, exitpt_col='time_lh',
                traj_to_plot= ['feedback'], separate_plots = False,
                verbose=0):
    assert isinstance(params,dict)
    for i,row in dfcc_all_sub.iterrows():
        tind = row['trial_index']
        subj = row['subject']
        # only mvt phase
        #dfpretraj = dfc_all.query('subject == @subj')
        #grp_perti = dfpretraj.groupby(['trial_index'])
        #lf = row2multierr(row, dfpretraj, grp_perti,
        #                  home_position,
        #            target_coords, params, revert_pert=0)
        if separate_plots:
            plt.figure()

        ax = plt.gca()
        plotTraj3(ax, row, dfc_all, df_es=None, colscols=None, params = params,
                  exitpt_col = exitpt_col, traj_to_plot=traj_to_plot,
                 verbose=verbose)


def plotAllTrajCouples(dfcc_all_sub, dfcc_all, dfc_all, params, exitpt_col='time_lh',
                traj_to_plot= ['feedback'],
                verbose=0):

    for i,row0 in dfcc_all_sub.iterrows():
        tind0 = row0['trial_index']
        subj = row0['subject']
        # only mvt phase
        if tind0 == 0:
            tinds == [tind0]
        else:
            tinds = [tind0-1, tind0]

        # curren traj has bigger markers
        mszs = [2,6]
        plt.figure()

        ttl = ''
        for tindi, tind in enumerate(tinds):
            df_ = dfcc_all.query('subject == @subj and trial_index == @tind')
            if len(df_) == 0:
                if row0['prev_trial_type'] == 'pause':
                    continue
                else:
                    print(row)
                    raise ValueError( f'{len(df_)}  {subj}  {tind}  {tind0} ' )
            row = df_.iloc[0]

            ax = plt.gca()
            plotTraj3(ax, row, dfc_all, df_es=None, colscols=None, params = params,
                      exitpt_col = exitpt_col, traj_to_plot=traj_to_plot,
                     verbose=verbose, markersize = mszs[tindi] )
            ttl += f'S{subj[-3:]} ti={tind} ({row["trialwb"]}); pert={row["perturbation"]};\n'

        ax.set_title(ttl[:-1] )

def plotAllTrajTimeRes(dfcc_all_sub, dfc_all):
    for i, row in dfcc_all_sub.iterrows():
        tind = row['trial_index']
        subj = row['subject']
        qs = 'subject == @subj and trial_index == @tind'
        dftr = dfc_all.query(qs).copy()

        #dftr['jax1_std'] = dftr['jax1'].expanding().std()
        #dftr['jax2_std'] = dftr['jax2'].expanding().std()
        #row = dfcc_all.query(qs).iloc[0]

    #     thr = 1e-3
    #     tmin1 = dftr.query('jax1_std > @thr')['time_since_trial_start'].min()
    #     tmin2 = dftr.query('jax2_std > @thr')['time_since_trial_start'].min()
    #     print(tmin1, tmin2, row['time_lh'])
    #     tmin = max(tmin1,tmin2)


    #     assert abs(tmin - row['time_tmstart']) < 1e-10
        #dftr['time_since_trial_start']

        ax = None # plt.gca()

        colnsy = ['jax1', 'jax2', ]
        stdcols = ['jax1_std', 'jax2_std']
        if stdcols[0] in dftr.columns:
            colnsy += stdcols
        dftr.plot(ax = ax, x='time_since_trial_start',
                  y=colnsy, figsize=(4,3))
        plt.vlines([row['time_lh'], row['time_tmstart2'] ], -0.5,0.2)
        print(row['time_lh'], row['time_tmstart2'])
        plt.xlim(row['time_lh'] - 0.1, 0.5)
        plt.legend(loc='lower right')

        ttl = f'S{subj[-3:]} ti={tind} ({row["trialwb"]}); pert={row["perturbation"]}'
        plt.title(ttl )


def printRejectInfo(df, coln = 'error_lh2_ang_deg', thr = 1.):
    c = df[coln].abs() < thr
    neasy      = (c & df['is_easy']).sum()
    neasy_wide = (c & df['is_easy_wide']).sum()
    n = c.sum()
    prop = n / len(df)
    prop_easy = neasy / len(df)
    prop_easy_wide = neasy_wide / len(df)
    print(f'thr={thr:.3f}, prop of all smaller={prop * 100:4.1f}%, '
         f'easy={prop_easy * 100:4.1f}% easy_wide={prop_easy_wide * 100:.1f}%')

    if 'err_sens' in df.columns:
        print('Err sens abs info of truncated')
        inf = np.isinf( df['err_sens'] )
        print( df.loc[c & (~inf), 'err_sens'].abs().describe()  )


def myttest(df_, qs1, qs2, varn, alt = ['two-sided','greater','less'], paired=False,
            cols_checkdup = []):
    # cols_checkdup = ['subject','trials'])
    from pingouin import ttest
    ttrs = []
    if isinstance(alt,str):
        alt = [alt]

    try:
        df1 = df_.query(qs1)
    except Exception as e:
        print(f'myttest: Exception {e} for {qs1}')
        raise ValueError(f'bad qs {qs1}')

    try:
        df2 = df_.query(qs2)
    except Exception as e:
        print(f'myttest: Exception {e} for {qs2}')
        raise ValueError(f'bad qs {qs2}')

    if len(cols_checkdup):
        assert not df1.duplicated(cols_checkdup).any()
        assert not df2.duplicated(cols_checkdup).any()
    for alt_ in alt:
        ttr = ttest(df1[varn].values, 
                    df2[varn].values, alternative=alt_, paired=paired)
        ttrs += [ttr]
    ttrs = pd.concat(ttrs)
    ttrs['paired'] = paired
    ttrs = ttrs.rename(columns={'p-val':'pval'})
    ttrs['varn'] = varn
    ttrs['qs1'] = qs1
    ttrs['qs2'] = qs2
    ttrs['N1'] = len(df1)
    ttrs['N2'] = len(df2)
    return ttrs

def compare0(df, varn, alt=['greater','less'], cols_addstat = []):
    from pingouin import ttest
    '''
        returns ttrs (not only sig)
    '''
    if isinstance(alt,str):
        alt = [alt]
    ttrs = []
    for alt_ in alt: #, 'two-sided']:
        ttr = ttest( df[varn], 0, alternative = alt_ )
        ttr['alt'] = alt_
        ttr['val1'] = varn
        ttr = ttr.rename(columns ={'p-val':'pval'})

        for coln in cols_addstat:
            ttr[coln + '_mean'] = df[coln].mean()
            ttr[coln + '_std' ] = df[coln].std()

        ttrs += [ttr]
    ttrs = pd.concat(ttrs, ignore_index = 1)
    ttrs['N1'] = len(df)
    decorateTtestRest(ttrs)
    return ttrs

def decorateTtestRest(ttrs):
    ttrs['ttstr']  = ''
    def f(row):
        alt = row['alternative']
        v1 = row['val1']
        v2 = row.get('val2', 0)
        if alt == 'greater':
            s = f'{v1} > {v2}'
        elif alt == 'less':
            s = f'{v1} < {v2}'
        elif alt == 'two-sided':
            s = f'{v1} != {v2}'
        return s

    ttrs['ttstr']  = ttrs.apply(f,axis=1)

def comparePairs(df_, varn, col, 
                 alt = ['two-sided','greater','less'], paired=False,
                 pooled = 2, updiag = True, qspairs = None):
    '''
    returns sig,all
    '''
    assert isinstance(paired, bool)
    assert len(df_), 'Given empty dataset'
    ttrs = []
    if int(pooled) == 1:
        ttrs = comparePairs_(df_,varn,col, pooled=True, 
                             alt=alt, paired=paired, qspairs = qspairs)
        ttrs += [ttrs]
    if (int(pooled) == 0) or (pooled == 2):
        ttrs_np = comparePairs_(df_,varn,col, pooled=False, 
            alt=alt, paired=paired, updiag = updiag, qspairs = qspairs)
        ttrs += [ttrs_np]
    ttrs = pd.concat(ttrs, ignore_index=1)

    ttrssig = ttrs.query('pval <= 0.05').copy()
    ttrssig['starcode'] = '*'
    ttrssig.loc[ ttrssig['pval'] <= 0.01  , 'starcode'] = '**'
    ttrssig.loc[ ttrssig['pval'] <= 0.001 , 'starcode'] = '***'
    ttrssig.loc[ ttrssig['pval'] <= 0.0001, 'starcode'] = '****'

    if len(ttrssig) == 0:
        return None,ttrs

    decorateTtestRest(ttrs)
    decorateTtestRest(ttrssig)
    return  ttrssig, ttrs

def comparePairs_(df_, varn, col, pooled=True , alt=  ['two-sided','greater','less'], paired=False, updiag = True, qspairs = None):
    '''
    all upper diag pairs of col values
    '''
    from behav_proc import myttest
    assert len(df_)

    ttrs = []

    if isinstance(col, (list,np.ndarray) ):
        cols = col
    else:
        cols = [col]

    if not pooled:
        if col is not None:
            s1 = ['subject'] + cols +  [varn]
            s2 = ['subject'] +  cols
        else:
            s1 = ['subject',  varn]
            s2 = ['subject' ]
        df_ = df_[s1].groupby(s2,observed=True).mean(numeric_only=1).reset_index()

    #print(df_.groupby()

    #colvals = colvals[~np.isnan(colvals)]
    if qspairs is None:
        colvals = df_[col].unique()
        for cvi,cv in enumerate(colvals):
            #vals1 = df.query('@col == @cv')
            if updiag:
                loop2 = enumerate(colvals[cvi+1:])
            else:
                loop2 = enumerate(colvals)
            for cvj,cv2 in loop2:
                # need if not updiag
                if cv == cv2:
                    continue

                #vals2 = df.query('@col == @cv2')
                cv_ = cv
                cv2_ = cv2
                if isinstance(cv,str):
                    cv_ = '"' + cv + '"'
                    cv2_ = '"' + cv2 + '"'
                qs1 = f'{col} == {cv_}'
                qs2 = f'{col} == {cv2_}'
                ttrs_ = myttest(df_,qs1, qs2, varn, alt=alt, paired=paired)
                ttrs_['val1'] = cv
                ttrs_['val2'] = cv2
                ttrs += [ttrs_]
    else:
        for qs1,qs2 in qspairs:
            ttrs_ = myttest(df_,qs1, qs2, varn, alt=alt, paired=paired)
            ttrs_['val1'] = qs1
            ttrs_['val2'] = qs2
            ttrs += [ttrs_]

    ttrs = pd.concat(ttrs, ignore_index=1)
    ttrs['pooled'] = pooled
    return ttrs

def reshiftPi(df,coln):
    def f(x):    
        if x > np.pi :
            x -= 2*np.pi
        elif x < -np.pi :
            x += 2*np.pi
        return x
    df[coln] = df[coln].apply(f)

        
def addWindowCols_DEPRECATED(df, cols = ['error', 'err_sens','error_pscadj', 'error_change'], 
                  window_sizes =  [3, 5,10,15], shift = True):
    #, 'org_feedback_pscadj'] ):
    assert not df.duplicated(['subject','trials']).any()      # maybe I can get oveer it but not now
    dfc = df.copy()
    dfc = dfc.sort_values(
        ['trial_group_col_calc', 'pert_seq_code', 'subject', 'trials'])
    #df['env'] = df['environment'].apply(lambda x: envcode2env[x])

    dfc['error_change'] = dfc.groupby(['subject'])['error'].diff()
    #dfc['error_prod']   = dfc.groupby(['subject'])['error'] * dfc.groupby(['subject']).shift(1)['error']

    adjustErrBoundsPi(dfc, ['feedback',   'error',      'prev_error', 'error_pscadj', 'error_change'] )

    #reshiftPi(dfc,'feedback')
    #reshiftPi(dfc,'error')
    #reshiftPi(dfc,'prev_error')
    #reshiftPi(dfc,'error_pscadj')
    #reshiftPi(dfc,'error_change')

    #dfc['trialwpertstage_wb'] = dfc['trialwpertstage_wb'].\
    #    where(dfc['environment'] == 0, dfc['trialwb'])
    #dfc['trialwpertstage_wb'] = dfc['trialwpertstage_wb'].astype(int)

    # nan-ify after pause
    dfc.loc[dfc['trialwb'] == 0, 'err_sens'] = np.nan

    # it's important to do things separately for different pert_seq_code althouth right here maybe the sorting wrt it is not so important
    grp = dfc.\
        groupby(['pert_seq_code', 'subject', 'trial_group_col_calc'],
               observed=True)

    
    for std_mavsz_ in window_sizes:
        #vars_to_plot = ['err_sens','error', 'org_feedback']
        for varn in cols:
            for g,gi in grp.groups.items():
                if shift:
                    dfc.loc[gi,f'{varn}_std{std_mavsz_}'] = dfc.loc[gi,varn].shift(1).rolling(std_mavsz_).std()   
                    dfc.loc[gi,f'{varn}_mav{std_mavsz_}'] = dfc.loc[gi,varn].shift(1).rolling(std_mavsz_).mean()   
                else:
                    dfc.loc[gi,f'{varn}_std{std_mavsz_}'] = dfc.loc[gi,varn].rolling(std_mavsz_).std()   
                    dfc.loc[gi,f'{varn}_mav{std_mavsz_}'] = dfc.loc[gi,varn].rolling(std_mavsz_).mean()   
                    
            dfc[f'{varn}_invstd{std_mavsz_}'] = 1/dfc[f'{varn}_std{std_mavsz_}']
            dfc[f'{varn}_var{std_mavsz_}']    = dfc[f'{varn}_std{std_mavsz_}'] ** 2
            dfc[f'{varn}_mavsq{std_mavsz_}']  = dfc[f'{varn}_mav{std_mavsz_}'] ** 2
            dfc[f'{varn}_mav_d_std{std_mavsz_}']  = dfc[f'{varn}_mav{std_mavsz_}'].abs() / dfc[f'{varn}_std{std_mavsz_}']
            dfc[f'{varn}_mav_d_var{std_mavsz_}']  = dfc[f'{varn}_mav{std_mavsz_}'].abs() / dfc[f'{varn}_var{std_mavsz_}']
            dfc[f'{varn}_Tan{std_mavsz_}']    = dfc[f'{varn}_mavsq{std_mavsz_}'] / dfc[f'{varn}_var{std_mavsz_}']

    return dfc.copy()

def truncateNIHDfFromErr(df, err_col = 'error', varn = 'err_sens', mult = 1):
    # add hit_mestd1 column and add 
    # err_sens_trunc column that has nan where hit happens

    # estimate error at second halfs of init stage
    qs_initstage = 'pert_stage_wb.abs() < 1e-10'
    df_init = df.query(qs_initstage + ' and trialwb >= 10')
    #grp = df_init.groupby(['subject','pert_stage'])

    stds = df_init.groupby(['subject'], observed=True)[err_col].std()#.std()
    mestd = stds.mean()
    if len(stds) > 1:
        stdstd = stds.std()
    else:
        stdstd = None
    #print('mestd mean = {} [rad], std ={}'.format(mestd,  stdstd))
    if err_col.find('_deg') < 0:
        if stdstd is not None:
            std_ = stdstd*180/np.pi
        else:
            std_ = None
        print('mestd mean = {} [deg], std ={} [deg]'.format(mestd*180/np.pi, std_))
    else:
        print('mestd mean = {} [deg], std ={} [deg]'.format(mestd,  stdstd))

    stds = stds.to_frame().reset_index().rename(columns={err_col: err_col + '_initstd'})
    df_ = df.merge(stds, on='subject')    

    shiftsz = 1
    df_['hit_mestd1'] = df_[err_col].shift(shiftsz).abs() <= \
            df_[err_col + '_initstd'] * mult 
    df_[varn + '_trunc'] = df_[varn]
    df_.loc[df_['hit_mestd1'], varn + '_trunc'] = np.inf
    
    return df_  # mamba install seaborn ipykernel shapely matplotlib statsmodels pyqt pingouin scipy pandas


def truncateNIHDfFromErr2(df_wthr, mults):
    for multi, mult in enumerate(mults):
        df_ = df_wthr.copy()
        df_wthrdf_ = df_[df_.error.shift(1).abs() >= df_['error_initstd'] * mult ].copy()
        #df_ = df_.query('error_deg.shift(1).abs() >= @thr')
        #thr_s = '{:.3f}'.format(thr)
        #thr_s = f'{maxmult}*mestd/{ NN - (thri)}'
        #thr_s = f'mestd/{ NN - (thri)}'
        thr_s = f'mestd*{mults[multi]}'
        df_wthrdf_['thr'] = thr_s
        dfs += [df_wthrdf_]
        print(thr_s, len(df_wthrdf_), len(df_))
        
    dfall_notclean = pd.concat(dfs)
    dfall = truncateDf(dfall_notclean, 'err_sens', q=0.0, infnan_handling='discard', 
                       cols_uniqify = ['subject','env','thr'],
                       verbose=True)

    return dfall

def calcESthr(df, mult):
    assert not np.isinf(df['err_sens']).any()
    dfni = df                                           
    dfni_d = dfni.groupby(['subject'],observed=True)\
        ['err_sens'].describe().reset_index()
    ES_thr = dfni_d[dfni_d.columns[1:]].mean().to_dict()['std'] * mult
    #ES_thr_single = ES_thr
    return ES_thr

def truncateNIHDfFromES(df_wthr, mult, ES_thr=None):
    # remove trials with error > std_mult * std of error
    std_mult = mult

    dfni = df_wthr[~np.isinf(df_wthr['err_sens'])]
    if ES_thr is None:
        ES_thr = calcESthr(dfni, mult)
        print(f'ES_thr (recalced) = {ES_thr}')

    dfni_g = dfni.query('err_sens.abs() <= @ES_thr')
    nremoved_pooled = len(dfni) - len(dfni_g)

    sz = dfni.groupby(['subject'],observed=True).size()
    sz_g = dfni_g.groupby(['subject'],observed=True).size()
    mpct = ((sz - sz_g) / sz).mean() * 100
    print(f'Mean percentage of removed trials = {mpct:.3f}%, '
          f'pooled = {nremoved_pooled / len(dfni) * 100:.3f}%')

    dfall = dfni_g.copy()
    dfall['thr'] = "mestd*0" # just for compat
    return dfall

# mamba install seaborn ipykernel shapely matplotlib statsmodels pyqt pingouin scipy pandas

#def addTrigPresentCol_NIH(df, events, environment=None):
#    '''
#    events is 2-dim
#    '''
#    assert events.ndim == 2
#    from base2 import int_to_unicode
#    from Levenshtein import editops
#    assert (np.diff( np.array(df.index) ) > 0).all()
#
#
#    # remove some of the behav inds
#    # WARNING: this will fail (give empty)if epochs are based of time_locked=feedback
#    # (or maybe it was only in old code?)
#    meg_targets_ev = events[:, 2].copy()
#    #target_inds_behav = np.array( df['target_inds'] )
#
#    target_codes_behav = np.array( df['target_codes'] )
#
#    assert len(target_codes_behav)
#    assert len(meg_targets_ev)
#    print(  len(target_codes_behav) , len(meg_targets_ev) )
#    #if environment is None:
#    #    environment = df['environment']
#
#    # for different environments target signals stored in .fif have different codes (20+ vs 30+) so
#    # we convert to unified indices
#
#    #codes_allowed = [20, 21, 22, 23, 25, 26, 27, 28]
#    from config2 import event_ids_tgt
#    nbad = (~np.isin(meg_targets_ev, event_ids_tgt) ).sum()
#    print(nbad , 'non target code events' )
#    assert nbad == 0
#    #for env,envcode in env2envcode.items():
#    #    #trial_inds = np.where(environment == envcode)[0]
#    #    trial_inds = np.where( np.isin(meg_targets_ev,
#    #        stage2evn2event_ids['target'][env] ) )[0]
#    #    # it uses that target codes are conequitive integers
#    #    meg_targets_ev[trial_inds] = meg_targets_ev[trial_inds] - env2subtr[env]
#
#    # one can have less triggers than behav
#    changes = editops(int_to_unicode(target_codes_behav),
#                        int_to_unicode(meg_targets_ev))
#    # we have missing triggers in MEG file so we delete stuff from behav file
#    delete_trials = [change[1] for change in changes] # these are indices of rows
#    print( changes )
#    df['trigger_present'] = True
#    colind = df.columns.get_loc('trigger_present')
#    df.iloc[delete_trials, colind] = False
#
#    assert df['trigger_present'].sum() > 0
#    print(df['trigger_present'].astype(float).describe() )
#
#    target_codes_behav2 = df.query('trigger_present == True')['target_codes']
#
#    # respects ordering
#    if np.array_equal(meg_targets_ev, target_codes_behav2):
#        if len(delete_trials):
#            print(f'addTrigPresentCol: {len(delete_trials)} triggers missing for {len(delete_trials)} trials')
#    else:
#        #warnings.warn('MEG events and behavior file do not match')
#        raise ValueError('MEG events and behavior file do not match')
#
#    return df

def checkErrBounds(df, cols=['error','prev_error','correction',
                             'error_deg','prev_error_deg','belief','perturbation' ]):
    # ,'target_locs' can be > Pi because they are 90 + smth, with smth \in [0,pi]
    # not feedback and org_feedback or target_locs because they are referenced not to 0 in NIH data
    bd = np.pi 
    bd_deg = 180
    badcols = []
    badcols_tuples = []
    for col in cols:
        if col in df.columns:
            mx = df[col].abs().max()
            if col.endswith('deg'):                
                if  mx > bd_deg:
                    badcols += [col]
                    badcols_tuples += [(col,mx, np.sum(df[col].abs() > bd_deg) )]

            else:
                if mx > bd:
                    badcols += [col]
                    #badcols_tuples += [(col,mx)]
                    badcols_tuples += [(col,mx, np.sum(df[col].abs() > bd) )]
    print('Bad columns found: ', badcols_tuples)
    return badcols


def adjustErrBoundsPi(df, cols):
    bd = np.pi 
    bd_deg = 180
    for col in cols:
        mx = df[col].abs().max()
        if  mx > bd_deg:
            bd_eff =  bd_deg
        else:
            bd_eff =  bd
        print('adjustErrBoundsPi', col, bd_eff)
            
        def f(x):    
            if x > bd_eff:
                x -= bd_eff * 2
            elif x < -bd_eff:
                x += bd_eff * 2
            return x
        df[col] = df[col].apply(f)


#####################

def corrMean(dfallst, coltocorr = 'trialwpertstage_wb', 
             stagecol = 'pert_stage_wb', 
             coln = 'err_sens', method = 'pearson', covar = None):
    '''
    compute correlation with p-value within subject and also mean across
    does it within condition defined by stagecol

    returns:
        tuple of correlation dataframes
        first is mean within across subjects, second with separate subjects
        
    '''
    # corr or partial correlation within participant, averaged across participants
    import pingouin as pg
    assert coltocorr in dfallst.columns
    assert stagecol in dfallst.columns
    assert coln in dfallst.columns

    def f(df_):
        try:
            if covar is None:
                r = pg.corr( df_[coltocorr], df_[coln],  method=method)
            else:
                r = pg.partial_corr( df_, coltocorr, coln, covar, method=method)
            r['method'] = method
            r['mean_x'] = df_[coltocorr].mean()
            r['mean_y'] = df_[coln].mean()
            r['std_x'] = df_[coltocorr].std()
            r['std_y'] = df_[coln].std()
        except ValueError as e:
            return None
        return r

    groupcols0 = []
    if 'thr' in dfallst.columns:
        groupcols0 = ['thr'] 
    groupcols = groupcols0 + [stagecol]
    groupcols2 = groupcols0 + ['subject', stagecol]

    # separate subjects
    corrs_per_subj_me0 = dfallst.groupby(groupcols2, observed=True).apply(f)
    corrs_per_subj_me0['method'] = method
    

    # mean over subjects
    corrs_per_subj_me = corrs_per_subj_me0.rename(columns={'p-val':'pval'})
    corrs_per_subj_me = corrs_per_subj_me.\
        groupby(groupcols, observed=True)[['r','pval',
            'mean_x','mean_y','std_x','std_y']].mean(numeric_only = 1)
    corrs_per_subj_me['method'] = method
    corrs_per_subj_me['varn'] = covar

    return corrs_per_subj_me, corrs_per_subj_me0


def formatRecentStatVarnames(isec, histlen_str=' (histlen='):
    '''
    takes list of varnames, outputs list of nice varnames
    '''
    isec_nice = []
    for s in isec:
        s2 = s.replace('error_pscadj_abs','Error magnitude')\
            .replace('error_pscadj','Signed error')\
            .replace('_Tan',' mean^2/var' + histlen_str)\
            .replace('_mavsq',' mean^2' + histlen_str)\
            .replace('_invstd',' 1/std' + histlen_str)\
            .replace('_mav_d_std',' mean/std' + histlen_str)\
            .replace('_std',' std' + histlen_str)\
            .replace('_mav_d_var',' mean/std' + histlen_str) 
        if len(histlen_str):
            s2 += ')' 
        isec_nice.append(s2 )
    return isec_nice

def compDf(dftmp1,dftmp2,cols, cols_to_comp=None, suffix_second='_ev'):
    '''
    to compared dataframes with same columns side by side
    '''
    #comp_numeric = False):
    assert len(dftmp1) == len(dftmp1), (len(dftmp1) , len(dftmp1))
    from pandas.api.types import is_string_dtype, is_numeric_dtype
    assert set(cols).issubset(dftmp1.columns), set(cols) ^ set(dftmp1.columns) 
    assert set(cols).issubset(dftmp2.columns)
    
    dftmp1 = dftmp1.reset_index(drop=True)#.drop(columns=['index'])
    dftmp2 = dftmp2.reset_index(drop=True)#.drop(columns=['index'])
    dftmp2= dftmp2.rename(dict(zip(dftmp2.columns, [col + '_ev' for col in dftmp2.columns] ))  ,axis=1)
    dftmp = pd.concat([dftmp1,dftmp2],axis=1)

    compcols = []
    if cols_to_comp is None:
        z = zip(dftmp1.columns,dftmp2.columns)
    else:
        z = zip(cols_to_comp, [col + suffix_second for col in cols_to_comp] )
    for col1,col2 in z:
        #if comp_numeric or (not is_numeric_dtype(dftmp[col1]) ):
        col_ = col1 + '_eq'
        dftmp[col_] = dftmp[col1] == dftmp[col2]
        compcols += [col_,col1,col2]

    restcols = [col for col in cols if (not col  in compcols ) ]
    z2 = zip(restcols, [col + suffix_second for col in restcols] )
    # double list comprehension!
    #colpairs = [element for pair in zip(compcols,dftmp1.columns,dftmp2.columns) for element in pair]
    colpairs = compcols
    colpairs += [element for pair in z2 for element in pair]
    return dftmp[colpairs]

def readTrialInfoSeqParams(file_path):
    import pandas as pd

    # Initialize lists to hold extracted data
    trial_indices = []
    trial_types = []
    target_indices = []
    vfts = []
    special_block_types = []
    block_inds = []

    # Read and process the file
    with open(file_path, 'r') as file:
        content = file.readlines()

        # Find the starting point for trial information
        start_index = 0
        for i, line in enumerate(content):
            if line.strip() == "# trial_infos =":
                start_index = i + 1
                break

        # Extract data from the relevant lines
        for line in content[start_index:]:
            parts = line.strip().split(' = ')
            if len(parts) == 2:
                trial_index, data = parts
                data_parts = data.split(', ')
                if len(data_parts) == 5:
                    trial_indices.append(int(trial_index[1:]) )
                    trial_types.append(data_parts[0])
                    target_indices.append((data_parts[1]))
                    vfts.append(str(data_parts[2]))
                    special_block_types.append(data_parts[3])
                    block_inds.append(int(data_parts[4]))

    # Create a DataFrame from the extracted data
    df_trialinfoseq_params = pd.DataFrame({
        'trial_index': trial_indices,
        'trial_type': trial_types,
        'tgti_to_show': target_indices,
        'vis_feedback_type': vfts,
        'special_block_type': special_block_types,
        'block_ind': block_inds
    })
    return df_trialinfoseq_params


def checkSavingsNIH(dfall, method = 'spearman' ):
    s1,s2 = set(['pert_stage','err_sens','trial_index','pert_stage']), set(dfall.columns) 
    assert s1 < s2, ( s1 - s2 )

    print(method)
    corrs_per_subj_me_,corrs_per_subj  = corrMean(dfall, 
                stagecol = 'pert_stage', coln='err_sens' ,method=method)

    # show stat signif
    stage_pairs = [(1,6),(3,8)]
    ttrs2 = []
    for s1,s2 in stage_pairs: 
        lst = [s1,s2]
        df_ = corrs_per_subj.reset_index().query('pert_stage.isin(@lst)')
        if len(df_) == 0:
            print(f'empty for {lst}')
        ttrs_sig, ttr = comparePairs(df_, 'r', 'pert_stage',
                                     paired=True)
        ttr['stage_pair'] = f'{s1}-{s2}'
        ttrs2 += [ttr]
    ttrs2 = pd.concat(ttrs2)
    display( ttrs2.query('pval <= 5e-2') )

    ###########################

    stage_pairs_nice = {"1-6":'first and last', "3-8":'second and third'}
    #display(ttrs2)
    
    some = False
    for irow,row in ttrs2.query('alternative == "two-sided"').iterrows():
        sp = row['stage_pair']
        pv=row['pval']
        T=row['T']

        s = ''
        if pv > 0.05:
            s = 'not '
        else:
            some |= True
        #print(sp,pv)
        print('ES during {} perturbations are {}significantly different, t={:.2f}, p-value = {:.2e}.'.\
                  format(stage_pairs_nice[sp],s,T,pv) )
    if not some:
        print(f'\n\nNo savings (we have used {method}) !')


    ##################   let's check for other two pairs as well
    stage_pairs = [(1,3),(6,8)]
    print(stage_pairs)
    ttrs2 = []
    for s1,s2 in stage_pairs: 
        lst = [s1,s2]
        ttrs_sig, ttr = comparePairs(corrs_per_subj.reset_index().query('pert_stage.isin(@lst)'), 'r', 'pert_stage',
                                     paired=True)
        ttr['stage_pair'] = f'{s1}-{s2}'
        ttrs2 += [ttr]
    ttrs2 = pd.concat(ttrs2).query('alternative != "two-sided"')
    display( ttrs2.query('pval <= 1e-2')[['pval','ttstr']] )#

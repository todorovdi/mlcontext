import os
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
 'trialwtgt_wpert_we']

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

        if skip_existing and ('block_name' not in df_all.columns):
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
    else:
        block_names = list(sorted( df_all['block_name'].unique() ))

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

        ############################

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
            assert df_all[tcn].max() <= tmax
            assert df_all[tcn].max() >= 0
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

#df = getSubDf(df_all, subj, pertv,tgti,env)
def calcQuantilesPerSubj(df_all, coln, q = 0.05 ):
    subj2qts = {}
    subjects     = df_all['subject'].unique()
    for subj in subjects:
        mask_subj = df_all['subject'] == subj
        dat = df_all.loc[mask_subj,coln]
        dat = dat[~np.isinf(dat)]
        qts = np.nanquantile(dat,[q,1-q])
        subj2qts[subj] = qts
    return subj2qts

def truncateDf(df, coln, q=0.05, verbose=0, clean_infnan=False, inplace=False,
               return_mask = False):
    if not inplace:
        df = df.copy()

    mask = np.ones(len(df), dtype=bool)
    if clean_infnan:
        # mask of good
        mask =  ~ ( df[coln].isna() | np.isinf( df[coln] ) )
        df =  df[ mask ]
        mask = np.array(mask)

    if (q is not None) and (q > 1e-10):
        subj2qts = calcQuantilesPerSubj(df, coln, q=q)
        #df = df.copy()
        for subj,qts in subj2qts.items():
            mask_cur = (df['subject'] == subj) & \
                ( ( df[coln] < qts[0]) | (df[coln] > qts[1]) )
            df.loc[mask_cur,coln] = np.nan
            if verbose:
                print(f'Setting {sum(mask_cur)} / {len(mask_cur)} points to NaN')
        assert np.any( ~(np.isnan(df[coln]) | np.isinf(df[coln]))  )

    if clean_infnan:
        mask2 = ~ ( df[coln].isna() | np.isinf( df[coln] ) )
        df =  df[ mask2]
        mask2 = np.array(mask2)
        indsgood = np.where(mask)[0]
        mask[ indsgood[~mask2] ] = False
        #mask[ indsgood[mask2] ] = True
        #mask[~mask2] = False

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
               colname_nh = 'non_hit_shifted', trial_shift_sizes = [1],
               DEBUG=0, allow_duplicating=True, time_locked = 'target',
                          addvars = None, target_info_type = 'inds' ):
    '''
        if allow_duplicating is False we don't allow creating copies
        of subsets of indices within subject (this can be useful for decoding)
    '''
    from config2 import block_names
    assert isinstance(dists_trial_from_prevtgt_cur, list)
    assert isinstance(dists_rad_from_prevtgt_cur, list)

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
    from error_sensitivity import computeErrSens2
    p = itprod(envs_cur,block_names_cur,pertvals_cur,gseqcs_cur,tgt_inds_cur,
               dists_rad_from_prevtgt_cur,dists_trial_from_prevtgt_cur)
    p = list(p)
    print('len(prod ) = ',len(p))
    print(p)

    if subj_list is None:
        subj_list = df_all['subject'].unique()

    #colns_set  = []; colns_skip = [];
    debug_break = 0
    dfs = []; #df_inds = []
    for subj in subj_list: #[:1]:
        for tpl in p:
            #print(len(tpl), tpl)
            (env,block_name,pertv,gseqc,tgti,drptgt,dtptgt) = tpl

            tpl = env,block_name,pertv,gseqc,tgti,\
                drptgt,dtptgt,\
                None,None
            print( sprintf_tpl_statcalc(tpl) )

            print(env,block_name,pertv,gseqc,tgti,drptgt,dtptgt)
            #df = df_all
            # if we take only non-hit then, since we'll compute err sens sequentially
            # we'll get wrong
            #if trial_group_col in ['trialwb']:
            #    raise ValueError('not implemented')
            df = getSubDf(df_all, subj, pertv,tgti,env,block_name,
                          non_hit = False)
            db_inds = df.index
            #df_inds += [db_inds]

            tgn = getTrialGroupName(pertv, tgti, env, block_name)
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

            for tsz in trial_shift_sizes:
                escoln = 'err_sens'
                #if tsz == 0:
                #    escoln = 'err_sens':
                #else:
                #    escoln = 'err_sens_-{tsz}t':
                # resetting index is important
                r = computeErrSens2(df.reset_index(), df_inds=None,
                                    error_type=error_type,
                                    colname_nh = colname_nh,
                                    correct_hit = 'inf', shiftsz = tsz,
                                    err_sens_coln=escoln,
                                    time_locked = time_locked, addvars=addvars,
                                    target_info_type = target_info_type,
                                    recalc_non_hit = False)
                nhna, df_esv, ndf2vn = r

                # if I don't convert to array then there is an indexing problem
                # even though I try to work wtih db_inds it assigns elsewhere
                # (or does not assigne at all)
                es_vals = np.array( df_esv[escoln] )
                assert np.any(~np.isnan(es_vals)), tgn  # at least one is not None

                #colns_set += [coln]

                dfcur = df.copy()
                dfcur['trial_shift_size'] = tsz  # NOT _nh, otherwise different number
                dfcur['time_locked'] = time_locked  # NOT _nh, otherwise different number
                dfcur[escoln] = es_vals  # NOT _nh, otherwise different number
                dfcur['correction'] = np.array( df_esv['correction'] )
                dfcur['trial_group_col_calc'] = tgn
                dfcur['error_type'] = error_type
                # here it means shfited by 1 within subset
                dfcur['non_hit_shifted'] = np.array( df_esv['non_hit_shifted'] )

                errn = ndf2vn['prev_error']
                dfcur[errn] = np.array( df_esv[errn] )

                for avn in addvars:
                    if avn in dfcur.columns:
                        continue
                    dfcur[avn] = np.array(df_esv[avn])


                #lbd(0.5)
                #print(dfcur['target_locs'].values, df_esv['prev_target'].values )
                #raise ValueError('f')
                dfcur['dist_rad_from_prevtgt2'] = dfcur['target_locs'].values -\
                    df_esv['prev_target'].values

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
    print('Finished successfully')

    df_all2 = pd.concat(dfs)
    df_all2.reset_index(inplace=True, drop=True)
    df_all2.drop(['feedbackX','feedbackY'],axis=1,inplace=True)
    if 'trajectoryX' in df_all2.columns:
        df_all2.drop(['trajectoryX','trajectoryY'],axis=1,inplace=True)


    df_all2.loc[df_all2['trials'] == 0, 'non_hit_shifted'] = False
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

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config2 import *
import seaborn as sns

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

                for bn in block_names:
                    mask_bn = mask_pert & (df_all['block_name'] == bn)
                    trials = df_all.loc[mask_bn, 'trials']
                    df_all.loc[mask_bn, 'trialwtgt_wpert_wb'] = np.arange(len(trials) )
                for envc in envcode2env:
                    mask_env = mask_pert & (df_all['environment'] == envc)
                    trials = df_all.loc[mask_env, 'trials']
                    df_all.loc[mask_env, 'trialwtgt_wpert_we'] = np.arange(len(trials) )



            for bn in block_names:
                mask_bn = mask & (df_all['block_name'] == bn)
                trials = df_all.loc[mask_bn, 'trials']
                df_all.loc[mask_bn, 'trialwtgt_wb'] = np.arange(len(trials) )
            for envc in envcode2env:
                mask_env = mask & (df_all['environment'] == envc)
                trials = df_all.loc[mask_env, 'trials']
                df_all.loc[mask_env, 'trialwtgt_we'] = np.arange(len(trials) )
    #df_all['trialwtgt_wpert_wb'] = df_all['trialwtgt_wpert_wb'].astype(int)

    # tcolnames = [s for s in df_all.columns if s.find('trial') >= 0]
    tmax = df_all['trials'].max()
    tcolnames = ['trialwb',
     'trialwe',
     'trialwpert_wb',
     'trialwpert_we',
     'trialwtgt',
     'trialwpert',
     'trialwtgt_we',
     'trialwtgt_wb',
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


def getSubDf(df, subj, pertv,tgti,env, block_name=None,
        pert_seq_code=None,
        dist_rad_from_prevtgt=None, dist_trial_from_prevtgt=None,
        non_hit=False, verbose=0, nonenan=False ):
    '''
    if nonenan is True, then NaN in numeric columns are treated as None
    '''
    assert env in ['stable','random','all'], env
    if pertv is not None:
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

def truncateDf(df, coln, q=0.05, verbose=0, clean_infnan=False):
    if clean_infnan:
        df =  df[  ~ ( df[coln].isna() | np.isinf( df[coln] ) ) ]

    if (q is not None) and (q > 0):
        subj2qts = calcQuantilesPerSubj(df, coln, q=q)
        df = df.copy()
        for subj,qts in subj2qts.items():
            mask = (df['subject'] == subj) & \
                ( ( df[coln] < qts[0]) | (df[coln] > qts[1]) )
            df.loc[mask,coln] = np.nan
            if verbose:
                print(f'Setting {sum(mask)} / {len(mask)} points to NaN')
        assert np.any( ~(np.isnan(df[coln]) | np.isinf(df[coln]))  )

    if clean_infnan:
        df =  df[  ~ ( df[coln].isna() | np.isinf( df[coln] ) ) ]

    return df

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


def computeErrSensVersions(df_all, envs_cur,block_names_cur,pertvals_cur,gseqcs_cur,tgt_inds_cur,
               dists_rad_from_prevtgt_cur,dists_trial_from_prevtgt_cur,
               subj_list=None, error_type='MPE',
               colname_nh = 'non_hit_shifted',
               DEBUG=0):
    dfme = []
    from itertools import product as itprod
    from error_sensitivity import computeErrSens2
    p = itprod(envs_cur,block_names_cur,pertvals_cur,gseqcs_cur,tgt_inds_cur,
               dists_rad_from_prevtgt_cur,dists_trial_from_prevtgt_cur)
    p = list(p)

    if subj_list is None:
        subj_list = df_all['subject'].unique()

    colns_set  = []; colns_skip = [];
    debug_break = 0
    dfs = []
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

            tgn = getTrialGroupName(pertv, tgti, env, block_name)
            coln = getColn(pertv, tgti, env, block_name, None) #, trial_group_col)
            print('  ',tgn, coln,len(df))
            if (len(df) == 0) or (len(db_inds) == 0):
                #rowi += 1
                print('skip',coln,subj)
                colns_skip += [coln]
                if DEBUG:
                    debug_break = 1
                    break
                continue

            # resetting index is important
            r = computeErrSens2(df.reset_index(), df_inds=None,
                                error_type=error_type,
                                colname_nh = colname_nh,
                                correct_hit = 'inf')
            nhna, df_esv, ndf2vn = r

            # if I don't convert to array then there is an indexing problem
            # even though I try to work wtih db_inds it assigns elsewhere
            # (or does not assigne at all)
            es_vals = np.array( df_esv['err_sens'] )

            assert np.any(~np.isnan(es_vals)), coln  # at least one is not None
            assert np.any(~np.isnan(es_vals)), coln  # at least one is not None

            colns_set += [coln]

            dfcur = df.copy()
            dfcur['err_sens'] = es_vals  # NOT _nh, otherwise different number
            dfcur['correction'] = np.array( df_esv['correction'] )
            dfcur['trial_group_col_calc'] = tgn
            dfcur['error_type'] = error_type
            dfcur['non_hit_shifted'] = np.array( df_esv['non_hit_shifted'] )
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
    df_all2.drop(['trajectoryX','trajectoryY'],axis=1,inplace=True)


    df_all2.loc[df_all2['trials'] == 0, 'non_hit_shifted'] = False
    #df_all2.loc[df_all2['trials'] == 0, 'non_hit_not_adj'] = False
    df_all2.loc[df_all2['trials'] == 0, 'err_sens'] = np.inf
    return df_all2

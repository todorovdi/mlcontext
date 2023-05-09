import pandas as pd
import numpy as np


def events2df(events, info, dat_time, dat_diode, trigger2phase, CONTEXT_TRIGGER_DICT, restmintime):
    dfev = pd.DataFrame(events, columns=['nsample', 'prev_trigger', 'trigger' ]  )
    CONTEXT_TRIGGER_DICT_inv = dict(zip(CONTEXT_TRIGGER_DICT.values(), CONTEXT_TRIGGER_DICT.keys()))
    def f(row): 
        trg = row['trigger']
        if trg in CONTEXT_TRIGGER_DICT:
            r = CONTEXT_TRIGGER_DICT[row['trigger']]
            phase = r[-1]
        elif trg in trigger2phase:
            r = trigger2phase[trg]
            phase = r
        else: 
            r = None
            phase = None
        
        return r,phase

    dfev[['trial_info','phase']] = dfev.apply(f,  1, result_type='expand')
    # this is 'naive time', when MEG is restarted (i.e. when multiple raws are concatenated)
    # it won't show any big difference between consequtive time values
    # in particular it will be inconsistent with the time values from the log
    dfev['time'] = dfev['nsample'] / info['sfreq'] 

    #### define better times
    dfev['time_megchan'] = dat_time[0,dfev['nsample']]

    restmintime_meg = dfev.query('phase == "REST"')['time_megchan'].min()
    print(restmintime_meg)

    restminsample_meg = dfev.loc[dfev['time_megchan'] == restmintime_meg, 'nsample'].values[0]

    dfev['time_since_first_REST'] = dfev['time_megchan'] - restmintime_meg

    def f(row):
        tpl = row.get('trial_info',None)
        r = None,None,-100,None
        if tpl is not None:
            if isinstance(tpl,tuple):
                r = tpl
        return r
            
    dfev[['trial_type', 'vis_feedback_type', 'tgti_to_show', 'phase']] = \
        dfev.apply(f,1, result_type='expand')

    # indexing
    ph = dfev['phase']

    # TODO: epochs plot for photodiode aligned to start of the mvt

    c0 = (ph == 'GO_CUE_WAIT_AND_SHOW') & (ph.shift(-1) == 'TARGET_AND_FEEDBACK')
    print(np.sum(c0))
    dfev['csr_GO_CUE_WAIT_AND_SHOW'] = c0.cumsum()

    c = (ph == 'REST') & (ph.shift(-1) == 'GO_CUE_WAIT_AND_SHOW') & (ph.shift(-2) == 'TARGET_AND_FEEDBACK')
    print(np.sum(c))
    assert np.sum(c) == np.sum(c0)
    dfev['csr_REST'] = c.cumsum()

    c |= (ph == 'TRAINING_END') | (ph == 'PAUSE') | (ph == 'BREAK')
    print(np.sum(c))
    dfev['csr_REST_ext'] = c.cumsum()

    c = (ph == 'REST') & (ph.shift(-1) == 'GO_CUE_WAIT_AND_SHOW') 
    print(np.sum(c))
    dfev['csr_REST_lax'] = c.cumsum()

    c = (ph == 'REST') | (ph == 'TRAINING_END') #| (ph == 'PAUSE') | (ph == 'BREAK')
    print(np.sum(c))
    dfev['csr_REST_lax_ext'] = c.cumsum()

    phprev = ph.shift()
    crest = (ph == 'REST') & ( phprev.isin( ['ITI','PAUSE','BREAK','TRAINING_END']) )
    c =(ph == 'TRAINING_END') | crest  #| (ph == 'PAUSE') | (ph == 'BREAK')
    print(np.sum(c))
    dfev['trial_index'] = c.cumsum() #- 1

    #######################

    # take first time we start movement from the point of view of dfev, 
    # take the value of the diode right before and in the middle of the mvt
    safe_delay = 200
    # we really need to use 'time' (which is raw.times basically) for time_as_index to work
    inds = raw.time_as_index(mvt_starts_megtrig['time'] - safe_delay)
    diode_pos = diode[0, inds]

    safe_delay2 = params['time_feedback'] / 2
    inds2 = raw.time_as_index(mvt_starts_megtrig['time'] + safe_delay2)
    diode_neg = diode[0, inds2]

    med = np.mean([ np.mean(diode_neg), np.mean(diode_pos)] )
    print('med = ', med)


    dfdi = pd.DataFrame({'diode':dat_diode[0], 'times': times} )
    #med = dfdi['diode'].median()
    #med = - 0.18 # MAY NEED TO BE TUNED MANUALLY! or I have to use info from dfev
    dfdi['diode_neg'] = dfdi['diode'] < med
    dfdi['diode_neg_csr'] = (dfdi['diode_neg'] != dfdi['diode_neg'].shift(1)) .cumsum()

    #############


    dfdi['time_since_first_REST'] = dat_time[0] - restmintime_meg
    grp = dfdi.query('diode_neg == True').groupby('diode_neg_csr')
    dfdisz = grp.size() 
    dfmvtdur_di = dfdisz / raw.info['sfreq']
    display('Mvt dur from diode ' ,dfmvtdur_di.describe())

    ###################
    # times when diode changed from off to on, first occurence
    ts_di = grp.min('time_since_first_REST')['time_mmvt_starts_megtrigegchan'].values

    # times when diode changed from ON to OFF, first occurence
    ts_di_off = grp.max('time_since_first_REST') #+ 1/raw.info['sfreq']
    ts_di_off['time_since_first_REST'] += 1/raw.info['sfreq']

    ts_ev = mvt_starts_megtrig['time_since_first_REST'].values
    ts_diev_diff = ts_di - ts_ev
    print('max,min,mean = ', np.max(ts_diev_diff), np.min(ts_diev_diff), np.mean(ts_diev_diff) )

    #################
    dfev['time_since_first_REST_diode'] = np.nan
    dfev.loc[mvt_starts_megtrig.index, 'time_since_first_REST_diode'] = ts_di
    dfev.loc[iti_starts_megtrig.index, 'time_since_first_REST_diode'] = ts_di_off


    ########################
    
    return dfev


def checkSeqConsist(df, dfev):
    cols = ['trial_type','vis_feedback_type','tgti_to_show','phase', 'time_since_first_REST']
    #dfevtmp = dfev_mx.query('~phase.isna()').sort_values('time_since_first_REST')[['csr_REST'] + cols].copy()
    # this removes BUTTON_PRESS and MEG start, MEG end and 255 trigger, nothing else
    dfevtmp = dfev.query('~phase.isna()')\
        [['trial_index','csr_REST','csr_REST_lax_ext'] + cols].copy()

    s = ''
    if not dfev.query('phase == "BREAK"').size:
        s = ' and phase != "BREAK"'
    dftmp = df.query('subphase_relation == "last"' + s)[['trial_index','block_ind'] + cols].copy()

    #dfevtmp = dfevtmp.drop(['phase','csr_REST'],1)
    assert dfevtmp.reset_index()['time_since_first_REST'].diff().max() >= 0

    d = {}
    dftmp_ = dftmp.reset_index()
    for cn in dftmp_.columns:
        cn2 = cn+ '_log'
        d[cn] = cn2
    #dftmp_ = dftmp.rename(columns=d).reset_index()
    dftmp_ = dftmp_.rename(columns=d)

    #dftmpmerge = pd.concat([dfevtmp.reset_index(),dftmp_], 1)
    # WARNING: reset_index destroys consistence with dfevtmp inds
    # WARNING: here shift creats index incosist with df and [:-1] creates inconsiste with lengths of dfev
    dftmpmerge = pd.concat([dfevtmp.reset_index(),dftmp_.shift(-1)], axis=1).iloc[:-1]

    ######  Check corrspondance of sequences of trial infos
    goods = ['TARGET_AND_FEEDBACK','GO_CUE_WAIT_AND_SHOW']
    bads = ['PAUSE','BREAK']
    bads2 = ['TRAINING_END']
    #qs = '~phase.isin(@bads) and ~phase.shift(1).isin(@bads) and ~phase.shift(2).isin(@bads)'
    #df_corresp_test = dftmpmerge.query(qs)
    c = dftmpmerge['phase'].isin(bads)

    # break contaminates also next two stages (inc REST of the first). 
    # Since we don't have BREAK present in dfev,
    # we have to detect it in an indirect way
    cb = (dftmpmerge['trial_type'] == 'break') & (dftmpmerge['phase'] == 'REST')
    c |= cb | cb.shift(1) | cb.shift(2) | dftmpmerge['phase'].isin(bads2)

    # PAUSE contaiminates preceding REST (it's this PAUSE's REST in fact), 
    # following TRAINING_END (we remove it anyway) 
    # and following REST (it is REST of the next trial)
    c |=  c.shift(-1) | c.shift(1) | c.shift(2) 

    # target on very first trial at REST is diff (at log it is not set, i.e. set to -1)
    c |= (dftmpmerge['trial_index'] == 0) & (dftmpmerge['phase'] == 'REST') &\
        (dftmpmerge['tgti_to_show_log'] == -1)

    # make sure we won't skip really important phases (they could have been exluded by shifts)
    c &= ~dftmpmerge['phase'].isin(goods)

    invalid_trigger_info_corresp = c
    df_corresp_test = dftmpmerge[~c]
    df_ = df_corresp_test # just for shortness

    # want phase agreement also for bad
    difph = np.where( dftmpmerge['phase'] != dftmpmerge['phase_log'] )[0]#[0]
    assert len(difph) == 0 , 'Phases differ!'
    # want trial agreement also for bad
    difti = np.where( dftmpmerge['trial_index'] != dftmpmerge['trial_index_log'] )[0]#[0]
    assert len(difti) == 0, f'trial_inds differ {difti}'
    diftgt = df_[ df_['tgti_to_show'] != df_['tgti_to_show_log']].index
    assert len(diftgt) == 0 , f'tgti differ! {diftgt}'
    difvft = np.where( df_['vis_feedback_type'] != df_['vis_feedback_type_log'] )[0]#[0]
    assert len(difvft) == 0 , f'vis_feedback_type differ! {difvft}'
    difttype = np.where( df_['trial_type'] != df_['trial_type_log'] )[0]#[0]
    assert len(difttype) == 0 , f'vis_feedback_type differ! {difttype}'

    print("Consistency check passed")
    r  = len(diftgt) == 0
    r  &= len(difvft) == 0
    r  &= len(difttype) == 0
    r  &= len(difti) == 0
    r  &= len(difph) == 0

    # these are indices in dfev
    bad_inds_dfev = dftmpmerge.loc[c,'index']
    bad_inds_df   = dftmpmerge.loc[c,'index_log']
    good_inds_dfev = dftmpmerge.loc[~c,'index']
    good_inds_df   = dftmpmerge.loc[~c,'index_log']
    #assert (dfev.loc[badis_dfev.values,'trial_index'].values ==  df.loc[badis_df.values,'trial_index'].values).all()
    return r, bad_inds_dfev, bad_inds_df, good_inds_dfev, good_inds_df

def decode_naive(X, y, nperm= 2, n_jobs = 10, nbfold=5, nbfold_perm=2,
                pbackend = 'multiprocessing' ):
    from mne.decoding import SlidingEstimator, LinearModel
    #from sklearn.linear_model import LinearRegression
    from jr.gat import scorer_spearman
    from scipy.stats import spearmanr
    from mne.decoding import cross_val_multiscore
    from sklearn.preprocessing import RobustScaler
    from joblib import parallel_backend
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.metrics import make_scorer
    import mne

    assert len(X) == len(y), (X.shape, y.shape)
    good =  ~np.isnan(y) 
    print( f'Removing {np.sum(~good)} bad (nan) datapoints')
    yg = y[good]
    Xg = X[good]  # trial x channel x time   

    print( np.sum( np.isinf(yg) ), np.sum( np.isnan(yg) ) )
    scoring = make_scorer(scorer_spearman)    
    
    regr = RidgeCV()
    pipeline = make_pipeline(RobustScaler(), LinearModel(regr))
    est = SlidingEstimator(pipeline, scoring = scoring,
                          n_jobs = n_jobs)


    cv     = KFold(nbfold, shuffle=True)
    cvperm = KFold(nbfold_perm, shuffle=True)
    classic_dec_verbose = 1

    # #%debug
    def sc(est,XX,YY): 
        print(XX.shape)
        return spearmanr(XX,YY)[0]
    yg_perm = np.random.permutation(yg)
    with mne.use_log_level('warning'):
        with parallel_backend(pbackend):
            scores = cross_val_multiscore(est, Xg, y=yg, cv=cv,                                                                                    
               scoring=scoring, n_jobs=n_jobs,                                                                                             
               verbose=classic_dec_verbose) 

            scs = []
            for i in range(nperm):
                yg_perm = np.random.permutation(yg)
                scores_perm = cross_val_multiscore(est, Xg, y=yg_perm, cv=cvperm,                                                                                    
                       scoring=scoring, n_jobs=n_jobs,                                                                                             
                       verbose=0) 
                scs += [scores_perm.mean(0)]
            scores_perm = np.array(scs)    
    
    return scores, scores_perm

def decode_naive2(X, y, nperm= 2, n_jobs = 10, nbfold=5, nbfold_perm=2,
        pbackend = 'multiprocessing'):
    from mne.decoding import SlidingEstimator, LinearModel
    #from sklearn.linear_model import LinearRegression
    from jr.gat import scorer_spearman
    from scipy.stats import spearmanr
    from mne.decoding import cross_val_multiscore
    from sklearn.preprocessing import RobustScaler
    from joblib import parallel_backend
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.metrics import make_scorer
    import mne




    assert len(X) == len(y), (X.shape, y.shape)
    good =  ~np.isnan(y) 
    print( f'Removing {np.sum(~good)} bad (nan) datapoints')
    yg = y[good]
    Xg = X[good]  # trial x channel x time   

    print('num inf, nan =', np.sum( np.isinf(yg) ), np.sum( np.isnan(yg) ) )
    scoring = make_scorer(scorer_spearman)    
    
    regr = RidgeCV()
    #pipeline = make_pipeline(RobustScaler(), LinearModel(regr))
    est = make_pipeline(RobustScaler(), LinearModel(regr) )
                            
#     est = SlidingEstimator(pipeline, scoring = scoring,
#                           n_jobs = n_jobs)


    cv     = KFold(nbfold, shuffle=True)
    cvperm = KFold(nbfold_perm, shuffle=True)
    classic_dec_verbose = 1

    # #%debug
    def sc(est,XX,YY): 
        print(XX.shape)
        return spearmanr(XX,YY)[0]
    yg_perm = np.random.permutation(yg)
    with mne.use_log_level('warning'):
        with parallel_backend(pbackend):
            Xg_ = Xg.reshape(len(Xg), -1)
            scores = cross_val_multiscore(est, Xg_, y=yg, cv=cv,                                                                                    
               scoring=scoring, n_jobs=n_jobs,                                                                                             
               verbose=classic_dec_verbose) 

            scs = []
            for i in range(nperm):
                yg_perm = np.random.permutation(yg)
                scores_perm = cross_val_multiscore(est, Xg_, y=yg_perm,
                       cv=cvperm, scoring=scoring, n_jobs=n_jobs,
                       verbose=0) 
                scs += [scores_perm.mean()]
            scores_perm = np.array(scs)    

    
    return scores, scores_perm

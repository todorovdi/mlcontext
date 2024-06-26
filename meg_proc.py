import pandas as pd
import numpy as np
import os
from os.path import join as pjoin
import mne

def cleanEvents_NIH(events):
    # code by Romain
    # check we have for stable events  tgtcode, 100, 30
    # make sure feedback phase goes after target, otherwise delete
    import warnings
    from config2 import event_ids_tgt_stable
    t = -1
    bad_trials = list()
    bad_events = list()
    for ii, event in enumerate(events):
        if event[2] in event_ids_tgt_stable:
            t += 1
            if events[ii+1, 2] == 100:
                if events[ii+2, 2] != 30:
                    bad_trials.append(t)
                    warnings.warn('Bad sequence of triggers')
                    # Delete bad events until the next beginning of a trial (10)
                    bad_events.append(ii - 1)
                    for iii in range(5):
                        if events[ii + iii, 2] == 10:
                            break
                        else:
                            bad_events.append(ii+iii)
            elif events[ii+1, 2] != 30:
                bad_trials.append(t)
                warnings.warn('Bad sequence of triggers')
                # Delete bad events until the next beginning of a trial (10)
                bad_events.append(ii - 1)
                for iii in range(5):
                    if events[ii + iii, 2] == 10:
                        break
                    else:
                        bad_events.append(ii+iii)
    print(f'cleanEvents_NIH: deleted {len(bad_events)} events')
    events_cleaned = np.delete(events, bad_events, 0)
    return events_cleaned

def events2df_NIH(events, advcols=True):
    ''' adv cols is whether adding prev columns'''
    from config2 import stage2evn2event_ids
    event_ids_all_tgt = stage2evn2event_ids['target']['all'] 
    event_ids_all_fb = stage2evn2event_ids['feedback']['all'] 
    dfev = pd.DataFrame({'code':events[:,2], 'sample':events[:,0]})
    def f(x): 
        if x in event_ids_all_tgt:
            r = 'target'
        elif x in event_ids_all_fb:
            r = 'feedback' 
        else:
            r = None
        return r
     
    dfev['type'] = dfev['code'].apply( f )
    

    if advcols:
        dfev['prev_type'] = dfev['type'].shift(1)
        dfev['prev_prev_type'] = dfev['type'].shift(2)
        # check alternation
        assert (dfev['sample'].diff()[1:] > 0).all()
        assert (dfev['prev_prev_type'] == dfev['type']).iloc[2:].all()

        dfev['prev_code'] = dfev['code'].shift(1, fill_value=-10000) # fill_value so that we have int and not float type
        dfev['target_code'] = dfev.apply( lambda x: x['code'] if x['type'] == 'target' else x['prev_code'], 1 )

        # here we may miss some of the trials because triggers did not always arrive
        dfev['trial_index_ev'] = (dfev['type'] == 'target').cumsum() - 1
    return dfev

def getTargetCodes_NIH(epochs_both, epochs_subset):
    # epochs_both should be created with minimal tmin tmax around zero to capture really all events
    # both epochs objects have to be 
    dfev = events2df_NIH(epochs_both.events)
    # this adds target_code, beyond other things
    dfev2 = events2df_NIH(epochs_subset.events, advcols =False)
    dfev2 = dfev2.drop(columns=['code','type'])

    # this assigns corresponding rows and does intersection
    dfev2_ = dfev2.merge(dfev, on='sample', how='left')
    return dfev2_

def events2df(events, raw, dat_diode, dat_time,
              trigger2phase, CONTEXT_TRIGGER_DICT, 
              params, restmintime=None, return_aux = False):
    assert isinstance(params, dict)
    dfev = pd.DataFrame(events, columns=['nsample', 'prev_trigger', 'trigger' ]  )

    if params['subject'] == '2023-SE2-002':
        cti = dfev['nsample'] == 1184635
        dfev.loc[cti, 'trigger'] = 44 #ITI
    #dfev['trigger'] =dfev['trigger'].astype(int)
    #dfev['prev_trigger'] =dfev['prev_trigger'].astype(int)
    CONTEXT_TRIGGER_DICT_inv = dict(zip(CONTEXT_TRIGGER_DICT.values(), CONTEXT_TRIGGER_DICT.keys()))
    CONTEXT_TRIGGER_DICT_ext = CONTEXT_TRIGGER_DICT.copy()

    CONTEXT_TRIGGER_DICT_ext.update({
         250 : (None,None,-100,'trigger_keypress'),
         255 : (None,None,-100,'MEG_start_trigger_native'),
         252 : (None,None,-100,'MEG_start_trigger'),
         253 : (None,None,-100,'MEG_stop_trigger')} )


    def f(row, col): 
        trg = row[col]
        if np.isnan(trg):
            r = None
            phase =None
        else:
            trg = int(trg)
            if trg in CONTEXT_TRIGGER_DICT_ext:
                r = CONTEXT_TRIGGER_DICT_ext[trg]
                phase = r[-1]
            elif trg in trigger2phase:
                r = trigger2phase[trg]
                phase = r
            else: 
                r = None
                phase = None
                print('skipping ',trg)
        
        return r,phase



    dfev[['trial_info','phase']] = dfev.apply(f,  1, result_type='expand', col='trigger')
    dfev['prev_phase'] = dfev['phase'].shift(1).copy()

    c = dfev['phase'] != dfev['prev_phase']
    c[0] = False
    dfev['phase_ind'] = np.cumsum(c)

    #print(sum(dfev['phase'] == 'MEG_start_trigger'))

    # this is 'naive time', when MEG is restarted (i.e. when multiple raws are concatenated)
    # it won't show any big difference between consequtive time values
    # in particular it will be inconsistent with the time values from the log
    dfev['time'] = dfev['nsample'] / raw.info['sfreq'] 
    assert dfev['time'].diff()[1:].all() > 0, 'sample based time not inc'

    #### define better times. --- NO! It is a bad idea because dat_time is not always increasing. And not clear when it jumps
    dfev['time_megchan'] = dat_time[0,dfev['nsample']]
    assert dfev['time_megchan'].diff()[1:].all() > 0, 'megchan based time not inc' 
    restmintime_meg = dfev.query('phase == "REST"')['time_megchan'].min()

    print(restmintime_meg)

    #restminsample_meg = dfev.loc[dfev['time_megchan'] == restmintime_meg, 'nsample'].values[0]

    # megchan time
    restmintime_meg0 = dfev.query('phase == "REST"')['time'].min()
    #dfev['time_since_first_REST'] = dfev['time'] - restmintime_meg0
    dfev['time_since_first_REST'] = dfev['time_megchan'] - restmintime_meg
    assert dfev['time_since_first_REST'].diff()[1:].all() > 0, 'since REST based on megchang does not inc'

    def f(row):
        tpl = row.get('trial_info',None)
        r = None,None,-100,None
        if tpl is not None:
            if isinstance(tpl,tuple):
                r = tpl
        return r
            
    dfev[['trial_type', 'vis_feedback_type', 'tgti_to_show', 'phase']] = \
        dfev.apply(f,1, result_type='expand')

    dfev.loc[ dfev['phase']      == 'MEG_stop_trigger' , 'phase' ] = 'BREAK'
    dfev.loc[ dfev['prev_phase'] == 'MEG_stop_trigger' , 'prev_phase' ] = 'BREAK'

    ph = dfev.query('~phase.isna() and phase != "BUTTON_PRESS"')['phase'] # otherwise setting trial index give inconsistency
    #print('We had {} BUTTON_PRESS events '.format(len(dfev) - len(ph) ) )


    # indexing

    # TODO: epochs plot for photodiode aligned to start of the mvt
    print( ph.value_counts() )

    c0 = (ph == 'GO_CUE_WAIT_AND_SHOW') & (ph.shift(-1) == 'TARGET_AND_FEEDBACK')
    print('Num GO_CUE_WAIT_AND_SHOW happening after TARGET_AND_FEEDBACK ' , np.sum(c0))
    dfev['csr_GO_CUE_WAIT_AND_SHOW'] = c0.cumsum()

    c = (ph == 'REST') & (ph.shift(-1) == 'GO_CUE_WAIT_AND_SHOW') & (ph.shift(-2) == 'TARGET_AND_FEEDBACK')
    print('Num REST after GO_CUE_WAIT_AND_SHOW after TARGET_AND_FEEDBACK ', np.sum(c))
    dfev['csr_REST'] = c.cumsum()
    c1 = c.copy()

    c |= (ph == 'TRAINING_END') | (ph == 'PAUSE') | (ph == 'BREAK')
    print('+ TRAINING_END or PAUSE or BREAK ', np.sum(c))
    dfev['csr_REST_ext'] = c.cumsum()

    c = (ph == 'REST') & (ph.shift(-1) == 'GO_CUE_WAIT_AND_SHOW') 
    print('Num REST following GO_CUE_WAIT_AND_SHOW ', np.sum(c))
    dfev['csr_REST_lax'] = c.cumsum()

    c = (ph == 'REST') | (ph == 'TRAINING_END') #| (ph == 'PAUSE') | (ph == 'BREAK')
    print('Num REST or TRAINING_END',np.sum(c))
    dfev['csr_REST_lax_ext'] = c.cumsum()

    # this is MEG event-based so no breaks are recorded here
    #c = (ph == 'BREAK') 
    #print('Num BREAK',np.sum(c))
    #dfev['csr_BREAK'] = c.cumsum()

    phprev = ph.shift()
    crest = (ph == 'REST') & ( phprev.isin( ['ITI','PAUSE','BREAK','TRAINING_END','MEG_start_trigger','MEG_start_trigger_native']) )
    cti =(ph == 'TRAINING_START') | crest  #| (ph == 'PAUSE') | (ph == 'BREAK')
    #cti = crest
    print('Num REST after ITI,PAUSE, BREAK or TRAINING_END ',np.sum(cti))
    #dfev['trial_index'] = -1
    dfev['trial_index'] = cti.cumsum() #- 1
    # TODO: extend to other indices after sorting wrt nsample. Or better check that nsample sorting did not change

    if np.sum(c1) != np.sum(c0): 
        print( f'WARNING: sum(c1) = {np.sum(c1)}, sum(c0) ={ np.sum(c0)} ')

    #######################

    #mvt_starts = df.query('phase == "TARGET_AND_FEEDBACK"').groupby('trial_index').min('time')
    mvt_starts_megtrig = dfev.query('phase == "TARGET_AND_FEEDBACK"')

    # take first time we start movement from the point of view of dfev, 
    # take the value of the diode right before and in the middle of the mvt
    safe_delay = 200
    # we really need to use 'time' (which is raw.times basically) for time_as_index to work and not time_megchan
    inds = raw.time_as_index(mvt_starts_megtrig['time'] - safe_delay)
    diode_pos = dat_diode[0, inds]

    safe_delay2 = params['time_feedback'] / 2
    inds2 = raw.time_as_index(mvt_starts_megtrig['time'] + safe_delay2)
    diode_neg = dat_diode[0, inds2]

    diode_extremes = [ np.mean(diode_neg), np.mean(diode_pos)]
    med = np.mean(diode_extremes )
    print('diode med = ', med, ' extremes ',diode_extremes)


    dfdi = pd.DataFrame({'diode':dat_diode[0], 'times': raw.times, 'time_megchan':dat_time[0,:], 'nsample':np.arange(len(raw.times)) } )
    #med = dfdi['diode'].median()
    #med = - 0.18 # MAY NEED TO BE TUNED MANUALLY! or I have to use info from dfev
    dfdi['diode_neg'] = dfdi['diode'] < med
    # when diode changes
    dc = (dfdi['diode_neg'] != dfdi['diode_neg'].shift(1))
    print(dc[0])
    dc[0] = False
    dfdi['diode_neg_csr'] =  dc.cumsum()

    #############


    #dfdi['time_since_first_REST'] = dat_time[0] - restmintime_meg
    dfdi['time_since_first_REST0'] = dfdi['times'] - restmintime_meg0
    dfdi['time_since_first_REST'] = dfdi['time_megchan'] - restmintime_meg
    grp = dfdi.query('diode_neg == True').groupby('diode_neg_csr')
    dfdisz = grp.size() 
    dfmvtdur_di = dfdisz / raw.info['sfreq']
    print('Mvt dur from diode ' ,dfmvtdur_di.describe())

    ###################
    # times when diode changed from off to on, first occurence
    ts_di_ = grp.min('time_since_first_REST')
    ts_di = ts_di_['time_since_first_REST'].values
    #ts_di = grp.min('time_since_first_REST')['time_mvt_starts_megtrigchan'].values

    #dfev[]

    # times when diode changed from ON to OFF, first occurence
    ts_di_off = grp.max('time_since_first_REST') #+ 1/raw.info['sfreq']
    ts_di_off['time_since_first_REST'] += 1/raw.info['sfreq']
    ts_di_off_ = ts_di_off
    ts_di_off = ts_di_off['time_since_first_REST'].values

    ts_di_    ['phase'] = 'diode_OFF_to_ON'
    ts_di_off_['phase'] = 'diode_ON_to_OFF'

    df_ = pd.concat([ts_di_.reset_index(), ts_di_off_.reset_index()])
    df_diode = df_.drop(columns=['diode','times','diode_neg','diode_neg_csr'])
    df_diode = df_diode.sort_values(['nsample'])

    ts_ev = mvt_starts_megtrig['time_since_first_REST'].values

    #mvt_starts_megtrig = dfev.query('phase == "TARGET_AND_FEEDBACK"').groupby('trial_index').min('time') 
    #iti_starts_megtrig = dfev.query('phase == "ITI"').groupby('trial_index').min('time')

    dfev['time_since_first_REST_diode'] = np.nan

    #mvt_starts_megtrig = dfev.query('phase == "TARGET_AND_FEEDBACK"').groupby('trial_index').min('time_megchan') 
    from behav_proc import aggRows 
    #just using .groupby('trial_index').min('time_megchan') 
    # (and perhaps same with 'time') gives incorrect row in the end, at least nsample is clearly wrong
    mvt_starts_megtrig = aggRows(dfev.query('phase == "TARGET_AND_FEEDBACK"'), 'time', 'min') 
    dfd,dfd_good,dfd_bad,paths     = find_correspondence(mvt_starts_megtrig['time_since_first_REST'].values, ts_di )
    mvt_starts_megtrig_a = mvt_starts_megtrig.iloc[dfd_good.ind1.values] 
    dfev.loc[mvt_starts_megtrig_a.index, 'time_since_first_REST_diode'] = ts_di    [dfd_good['ind2'].values]

    iti_starts_megtrig = aggRows(dfev.query('phase == "ITI"'), 'time', 'min') 
    dfd2,dfd_good2,dfd_bad2,paths2 = find_correspondence(iti_starts_megtrig['time_since_first_REST'].values, ts_di_off )
    iti_starts_megtrig_a = iti_starts_megtrig.iloc[dfd_good2.ind1.values]
    dfev.loc[iti_starts_megtrig_a.index, 'time_since_first_REST_diode'] = ts_di_off[dfd_good2['ind2'].values]

    df_ = pd.concat( [ts_di_.iloc[dfd_good['ind2'].values].reset_index(), ts_di_off_.iloc[dfd_good2['ind2'].values].reset_index()  ] )
    df_diode_clean = df_.drop(columns=['diode','times','diode_neg','diode_neg_csr'])
    df_diode_clean = df_diode_clean.sort_values(['nsample'])

    #iti_ev = iti_starts_megtrig['time_since_first_REST'].values

    #ts_di_unshift = ts_di
    #if len(ts_di) !=  len(ts_ev):
    #    if len(ts_di) - len(ts_ev)  == 1:
    #        difs = np.abs(ts_di[1:] - ts_ev)
    #        print('discrep after shifting diode one trial min={:.3f}s, max={:.3f}s'.format(
    #            np.min(difs),np.max(difs)) )
    #        assert np.max(difs) < 1., 'ts_di - ts_ev difs are large'
    #        ts_di_unshift = ts_di.copy() # it will go to locals() output for debugc:w
    #        ts_di = ts_di[1:]


    #if len(ts_di_off) !=  len(iti_ev):
    #    if len(ts_di_off) - len(iti_ev)  == 1:
    #        difs = np.abs(ts_di_off[:-1] - iti_ev)
    #        print('discrep after shifting diode one trial min={:.3f}s, max={:.3f}s'.format(
    #            np.min(difs),np.max(difs)) )
    #        ts_di_off = ts_di_off[:-1]

    #if ( len(ts_di) ==  len(mvt_starts_megtrig) ) and ( len(iti_starts_megtrig) ==  len(ts_ev) ):
    #    ts_diev_diff = ts_di - ts_ev
    #    print('ts_diev_diff = max={:.3f}, min={:.3f}, mean={:.3f}'.format( np.max(ts_diev_diff), 
    #                    np.min(ts_diev_diff), np.mean(ts_diev_diff) ) )

    #    #################

    #    dfev['time_since_first_REST_diode'] = np.nan
    #    dfev.loc[mvt_starts_megtrig.index, 'time_since_first_REST_diode'] = ts_di
    #    dfev.loc[iti_starts_megtrig.index, 'time_since_first_REST_diode'] = ts_di_off
    #else:
    #    print(f'WARNING: number of diode switches {(len(ts_di)), (len(ts_di_off))} not equals '
    #        f'number of triggers of start of movement, ITI {len(mvt_starts_megtrig), len(iti_starts_megtrig)}')
    #    print('Therefore cannot set time_since_first_REST_diode column')


    ########################
    
    if return_aux:
        return dfev, locals()
    else:
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

def addTrigPresentCol_NIH(df, meg_targets_ev, environment=None):
    '''
    events is 2-dim
    '''
    assert meg_targets_ev.ndim == 1
    from base2 import int_to_unicode
    from Levenshtein import editops
    assert (np.diff( np.array(df.index) ) > 0).all()

    # remove some of the behav inds
    # WARNING: this will fail (give empty)if epochs are based of time_locked=feedback
    # (or maybe it was only in old code?)
    #meg_targets_ev = events[:, 2].copy()
    #target_inds_behav = np.array( df['target_inds'] )

    target_codes_behav = np.array( df['target_codes'] )

    assert len(target_codes_behav)
    assert len(meg_targets_ev)
    print(  len(target_codes_behav) , len(meg_targets_ev) )
    #if environment is None:
    #    environment = df['environment']

    # for different environments target signals stored in .fif have different codes (20+ vs 30+) so
    # we convert to unified indices

    #codes_allowed = [20, 21, 22, 23, 25, 26, 27, 28]
    from config2 import event_ids_tgt
    nbad = (~np.isin(meg_targets_ev, event_ids_tgt) ).sum()
    print(nbad , 'non target code events' )
    assert nbad == 0
    #for env,envcode in env2envcode.items():
    #    #trial_inds = np.where(environment == envcode)[0]
    #    trial_inds = np.where( np.isin(meg_targets_ev,
    #        stage2evn2event_ids['target'][env] ) )[0]
    #    # it uses that target codes are conequitive integers
    #    meg_targets_ev[trial_inds] = meg_targets_ev[trial_inds] - env2subtr[env]

    # one can have less triggers than behav
    changes = editops(int_to_unicode(target_codes_behav),
                        int_to_unicode(meg_targets_ev))
    # we have missing triggers in MEG file so we delete stuff from behav file
    delete_trials = [change[1] for change in changes] # these are indices of rows
    print('addTrigPresentCol: ediops changes = ', changes )
    df['trigger_present'] = True
    colind = df.columns.get_loc('trigger_present')
    # we can use iloc becahse editoops was computed on the same-indexed dataset
    df.iloc[delete_trials, colind] = False

    assert df['trigger_present'].sum() > 0
    print(df['trigger_present'].astype(float).describe() )

    target_codes_behav2 = df.query('trigger_present == True')['target_codes']

    # respects ordering
    if np.array_equal(meg_targets_ev, target_codes_behav2):
        if len(delete_trials):
            print(f'addTrigPresentCol: {len(delete_trials)} triggers missing for {len(df)} trials')
    else:
        #warnings.warn('MEG events and behavior file do not match')
        raise ValueError('MEG events and behavior file do not match')

    return df

def ML_classic(X, subdf, varname, mask_good, regression_type,
        scale_Y_robust, scale_X_robust,  nskip_trial,
        covs=None, use_non_hit_recalc=0, ):
    print(f'-- classic decoding {varname}')
    assert len(varname)
    vals    = np.array(subdf[varname])
    non_hit = np.ones(len(subdf), dtype=bool)

    if use_non_hit_recalc:
        non_hit      = prepNonHit(subdf, varname, env, time_locked)
    if par['discard_hit_twice']:
        non_hit = subdf['non_hit_not_adj']

    res = {}
    y    = y[mask_good]
    Xcur = Xcur[mask_good]

    print('Total = {}, noninf = {} notnan = {}'.format( len(y),  mask_not_inf.sum(),  ( ~np.isnan(y) ).sum()  ) )
    if len(y) < 5:
        print('Too short y, return None')
        res['dec_error'] = str(f'Too short y {len(y)}')
        return res

    if par['scale_Y_robust'] < 3:
        Xcur, y = rescaleIfNeeded(Xcur, y, par, centering = par.get('XYcentering',0) )

    if exit_after == 'rescale':
        print(y.shape)
        print(y[:10])
        return y
        #sys.exit(0)

    if nskip_trial > 1:
        y = y[::nskip_trial]
        Xcur = Xcur[::nskip_trial]

    # for quicker tests, -1 means all channels

    #if debug_save_Xy_exit:
    #    fnf =pjoin( results_folder , f'Xy_classical_{env}_{varname}.npz')
    #    np.savez( fnf, X=Xcur, y=y, varname = varname  )
    #    print(f'DEBUG: saved Xy classic to {fnf}, return')
    #    return {}


    ##############################
    # Regression for classic decoding
    if regression_type == 'Ridge_noCV': #this is ungly but just for back compat, earlier I was using 'Ridge' for 'RidgeCV'
        alpha_ind = len(alphas) // 2 
        reg_step = Ridge(alpha = alphas[alpha_ind], fit_intercept = fit_intercept_classic_dec)
    elif regression_type in ('RidgeCV','Ridge'):
        # def alphas = (0.1, 1, 10), in the original code Romain was not supplying them in classical case for some reason
        # alpha_per_target = False by def meaning that it will take the best alpha only
        #est = make_pipeline(spoc_est, RidgeCV(alphas = alphas, fit_intercept = fit_intercept_classic_dec))
        reg_step = RidgeCV(alphas = alphas, fit_intercept = fit_intercept_classic_dec)
    elif regression_type == 'xgboost':
        import xgboost
        from xgboost import XGBRegressor
        xgb = XGBRegressor(**add_clf_creopts)

        add_clf_creopts['n_jobs'] = n_jobs_per_dim_classical_dec
        xgb_est = XGBRegressor(**add_clf_creopts_est)
        reg_step = xgb_est
    else:
        raise ValueError('wrong regression value')

    from sklearn.preprocessing import StandardScaler, RobustScaler

    ##############################

    y_preds = np.zeros(len(y))
    scores = list()
    print(f'{varname}: Starting CV for regression_type={regression_type}')
    nsplit = 0
    alphas_found = []
    patterns_found = []
    filters_found = []
    try:
        for train, test in cv.split(Xcur, y):

            precalc_covs_cur = precalc_covs[foldi]
            spoc_est = SPoC(n_components=SPoC_n_components, log=True, reg='oas',
                rank='full', n_jobs=n_jobs_per_dim_classical_dec,
                fit_log_level=mne_fit_log_level, precalc_covs = precalc_covs_cur)
            centering = par.get('XYcentering',0)
            if par['scale_Y_robust'] == 3:
                scaler = RobustScaler(with_centering=centering)
                est = make_pipeline(scaler, spoc_est, reg_step)
            elif par['scale_Y_robust'] == 4:
                scaler = StandardScaler(with_centering=centering)
                est = make_pipeline(scaler, spoc_est, reg_step)
            else:
                est = make_pipeline(spoc_est, reg_step)


            print(f'Starting split N={nsplit}')
            with mne.use_log_level(mne_fit_log_level):
                est.fit(Xcur[train], y[train])
            y_preds[test] = est.predict(Xcur[test])
            score = spearmanr(y_preds[test], y[test])
            scores.append(score[0])
            nsplit += 1
            if regression_type == 'Ridge_noCV':
                alphas_found += [est.named_steps['ridge'].alpha]
            else:
                alphas_found += [est.named_steps['ridgecv'].alpha_]
            # before Oct 15 it was like this, inverted
            #patterns_found += [est.named_steps['spoc'].filters_]
            #filters_found += [est.named_steps['spoc'].patterns_]

            patterns_found += [est.named_steps['spoc'].patterns_]
            filters_found += [est.named_steps['spoc'].filters_]
        diff = np.abs(y - y_preds)

        res['non_hit'] = non_hit
        res['diff']   = diff

        res['alphas']   = alphas_found
        res['filters']   = filters_found
        res['patterns']   = patterns_found

        res['scores'] = scores  # scores per (outer) fold
        res['vals']   = y  # non_hit
        res['mask_valid'] = mask_not_inf
        res['dec_type'] = 'classic'
        res['Xshape'] = Xcur.shape
        res['dec_error'] = None
        print(f'Finished classical {varname} scores average = {np.mean(scores):.4f}' )
    except Exception as e:
        print(f'!!!! Error during fit for {varname}: ',str(e) )
        if dec_error_handling == 'raise':
            raise e
        else:
            res['dec_error'] = str(e)

    return res

def getCleanEpochsMaskForDec(subdf, varnames, env, time_locked,
        discard_hit_twice):
    non_hit = np.ones(len(subdf), dtype=bool)
    if use_non_hit_recalc:
        non_hit &= prepNonHit(subdf, varnames[0], env, time_locked)
        for varname in varnames[1:]:
            non_hit &= prepNonHit(subdf, varname, env, time_locked)
    if discard_hit_twice:
        non_hit &= subdf['non_hit_not_adj']

    after_break = (subdf['trials'] % 192 == 0).values

    target_vals      = subdf[varnames]._values
    if len(varnames) > 1:
        mask_good = getMaskNotNanInf(target_vals, axis = 1)
        #mask_not_inf = ~np.any( np.isinf(y), axis=1)
        #mask_not_nan = ~np.any( np.isnan(y), axis=1)
    else:
        mask_good = getMaskNotNanInf(target_vals[:,0] )
        #mask_not_inf = ~np.any( np.isinf(y))
        #mask_not_nan = ~np.any( np.isnan(y))

    mask_good &= non_hit
    mask_good &= (~after_break)

    return mask_good

def ML_b2b(varnames, X, covs):
    print(f'-- decoding {varnames}')
    assert len(varnames)
    vals       = subdf[varnames]._values
    assert vals.shape[0] == X.shape[0]

    non_hit = np.ones(len(subdf), dtype=bool)
    if use_non_hit_recalc:
        non_hit &= getNonhitForDec(subdf, varnames, env, time_locked)

    if par['discard_hit_twice']:
        non_hit &= subdf['non_hit_not_adj']

    res = {}
    vals_non_hit = vals[non_hit]  # already non_hit
    Xhn = X[non_hit]

    y = vals_non_hit
    Xcur = Xhn

    mask_not_inf = ~np.any( np.isinf(y), axis=1)
    mask_not_nan = ~np.any( np.isnan(y), axis=1)
    #print('infs ', (~mask_not_inf).sum(), len(mask_not_inf) )

    mask_good = getMaskNotNanInf(y, axis=1)
    y    = y[mask_good]
    Xcur = Xcur[mask_good]

    print('B2B: Total = {}, noninf = {} notnan = {}'.format( len(y),  mask_not_inf.sum(),  mask_not_nan.sum()  ) )

    if debug_save_Xy_exit:
        np.savez( pjoin( results_folder , f'Xy_b2b_{env}.npz'),
                 X=Xcur, y=y, varnames=varnames )
        print('DEBUG: saved Xy b2b and exiting')
        return {}

    if len(y) < 5:
        print('Too short y, return None')
        res['dec_error'] = str(f'Too short y {len(y)}')
        return res

    if par['scale_Y_robust'] < 3:
        Xcur, y = rescaleIfNeeded(Xcur, y, par, centering = par.get('XYcentering',0))

    if nskip_trial > 1:
        y = y[::nskip_trial]
        Xcur = Xcur[::nskip_trial]

    # for quicker tests, -1 means all channels
    if n_channels_to_use > 0:
        Xcur = Xcur[:,:n_channels_to_use]
    ##############################
    spoc = SPoC(n_components=SPoC_n_components, log=True, reg='oas',
                rank='full', n_jobs=min(n_jobs_SPoC, Xcur.shape[0] ),
                fit_log_level=mne_fit_log_level)
    # Regression for classic decoding
    if regression_type == 'Ridge_noCV':
        #est = make_pipeline(spoc_est, Ridge(alpha = alphas[5], fit_intercept = fit_intercept_classic_dec))
        alpha_ind = len(alphas) // 2 
        reg_step = Ridge(alpha=alphas[alpha_ind], fit_intercept=fit_intercept_b2b)
    elif regression_type in ('RidgeCV','Ridge'):
        # def alphas = (0.1, 1, 10)
        reg_step = RidgeCV(alphas=alphas, fit_intercept=fit_intercept_b2b)
    elif regression_type == 'xgboost':
        from xgboost import XGBRegressor
        reg_step = XGBRegressor(**add_clf_creopts)
        #param_grid = {
        #    'pca__n_components': [5, 10, 15, 20, 25, 30],
        #    'model__max_depth': [2, 3, 5, 7, 10],
        #    'model__n_estimators': [10, 100, 500],
        #}
        #grid = GridSearchCV(pipeline, param_grid,
        #  cv=5, n_jobs=-1, scoring='roc_auc')
    else:
        raise ValueError('wrong regression value')

    from sklearn.preprocessing import StandardScaler, RobustScaler
    centering = par.get('XYcentering',0)
    if par['scale_Y_robust'] == 3:
        scaler = RobustScaler(with_centering=centering)
        G = make_pipeline(scaler, spoc, reg_step)
    elif par['scale_Y_robust'] == 4:
        scaler = StandardScaler(with_centering=centering)
        G = make_pipeline(scaler, spoc, reg_step)
    else:
        G = make_pipeline(spoc, reg_step)
    #G = make_pipeline(spoc, reg_step )

    H = LinearRegression(fit_intercept=False, n_jobs= n_jobs_SPoC)
    #G = direct pipeline (spoc + regerssor)
    #H = back pipeline
    b2b = B2B_SPoC(G=G, H=H, n_splits=n_splits_B2B,
        parallel_type=B2B_SPoC_parallel_type,n_jobs=n_jobs)

    try:
        b2b.fit(Xcur,y)
        ##############################
        # E_ is a mean of H_hats over 0 axis
        partial_scores = np.diag(b2b.E_)

        #b2b.G -- get alpha
        res['b2b.H'] = b2b.H
        res['b2b.G'] = b2b.G
        res['non_hit'] = non_hit
        #res['diff']   = diff
        res['scores'] = partial_scores
        res['scores_std'] = np.diag(np.std(b2b.H_hats,0 ) ) 
        res['vals']   = y  # non_hit
        res['mask_valid'] = mask_not_inf
        res['varnames'] = varnames
        res['dec_type'] = 'b2b'
        res['Xshape'] = Xcur.shape
        res['dec_error'] = None

        print(f'Finished b2b for {varnames} {name} partial_scores = {partial_scores}' )
    except Exception as e:
        print(f'!!!! Error during fit for {varnamas}: ',str(e) )
        res['dec_error'] = str(e)
    return res

def find_folder_with_ds_subfolders(base_path, subj = None, verbose = 0):
    ''' 
    Context change experiment
    give parent dir containing subject folders collects paths to folders containing .df subfolders.
    Returns list of full paths to directories'''
    target_paths = []
    for first_level in [os.path.join(base_path, d) for d in os.listdir(base_path)]:
        #print(first_level)
        if (subj is not None) and (first_level.find(subj) < 0 ):
            continue
        if not os.path.isdir(first_level) :
            print("A required directory is missing or does not follow the expected single-folder structure, or no folder starts with 'MEG_MLEARN'.")
        
        second_level = next(os.path.join(first_level, d) for d in \
                           os.listdir(first_level) if d.startswith("MEG_MLEARN_"))
        #print(first_level)
        # Get the first (and only) folder inside the base path that starts with "MEG_MLEARN"
#         first_level = next(os.path.join(base_path, d) for d in os.listdir(base_path)\
#                 if os.path.isdir(os.path.join(base_path, d))\
#                            and d.startswith("MEG_MLEARN"))
        
        # Step into 'scans' inside the first level
        scans_path = os.path.join(second_level, 'scans')
        # Assuming 'scans' exists and has exactly one subdirectory named 'MEGSCAN_CTF'
        megscan_ctf_path = os.path.join(scans_path, 'MEGSCAN_CTF')
        if not os.path.isdir(megscan_ctf_path) :
            megscan_ctf_path = os.path.join(scans_path, 'MEGSCAN-CTF')
            print('!! needed to change')
        assert os.path.isdir(megscan_ctf_path)
        
        # Inside 'MEGSCAN_CTF', find the final target directory (assumed to be the only one)
        target_path = next(os.path.join(megscan_ctf_path, d) for d in os.listdir(megscan_ctf_path) if os.path.isdir(os.path.join(megscan_ctf_path, d)))
        
        if verbose > 0:
            print(f"Found target path: {target_path}")
        target_paths += [target_path]
    return target_paths

def _getSubj2dsnames(dmeg, verbose = 0):
    #/2023-SE2-035/MEG_MLEARN_2023-SE2-035_20240318_mlearn_BE11746
    fnbs = []
    subjects = []
    subj2path = {}
    subj2dsnames = {}
    for fnf in os.scandir(dmeg):
        if fnf.is_dir():
            if verbose:
                print(fnf, fnf.name)
            subjects += [fnf.name]
            subj2path[fnf.name] = fnf.path

            fld = find_folder_with_ds_subfolders(dmeg, fnf.name, verbose=verbose-1)[0]

            subj2dsnames[fnf.name] = []

            for fnf2 in os.scandir(fld):
                #print(fnf2.name)
                if fnf2.name.endswith('.ds'):
                    subj2dsnames[fnf.name] += [fnf2.path]
            subj2dsnames[fnf.name] = list(sorted(subj2dsnames[fnf.name]))
    return subj2dsnames,subj2path


#find_folder_with_ds_subfolders(dmeg, subjects[0])

def loadBehavOneSubj(path_behav, multi_param_read_mode = 'read_last'):
    'path_behav is path to behav folder all the way'
    #multi_param_read_mode = 'read_all'

    from behav_proc import readParamFiles
    from glob import glob
    from pathlib import Path
        
    fnbs = []
    for fnf in glob(pjoin(path_behav,'*.param') ):
        name = Path(fnf).name
        fnb = name.replace('.param','')
        fnbs += [fnb]
    
    print('Number of behav files = ',len(fnbs))

    fnbs = list( sorted(fnbs, key=lambda fn: int(fn.split('_')[-1].split('.')[0] ) ) )
    if multi_param_read_mode == 'read_last':
        if len(fnbs) > 1:
            print('Selecting last param file')
        paramfiles = [fn for fn in list( os.listdir(path_behav) ) if fn.endswith('.param')]
        last = list( sorted( [ int(fn.split('_')[-1].split('.')[0]) for fn in paramfiles] ) )[-1]
        lastpf = [fn for fn in paramfiles if fn.find(str(last)) >= 0][0]
        fnbs = [lastpf.split('.')[0]]
    else:
        assert multi_param_read_mode == 'read_all'

    dfs = []
    dftriglogs = []
    rs = []
    for fnb in fnbs:
        fnp = fnb + '.param'

        #fnf_par = pjoin(path_behav,fnp)
        b = False
        fnf_trig = pjoin(path_behav,fnb + '_trigger.log')
        fnf_log  = pjoin(path_behav,fnb + '.log')
        if os.path.exists(fnf_trig):
            b = os.path.getsize(fnf_trig) > 0
        else:
            print('f1')
            b = False
        if os.path.exists(fnf_log):
            b &= ( os.path.getsize(fnf_log) > 0 )
        else:
            print('f2')
            b = False

        if not b:
            print(f'Task did not start for fnb={fnb}, skipping')
            continue
        params, phase2trigger, trigger2phase, CONTEXT_TRIGGER_DICT =\
            readParamFiles(fnp, path_behav)

        fn = fnb + '.log'
        fnp = fnb + '.param'

        with open(pjoin(path_behav,fn), 'r') as f:
            l = f.readline()
            truelen = len( l.split(',') )

        # time is regular time MINUS initial time, meaning when software starts (it is NOT the same as time of first REST used in analysis later
        r = ('trial_index, current_phase_trigger, tgti_to_show,'
           ' vis_feedback_type, trial_type, special_block_type, block_ind, '
            ' feedbackX, feedbackY, unpert_feedbackX, unpert_feedbackY,'
             ' error_distance, target_coordX, target_coordY, '
             'feedbackX_when_crossing, feedbackY_when_crossing, '
             'jax1, jax2, reward, time, time_abs')
        r = r.replace(' ','')
        colnames = r.split(',')
        print('len(colnames) = ',len(colnames),'truelen = ',truelen,colnames)
        assert truelen == len(colnames)

        nbad = 3  # recompense strip
        
        fn_nofoot = fnb + '__nofoot.log' 
        create_copy_without_footer(pjoin(path_behav,fn),
            pjoin(path_behav,fn_nofoot), nbad)

        from numpy import dtype
        # 'O' types are there because we may have None
        types = {'trial_index': dtype('int64'),
             'current_phase_trigger': dtype('int64'),
             'tgti_to_show': dtype('float64'),
             'vis_feedback_type': dtype('O'),
             'trial_type': dtype('O'),
             'special_block_type': dtype('O'),
             'block_ind': dtype('int64'),
             'feedbackX': dtype('int64'),
             'feedbackY': dtype('int64'),
             'unpert_feedbackX': dtype('float64'),
             'unpert_feedbackY': dtype('float64'),
             'error_distance': dtype('float64'),
             'target_coordX': dtype('int64'),
             'target_coordY': dtype('int64'),
             'feedbackX_when_crossing': dtype('int64'),
             'feedbackY_when_crossing': dtype('int64'),
             'jax1': dtype('float64'),
             'jax2': dtype('float64'),
             'reward': dtype('float64'),
             'time': dtype('float64'),
             'time_abs': dtype('float64'),
             'subject': dtype('O'),
             'phase': dtype('O')}

        #colnames = ['trial_index', 'current_phase_']
        # skipfooter = nbad,
        # comment = '#' to ingore "Task restarted" line
        df = pd.read_csv(pjoin(path_behav,fn_nofoot),  
             on_bad_lines='warn', header=0,
            names=colnames, encoding='latin1', comment = '#', 
                         engine='c', dtype=types)
        #encoding='latin-1'
        subj_ = fnb.split('_')[0]    
        df['subject'] = subj_        
        df['phase'] = df.apply(lambda row: trigger2phase[row['current_phase_trigger']], 1)
        
        dfs += [df]

        fnf = pjoin(path_behav, fnb + '_trigger.log')
        from behav_proc import loadTriggerLog
        dftriglog = loadTriggerLog(fnf,CONTEXT_TRIGGER_DICT)
        dftriglogs += [dftriglog]

        fnf = pjoin(path_behav,fnb+'.param')
        from behav_proc import readTrialInfoSeqParams
        df_trialinfoseq_params = readTrialInfoSeqParams(fnf)
#     if len(dfs) > 1:
#         dfr = pd.concat(dfs)
#     else:
#         dfr = df

        rs += [( df,dftriglog, df_trialinfoseq_params, params, phase2trigger, trigger2phase, CONTEXT_TRIGGER_DICT )]
    for df_ in dfs:
        print( 'len = {}, maxtime in min = {}'.format( len(df_), df_['time'].max() / 60 ) )
    
    return rs

def create_copy_without_footer(input_file, output_file, num_rows_to_skip=3, skip_if_exists=True):
    """
    Reads a CSV file using readlines, skips the last n rows (footer), and writes it to a new file.

    Args:
      input_file (str): Path to the input CSV file.
      output_file (str): Path to the output CSV file.
      num_rows_to_skip (int, optional): Number of rows from the end to skip (default: 3).
      skip_if_exists (bool, optional): Skip creating the output file if it already exists (default: True).
    """
    
    if skip_if_exists and os.path.exists(output_file):
        print(f"create_copy_without_footer: Output file {output_file} already exists. Skipping creation.")
        return

    with open(input_file, 'r', encoding='latin1') as infile, open(output_file, 'w', encoding='latin1') as outfile:
        lines = infile.readlines()        
        outfile.writelines(lines[:-num_rows_to_skip])
        print(f"create_copy_without_footer: Successfully created copy of {input_file} without footer in {output_file}")

def read_final_rwd_info_MEG_CC(filename, num_lines=3):
    """
    returns something like 
        {'recompense_bonus_totale': 10.0,
     'reward_accrued': 779.6860704593224,
     'monetary_value_tot': 10.073463442626904}
    """
    import re
    try:
        with open(filename, 'r', encoding = 'latin1') as f:
            lines = f.readlines()
            if len(lines) < num_lines:
                return lines  # Return all lines if there are less than num_lines
        lines = lines[-num_lines:]  # Get the last num_lines elements
        line = lines[-1]
        print(line)

        values_dict = {}
        # Regular expression pattern to find key-value pairs where value is a float
        pattern = re.compile(r'([^=;\n]+)=\s*([0-9]*\.?[0-9]+)')
        # Find all matches of the pattern in the input line 
        matches = pattern.findall(line)

        # Populate the dictionary with key-value pairs from matches
        for key, value in matches:
            try:
                # Attempt to convert value to float and assign to the corresponding key in the dictionary
                k=key.strip()
                if k == '#Récompense bonus totale':
                    k = 'recompense_bonus_totale'
                values_dict[k] = float(value)
            except ValueError:
                # Skip any matches that cannot be converted to float
                continue

        return values_dict

    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None  # Return empty list on error

def read_raw_MEG_CC(fnfs_ds, fewch = False, verbose=None):
    raws0 = []
    sfreqs = []
    for fnf_ds in fnfs_ds:
        print(fnf_ds)

        raw = mne.io.read_raw_ctf(fnf_ds, verbose=verbose)
        print(raw.n_times, raw.times[-1], raw.times[-1]/60 )
        raws0 += [raw]
        sfreqs += [raw.info['sfreq']]

        # if I drop ref meg it writes "". So maybe I won't for now
        #Removing 5 compensators from info because not all compensation channels were picked.
        #raw = raw.drop_channels(chns_ref_meg)    
    assert np.std(sfreqs) <= 1e-10

    # Here we ignore that these have different device<->head transforms
    raw = mne.io.concatenate_raws(raws0, on_mismatch='ignore')
    #raw_erm = read_raw_ctf(erm_fname)
    m = {'SCLK01-177':'syst',
     'EEG063-2800':'emg',
     'EEG064-2800':'emg',
     'UADC001-2800':'misc', 
     'UADC010-2800':'eog',
     'UADC011-2800':'eog',
     'UADC012-2800':'misc',
     'UPPT001':'stim',
     'UPPT002':'syst'}

    # The EEG063 and 64 are EMG
    # UPPT002 is the parralel port from the interface box (only registers button presses)
    # UADC0{x}-2800 for x=10, 11 and 12 are EOG eyelink for x, y position and pupil size
    chn_diode = 'UADC001-2800'
    chn_pupil_size = 'UADC012-2800'
    chn_time = 'SCLK01-177'

    to_drop =['EEG063-2800']
    raw = raw.set_channel_types(m)

    chn2cht = dict( zip( raw.ch_names, raw.get_channel_types() ) )
    chns_nontriv = [chn for chn,cht in chn2cht.items() if cht not in ['mag', 'ref_meg']]
    chns_ref_meg = [chn for chn,cht in chn2cht.items() if cht in ['ref_meg']]
    
    if fewch:
        raw = raw.pick([chn_diode,chn_time,'UPPT002','UPPT001'])

    events = mne.find_events(raw, shortest_event=2)


    del raws0
    return raw, events

def find_correspondence(arr1, arr2, std_mult = 3.):
    '''
    tries to match arr1 to arr2 using dynamic time warping (DTW)
    outputs 3 dataframes and matching paths
    '''
    from dtaidistance import dtw
    print('find_correspondence: Lens of input = ', len(arr1),len(arr2))
    # Ensure arr1 is the shorter array
    rev = False
    #if len(arr1) > len(arr2):
    #    arr1, arr2 = arr2, arr1
    #    rev = True
    
    # Calculate the DTW distance and the best path
    # psi: Up to psi number of start and end points of a sequence can be 
    # ignored if this would lead to a lower distance. This is also called psi-relaxation (for cyclical sequences) [2].
    distance, paths = dtw.warping_paths_fast(arr1, arr2, window=min(len(arr1),len(arr2)), psi=0)
    best_path = dtw.best_path(paths)
    
    #print(distance)
    
#     ds = []
#     for i, j in best_path:
#         d = abs(arr1[i] - arr2[j])
#         ds += [d]
#     print( pd.DataFrame( ds).describe() )
    
    # Filter the path for the correspondences within the max_distance
    correspondences = []
    for i, j in best_path:
        d = abs(arr1[i] - arr2[j])
        #if d <= max_distance:
        correspondences.append((i, j, d))
    
    dfd = pd.DataFrame(correspondences, columns=['ind1','ind2','dist'])
    dfd['ind_diff'] = dfd['ind1'] - dfd['ind2']
    v = dfd['dist'].mean() + dfd['dist'].std() * std_mult
    dfd_bad = dfd[dfd['dist'] > v]
    dfd_good = dfd[dfd['dist'] <= v]

    percentiles = [.5, 0.7, 0.8, 0.95, 0.98]
    print(dfd['dist'].describe(percentiles = percentiles))
    print(f'find_correspondence: For thr={v:.4f} Num bad = ',len(dfd_bad) )
    
    return dfd,dfd_good,dfd_bad,paths

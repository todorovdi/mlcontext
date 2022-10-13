import os
import os.path as op
import numpy as np
import mne
from mne.io import read_raw_fif
from mne import Epochs
# import pandas as pd
import warnings
import sys
from base2 import (int_to_unicode, point_in_circle,
                   calc_target_coordinates_centered, radius_target,
                   radius_cursor)
from config2 import path_data
from config2 import event_ids_tgt,event_ids_feedback, env2envcode, env2subtr
from config2 import target_angs
from Levenshtein import editops
from config2 import delay_trig_photodi, min_event_duration

target_coords = calc_target_coordinates_centered(target_angs)

#subject = sys.argv[1]
#output_folder = 'beh_proc2'
#
#results_folder = op.join(path_data,subject,'results',output_folder)
#if not os.path.exists(results_folder):
#    os.makedirs(results_folder)


#time_locked = 'target'

#use_preloaded_raw = True


# task = 'LocaError_'  # 'VisuoMotor_' or 'LocaError_'
# time_locked = 'target'  # 'target' or 'reach'
# Read raw to extra MEG event triggers

# cycle over frequencies
#print(freq_name + '--------------------------')
# read behav data
#fname = op.join(path_data, subject, 'behavdata',
#                'behav_%sdf.pkl' % task)
#behav_df = pd.read_pickle(fname)
# read raw data
#run = list()
#files = os.listdir(op.join(path_data, subject))
#run.extend(([op.join(
#            path_data, subject + '/')
#            + f for f in files if task in f]))
#fname_raw = run[0]
# get raw and events from raw file
# raw = read_raw_ctf(fname_raw, preload=True, system_clock='ignore')

#modify behav_df in place
def enforceTargetTriggerConsistency(behav_df, epochs, environment,
                                    do_delete_trials=1,
                                    save_fname=None):
    # WARNING: this will fail (give empty) if epochs are based of time_locked=feedback
    meg_targets = epochs.events[:, 2].copy()
    #targets = np.array( behav_df['target_codes'] )
    targets = np.array( behav_df['target_inds'] )
    assert len(targets)
    assert len(meg_targets)
    print(  len(targets) , len(meg_targets) )
    if environment is None:
        environment = behav_df['environment']

    for env,envcode in env2envcode.items():
        trial_inds = np.where(environment == envcode)[0]
        meg_targets[trial_inds] = meg_targets[trial_inds] - env2subtr[env]

    behav_df_res = None
    if do_delete_trials:
        changes = editops(int_to_unicode(targets),
                          int_to_unicode(meg_targets))
        # we have missing triggers in MEG file so we delete stuff from behav file
        delete_trials = [change[1] for change in changes]
        # read behav.pkl and remove bad_trials if bad_trials is not empty
        if len( delete_trials) :
            print(f'enforceTargetTriggerConsistency: removing '
                  f'{len(delete_trials)} trials' )
            #behav_df = behav_df_full.copy().drop(delete_trials, errors='ignore')
            behav_df_res = behav_df.drop(delete_trials, errors='ignore')
            if save_fname is not None:
                behav_df.to_pickle(save_fname)

            targets_old = np.array( behav_df['target_inds'] )
            targets = np.array( behav_df_res['target_inds'] )
            print(len(targets), len(targets_old))

            assert len(targets)
        else:
            behav_df_res = behav_df


        if np.array_equal(meg_targets, targets):
            if len(delete_trials):
                print(f'enforceTargetTriggerConsistency: Deleted {len(delete_trials)} trials')
        else:
            warnings.warn('MEG events and behavior file do not match')

        assert behav_df is not None
    else:
        behav_df_res = behav_df
    return behav_df_res

# legacy
def getErrSensVals(error,target,movement, time_locked='target',
                   ret_df=False):
    # here target should be indices of target
    # shift error by -1

    assert len( set([ len(error), len(target), len(movement) ] ) ) == 1

    #getAnalysisData('all',time_locked,control_type,behad_df)
    varnames_def = ['target','prev_target', 'movement', 'prev_movement',
                'prev_error']

    if time_locked == 'target':
        # add zero in the beginning
        prev_errors   = np.insert(error,   0, 0)[:-1]
        prev_targets  = np.insert(target,  0, 0)[:-1]
        prev_movement = np.insert(movement, 0, 0)[:-1]
        analysis_value = [prev_targets, prev_movement,
                        error,
                        prev_errors]

        values_for_es = [target_angs[target],
                        target_angs[prev_targets],
                        movement,
                        prev_movement,
                        prev_errors]
        varnames = varnames_def
    else:
        # add zero in the end
        next_errors    = np.insert(error,  len(error), 0)[1:]
        next_target    = np.insert(target, len(target), 0)[1:]
        next_movement  = np.insert(movement,  len(movement), 0)[1:]

        analysis_value = [target, movement,
                        next_errors,
                        error]

        values_for_es = [target_angs[next_target],
                        target_angs[target],
                        next_movement,
                        movement,
                        error]

        varnames = ['next_target','target', 'next_movement', 'movement',
                    'error']
    if 'error' not in varnames: # I want to have current error as well
        varnames += ['error']
        values_for_es += [error]

    vndef2vn = dict(zip(varnames_def,varnames) )

    if ret_df:
        import pandas as pd
        df = pd.DataFrame()
        for vn,vs in zip(varnames,values_for_es):
            df[vn] = vs

        return df,vndef2vn
    else:
        return np.array(values_for_es), np.array(analysis_value), \
            varnames, varnames_def


def computeErrSens2(behav_df, df_inds=None, epochs=None, do_delete_trials=1,
                    enforce_consistency=0, time_locked='target',
                    correct_hit = 'inf', error_type = 'MPE',
                    colname_nh = 'non_hit_not_adj' ):
    '''
    computes error sensitiviy across dataset. So indexing is very important here.
    '''
    #colname_nh = 'non_hit_shifted'

    # in read_beahav we compute
    # errors.append(feedback[trial] - target_locs[trial])
    # target_locs being target angles

    assert correct_hit in ['prev_valid', 'zero', 'inf', 'nan' ]
    # modifies behav_df in place
    if enforce_consistency:
        # before enforcing consistencey. VERY DANGEROUS HERE since it DOES NOT take
        # only df_inds
        environment  = np.array(behav_df['environment'])
        assert epochs is not None
        behav_df = enforceTargetTriggerConsistency(behav_df, epochs, environment,
                                        do_delete_trials=do_delete_trials )

    if df_inds is None:
        df_inds = behav_df.index

    dis = np.diff(behav_df.index.values)
    np.max( dis ) == np.max( dis ) and np.max( dis ) == 1

    targets_cur      = np.array(behav_df.loc[df_inds,'target_inds'])
    targets_locs_cur      = np.array(behav_df.loc[df_inds,'target_locs'])
    org_feedback_cur      = np.array(behav_df.loc[df_inds,'org_feedback'])
    feedback_cur      = np.array(behav_df.loc[df_inds,'feedback'])
    # after enforcing consistencey
    environment_cur  = np.array(behav_df.loc[df_inds,'environment'])
    #feedback     = np.array(behav_d.locf['feedback'])
    feedbackX_cur    = np.array(behav_df.loc[df_inds, 'feedbackX'])
    feedbackY_cur    = np.array(behav_df.loc[df_inds, 'feedbackY'])
    movement_cur     = np.array(behav_df.loc[df_inds, 'org_feedback'])

    if error_type == 'MPE':
        errors_cur       = np.array(behav_df.loc[df_inds, 'error'])
    elif error_type == 'SPE_naive':
        # observed - predicted
        errors_cur       = feedback_cur - org_feedback_cur
    elif error_type == 'EGE_naive':
        errors_cur       = org_feedback_cur - targets_locs_cur
        # prediceted - goal

    ## after deleting wrong triggers
    #targets_cur = targets[df_inds]
    ## Feedback positions
    ##feedback_cur = feedback[df_inds]
    #feedbackX_cur = feedbackX[df_inds]
    #feedbackY_cur = feedbackY[df_inds]
    ## Movement positions [after deleting wrong trials]
    #movement_cur = movement[df_inds]
    ## Error positions
    #errors_cur = errors[df_inds]

    # it is a _mask_ of non_hit, not indices
    # mask is on df_inds only, not on entire behav_df
    non_hit_not_adj = point_in_circle(targets_cur, target_coords,
                                        feedbackX_cur, feedbackY_cur,
                                        radius_target + radius_cursor)
    non_hit_not_adj = np.array(non_hit_not_adj)

    # not that non-hit has different effect on error sens calc depending on
    # which time_locked is used
    if time_locked == 'feedback':
        #valid = np.insert(non_hit_not_adj, 0 ,0)[:-1]
        valid = np.insert(non_hit_not_adj, len(non_hit_not_adj), 0)[1:]
    elif time_locked == 'target':
        valid = np.insert(non_hit_not_adj, 0 ,0)[:-1]

    #non_hit = non_hit_cur
    #non_hit = np.array(non_hit)
    ## remove trials following hit (because no previous error)
    ## non_hit = ~(~non_hit | ~np.insert(non_hit, 0, 1)[:-1])
    ## normally it should depend on env
    #non_hit = np.insert(non_hit, 0, 0)[:-1]
    #non_hit[[0, N]] = False  # Removing first trial of each block
    # Q -- which non-hit should we use here?

    #non_hit = adjustNonHit(non_hit_not_adj,env,time_locked)

    # returs database of shfited data (either forward of backward, depending on
    # time_locked)
    df_esv,vndef2vn = \
        getErrSensVals(errors_cur,targets_cur, movement_cur,
            time_locked = time_locked, ret_df = True)

    #target_angs_next, target_angs, next_movement, movement, errors = values_for_es

    target_angs_next = df_esv[ vndef2vn['target'] ]
    target_angs      = df_esv[ vndef2vn['prev_target'] ]
    next_movement    = df_esv[ vndef2vn['movement'] ]
    movement         = df_esv[ vndef2vn['prev_movement'] ]
    errors           = df_esv[ vndef2vn['prev_error'] ]

    #= values_for_es


    #df = pd.DataFrame( values_for_es, columns=varnames)

    #Y_es = np.array(values_for_es)
    #analysis_value = analysis_value[:, non_hit].T

    #values_for_es = values_for_es[:, non_hit]
    # 0 -- target angs, 1 -- prev target angs, 2 -- current movemnet, 3 -- prev movements, 4 -- prev error
    # es = (target - current) - (prev tgt - prev movemnet) / prev_error
    # prev_error is taken from df['error']
    # movement _relative_ to the target

    #Q: why we divide by prev_error instead of by (Y_es[1]-Y_es[3])

    # this if for time_locked != 'target', for others it can be shifted
    correction = (target_angs_next - next_movement) - (target_angs - movement)
    df_esv['non_hit_not_adj'] = non_hit_not_adj
    df_esv['non_hit_shifted'] = valid
    df_esv.loc[df_esv[colname_nh],'err_sens']  = correction / errors

    #import pdb; pdb.set_trace()

    if correct_hit == 'prev_valid':
        df_esv.loc[~df_esv[colname_nh],'err_sens']  = np.inf
        hit_inds = np.where(~df_esv[colname_nh] )[0]
        for hiti in hit_inds:
            prev = df_esv.loc[ :hiti, 'err_sens' ]
            good = np.where( ~ (np.isinf( prev ) | np.isnan(np.isinf) ) )[0]
            if len(good):
                lastgood = good[-1]
                df_esv.loc[ hiti, 'err_sens' ] = df_esv.loc[ lastgood, 'err_sens' ]
        #df_esv.loc[~df_esv[colname_nh],'err_sens']  =
    elif correct_hit == 'zero':
        df_esv.loc[~df_esv[colname_nh],'err_sens']  = 0
    elif correct_hit == 'inf':
        df_esv.loc[~df_esv[colname_nh],'err_sens']  = np.inf
    elif correct_hit == 'nan':
        df_esv.loc[~df_esv[colname_nh],'err_sens']  = np.nan
    df_esv['correction'] = correction
    df_esv['error_type'] = error_type

    df_esv['environment']  = np.array( behav_df.loc[df_inds, 'environment'] )
    df_esv['perturbation'] = np.array( behav_df.loc[df_inds, 'perturbation'])

    df_esv['trial_inds_glob'] = np.array( behav_df.loc[df_inds, 'trials'])

    #corr = (values_for_es[0]-values_for_es[2]) - \
    #    (values_for_es[1]-values_for_es[3])
    #es = corr / values_for_es[4]

    #values_for_es[4]  ==  df['feedback'] - df['target']

    #return non_hit_not_adj, corr, es, values_for_es, varnames, varnames_def
    return non_hit_not_adj, df_esv, vndef2vn

# ICA can be also ''
# here value of hpass is not imporant because we use only stim channel. Still we need to set something so that we can read it
def computeErrSens(behav_df, subject, task = 'VisuoMotor_' ,fname=None, raw=None,
        force_resave_raw = 0,  ICA = 'with_ICA', hpass=0.1  ):
    from config2 import cleanEvents,delay_trig_photodi

    tmin = -0.5
    tmax = 0
    stim_chn = 'UPPT001'
    stim_chn_regex = '^((?!UPPT).)*$'

    if raw is None:
        fname_raw_full = op.join(path_data, subject,
                           f'raw_{task}{hpass}_{ICA}.fif')
        fname_raw_full_onlystim = op.join(path_data, subject,
                           f'raw_{task}{hpass}_{ICA}_onlystim.fif' )

        if not os.path.exists(fname_raw_full):
            raise ValueError(f'{fname_raw_full} does not exist, exiting')

        if os.path.exists(fname_raw_full_onlystim) and not force_resave_raw:
            raw = read_raw_fif(fname_raw_full_onlystim, preload=True)
        else:
            raw = read_raw_fif(fname_raw_full, preload=False)
            chinds = mne.pick_channels_regexp( raw.info['ch_names'], stim_chn_regex )
            #len(chinds)
            not_stim_chans = np.array(raw.info['ch_names'] )[chinds]
            #raw.drop_channels(not_stim_chans)
            raw.pick_channels([stim_chn])
            rawold = raw
            raw = mne.io.RawArray(rawold.get_data(),
                    mne.create_info(rawold.ch_names,rawold.info['sfreq']))
            raw.save(fname_raw_full_onlystim, overwrite=True )
    assert raw is not None

    events = mne.find_events(raw, stim_channel=stim_chn,
                             min_duration=min_event_duration)
    events[:, 0] += delay_trig_photodi  # to account for delay between trig. & photodi.
    # check that a target trigger is always followed by a reach trigger
    #t = -1
    #bad_trials = list()
    #bad_events = list()
    #for ii, event in enumerate(events):
    #    if event[2] in [20, 21, 22, 23]:
    #        t += 1
    #        if events[ii+1, 2] == 100:
    #            if events[ii+2, 2] != 30:
    #                bad_trials.append(t)
    #                warnings.warn('Bad sequence of triggers')
    #                # Delete bad events until the next beginning of a trial (10)
    #                bad_events.append(ii - 1)
    #                for iii in range(5):
    #                    if events[ii + iii, 2] == 10:
    #                        break
    #                    else:
    #                        bad_events.append(ii+iii)
    #        elif events[ii+1, 2] != 30:
    #            bad_trials.append(t)
    #            warnings.warn('Bad sequence of triggers')
    #            # Delete bad events until the next beginning of a trial (10)
    #            bad_events.append(ii - 1)
    #            for iii in range(5):
    #                if events[ii + iii, 2] == 10:
    #                    break
    #                else:
    #                    bad_events.append(ii+iii)
    #events = np.delete(events, bad_events, 0)
    events = cleanEvents(events)

    epochs = Epochs(raw, events, event_id=event_ids_tgt,
                    tmin=tmin, tmax=tmax, preload=True,
                    baseline=None, decim=2)

        #raw.filter(freq[0], freq[1], n_jobs=n_jobs)


    behav_df_full = behav_df.copy()

    environment  = np.array(behav_df_full['environment'])
    # modifies behav_df in place
    enforceTargetTriggerConsistency(behav_df, epochs, environment, do_delete_trials=1 )
    targets      = np.array(behav_df['target_inds'])
    environment  = np.array(behav_df['environment'])
    feedback     = np.array(behav_df['feedback'])
    feedbackX    = np.array(behav_df['feedbackX'])
    feedbackY    = np.array(behav_df['feedbackY'])
    errors       = np.array(behav_df['error'])
    movement = np.array(behav_df['org_feedback'])


    env2err_sens                     = {}
    env2corr                     = {}
    env2pre_err_sens           = {}
    env2pre_dec_data           = {}
    env2non_hit                = {}

    for env in ['stable','random','all']:
        if env in env2envcode:
            envcode = env2envcode[env]
            trial_inds = np.where(environment == envcode)[0]
        else:
            trial_inds = np.arange(len(environment))

        # after deleting wrong triggers
        targets_cur = targets[trial_inds]
        # Feedback positions
        feedback_cur = feedback[trial_inds]
        feedbackX_cur = feedbackX[trial_inds]
        feedbackY_cur = feedbackY[trial_inds]
        # Movement positions [after deleting wrong trials]
        movement_cur = movement[trial_inds]
        # Error positions
        errors_cur = errors[trial_inds]

        # keep only non_hit trials
        non_hit_cur = point_in_circle(targets_cur, target_coords,
                                         feedbackX_cur, feedbackY_cur,
                                         radius_target + radius_cursor)
        non_hit = non_hit_cur
        non_hit = np.array(non_hit)
        # remove trials following hit (because no previous error)
        # non_hit = ~(~non_hit | ~np.insert(non_hit, 0, 1)[:-1])
        non_hit = np.insert(non_hit, 0, 0)[:-1]
        non_hit[[0, 192]] = False  # Removing first trial of each block

        values_for_es,analysis_value,varnames,varnames_def = getErrSensVals(errors_cur,targets_cur,movement_cur)
        #Y_es = np.array(values_for_es)
        #analysis_value = analysis_value[:, non_hit].T

        values_for_es = values_for_es[:, non_hit]
        # 0 -- target angs, 1 -- prev target angs, 2 -- current movemnet, 3 -- prev movements, 4 -- prev error
        # es = (target - current) - (prev tgt - prev movemnet) / prev_error
        # prev_error is taken from df['error']
        # movement _relative_ to the target

        #Q: why we divide by prev_error instead of by (Y_es[1]-Y_es[3])
        corr = (values_for_es[0]-values_for_es[2]) - (values_for_es[1]-values_for_es[3])
        es = corr/values_for_es[4]
        #values_for_es[4]  ==  df['feedback'] - df['target']

        env2non_hit[env]        = non_hit
        env2corr[env]            = corr
        env2err_sens[env]            = es
        env2pre_err_sens[env]  = values_for_es
        env2pre_dec_data[env]           = analysis_value

    #results_folder = folder
    #fname = op.join(folder, f'{subject}_{env}.npy')
    if fname is not None:
        print(f'Saving to {fname}')
        np.savez(fname, dict(env2err_sens=env2err_sens,
           env2pre_err_sens=env2pre_err_sens,env2pre_dec_data=env2pre_dec_data,
           env2non_hit=env2non_hit) )

    return env2err_sens, env2pre_err_sens,env2pre_dec_data, env2non_hit

def adjustNonHitDf(df, env, time_locked, hits_to_use = ['current_trial']):
    raise ValueError('not finished implementation')
    assert np.all( np.isin(hits_to_use, ['current_trial','shifted_trial'] ) )

    non_hit = df['non_hit']
    # return a mask
    # some adjustment of non_hit, based on time lock and whether we are in
    # stable or random environment
    #non_hit = non_hit.copy()
    from config2 import n_trials_in_block as N
    # it is painful to do those inserts in case of pandas
    raise ValueError('not implemented')
    if 'current_trial' in hits_to_use:
        nhr = non_hit
    else:
        nhr = np.ones(len(non_hit), dtype=bool )

    if 'shifted_trial' in hits_to_use:
        if env == 'all':
            if time_locked == 'feedback':
                # remove trials preceding hit (because no next error)
                nhr &= np.insert(nhr, len(nhr), 1)[1:]
                nhr[N*4 - 1] = False  # Removing last trial of each block
            elif time_locked == 'target':
                # remove trials following hit (because no previous error)
                nhr &= np.insert(nhr, 0, 1)[:-1]
                nhr[0] = False  # Removing first trial of each block
        elif env == 'stable':
            if time_locked == 'feedback':
                # remove trials preceding hit (because no next error)
                nhr &= np.insert(nhr, len(nhr), 1)[1:]
                nhr[[N-1, N*2-1]] = False  # Removing last trial of each block
            elif time_locked == 'target':
                # remove trials following hit (because no previous error)
                nhr &= np.insert(nhr, 0, 1)[:-1]
                nhr[[0, N]] = False  # Removing first trial of each block
        elif env == 'random':
            if time_locked == 'feedback':
                # remove trials preceding hit (because no next error)
                nhr &= np.insert(nhr, len(nhr), 1)[1:]
                nhr[[N-1, N*2-1]] = False  # Removing last trial of each block
            elif time_locked == 'target':
                # remove trials following hit (because no previous error)
                nhr &= np.insert(nhr, 0, 1)[:-1]
                nhr[[0, N]] = False  # Removing first trial of each block

    df['non_hit_adj'] = nhr
    return nhr

def adjustNonHitDf_old(df, env, time_locked, hits_to_use = ['current_trial']):
    raise ValueError('not finished implementation')
    #hits_to_use

    non_hit = df['non_hit']
    # return a mask
    # some adjustment of non_hit, based on time lock and whether we are in
    # stable or random environment
    non_hit = non_hit.copy()
    from config2 import n_trials_in_block as N
    # it is painful to do those inserts in case of pandas
    raise ValueError('not implemented')
    if env == 'all':
        if time_locked == 'feedback':
            # remove trials preceding hit (because no next error)
            non_hit = ~(~non_hit | ~np.insert(non_hit, len(non_hit), 1)[1:])
            non_hit[N*4 - 1] = False  # Removing last trial of each block
        elif time_locked == 'target':
            # remove trials following hit (because no previous error)
            non_hit = ~(~non_hit | ~np.insert(non_hit, 0, 1)[:-1])
            non_hit[0] = False  # Removing first trial of each block
    elif env == 'stable':
        if time_locked == 'feedback':
            # remove trials preceding hit (because no next error)
            non_hit = ~(~non_hit | ~np.insert(non_hit, len(non_hit), 1)[1:])
            non_hit[[N-1, N*2-1]] = False  # Removing last trial of each block
        elif time_locked == 'target':
            # remove trials following hit (because no previous error)
            non_hit = ~(~non_hit | ~np.insert(non_hit, 0, 1)[:-1])
            non_hit[[0, N]] = False  # Removing first trial of each block
    elif env == 'random':
        if time_locked == 'feedback':
            # remove trials preceding hit (because no next error)
            non_hit = ~(~non_hit | ~np.insert(non_hit, len(non_hit), 1)[1:])
            non_hit[[N-1, N*2-1]] = False  # Removing last trial of each block
        elif time_locked == 'target':
            # remove trials following hit (because no previous error)
            non_hit = ~(~non_hit | ~np.insert(non_hit, 0, 1)[:-1])
            non_hit[[0, N]] = False  # Removing first trial of each block
    return non_hit

def adjustNonHit(non_hit,env,time_locked):
    # return a mask
    # some adjustment of non_hit, based on time lock and whether we are in
    # stable or random environment
    non_hit = non_hit.copy()
    from config2 import n_trials_in_block as N
    if env == 'all':
        if time_locked == 'feedback':
            # remove trials preceding hit (because no next error)
            non_hit = ~(~non_hit | ~np.insert(non_hit, len(non_hit), 1)[1:])
            non_hit[N*4 - 1] = False  # Removing last trial of each block
        elif time_locked == 'target':
            # remove trials following hit (because no previous error)
            non_hit = ~(~non_hit | ~np.insert(non_hit, 0, 1)[:-1])
            non_hit[0] = False  # Removing first trial of each block
    elif env == 'stable':
        if time_locked == 'feedback':
            # remove trials preceding hit (because no next error)
            non_hit = ~(~non_hit | ~np.insert(non_hit, len(non_hit), 1)[1:])
            non_hit[[N-1, N*2-1]] = False  # Removing last trial of each block
        elif time_locked == 'target':
            # remove trials following hit (because no previous error)
            non_hit = ~(~non_hit | ~np.insert(non_hit, 0, 1)[:-1])
            non_hit[[0, N]] = False  # Removing first trial of each block
    elif env == 'random':
        if time_locked == 'feedback':
            # remove trials preceding hit (because no next error)
            non_hit = ~(~non_hit | ~np.insert(non_hit, len(non_hit), 1)[1:])
            non_hit[[N-1, N*2-1]] = False  # Removing last trial of each block
        elif time_locked == 'target':
            # remove trials following hit (because no previous error)
            non_hit = ~(~non_hit | ~np.insert(non_hit, 0, 1)[:-1])
            non_hit[[0, N]] = False  # Removing first trial of each block
    return non_hit

def getAnalysisData(env, time_locked, control_type, behav_df):
    control_types_all = ['feedback', 'movement' , 'target', 'belief']
    assert control_type in control_types_all
    assert time_locked in ['feedback', 'target']
    environment_all = np.array(behav_df['environment'])
    if env != 'all':
        envcode = env2envcode[env]
        trial_inds = np.where(environment_all == envcode)[0]
    else:
        trial_inds = np.arange(len(environment_all))

    errors = np.array(behav_df['error'])[trial_inds]

    targets = np.array(behav_df['target'])[trial_inds]
    feedback = np.array(behav_df['feedback'])[trial_inds]
    feedbackX = np.array(behav_df['feedbackX'])[trial_inds]
    feedbackY = np.array(behav_df['feedbackY'])[trial_inds]
    movement = np.array(behav_df['org_feedback'])[trial_inds]
    errors = np.array(behav_df['error'])     [trial_inds]
    belief = np.array(behav_df['belief'])    [trial_inds]

    prev_targets  = np.insert(targets, 0, 0)[:-1]
    prev_errors  = np.insert(errors, 0, 0)[:-1]
    next_errors  = np.insert(errors, len(errors), 0)[1:]
    prev_feedback  = np.insert(feedback, 0, 0)[:-1]
    prev_movement  = np.insert(movement, 0, 0)[:-1]
    prev_belief  = np.insert(belief, 0, 0)[:-1]
    next_belief  = np.insert(belief, len(belief), 0)[1:]

    non_hit = point_in_circle(targets, target_coords,
                          feedbackX, feedbackY,
                          radius_target + radius_cursor)
    non_hit = np.array(non_hit)

    from config2 import n_trials_in_block as N
    non_hit = adjustNonHit(non_hit,env,time_locked)
    if env == 'all':
        #ep = epochs['20', '21', '22', '23', '30',
        #            '25', '26', '27', '28', '35']
        if time_locked == 'feedback':
            if control_type == 'feedback':
                analysis_name = 'feedback_errors_next_errors_belief'
                analysis_value = [feedback, errors, next_errors,
                                  belief]
            elif control_type == 'movement':
                analysis_name = 'movement_errors_next_errors_belief'
                analysis_value = [movement, errors, next_errors,
                                  belief]
            elif control_type == 'target':
                analysis_name = 'target_errors_nexterrors_belief'
                analysis_value = [targets, errors, next_errors,
                                  belief]
            elif control_type == 'belief':
                analysis_name = 'belief_errors_nexterrors'
                analysis_value = [belief, errors, next_errors]
            # remove trials preceding hit (because no next error)
            #non_hit = ~(~non_hit | ~np.insert(non_hit, len(non_hit), 1)[1:])
            #non_hit[N*4 - 1] = False  # Removing last trial of each block
        elif time_locked == 'target':
            if control_type == 'feedback':
                analysis_name = 'prevfeedback_preverrors_errors_prevbelief'
                analysis_value = [prev_feedback, prev_errors, errors,
                                  prev_belief]
            elif control_type == 'movement':
                analysis_name = 'prevmovement_preverrors_errors_prevbelief'
                analysis_value = [prev_movement, prev_errors, errors,
                                  prev_belief]
            elif control_type == 'target':
                analysis_name = 'prevtarget_preverrors_errors_prevbelief'
                analysis_value = [prev_targets, prev_errors, errors,
                                  prev_belief]
            elif control_type == 'belief':
                analysis_name = 'prevbelief_preverrors_errors'
                analysis_value = [prev_belief, prev_errors, errors]
            # remove trials following hit (because no previous error)
            #non_hit = ~(~non_hit | ~np.insert(non_hit, 0, 1)[:-1])
            #non_hit[0] = False  # Removing first trial of each block
    elif env == 'stable':
        if time_locked == 'feedback':
            if control_type == 'feedback':
                analysis_name = 'feedback_errors_next_errors_belief'
                analysis_value = [feedback, errors,
                                  next_errors, belief]
            elif control_type == 'movement':
                analysis_name = 'movement_errors_next_errors_belief'
                analysis_value = [movement, errors,
                                  next_errors, belief]
            elif control_type == 'target':
                analysis_name = 'target_errors_nexterrors_belief'
                analysis_value = [targets, errors,
                                  next_errors, belief]
            elif control_type == 'belief':
                analysis_name = 'belief_errors_nexterrors'
                analysis_value = [belief, errors,
                                  next_errors]
            # remove trials preceding hit (because no next error)
            #non_hit = ~(~non_hit | ~np.insert(non_hit, len(non_hit), 1)[1:])
            #non_hit[[N-1, N*2-1]] = False  # Removing last trial of each block
        elif time_locked == 'target':
            if control_type == 'feedback':
                analysis_name = 'prevfeedback_preverrors_errors_prevbelief'
                analysis_value = [prev_feedback, prev_errors,
                                  errors, prev_belief]
            elif control_type == 'movement':
                analysis_name = 'prevmovement_preverrors_errors_prevbelief'
                analysis_value = [prev_movement, prev_errors,
                                  errors, prev_belief]
            elif control_type == 'target':
                analysis_name = 'prevtarget_preverrors_errors_prevbelief'
                analysis_value = [prev_targets, prev_errors,
                                  errors, prev_belief]
            elif control_type == 'belief':
                analysis_name = 'prevbelief_preverrors_errors'
                analysis_value = [prev_belief, prev_errors,
                                  errors]
            # remove trials following hit (because no previous error)
            #non_hit = ~(~non_hit | ~np.insert(non_hit, 0, 1)[:-1])
            #non_hit[[0, N]] = False  # Removing first trial of each block
    elif env == 'random':
        if time_locked == 'feedback':
            if control_type == 'feedback':
                analysis_name = 'feedback_errors_next_errors_belief'
                analysis_value = [feedback, errors,
                                  next_errors, belief]
            elif control_type == 'movement':
                analysis_name = 'movement_errors_next_errors_belief'
                analysis_value = [movement, errors,
                                  next_errors, belief]
            elif control_type == 'target':
                analysis_name = 'target_errors_nexterrors_belief'
                analysis_value = [targets, errors,
                                  next_errors, belief]
            elif control_type == 'belief':
                analysis_name = 'belief_errors_nexterrors'
                analysis_value = [belief, errors,
                                  next_errors]
            # remove trials preceding hit (because no next error)
            #non_hit = ~(~non_hit | ~np.insert(non_hit, len(non_hit), 1)[1:])
            #non_hit[[N-1, N*2-1]] = False  # Removing last trial of each block
        elif time_locked == 'target':
            if control_type == 'feedback':
                analysis_name = 'prevfeedback_preverrors_errors_prevbelief'
                analysis_value = [prev_feedback, prev_errors,
                                  errors, prev_belief]
            elif control_type == 'movement':
                analysis_name = 'prevmovement_preverrors_errors_prevbelief'
                analysis_value = [prev_movement, prev_errors,
                                  errors, prev_belief]
            elif control_type == 'target':
                analysis_name = 'prevtarget_preverrors_errors_prevbelief'
                analysis_value = [prev_targets, prev_errors,
                                  errors, prev_belief]
            elif control_type == 'belief':
                analysis_name = 'prevbelief_preverrors_errors'
                analysis_value = [prev_belief, prev_errors,
                                  errors]
            # remove trials following hit (because no previous error)
            #non_hit = ~(~non_hit | ~np.insert(non_hit, 0, 1)[:-1])
            #non_hit[[0, N]] = False  # Removing first trial of each block

    return analysis_name, analysis_value, non_hit


#def getEpochs_custom(raw,event_ids, tmin=None,tmax=None, bsl=None ):
#    events = mne.find_events(raw, stim_channel='UPPT001',
#                            min_duration=0.02)
#    events[:, 0] += delay_trig_photodi  # to account for delay between trig. & photodi.
#
#    epochs = Epochs(raw, events, event_id=event_ids,
#                            tmin=-0.2, tmax=3, preload=True,
#                            baseline=(-0.2, 0), decim=6)
#    return [ ('custom',epochs) ]

# from td_long2
def getEpochs(raw,is_short,bsl):
    events = mne.find_events(raw, stim_channel='UPPT001',
                            min_duration=min_event_duration)
    events[:, 0] += delay_trig_photodi  # to account for delay between trig. & photodi.
    # diffence is tmax and which events are taken into acc
    # also how baseline is computed (what is considered to be baseline data)
    if is_short:
        if bsl:
            epochs_feedback = Epochs(raw, events, event_id=event_ids_feedback,
                                    tmin=-0.2, tmax=3, preload=True,
                                    baseline=(-0.2, 0), decim=6)
        else:
            epochs_feedback = Epochs(raw, events, event_id=event_ids_feedback,
                                    tmin=-0.2, tmax=3, preload=True,
                                    baseline=None, decim=6)
        epochs_feedback.pick_types(meg=True, misc=False)
    else:
        if bsl:
            epochs_feedback = Epochs(raw, events, event_id=event_ids_feedback,
                                    tmin=-2, tmax=5, preload=True,
                                    baseline=None, decim=6)
            epochs_target = Epochs(raw, events, event_id=event_ids_tgt,
                                tmin=-5, tmax=2, preload=True,
                                baseline=None, decim=6)
            epochs_bsl = Epochs(raw, events, event_id=event_ids_tgt,
                                tmin=-0.46, tmax=-0.05, preload=True,
                                baseline=None, decim=6)
            # Apply baseline before the target to the epochs time-locked on feedback
            bsl_channels = mne.pick_types(epochs_feedback.info, meg=True)
            bsl_data = epochs_bsl.get_data()[:, bsl_channels, :]
            bsl_data = np.mean(bsl_data, axis=2)
            epochs_feedback._data[:, bsl_channels, :] -= bsl_data[:, :, np.newaxis]
            # Apply baseline before the target to the epochs time-locked on targets
            # we use the baseline from trials n-1 (the first trial is remove in
            # subsequent analysis)
            epochs_target._data[1:, bsl_channels, :] -= bsl_data[:-1, :, np.newaxis]
        else:
            epochs_feedback = Epochs(raw, events, event_id=event_ids_feedback,
                                    tmin=-2, tmax=5, preload=True,
                                    baseline=None, decim=6)
            epochs_target = Epochs(raw, events, event_id=event_ids_tgt,
                                tmin=-5, tmax=2, preload=True,
                                baseline=None, decim=6)
        epochs_feedback.pick_types(meg=True, misc=False)
        epochs_target.pick_types(meg=True, misc=False)

    if is_short:
        epochs_type = zip(['feedback'],
                        [epochs_feedback])
    else:
        epochs_type = zip(['feedback', 'target'],
                        [epochs_feedback, epochs_target])

    return epochs_type

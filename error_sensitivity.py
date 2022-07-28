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
    meg_targets = epochs.events[:, 2].copy()
    #targets = np.array( behav_df['target_codes'] )
    targets = np.array( behav_df['target_inds'] )
    assert len(targets)
    assert len(meg_targets)
    print(  len(targets) , len(meg_targets) )
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
def getErrSensVals(errors_cur,targets_cur,movement_cur):
    # here targets_cur should be indices of targets
    # shit errors_cur by -1
    prev_errors_cur   = np.insert(errors_cur, 0, 0)[:-1]
    prev_targets_cur  = np.insert(targets_cur, 0, 0)[:-1]
    prev_movement_cur = np.insert(movement_cur, 0, 0)[:-1]
    analysis_value = [prev_targets_cur, prev_movement_cur,
                      errors_cur,
                      prev_errors_cur]
    values_for_es = [target_angs[targets_cur],
                     target_angs[prev_targets_cur],
                     movement_cur,
                     prev_movement_cur,
                     prev_errors_cur]
    return np.array(values_for_es), np.array(analysis_value)


def computeErrSens2(behav_df, epochs, subject, trial_inds, do_delete_trials=1,
                    enforce_consistency=0):
    # before enforcing consistencey
    environment  = np.array(behav_df['environment'])
    # modifies behav_df in place
    if enforce_consistency:
        behav_df = enforceTargetTriggerConsistency(behav_df, epochs, environment,
                                        do_delete_trials=do_delete_trials )
    targets      = np.array(behav_df['target_inds'])
    # after enforcing consistencey
    environment  = np.array(behav_df['environment'])
    errors       = np.array(behav_df['error'])
    #feedback     = np.array(behav_df['feedback'])
    feedbackX    = np.array(behav_df['feedbackX'])
    feedbackY    = np.array(behav_df['feedbackY'])
    movement = np.array(behav_df['org_feedback'])

    # after deleting wrong triggers
    targets_cur = targets[trial_inds]
    # Feedback positions
    #feedback_cur = feedback[trial_inds]
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

    values_for_es,analysis_value = getErrSensVals(errors_cur,targets_cur,
                                                  movement_cur)
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

    return non_hit,corr,es,values_for_es

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

    events = mne.find_events(raw, stim_channel=stim_chn, min_duration=0.02)
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

        values_for_es,analysis_value = getErrSensVals(errors_cur,targets_cur,movement_cur)
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

def getAnalysisData(env, time_locked, control_type, behav_df):
    control_types_all = ['feedback', 'movement' , 'target', 'belief']
    assert control_type in control_types_all
    assert time_locked in ['feedback', 'target']
    environment = np.array(behav_df['environment'])
    if env != 'all':
        envcode = env2envcode[env]
        trial_inds = np.where(environment == envcode)[0]
    else:
        trial_inds = np.arange(len(environment))
    errors_cur = np.array(behav_df['error'])[trial_inds]

    targets_cur = np.array(behav_df['target'])[trial_inds]
    feedback_cur = np.array(behav_df['feedback'])[trial_inds]
    feedbackX_cur = np.array(behav_df['feedbackX'])[trial_inds]
    feedbackY_cur = np.array(behav_df['feedbackY'])[trial_inds]
    movement_cur = np.array(behav_df['org_feedback'])[trial_inds]
    errors_cur = np.array(behav_df['error'])     [trial_inds]
    belief_cur = np.array(behav_df['belief'])    [trial_inds]

    prev_targets_cur  = np.insert(targets_cur, 0, 0)[:-1]
    prev_errors_cur  = np.insert(errors_cur, 0, 0)[:-1]
    next_errors_cur  = np.insert(errors_cur, len(errors_cur), 0)[1:]
    prev_feedback_cur  = np.insert(feedback_cur, 0, 0)[:-1]
    prev_movement_cur  = np.insert(movement_cur, 0, 0)[:-1]
    prev_belief_cur  = np.insert(belief_cur, 0, 0)[:-1]
    next_belief_cur  = np.insert(belief_cur, len(belief_cur), 0)[1:]

    non_hit_cur = point_in_circle(targets_cur, target_coords,
                          feedbackX_cur, feedbackY_cur,
                          radius_target + radius_cursor)
    non_hit_cur = np.array(non_hit_cur)


    N = 192 # ntrials in the block
    if env == 'all':
        #ep = epochs['20', '21', '22', '23', '30',
        #            '25', '26', '27', '28', '35']
        if time_locked == 'feedback':
            if control_type == 'feedback':
                analysis_name = 'feedback_errors_next_errors_belief'
                analysis_value = [feedback_cur, errors_cur, next_errors_cur,
                                  belief_cur]
            elif control_type == 'movement':
                analysis_name = 'movement_errors_next_errors_belief'
                analysis_value = [movement_cur, errors_cur, next_errors_cur,
                                  belief_cur]
            elif control_type == 'target':
                analysis_name = 'target_errors_nexterrors_belief'
                analysis_value = [targets_cur, errors_cur, next_errors_cur,
                                  belief_cur]
            elif control_type == 'belief':
                analysis_name = 'belief_errors_nexterrors'
                analysis_value = [belief_cur, errors_cur, next_errors_cur]
            # remove trials preceding hit (because no next error)
            non_hit_cur = ~(~non_hit_cur | ~np.insert(non_hit_cur,
                len(non_hit_cur), 1)[1:])
            non_hit_cur[767] = False  # Removing last trial of each block
        elif time_locked == 'target':
            if control_type == 'feedback':
                analysis_name = 'prevfeedback_preverrors_errors_prevbelief'
                analysis_value = [prev_feedback_cur, prev_errors_cur, errors_cur,
                                  prev_belief_cur]
            elif control_type == 'movement':
                analysis_name = 'prevmovement_preverrors_errors_prevbelief'
                analysis_value = [prev_movement_cur, prev_errors_cur, errors_cur,
                                  prev_belief_cur]
            elif control_type == 'target':
                analysis_name = 'prevtarget_preverrors_errors_prevbelief'
                analysis_value = [prev_targets_cur, prev_errors_cur, errors_cur,
                                  prev_belief_cur]
            elif control_type == 'belief':
                analysis_name = 'prevbelief_preverrors_errors'
                analysis_value = [prev_belief_cur, prev_errors_cur, errors_cur]
            # remove trials following hit (because no previous error)
            non_hit_cur = ~(~non_hit_cur | ~np.insert(non_hit_cur, 0, 1)[:-1])
            non_hit_cur[0] = False  # Removing first trial of each block
    elif env == 'stable':
        if time_locked == 'feedback':
            if control_type == 'feedback':
                analysis_name = 'feedback_errors_next_errors_belief'
                analysis_value = [feedback_cur, errors_cur,
                                  next_errors_cur, belief_cur]
            elif control_type == 'movement':
                analysis_name = 'movement_errors_next_errors_belief'
                analysis_value = [movement_cur, errors_cur,
                                  next_errors_cur, belief_cur]
            elif control_type == 'target':
                analysis_name = 'target_errors_nexterrors_belief'
                analysis_value = [targets_cur, errors_cur,
                                  next_errors_cur, belief_cur]
            elif control_type == 'belief':
                analysis_name = 'belief_errors_nexterrors'
                analysis_value = [belief_cur, errors_cur,
                                  next_errors_cur]
            # remove trials preceding hit (because no next error)
            non_hit_cur = ~(~non_hit_cur | ~np.insert(non_hit_cur, len(non_hit_cur), 1)[1:])
            non_hit_cur[[N-1, N*2-1]] = False  # Removing last trial of each block
        elif time_locked == 'target':
            if control_type == 'feedback':
                analysis_name = 'prevfeedback_preverrors_errors_prevbelief'
                analysis_value = [prev_feedback_cur, prev_errors_cur,
                                  errors_cur, prev_belief_cur]
            elif control_type == 'movement':
                analysis_name = 'prevmovement_preverrors_errors_prevbelief'
                analysis_value = [prev_movement_cur, prev_errors_cur,
                                  errors_cur, prev_belief_cur]
            elif control_type == 'target':
                analysis_name = 'prevtarget_preverrors_errors_prevbelief'
                analysis_value = [prev_targets_cur, prev_errors_cur,
                                  errors_cur, prev_belief_cur]
            elif control_type == 'belief':
                analysis_name = 'prevbelief_preverrors_errors'
                analysis_value = [prev_belief_cur, prev_errors_cur,
                                  errors_cur]
            # remove trials following hit (because no previous error)
            non_hit_cur = ~(~non_hit_cur | ~np.insert(non_hit_cur, 0, 1)[:-1])
            non_hit_cur[[0, N]] = False  # Removing first trial of each block
    elif env == 'random':
        if time_locked == 'feedback':
            if control_type == 'feedback':
                analysis_name = 'feedback_errors_next_errors_belief'
                analysis_value = [feedback_cur, errors_cur,
                                  next_errors_cur, belief_cur]
            elif control_type == 'movement':
                analysis_name = 'movement_errors_next_errors_belief'
                analysis_value = [movement_cur, errors_cur,
                                  next_errors_cur, belief_cur]
            elif control_type == 'target':
                analysis_name = 'target_errors_nexterrors_belief'
                analysis_value = [targets_cur, errors_cur,
                                  next_errors_cur, belief_cur]
            elif control_type == 'belief':
                analysis_name = 'belief_errors_nexterrors'
                analysis_value = [belief_cur, errors_cur,
                                  next_errors_cur]
            # remove trials preceding hit (because no next error)
            non_hit_cur = ~(~non_hit_cur | ~np.insert(non_hit_cur, len(non_hit_cur), 1)[1:])
            non_hit_cur[[N-1, N*2-1]] = False  # Removing last trial of each block
        elif time_locked == 'target':
            if control_type == 'feedback':
                analysis_name = 'prevfeedback_preverrors_errors_prevbelief'
                analysis_value = [prev_feedback_cur, prev_errors_cur,
                                  errors_cur, prev_belief_cur]
            elif control_type == 'movement':
                analysis_name = 'prevmovement_preverrors_errors_prevbelief'
                analysis_value = [prev_movement_cur, prev_errors_cur,
                                  errors_cur, prev_belief_cur]
            elif control_type == 'target':
                analysis_name = 'prevtarget_preverrors_errors_prevbelief'
                analysis_value = [prev_targets_cur, prev_errors_cur,
                                  errors_cur, prev_belief_cur]
            elif control_type == 'belief':
                analysis_name = 'prevbelief_preverrors_errors'
                analysis_value = [prev_belief_cur, prev_errors_cur,
                                  errors_cur]
            # remove trials following hit (because no previous error)
            non_hit_cur = ~(~non_hit_cur | ~np.insert(non_hit_cur, 0, 1)[:-1])
            non_hit_cur[[0, 192]] = False  # Removing first trial of each block

    return analysis_name, analysis_value, non_hit_cur


def getEpochs_custom(raw,event_ids, tmin=None,tmax=None, bsl=None ):
    events = mne.find_events(raw, stim_channel='UPPT001',
                            min_duration=0.02)
    events[:, 0] += 18  # to account for delay between trig. & photodi.

    epochs = Epochs(raw, events, event_id=event_ids,
                            tmin=-0.2, tmax=3, preload=True,
                            baseline=(-0.2, 0), decim=6)
    return [ ('custom',epochs) ]

# from td_long2
def getEpochs(raw,is_short,bsl):
    events = mne.find_events(raw, stim_channel='UPPT001',
                            min_duration=0.02)
    events[:, 0] += 18  # to account for delay between trig. & photodi.
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

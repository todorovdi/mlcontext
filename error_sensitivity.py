import os
import os.path as op
import numpy as np
#import mne
#from mne.io import read_raw_fif
#from mne import Epochs
import pandas as pd
import warnings
import sys
from base2 import (int_to_unicode, point_in_circle,
                   calc_target_coordinates_centered, radius_target,
                   radius_cursor)
from config2 import path_data
from config2 import event_ids_tgt,event_ids_feedback, env2envcode, env2subtr
from config2 import target_angs, stage2evn2event_ids
from config2 import delay_trig_photodi, min_event_duration
from base2 import subAngles

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
    from Levenshtein import editops
    assert (np.diff( np.array(behav_df.index) ) > 0).all()

    # remove some of the behav inds
    # WARNING: this will fail (give empty)if epochs are based of time_locked=feedback
    # (or maybe it was only in old code?)
    meg_targets = epochs.events[:, 2].copy()
    target_inds_behav = np.array( behav_df['target_inds'] )

    assert len(target_inds_behav)
    assert len(meg_targets)
    print(  len(target_inds_behav) , len(meg_targets) )
    if environment is None:
        environment = behav_df['environment']

    # for different environments target signals stored in .fif have different codes (20+ vs 30+) so
    # we have to separate
    for env,envcode in env2envcode.items():
        #trial_inds = np.where(environment == envcode)[0]
        trial_inds = np.where( np.isin(meg_targets,
            stage2evn2event_ids['target'][env] ) )[0]
        # it uses that target codes are conequitive integers
        meg_targets[trial_inds] = meg_targets[trial_inds] - env2subtr[env]

    behav_df_res = None
    if do_delete_trials:
        # one can have less triggers than behav
        changes = editops(int_to_unicode(target_inds_behav),
                          int_to_unicode(meg_targets))
        # we have missing triggers in MEG file so we delete stuff from behav file
        delete_trials = [change[1] for change in changes]
        # read behav.pkl and remove bad_trials if bad_trials is not empty
        if len( delete_trials) :
            print(f'enforceTargetTriggerConsistency: removing '
                  f'{len(delete_trials)} trials' )
            #behav_df = behav_df_full.copy().drop(delete_trials, errors='ignore')
            behav_df_res = behav_df.drop(delete_trials, errors='ignore', axis=0)
            behav_df_res = behav_df_res.reset_index()
            if save_fname is not None:
                behav_df_res.to_pickle(save_fname)

            targets_old = np.array( behav_df['target_inds'] )
            target_inds_behav = np.array( behav_df_res['target_inds'] )
            print('old len = ',len(target_inds_behav), 'new len = ',len(targets_old))

            assert len(target_inds_behav)
        else:
            behav_df_res = behav_df


        # respects ordering
        if np.array_equal(meg_targets, target_inds_behav):
            if len(delete_trials):
                print(f'enforceTargetTriggerConsistency: Deleted {len(delete_trials)} trials')
        else:
            warnings.warn('MEG events and behavior file do not match')

        assert behav_df is not None
    else:
        behav_df_res = behav_df
    return behav_df_res

def checkTgtConsistency(behav_df, epochs):
    assert len(behav_df) == len(epochs), (len(behav_df), len(epochs) )

    meg_targets = epochs.events[:, 2].copy()
    target_inds_behav = np.array( behav_df['target_inds'] )

    assert len(target_inds_behav)
    assert len(meg_targets)
    # for different environments target signals stored in .fif have different codes (20+ vs 30+) so
    # we have to separate
    for env,envcode in env2envcode.items():
        trial_inds = np.where( np.isin(meg_targets,
            stage2evn2event_ids['target'][env] ) )[0]
        # it uses that target codes are conequitive integers
        meg_targets[trial_inds] = meg_targets[trial_inds] - env2subtr[env]

    return np.array_equal(meg_targets, target_inds_behav)

def getPrev(vals, defval = np.nan, shiftsz=1):
    ins = [ defval ] * shiftsz
    outvals = np.insert(vals, 0, ins)[:-shiftsz]
    return outvals

def getIndShifts(trial_inds, time_locked='target', shiftsz=1):
    ''' can be both global inds and local inds'''
    insind = [ -1000000 ] * shiftsz

    valid_inds_cur  = np.ones( len(trial_inds), dtype=bool )
    valid_inds_next = np.ones( len(trial_inds), dtype=bool )
    if time_locked in ['target', 'trial_end']:
        # insert in the beginning, i.e. shift right, i.e.
        # if trial_inds_cur[i] = trial_inds[i-1]
        trial_inds_cur = np.insert(trial_inds, 0, insind)[:-shiftsz]
        trial_inds_next      = trial_inds
    elif time_locked == 'feedback':
        trial_inds_cur       = trial_inds
        # insert in the end
        trial_inds_next      = np.insert(trial_inds, len(trial_inds), insind)[shiftsz:]
        # if trial_inds_next[i] = trial_inds[i+1]
    else:
        raise ValueError(time_locked)

    valid_inds_cur[trial_inds_cur < 0] = False
    valid_inds_next[trial_inds_next < 0] = False

    return  trial_inds_cur, valid_inds_cur, trial_inds_next, valid_inds_next

def shiftVals(vals, trial_inds_cur, valid_inds_cur,
              invalid_val = np.nan):
    '''
    only applies indices (only valid ones)
    '''
    assert len(vals) == len(trial_inds_cur)
    vals_cur  = np.ones(len(vals) ) * invalid_val
    vals_cur[valid_inds_cur] = vals[trial_inds_cur[valid_inds_cur] ]

    return vals_cur


# legacy
def getErrSensVals(error, target_inds, movement, time_locked='target',
                   target_locs = None,
                   ret_df=False, shiftsz = 1):
    # here target should be indices of target
    # shift error by -1

    assert shiftsz >= 1
    #ins = [ 0 ] * shiftsz
    ins = [ np.nan ] * shiftsz
    ins_int = [ -10000 ] * shiftsz

    assert len( set([ len(error), len(target_inds), len(movement) ] ) ) == 1

    #getAnalysisData('all',time_locked,control_type,behad_df)
    varnames_def = ['target','prev_target', 'movement', 'prev_movement',
                'prev_error']

    if target_locs is None:
        target_locs = target_angs[target_inds]
    target_locs = target_locs.astype(float)

    if time_locked == 'target':
        # add zero in the beginning
        prev_errors   = np.insert(error,         0, ins)[:-shiftsz]
        prev_target_ind   = np.insert(target_inds,   0, ins_int)[:-shiftsz]
        prev_movement = np.insert(movement,      0, ins)[:-shiftsz]
        analysis_value = [prev_target_ind, prev_movement,
                        error,
                        prev_errors]

        #tas = target_angs[prev_target_ind]
        #tas[:shiftsz] = np.nan
        prev_target_locs   = np.insert(target_locs,   0, ins)[:-shiftsz]
        prev_target_locs[:shiftsz] = np.nan

        values_for_es = [target_locs,
                        prev_target_locs,
                        movement,
                        prev_movement,
                        prev_errors]
        varnames = varnames_def
    else:
        # add zero in the end
        next_errors      = np.insert(error,     len(error),    ins)[shiftsz:]
        next_target_ind  = np.insert(target_inds,
                len(target_inds),   ins_int)[shiftsz:]
        next_movement    = np.insert(movement,  len(movement), ins)[shiftsz:]

        analysis_value = [target_inds, movement,
                        next_errors,
                        error]

        #tas = target_angs[next_target_ind]
        #tas[-shiftsz:] = np.nan

        next_target_locs   = np.insert(target_locs,
                len(target_locs), ins) [shiftsz:]
        next_target_locs[-shiftsz:] = np.nan

        values_for_es = [next_target_locs,
                        target_locs,
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
        df = pd.DataFrame()
        for vn,vs in zip(varnames,values_for_es):
            df[vn] = vs

        return df,vndef2vn
    else:
        return np.array(values_for_es), np.array(analysis_value), \
            varnames, varnames_def

def computeErrSens3(behav_df, df_inds=None, epochs=None,
                    do_delete_trials=1,
                    enforce_consistency=0, time_locked='target',
                    correct_hit = 'inf', error_type = 'MPE',
                    colname_nh = 'non_hit_not_adj',
                    coln_nh_out = None,
                    shiftsz=1,
                    err_sens_coln = 'err_sens',
                    addvars = [], recalc_non_hit = True, target_info_type = 'inds',
                    coln_correction_calc = None,
                    coln_error = 'error',
                    df_fulltraj = None,
                    trajPair2corr = None,
                    verbose = 0, use_sub_angles = 0, retention_factor = 1.,
                    reref_target_locs = True,
                   long_shift_numerator = False ):
    '''
    computes error sensitiviy across dataset. So indexing is very important here.
    '''

    assert shiftsz >= 1

    # in read_beahav we compute
    # errors.append(feedback[trial] - target_locs[trial])
    # target_locs being target angles

    #os.path
    # telecope and coc

    assert correct_hit in ['prev_valid', 'zero', 'inf', 'nan' ]
    # modifies behav_df in place
    if enforce_consistency:
        # before enforcing consistencey. VERY DANGEROUS HERE since it DOES NOT take
        # only df_inds
        environment  = np.array(behav_df['environment'])
        assert epochs is not None
        behav_df = enforceTargetTriggerConsistency(behav_df,
            epochs, environment, do_delete_trials=do_delete_trials )

    if df_inds is None:
        df_inds = behav_df.index

    dis = np.diff(behav_df.index.values)
    np.max( dis ) == np.max( dis ) and np.max( dis ) == 1

    if error_type == 'MPE':
        errors0       = np.array(behav_df.loc[df_inds, coln_error])
    else:
        raise ValueError(f'error_type={error_type} not implemnted')

    if recalc_non_hit or (colname_nh not in behav_df.columns):
        raise ValueError(f'{colname_nh} is not present and recalc_non_hit not implemneted')

    nonhit = behav_df[colname_nh].to_numpy()

    # not that non-hit has different effect on error sens calc depending on
    # which time_locked is used
    ins_nh = [ 0 ] * shiftsz
    # in prev vers it was called 'valid'
    # should correspond to the shift of ther errors over which we will divide
    #if time_locked == 'feedback':
    #    nonhit_shifted = np.insert(nonhit, len(nonhit), ins_nh)[shiftsz:]
    #elif time_locked == 'target':
    #    nonhit_shifted = np.insert(nonhit, 0 , ins_nh)[:-shiftsz]

    #############################
    tind_coln = 'trials'

    trial_inds_glob = np.array( behav_df.loc[df_inds, tind_coln])

    # replicating getIndShifts here because it's easier than change there
    if (df_fulltraj is not None) and shiftsz == 1:

        prev_trial_inds_glob = np.array( behav_df.loc[df_inds, 'prev_trial_index_valid'])

        if time_locked in ['target', 'trial_end']:
            print('Using prev_trial_index_valid')
            trial_inds_next      = trial_inds_glob
            valid_inds_cur  = np.ones( len(trial_inds_glob), dtype=bool )
            valid_inds_next = np.ones( len(trial_inds_glob), dtype=bool )
            valid_inds_next[trial_inds_next < 0] = False

            trial_inds_cur = prev_trial_inds_glob
            trial_inds_cur[np.isnan(prev_trial_inds_glob)  ] = -1e6
            trial_inds_cur = trial_inds_cur.astype(int)

            #trial_inds_cur = np.insert(trial_inds, 0, insind)[:-shiftsz]
            valid_inds_next[trial_inds_cur < 0] = False
            valid_inds_next[trial_inds_next < 0] = False

            trial_inds_glob1, valid_inds1, trial_inds_glob2, valid_inds2 = \
                trial_inds_cur, valid_inds_cur, trial_inds_next, valid_inds_next
        else:
            raise ValueError('not impl')
    else:
        trial_inds_glob1, valid_inds1, trial_inds_glob2, valid_inds2 = \
                getIndShifts(trial_inds_glob, time_locked=time_locked, shiftsz=shiftsz)

    trial_inds_loc0 = np.arange(len(trial_inds_glob ) )

    # suffix 1 is for those in the relative past and 2 for those in the relative future
    trial_inds_loc1, valid_inds1, trial_inds_loc2, valid_inds2 = \
        getIndShifts(trial_inds_loc0, time_locked=time_locked, shiftsz=shiftsz)
    trial_inds_loc1_s1, valid_inds1_s1, _, _ = \
        getIndShifts(trial_inds_loc0, time_locked=time_locked, shiftsz=1)
    #_,_, trial_inds_loc2_s1, valid_inds2_s1 = \
    #    getIndShifts(trial_inds_loc, time_locked=time_locked, shiftsz=1)

    target_locs  = np.array(behav_df.loc[df_inds,'target_locs']).copy()
    if reref_target_locs:
        print('Reref target locs')
        target_locs -= np.pi
    movement      = np.array(behav_df.loc[df_inds, 'org_feedback'])

    # which values will be in the numerator of the ES  
    if coln_correction_calc is None:
        #correction = (target_angs2 - next_movement) - (target_angs - movement)
        # -belief
        if use_sub_angles:
            vals_for_corr = subAngles(target_locs, movement)
        else:
            vals_for_corr = target_locs - movement
    else:
        vals_for_corr = behav_df.loc[df_inds, coln_correction_calc].to_numpy()

    # if true then we compute how error changes between now and far past changes wrt error in the far past, which is strange
    # if false, then we compute how error changes between now and PREVIOUS trials wrt error in the far past 
    #print(f'{long_shift_numerator=}, {shiftsz=}')
    if long_shift_numerator:
        vals_for_corr1  = shiftVals(vals_for_corr,
                                   trial_inds_loc1, valid_inds1)
    else:
        vals_for_corr1  = shiftVals(vals_for_corr,
                                   trial_inds_loc1_s1, valid_inds1_s1)

    vals_for_corr2  = shiftVals(vals_for_corr,
                        trial_inds_loc2, valid_inds2)


    errors1  = shiftVals(errors0, trial_inds_loc1, valid_inds1)
    # this should not NOT _s1
    errors2  = shiftVals(errors0, trial_inds_loc2, valid_inds2)

    nonhit_err1_compat = shiftVals(nonhit,
                            trial_inds_loc1, valid_inds1,
                                 invalid_val = False)
    nonhit_err2_compat = shiftVals(nonhit,
                            trial_inds_loc2, valid_inds2,
                                 invalid_val = False)

    targets_locs1  = shiftVals(target_locs,
                            trial_inds_loc1, valid_inds1)
    targets_locs2 = shiftVals(target_locs,
                                   trial_inds_loc2, valid_inds2)

    if df_fulltraj is not None:

        correction = []
        #raise ValueError('DEBUG')
        #for trial_inds_glob1, valid_inds1, trial_inds_glob2
        for gti1,gti2 in zip(trial_inds_glob1, trial_inds_glob2):
            dftraj1 = df_fulltraj.query( 'trial_index == @gti1' )
            dftraj2 = df_fulltraj.query( 'trial_index == @gti2' )

            dftraj1 = dftraj1.loc[dftraj1.index[1: ] ]
            dftraj2 = dftraj2.loc[dftraj2.index[1: ] ]

            ab = trajPair2corr(dftraj1, dftraj2  )
            #print(gti1,gti2, ab)

            correction_cur = ab
            correction += [correction_cur]
            #def areaBetween(xs,ys, xs2, ys2, start ,end):
        correction = np.array(correction)
    else:
        # it HAS to be target-centered here,
        # otherwise multiplying by retention factor is bad
        # so assuming it is ofb - target
        if use_sub_angles:
            correction = subAngles(retention_factor * vals_for_corr1, vals_for_corr2)
        else:
            correction =  retention_factor * vals_for_corr1 -  vals_for_corr2

    df_esv = {}
    df_esv['trial_inds_glob'] = trial_inds_glob
    df_esv['trial_inds_glob_prevlike_error'] = trial_inds_glob1
    df_esv['trial_inds_glob_nextlike_error'] = trial_inds_glob2
    df_esv = pd.DataFrame( df_esv )

    if len(addvars):
        #raise ValueError('Not implemneted')
        #df_esv['prevlike_error'] = errors1
        #df_esv['nextlike_error'] = errors2

        for vn in addvars:
            if vn.startswith('prev_'):
                valn = vn[5:]
                vals = behav_df.loc[df_inds, valn ]
                vals = shiftVals(vals, trial_inds_loc1, valid_inds1)
            elif vn.startswith('next_'):
                valn = vn[5:]
                vals = behav_df.loc[df_inds, valn ]
                vals = shiftVals(vals, trial_inds_loc2, valid_inds2)
            else:
                valn = vn
                print(f'computeErrSens3: boring add variable {vn}')
                vals = behav_df.loc[df_inds, valn ]

            df_esv[vn] = vals



    df_esv[colname_nh] = nonhit.astype(bool)
    if coln_nh_out is None:
        coln_nh_out = colname_nh + '_shifted'

    # todo rename cur to 1 and next to 2, otherwise it is confusing

    # trial end means that we use all the info from the current trial that is
    # available at the end

    if time_locked == 'trial_end':
        nonhit_err_compat = nonhit_err2_compat  # current trial error
    elif time_locked == 'target':
        nonhit_err_compat = nonhit_err1_compat  # prev error
    elif time_locked == 'feedback':
        nonhit_err_compat = nonhit_err1_compat  # current trial error


    #df_esv[coln_nh_out] = nonhit_shifted
    df_esv[coln_nh_out] = nonhit_err_compat.astype(bool)

    df_esv[err_sens_coln] = np.nan
    c = df_esv[coln_nh_out]

    if time_locked == 'trial_end':
        errors_denom = errors2
    else:
        errors_denom = errors1

    #errors_denom[c]
    df_esv.loc[c,err_sens_coln]  = (correction[c] / errors_denom[c])

    df_esv['retention_factor'] = retention_factor
    df_esv['retention_factor_s'] = df_esv['retention_factor'].apply(lambda x: f'{x:.3f}')

    # recal that in the experiment code what goes to "errors" columns is
    # computed this way
    # self.error_distance = np.sqrt((self.feedbackX - self.target_types[self.target_to_show][0])**2 +
    #                               (self.feedbackY - self.target_types[self.target_to_show][1])**2)

    #import pdb; pdb.set_trace()


    nh = np.sum( ~df_esv[coln_nh_out] )
    if correct_hit == 'prev_valid':
        df_esv.loc[~df_esv[coln_nh_out],err_sens_coln]  = np.inf
        hit_inds = np.where(~df_esv[coln_nh_out] )[0]
        for hiti in hit_inds:
            prev = df_esv.loc[ :hiti, err_sens_coln ]
            good = np.where( ~ (np.isinf( prev ) | np.isnan(np.isinf) ) )[0]
            if len(good):
                lastgood = good[-1]
                df_esv.loc[ hiti, err_sens_coln ] = df_esv.loc[ lastgood, err_sens_coln ]
        #df_esv.loc[~df_esv[coln_nh_out],err_sens_coln]  =
    elif correct_hit == 'zero':
        if verbose:
            print(f'correct_hit == {correct_hit}: setting {nh} out of {len(df_esv)}')
        df_esv.loc[~df_esv[coln_nh_out],err_sens_coln]  = 0
    elif correct_hit == 'inf':
        if verbose:
            print(f'correct_hit == {correct_hit}: setting {nh} out of {len(df_esv)}')
        df_esv.loc[~df_esv[coln_nh_out],err_sens_coln]  = np.inf
    elif correct_hit == 'nan':
        if verbose:
            print(f'correct_hit == {correct_hit}: setting {nh} out of {len(df_esv)}')
        df_esv.loc[~df_esv[coln_nh_out],err_sens_coln]  = np.nan

    #raise ValueError('debug')
    df_esv['correction'] = correction
    df_esv['error_type'] = error_type

    # '_like' means that it is not necessarily stricly prev/next and depends on
    # time_locked
    df_esv['prevlike_error'] = errors1
    df_esv['nextlike_error'] = errors2

    df_esv['prevlike_target_loc'] = targets_locs1
    df_esv['nextlike_target_loc'] = targets_locs2

    # for decoding later
    df_esv['belief_']      = -vals_for_corr
    # corr = 1-2
    df_esv['vals_for_corr1']      = vals_for_corr1
    df_esv['vals_for_corr2']      = vals_for_corr2
    # this should be set here (not in behav_proc)
    # because it depends on the coln_corr_calc
    df_esv['prev_belief_'] = -getPrev(vals_for_corr.astype(float) )
    df_esv['prev_movement'] = getPrev(movement.astype(float))
    #####

    df_esv['environment']  = np.array( behav_df.loc[df_inds, 'environment'] )
    df_esv['perturbation'] = np.array( behav_df.loc[df_inds, 'perturbation'])
    df_esv['prev_error']      = getPrev(errors0)
    df_esv['target_loc']      = target_locs.astype(float)
    df_esv['prev_target_loc'] = getPrev(target_locs.astype(float) )
    df_esv[f'prev_{err_sens_coln}'] = getPrev( df_esv[err_sens_coln].to_numpy() )

    #raise ValueError('debug')

    return nonhit, df_esv


def computeErrSens2(behav_df, df_inds=None, epochs=None,
                    do_delete_trials=1,
                    enforce_consistency=0, time_locked='target',
                    correct_hit = 'inf', error_type = 'MPE',
                    colname_nh = 'non_hit_not_adj', shiftsz=1,
                    coln_nh_out = 'non_hit_shifted', 
                    err_sens_coln = 'err_sens',
                    addvars = [], recalc_non_hit = True, target_info_type = 'inds',
                    coln_correction_calc = None,
                    coln_error = 'error',
                    verbose = 0, retention_factor = 1.):
    '''
    computes error sensitiviy across dataset. So indexing is very important here.
    '''

    assert shiftsz >= 1

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
        behav_df = enforceTargetTriggerConsistency(behav_df,
            epochs, environment, do_delete_trials=do_delete_trials )

    if df_inds is None:
        df_inds = behav_df.index

    # dis = np.diff(behav_df.index.values)
    # np.max( dis ) == np.max( dis ) and np.max( dis ) == 1

    targets_locs_cur      = np.array(behav_df.loc[df_inds,'target_locs'])
    target_inds      = np.array(behav_df.loc[df_inds,'target_inds'])
    org_feedback_cur      = np.array(behav_df.loc[df_inds,'org_feedback'])
    feedback_cur      = np.array(behav_df.loc[df_inds,'feedback'])
    # after enforcing consistencey
    # environment_cur  = np.array(behav_df.loc[df_inds,'environment'])
    # feedback     = np.array(behav_d.locf['feedback'])
    movement_cur     = np.array(behav_df.loc[df_inds, 'org_feedback'])

    if error_type == 'MPE':
        errors_cur       = np.array(behav_df.loc[df_inds, coln_error])
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
    if recalc_non_hit or ('non_hit_not_adj' not in behav_df.columns):
        feedbackX_cur    = np.array(behav_df.loc[df_inds, 'feedbackX'])
        feedbackY_cur    = np.array(behav_df.loc[df_inds, 'feedbackY'])
        if verbose:
            print('Creating new  "non_hit_not_adj"')
        non_hit_not_adj = point_in_circle(target_inds, target_coords,
                                            feedbackX_cur, feedbackY_cur,
                                            radius_target + radius_cursor)
        non_hit_not_adj = np.array(non_hit_not_adj)
    else:
        if verbose:
            print('Using existing "non_hit_not_adj"')
        non_hit_not_adj = behav_df['non_hit_not_adj'].to_numpy()

    # not that non-hit has different effect on error sens calc depending on
    # which time_locked is used
    ins = [ 0 ] * shiftsz
    if time_locked == 'feedback':
        #valid = np.insert(non_hit_not_adj, 0 ,0)[:-1]
        valid = np.insert(non_hit_not_adj, len(non_hit_not_adj), ins)[shiftsz:]
    elif time_locked == 'target':
        valid = np.insert(non_hit_not_adj, 0 , ins)[:-shiftsz]

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
    if target_info_type == 'inds':
        target_locs = None
    elif target_info_type == 'locs':
        target_locs = targets_locs_cur
    else:
        raise ValueError('Wrong target info type')
    df_esv,vndef2vn = \
        getErrSensVals(errors_cur,target_inds, movement_cur,
            time_locked = time_locked, ret_df = True, shiftsz = shiftsz,
                       target_locs = target_locs)

    #target_angs_next, target_angs, next_movement, movement, errors = values_for_es

    target_angs_next = df_esv[ vndef2vn['target'] ]
    target_angs      = df_esv[ vndef2vn['prev_target'] ]
    next_movement    = df_esv[ vndef2vn['movement'] ]
    movement         = df_esv[ vndef2vn['prev_movement'] ]
    errors           = df_esv[ vndef2vn['prev_error'] ]

    # this , look how it is called in varnames_def,  look indes what is put in values_for es
    # next_movement -> movement index 2  -> movement [tgt]      | next_movement [ fb ]
    # movement -> prev_movement index 3  -> prev_movement [tgt] | movement [fb]

    #corrcalc_next =
    #corrcalc      =

    for vn in addvars:
        if vn not in vndef2vn.values():
            if vn.startswith('prev_'):
                vals = behav_df.loc[df_inds, vn[5:] ]
                vals = np.array(vals)
                vals = np.insert(vals, 0, 0)[:-1]
            elif vn.startswith('next_'):
                vals = behav_df.loc[df_inds, vn[5:] ]
                vals = np.array(vals)
                vals = np.insert(vals,len(vals),np.nan)[1:]
            else:
                assert vn in behav_df.columns
                vals = behav_df.loc[df_inds, vn ]
                vals = np.array(vals)
            df_esv[vn] = vals

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
    # movement vs feedback is diff only when we have change of context between
    # prev and current
    if coln_correction_calc is None:
        # -cur - (-prev)
        correction = (target_angs_next - next_movement) - retention_factor * (target_angs - movement)
    else:
        corr = behav_df.loc[df_inds, coln_correction_calc].to_numpy()

        ins = [ np.nan ] * shiftsz
        if time_locked == 'target':
            corr_next = corr
            corr      = np.insert(corr, 0, ins)[:-shiftsz]
        else:
            corr_next = np.insert(corr, len(corr), ins)[shiftsz:]
            corr      = corr


        correction = corr_next - retention_factor * corr
        #df_esv[ ]

    df_esv['non_hit_not_adj'] = non_hit_not_adj
    df_esv[coln_nh_out] = valid
    df_esv.loc[df_esv[colname_nh],err_sens_coln]  = correction / errors
    # recal that in the experiment code what goes to "errors" columns is
    # computed this way
    # self.error_distance = np.sqrt((self.feedbackX - self.target_types[self.target_to_show][0])**2 +
    #                               (self.feedbackY - self.target_types[self.target_to_show][1])**2)

    #import pdb; pdb.set_trace()


    nh = np.sum( ~df_esv[colname_nh] )
    if correct_hit == 'prev_valid':
        df_esv.loc[~df_esv[colname_nh],err_sens_coln]  = np.inf
        hit_inds = np.where(~df_esv[colname_nh] )[0]
        for hiti in hit_inds:
            prev = df_esv.loc[ :hiti, err_sens_coln ]
            good = np.where( ~ (np.isinf( prev ) | np.isnan(np.isinf) ) )[0]
            if len(good):
                lastgood = good[-1]
                df_esv.loc[ hiti, err_sens_coln ] = df_esv.loc[ lastgood, err_sens_coln ]
        #df_esv.loc[~df_esv[colname_nh],err_sens_coln]  =
    elif correct_hit == 'zero':
        if verbose:
            print(f'correct_hit == {correct_hit}: setting {nh} out of {len(df_esv)}')
        df_esv.loc[~df_esv[colname_nh],err_sens_coln]  = 0
    elif correct_hit == 'inf':
        if verbose:
            print(f'correct_hit == {correct_hit}: setting {nh} out of {len(df_esv)}')
        df_esv.loc[~df_esv[colname_nh],err_sens_coln]  = np.inf
    elif correct_hit == 'nan':
        if verbose:
            print(f'correct_hit == {correct_hit}: setting {nh} out of {len(df_esv)}')
        df_esv.loc[~df_esv[colname_nh],err_sens_coln]  = np.nan


    df_esv['correction'] = correction
    df_esv['error_type'] = error_type

    df_esv['environment']  = np.array( behav_df.loc[df_inds, 'environment'] )
    df_esv['perturbation'] = np.array( behav_df.loc[df_inds, 'perturbation'])

    df_esv['trial_inds_glob'] = np.array( behav_df.loc[df_inds, 'trials'])


    #df_esv.loc[ vndef2vn['prev_target'] ]

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
    import mne
    from mne.io import read_raw_fif

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

    from mne import Epochs
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

def adjustNonHitDf(df, env, time_locked, hits_to_use = ['current_trial'],
        inplace=False):
    #raise ValueError('not finished implementation')
    assert np.all( np.isin(hits_to_use, ['current_trial','shifted_trial'] ) )
    # I'd need to make do separately for every subject
    assert df['subject'].nunique() == 1, 'for N subjects > 1 need more care'

    non_hit = df['non_hit_not_adj']
    # return a mask
    # some adjustment of non_hit, based on time lock and whether we are in
    # stable or random environment
    #non_hit = non_hit.copy()
    from config2 import n_trials_in_block as N
    # it is painful to do those inserts in case of pandas
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

    if inplace:
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

def getAnalysisVarnames(time_locked, control_type):
    from config2 import control_types_all, time_lockeds_all
    assert control_type in control_types_all
    assert time_locked in time_lockeds_all

    if time_locked == 'feedback':
        if control_type == 'feedback':
            analysis_name = 'feedback_error_next_error_belief'
            varnames = ['feedback', 'error', 'next_error', 'belief']
        elif control_type == 'movement':
            analysis_name = 'movement_error_next_error_belief'
            varnames = ['movement', 'error', 'next_error', 'belief']
        elif control_type == 'target':
            analysis_name = 'target_error_nexterror_belief'
            varnames = ['targets', 'error', 'next_error', 'belief']
        elif control_type == 'belief':
            analysis_name = 'belief_error_nexterror'
            varnames = ['belief', 'error', 'next_error']
    elif time_locked == 'target':
        if control_type == 'feedback':
            analysis_name = 'prevfeedback_preverror_error_prevbelief'
            varnames = ['prev_feedback', 'prev_error', 'error', 'prev_belief']
        elif control_type == 'movement':
            analysis_name = 'prevmovement_preverror_error_prevbelief'
            varnames = ['prev_movement', 'prev_error', 'error', 'prev_belief']
        elif control_type == 'target':
            analysis_name = 'prevtarget_preverror_error_prevbelief'
            varnames = ['prev_targets', 'prev_error', 'error', 'prev_belief']
        elif control_type == 'belief':
            analysis_name = 'prevbelief_preverror_error'
            varnames = ['prev_belief', 'prev_error', 'error']

    return varnames, analysis_name

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

    prev_targets   = np.insert(targets, 0, 0)[:-1]
    prev_errors    = np.insert(errors, 0, 0)[:-1]
    next_errors    = np.insert(errors, len(errors), 0)[1:]
    prev_feedback  = np.insert(feedback, 0, 0)[:-1]
    prev_movement  = np.insert(movement, 0, 0)[:-1]
    prev_belief    = np.insert(belief, 0, 0)[:-1]
    next_belief    = np.insert(belief, len(belief), 0)[1:]

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
    import mne
    from mne import Epochs
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


def getTruncDep(df, qs_to_check):
    # dependence on truncation params
    from behav_proc import truncateDf
    import pinguin as pg
    #truncateDf?

    #dfes_goog_noq = truncateDf( dfes, 'err_sens', q = None, infnan_handling='discard' )
    print( qs_to_check )

    pds = [dict(q=None), dict(q=None, low=-50, hi=50),dict(q=None, low=-30, hi=30), dict(q=None, low=-10, hi=10),
           dict(q=None, low=-1, hi=1),
        dict(q=0.05),dict(q=0.02), dict(q=None),
          dict(q=0.05, trunc_hi = 0),dict(q=0.02, trunc_hi =0), dict(q=None, trunc_hi =0),
          dict(q=0.05, trunc_low = 0),dict(q=0.02, trunc_low =0), dict(q=None, trunc_low =0),
          dict(q=0.05, trunc_low = 0, abs=True),dict(q=0.02, trunc_low =0, abs=True), dict(q=None, trunc_low =0, abs=True)]
    rs = []
    for truncapd in pds:
        print('   truncapd=',truncapd)
        dfes_goog, locs = truncateDf( df, 'err_sens', **truncapd,
                            infnan_handling='discard',
                             verbose=1, retloc=1 )
        for alt in ['greater','less','two-sided']:
            r = pg.ttest(dfes_goog.query(qs_to_check)['err_sens'], 0 , alternative=alt)
            r['truncaprs'] = repr(truncapd)
            r['hi_mean'] = None
            r['low_mean'] = None
            low = locs.get('qlow',None)
            hi = locs.get('qhi',None)
            if low is not None:
                r['low_mean'] = low.mean()
            else:
                r['low_mean'] = locs.get('low',None)
            if hi is not None:
                r['hi_mean'] = hi.mean()
            else:
                r['hi_mean'] = locs.get('hi',None)

            r['qs'] = qs_to_check
            rs += [r]

        del dfes_goog
        #break
        #display(pg.ttest(dfes_goog.query(qs_notspec)['err_sens'], 0 , alternative='less'))
    ttrs = pd.concat(rs)

    return ttrs

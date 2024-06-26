import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import os.path as op
import numpy as np
import mne
from mne.io import read_raw_fif
from csp_my import SPoC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
#from sklearn.metrics import make_scorer
from scipy.stats import spearmanr
from mne import Epochs
import pandas as pd
import warnings
import sys
from base2 import (getXGBparams,
                   calc_target_coordinates_centered,
                   target_angs, getGPUavail, B2B_SPoC, rescaleIfNeeded)
from config2 import n_jobs as n_jobs_def
from config2 import (min_event_duration,genFnSliding,
                     path_data,path_data_tmp,freq_name2freq,
                     stim_channel_name,delay_trig_photodi,
                     stage2evn2event_ids, stage2evn2event_ids_str,
                     paramFileRead, env2envcode, env2subtr,
                     stage2time_bounds)
from error_sensitivity import (enforceTargetTriggerConsistency,
                                checkTgtConsistency,
                               computeErrSens2,adjustNonHit, adjustNonHitDf,
                               getAnalysisVarnames)
from meg_proc import addTrigPresentCol_NIH
#from error_sensitivity import enforceTargetTriggerConsistency

from behav_proc import addBehavCols, computeErrSensVersions, truncateDf
from os.path import join as pjoin
from behav_proc import getMaskNotNanInf

###########
from datetime import datetime  as dt
from config2 import paramFileRead, genArgParser_decodeNIH

parser = genArgParser_decodeNIH()
print(sys.argv)

args = parser.parse_args()
par = vars(args)

if par['runpar_line_ind'] is not None:
    from config2 import path_code
    with open(op.join(path_code,'__runpars.txt'), 'r' ) as f:
        lines = f.readlines()
    line = lines[args.runpar_line_ind]
    from config2 import parline2par
    par = parline2par(line)

    par.update( vars(args)  )
else:
    par = vars(args)
    if 'param_file' in par:
        par_from_file = paramFileRead(par['param_file'])
        par_cmd = par
        par = par_from_file
        for pn,pv in par_cmd.items():
            if pv is not None:
                par[pn] = pv

#################################################
dec_error_handling = par.get('dec_error_handling', 'ignore' )

if par['scale_Y_robust'] >= 3:
    # normal scalers cannot work with 3 dim arrays (which is the case of epochs)
    # puttting it in the middle of the pipeline is weird
    # at the end also not possible because we need transform of target vars, not features
    raise ValueError('Not implemented')

target_coords = calc_target_coordinates_centered(target_angs)

n_jobs            = int( par.get('n_jobs', n_jobs_def) )
subject           = par['subject']
env_to_run        = par['env_to_run']; env = env_to_run  #
regression_type   = par['regression_type']               #
freq_name         = par['freq_name']
fl = par.get('freq_limits', 'auto')
if fl == 'auto':
    freq_limits       = freq_name2freq[freq_name]
else:
    if fl is None:
        raise ValueError('none freq limits')
    freq_limits = eval(fl)
if isinstance(freq_limits,str):
    freq_limits = eval(freq_limits)
hpass             = par['hpass']  # '0.1', no_hpass, no_filter

assert ',' not in regression_type, regression_type
#if ',' in regression_type:
#    regression_type = regression_type.split(',')
#else:
#    regression_type = [regression_type]

if ',' in env_to_run:
    env_to_run = env_to_run.split(',')
else:
    env_to_run = [env_to_run]

########### groupby   -- to compute err_sens
# indep of grouping for any (or almost any) poitn I will have a value of
# err sens. But if I perform SPoC directly it will search for
# stuff in the same brain regions. And maybe for different targets
# it happens in diff brain regions


cs = ['current_trial', 'shifted_trial']
behvar2nhtype = {'err_sens': cs,
                 'correction':cs, 'prev_error':['shifted_trial'],
                 'error':['current_trial'] }

def prepNonHit(subdf, var, env, time_locked):
    nhts = behvar2nhtype[var]
    nh = np.array(subdf['non_hit_not_adj'] ).astype(bool)
    non_hit_adj_curtr = adjustNonHitDf(subdf, env, time_locked,
            hits_to_use=['current_trial'], inplace=False )
    non_hit_adj_prevtr = adjustNonHitDf(subdf, env, time_locked,
            hits_to_use=['shifted_trial'], inplace=False )
    if 'current_trial' in nhts:
        nh &= non_hit_adj_curtr
    if 'shifted_trial' in nhts:
        nh &= non_hit_adj_prevtr
    return nh

n_channels_to_use = par.get('n_channels_to_use', -1)

target_inds_to_use = par.get('target_inds_to_use' )

trial_group_col_calc = par.get('trial_group_col_calc' )
tgt_inds_cur = target_inds_to_use
if len(target_inds_to_use) == 1 and target_inds_to_use[0] is None:
    #if trial_group_col_calc == 'trialwe':
    #    tgt_inds_cur = [None]
    #elif trial_group_col_calc == 'trialwtgt_we':
    #    tgt_inds_cur = np.arange(4, dtype=int)
    if trial_group_col_calc == 'trialwtgt_we':
        tgt_inds_cur = np.arange(4, dtype=int)
#else:
    #tgt_inds_cur = map(int,target_inds_to_use.split(',') )

trim_outliers = par.get('trim_outliers',0)
block_names_cur = par.get('block_names_to_use','all')
if ',' in block_names_cur:
    block_names_cur = block_names_cur.split(',')
else:
    block_names_cur = [block_names_cur]

dists_trial_from_prevtgt_cur = par.get('dists_trial_from_prevtgt',[None])
dists_rad_from_prevtgt_cur = par.get('dists_rad_from_prevtgt',[None])
pertvals_cur = par.get('pertvals')

use_non_hit_recalc = 0 # since I use truncateDF which also gets rid of NaNs

# TODO maybe rename to remove _cur but careful not to inflice name clash
error_type = par.get('error_type','MPE')  # observed - goal, motor performance error
gseqcs_cur = [ (0,1) ]

###########

output_folder = par.get('output_folder',f'corr_spoc_es_sliding2_{hpass}')
ICAstr = par.get('ICAstr','with_ICA'  )  # empty string is allowed
time_locked = par.get('time_locked')
control_type = par.get('control_type')
coln_error = 'error'
colns_ES = par.get('colns_ES', 'err_sens').split(',')

#,error_pred_Herz
defcolns_classic = [ 'prev_error,error,trials' ]   #correction,
colns_classic = par.get('colns_classic', defcolns_classic).split(',')

vars_to_decode_classic_def = colns_ES + colns_classic
#vars_to_decode_classic_def = [] # DEBUG
#vars_to_decode_b2b_def     = [colns_ES[0] ] + [ 'prev_error', 'error' ] # b2b does not allow full linear dependence, so cannot put correction here
#vars_to_decode_classic_def = []
#vars_to_decode_b2b_def = []
vars_to_decode_b2b_def =  [  [coln_ES_cur] + [ 'prev_error', 'error' ] for coln_ES_cur in colns_ES ]


print('vars_to_decode_classic_def = ', vars_to_decode_classic_def)
print('vars_to_decode_b2b_def = ', vars_to_decode_b2b_def)

# home position
if par['slide_windows_type'] == 'auto':
    start, end = stage2time_bounds[time_locked]
    start, end = eval( par.get(f'time_bounds_slide_{time_locked}', (start,end) ) )
    shift = float( par['slide_window_shift'] )
    dur =   float( par['slide_window_dur'  ] )
    tmins = np.arange(start,end,shift)
    tmaxs = dur + tmins

    tminmax = zip(tmins,tmaxs)
elif par['slide_windows_type'] == 'explicit':
    tmin = par.get('tmin',None)
    tmax = par.get('tmax',None)

    if ',' in tmin:
        tmin = tmin.split(',')
        tmin = map(float,tmin)
    else:
        tmin = [float(tmin) ]

    if ',' in tmax:
        tmax = tmax.split(',')
        tmax = map(float,tmax)
    else:
        tmax = [float(tmax)]
    tminmax = zip(tmin,tmax)

tminmax = list(tminmax)
# to save to the final file
par['tminmax'] = tminmax

DEBUG = int(par.get('debug',0) )
if DEBUG:
    print('---------------------- DEBUG MODE --------------------- ')
    print('---------------------- DEBUG MODE --------------------- ')
    print('---------------------- DEBUG MODE --------------------- ')
# task = 'LocaError'  # 'VisuoMotor' or 'LocaError'
task = par.get('task','VisuoMotor')

do_classic_dec            = int( par.get('do_classic_dec',1)            )
do_b2b_dec            = int( par.get('do_b2b_dec',1)            )
est_parallel_across_dims  = int( par.get('est_parallel_across_dims',1)  )
est_parallel_within_dim   = int( par.get('est_parallel_within_dim',0)   )
b2b_each_fit_is_parallel  = int( par.get('b2b_each_fit_is_parallel',0)   )
each_SPoC_fit_is_parallel = int( par.get('each_SPoC_fit_is_parallel',1)   )
classic_dec_verbose       = int( par.get('classic_dec_verbose',3)       )


#decode_merge_pert = int( par.get('decode_merge_pert',1)   )
#decode_per_pert   = int( par.get('decode_per_pert',1)   )

B2B_SPoC_parallel_type = par.get('B2B_SPoC_parallel_type', 'across_splits')
# 'across_splits_and_dims'

nskip_trial               = int( par.get('nskip_trial',1)                   )
nb_fold                   = int( par.get('nb_fold',6)  ) # generavl CV fold num
decim_epochs              = int( par.get('decim_epochs',2)                   )
n_splits_B2B              = int( par.get('n_splits_B2B',30)             )
SPoC_n_components         = int( par.get('SPoC_n_components',5)         )
safety_time_bound         = float( par.get('safety_time_bound',0.05) )
random_seed               = int( par.get('random_seed',0) )
save_result               = int( par.get('save_result',1) )
crop                      = par.get('crop',None) ;
groupcols                 = par.get('groupcols',[ 'environment' ]) ;
an_suffix                 = par.get('custom_suffix','err_sens') ;
assert isinstance(groupcols, list)
if crop is not None:
    crop = eval(crop)

#analysis_name_from_par = par.get('an_suffix', 'prevmovement_preverrors_errors_prevbelief')
use_preloaded_raw = int( par.get('use_preloaded_raw', 0) )
use_preloaded_flt_raw = int( par.get('use_preloaded_flt_raw', 0) )
mne_fit_log_level         = par.get('mne_fit_log_level', 'warning')

load_epochs  = int( par.get('load_epochs',0)  ) # only reg type and env would be optimized
save_epochs  = int( par.get('save_epochs',0)  ) # only reg type and env
load_flt_raw = int( par.get('load_flt_raw',1) )
save_flt_raw = int( par.get('save_flt_raw',1) )

exit_after = par.get('exit_after','end')

##########################################################################
##########################################################################

np.random.seed(random_seed)

print(f'__START: {__file__} subj={subject}, hpass={hpass}, '
      f'regression_type={regression_type}, freq_name={freq_name}, '
      f'env={env_to_run} at {dt.now()}')
print('par=',par)
#mne_fit_log_level = 'debug'

# both on will give errors, both off are fine
assert not (est_parallel_across_dims and est_parallel_within_dim)

allow_CUDA = 1
if allow_CUDA and len(getGPUavail() ):
    mne.cuda.init_cuda()
    n_jobs_MNE='cuda'
else:
    n_jobs_MNE=n_jobs
n_jobs_XGB=n_jobs

results_folder = op.join(path_data,subject,'results',output_folder)
if not os.path.exists(results_folder):
    try:
        os.makedirs(results_folder)
    except OSError as e:
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)


if each_SPoC_fit_is_parallel:
    n_jobs_SPoC = n_jobs
else:
    n_jobs_SPoC = 1

#if b2b_each_fit_is_parallel:
#    n_jobs_SPoC = n_jobs
#else:
#    n_jobs_SPoC = 1

# if we parallelize across dims, we want it to be 1
# NOT USED
if est_parallel_across_dims:
    n_jobs_per_dim_classical_dec = 1
else:
    if est_parallel_within_dim:
        n_jobs_per_dim_classical_dec = n_jobs
    else:
        n_jobs_per_dim_classical_dec = 1
add_clf_creopts_est = getXGBparams(n_jobs = n_jobs_per_dim_classical_dec)

#regression_type = 'Ridge'
#regression_type = 'xgboost'

if b2b_each_fit_is_parallel:
    add_clf_creopts = getXGBparams(n_jobs = None)
else:
    add_clf_creopts = getXGBparams(n_jobs = 1)

##############################

# read behav data
fname_full_def = op.join(path_data, subject, 'behavdata',
                f'behav_{task}_df.pkl' )
behav_fnf = par.get('behav_file', fname_full_def)
# where we would save cleaned df later
fname_EC_full = op.join(path_data, subject, 'behavdata',
                f'behav_{task}_df_EC.pkl' )
behav_df_full = pd.read_pickle(behav_fnf)
if 'subject' not in behav_df_full.columns:
    behav_df_full['subject'] = subject
else:
    behav_df_full = behav_df_full.query('subject == @subject').copy()

if 'pert_seq_code' not in behav_df_full.columns:
    addBehavCols(behav_df_full, inplace=True)

print('Behav file has {} dup entries'.format(
    behav_df_full.duplicated(['subject','trials']).sum() ) )
if 'trial_shift_size' in behav_df_full:
    tsz = par['trial_shift_size']
    behav_df_full = behav_df_full.query('trial_shift_size == @tsz')
if 'trial_group_col_calc' in behav_df_full:
    trgc = par['trial_group_col_calc']
    behav_df_full = behav_df_full.query('trial_group_col_calc == @trgc')
print('Behav file has {} dup entries, total len = {}'.format(
    behav_df_full.duplicated(['subject','trials']).sum(), len(behav_df_full) ) )

behav_df_full = behav_df_full.sort_values('trials').reset_index(drop=True)
assert (behav_df_full['trials'].diff().iloc[1:] > 0).all()

behav_df_full['movement'] = behav_df_full['org_feedback']
targets_codes_full      = np.array(behav_df_full['target_codes'])

#####
for coln in vars_to_decode_classic_def:
    assert coln in behav_df_full.columns, coln
for colnset in vars_to_decode_b2b_def:
    for coln in colnset:
        assert coln in behav_df_full.columns, coln

# argument in RidgeCV
n_ridgeCV_alphas = par.get('n_ridgeCV_alphas', 12)
alphas = np.logspace(-5, 5, n_ridgeCV_alphas)
fit_intercept_classic_dec = True
fit_intercept_b2b = False

# time_locked = 'target'  # 'target' or 'reach'
# Read raw to extra MEG event triggers
#for freq_name, freq in freq_names_freqs:
freq = freq_limits
print(f'---------- Starting freq = {freq_name}')


############################################################################
################################## read raw data
############################################################################
print(f'use_preloaded_raw = {use_preloaded_raw}, use_preloaded_flt_raw={use_preloaded_flt_raw}')
#import sys; sys.exit(1)
# get raw and events from raw file

try:
    raw is None
    preloaded_raw_is_present = True
    preloaded_flt_raw_is_present = True
except NameError as e:
    preloaded_raw_is_present = False
    preloaded_flt_raw_is_present = False
    raw = None

print(f'preloaded_raw_is_present = {preloaded_raw_is_present}, preloaded_flt_raw_is_present={preloaded_flt_raw_is_present}')

path_data_tmp_cursubj = op.join(path_data_tmp, subject)
fn_flt_raw = f'raw_{task}_{hpass}_{ICAstr}_{freq_name}.fif'
fn_flt_raw_full = op.join(path_data_tmp_cursubj, fn_flt_raw )

fn_events = f'{task}_{hpass}_{ICAstr}_eve.txt'
fn_events_full = op.join(path_data, subject, fn_events )

#from config2 import cleanEvents
from meg_proc import cleanEvents_NIH

#################  load filtered raw if present
loaded_flt_raw = False
if load_flt_raw and os.path.exists(fn_flt_raw_full) \
    and os.path.exists(fn_events_full) and \
    (not (use_preloaded_flt_raw and preloaded_flt_raw_is_present) ):
    print(f'INFO: Found filtered raw, loading {fn_flt_raw_full}')
    try:
        raw = read_raw_fif(fn_flt_raw_full, preload=True)
        print('raw.times[-1] = ',raw.times[-1] )

        if crop is not None:
            raw.crop(crop[0],crop[1])
            events = mne.find_events(raw, stim_channel=stim_channel_name,
                                        min_duration=min_event_duration)
            events[:, 0] += delay_trig_photodi

            events = cleanEvents_NIH(events)
        else:
            events = mne.read_events(fn_events_full)
        loaded_flt_raw = True
    except Exception as e:
        print('loading filtered raw failed, got exception ',str(e))


# if raw is not present, re-filter it from zero
if (raw is None) or ( (not loaded_flt_raw) and \
    ( not (use_preloaded_flt_raw and preloaded_flt_raw_is_present) ) ):

    fif_fnf = op.join(path_data, subject, f'raw_{task}_{hpass}_{ICAstr}.fif' )
    fif_fnf = par.get('fif_file', fif_fnf)

    raw = read_raw_fif(fif_fnf, preload=True)
    if crop is not None:
        raw.crop(crop[0],crop[1])
    events = mne.find_events(raw, stim_channel=stim_channel_name,
                                min_duration=min_event_duration)
    events[:, 0] += delay_trig_photodi
    events = cleanEvents_NIH(events)

    raw.filter(freq[0], freq[1], n_jobs=n_jobs_MNE)

    if save_flt_raw:
        if not os.path.exists(path_data_tmp_cursubj):
            os.makedirs(path_data_tmp_cursubj)
        try:
            raw.save(fn_flt_raw_full,overwrite=True)
            print(f'Saved filtered raw {fn_flt_raw_full}')
        except OSError as e:
            print(f'ERROR: got OSerror during saving raw {str(e)}')

        mne.write_events(fn_events_full, events, overwrite=True)

# take epochs for all events
# ensure target trigger consistency
# get trial inds
# extrace error sens using behav data
# adjust Non Hit -- ?
# add safety bound for time
# take only necessary epochs (corresp to time_lock and to env)
# take non-hit es and non-hit X
# create SPoC and attach ridge CV to it
# fit and predict, compute diff and score

#groupby

#############################################################
##################################    Start calc
#############################################################
# this needed for alignment later, does not dep on tmin,tmax
event_ids_all_both = stage2evn2event_ids['feedback']['all'] + stage2evn2event_ids['target']['all']
epochs_for_EC = Epochs(raw, events, event_id = event_ids_all_both,
            tmin=-0.01, tmax=0.01, preload=True, decim=decim_epochs)

for tmin_cur,tmax_cur in tminmax:
    print('------------- Starting {:.2f},{:.2f}'.format(tmin_cur,tmax_cur) )
    bsl = par.get('baseline','None')
    bsl = eval(bsl)

    ####################   ensure consistency ################
    event_ids_all = stage2evn2event_ids[time_locked]['all']
    epochs = Epochs(raw, events, event_id = event_ids_all,
                    tmin=tmin_cur, tmax=tmax_cur, preload=True,
                    baseline=bsl, decim=decim_epochs)


    # if tmin and/or tmax are large then we can lose correspondance between target and feedback target indices, which in turn would make
    # it difficult to insure correspondance between log target sequence and trigger target sequence
    # so here we get target codes from epochs regardless whether for which time locked they were (I use sample to sample correspondance)
    from meg_proc import getTargetCodes_NIH
    dfev = getTargetCodes_NIH(epochs_for_EC, epochs)
    target_codes = dfev['target_code'].values

    #environment_full = np.array(behav_df_full['environment'])
    ## modifies in place
    #behav_df_cur = enforceTargetTriggerConsistency(behav_df_full.copy(),
    #                                epochs_for_EC,
    #                                environment_full, save_fname = fname_EC_full)

    #behav_df_cur = addTrigPresentCol_NIH(behav_df_full, epochs_for_EC.events)
    behav_df_cur = addTrigPresentCol_NIH(behav_df_full, target_codes)
    behav_df_cur = behav_df_cur.query('trigger_present == True')
    print('Trigger consist: before={}, after={}'.format( len(behav_df_full) ,
        len(behav_df_cur) ))

    assert len(behav_df_cur) > 0
    #assert len(behav_df_cur) == len(epochs_for_EC)
    assert len(behav_df_cur) == len(epochs)

    behav_df_cur =behav_df_cur.sort_values('trials')
    assert np.all( behav_df_cur['trials'].diff()[1:] > 0 )

    # this assertion is violated if ediops asks to delete first entry in behav_df
    #assert np.array_equal( dfev['trial_index_ev'].values , behav_df_cur['trials'].values)
    dfev['trial_index'] =  behav_df_cur['trials'].values

    if exit_after == 'enforce_consist':
        sys.exit(0)

    # after this behav_df['target_inds'] and epochs.events[:,2] are equal
    vars_to_decode_b2b_tmp, aname = \
            getAnalysisVarnames(time_locked, control_type)
    # list of lists
    if len(vars_to_decode_b2b_def):
        vars_to_decode_b2b = [vars_to_decode_b2b_tmp ] +  vars_to_decode_b2b_def
    else:
        vars_to_decode_b2b = [vars_to_decode_b2b_tmp ]

    if len(vars_to_decode_classic_def):
        vars_to_decode_classic = list(set(vars_to_decode_b2b_tmp + vars_to_decode_classic_def) )
    else:
        vars_to_decode_classic = vars_to_decode_b2b_tmp

    #vars_to_decode_classic = []; print('DEBUGGGG no classic')
    #behav_df_cur['non_hit'] = True;  print('DEBUG VER OF NONHIT!!!!')


    computation_ver = 'computeErrSens3'
    #time_locked_EScalc = 'target'
    time_locked_EScalc = time_locked
    if par['recalc_err_sens']:
        print('Recalc err sens')
        behav_df_cur, vndef2vn = computeErrSensVersions(behav_df_cur,
            env_to_run,block_names_cur,pertvals_cur,gseqcs_cur,
            tgt_inds_cur, dists_rad_from_prevtgt_cur,
            dists_trial_from_prevtgt_cur,
            error_type = error_type, allow_duplicating = False,
            computation_ver = computation_ver,
            coln_nh='non_hit_not_adj',
            coln_nh_out='non_hit_shfited',
            coln_error = coln_error,
            addvars=vars_to_decode_b2b_tmp, time_locked = time_locked_EScalc)
        # output can be not properly ordered because of filtering used inside
        behav_df_cur = behav_df_cur.sort_values('trials')
        if vndef2vn is None:
            vndef2vn = {}
    else:
        assert  set(vars_to_decode_b2b_tmp ) < set(behav_df_cur.columns)
        vndef2vn = {}

    #assert len(behav_df_cur) == len(behav_df_full)
    # it can be smaller than behav_df_full because we filter inside
    assert len(behav_df_cur) > 0

    # make sure trial index consistency was not destroyed in compute err sens
    assert np.array_equal( dfev['trial_index'].values , behav_df_cur['trials'].values)

    for vn in vars_to_decode_b2b_tmp:
        assert vn in behav_df_cur.columns
    # TODO: check that each 'trials' col element appears only once

    ####################################

    if exit_after == 'recalc_ES':
        sys.exit(0)

    # let's reread after ensuring consistency
    environment = np.array(behav_df_cur['environment'])

    #targets      = np.array(behav_df_cur['target'])
    #movement     = np.array(behav_df_cur['org_feedback'])
    for env in env_to_run:
        print('--- Starting {:.2f},{:.2f} env = {}'.format(tmin_cur,tmax_cur,env) )
        if env in env2envcode:
            envcode = env2envcode[env]
            subdf0 = behav_df_cur[ behav_df_cur['environment'] == envcode ]
        else:
            assert env == 'all'
            subdf0 = behav_df_cur

        fname_full = genFnSliding(results_folder, env,
                        regression_type,time_locked,
                        an_suffix,freq_name,tmin_cur,tmax_cur,
                        trial_group_col_calc)

        # whether we remove only inf nan or also outliers
        if trim_outliers:
            subdf, mask_trunc = truncateDf(subdf0, colns_ES[0],
                        q=0.05, verbose=0, infnan_handling='discard',
                       return_mask = True)
        else:
            subdf, mask_trunc = truncateDf(subdf0, colns_ES[0],
                        q=0, verbose=0, infnan_handling='discard',
                       return_mask = True)

        print(f'After cleaning outliers (trim_outliers={trim_outliers}) and nan/infs'
              f': {sum(mask_trunc)} / {len(mask_trunc) } ')

        ################################  prepare X
        trials_left = subdf['trials']
        dfev_cur_trunc = dfev.query('trial_index_ev in @trials_left')
        # idk why but its really important to have str values here instead of ints
        #ep = epochs[ stage2evn2event_ids_str[time_locked][env] ]

        #dfev_cur = getTargetCodes_NIH(epochs_for_EC, ep)
        #dfev_cur = dfev_cur.sort_values('sample')
        #dfev_cur_trunc = dfev_cur.query('trial_index in @subdf.trial_index.values')

        # remember that we ensured consistency of indices earlier
        epochs_curenv = epochs[dfev_cur_trunc.index.values ]
        checkTgtConsistency(subdf, epochs_curenv) # checks if epochs and df are consistent

        tmin_safe = tmin_cur + safety_time_bound
        tmax_safe = tmax_cur - safety_time_bound

        X = epochs_curenv.pick_types(meg=True, ref_meg=False)._data
        times = epochs_curenv.times

        assert X.shape[0] == len(subdf), (X.shape, len(subdf) )

        wh = (times > tmin_safe) & (times < tmax_safe)
        #X = X[non_hit] # no, we do it later, depending on which war we decode
        X = X[:, :, wh]

        ##################################  prepare classif
        # Cross-validation
        cv = KFold(nb_fold, shuffle=True)
        #cv = GroupKFold(nb_fold, shuffle=True)

        ##################################
        svd = {'par':par}
        svd['varnames'] = list( vndef2vn.values() )
        svd['varnames_def2varnames'] = vndef2vn
        ##################################  run classif

        #debug_save_Xy_exit = 1
        debug_save_Xy_exit = 0

        from meg_proc import precalcRegCov, ML_classic, ML_b2b
        #covs = np.empty((n_epochs, n_channels, n_channels))
        
        precalc_covs_only_notnaninf = False
        if precalc_covs_only_notnaninf:
            # just need to getMaskNotNanInf for every variable and make &
            # should not loose that much because mostly they fail together
            # need to take into acc discard_hit_twice
            raise ValueError('not implemented')

        varnames_good_epochs_calc = list(set(vars_to_decode_classic +\
                sum(vars_to_decode_b2b, [])))
        mask_good_epochs = getCleanEpochsMaskForDec(subdf, 
                varnames_good_epochs_calc, env, time_locked, 
                par['discard_hit_twice']) 

        print('precalcRegCov')
        # n_skip_trial will be done insde fit functions
        X_for_fit = X[mask_good_epochs,:n_channels_to_use, :] 
        covs = precalcRegCov(X_for_fit, reg, cov_method_params,
                  rank, fit_log_level, n_jobs)
        print('precalcRegCov finishsed')

        #if decode_merge_pert:
        ##################################################################
        if do_classic_dec and len(vars_to_decode_classic):
            print('-------------  Classical decoding')
        ##################################################################
            addvar_dict = {}
            for addvar in vars_to_decode_classic:
                res = ML_classic(addvar)
                if exit_after == 'rescale':
                    sys.exit(0)

                addvar_dict[addvar] = res


                if exit_after == 'classic_1st':
                    sys.exit(0)

            #if debug_save_Xy_exit:
            #    print('debug_save_Xy_exit')
            #    sys.exit(0)

            svd['decoding_per_var'] = addvar_dict

            if save_result:
                dfpath = fname_full.replace('.npz','.pkl')
                subdf.to_pickle(dfpath )
                svd['dfpath'] = dfpath
                np.savez(fname_full, **svd)
                print(f'after classic decoding and saved '
                        'results to {fname_full}')


        ##################################################################
        print('-------------  B2B decoding')
        ##################################################################

        if do_b2b_dec and len(vars_to_decode_b2b):
            addvar_dict = {}
            for addvars in vars_to_decode_b2b:
                res = ML_b2b(addvars)
                addvar_dict[','.join(addvars) ] = res

                if exit_after == 'b2b_1st':
                    sys.exit(0)
            svd['decoding_per_var_b2b'] = addvar_dict

            if debug_save_Xy_exit:
                print('debug_save_Xy_exit')
                sys.exit(0)

        #if decode_per_pert:
        #    addvar_dict = {}
        #    for addvar in vars_to_decode_classic:
        #        for pertval in behav_df_cur['perturbation'].unique():
        #            print(f'-------------  per pert decoding, pert = {pertval}')
        #            mask = df_esv['perturbation'][non_hit] == pertval
        #            res = ML(addvar, mask, name= f'pert={pertval}')
        #            addvar_dict[ (addvar, int(pertval) ) ] = res
        #    svd['decoding_per_var_and_pert'] = addvar_dict

        ##################################  save res

        # 0 -- target angs, 1 -- prev target angs,
        # 2 -- current movemnet, 3 -- prev movements, 4 -- prev error
        #values_for_es = values_for_es[:, non_hit]
        # I'm not sure I really need it, just legacy
        #prev_error    = np.array(values_for_es[4] )
        #prev_movement = np.array(values_for_es[3] )
        #prev_error    = df_esv[vndef2vn['prev_error'] ]    [ non_hit]
        #prev_movement = df_esv[vndef2vn['prev_movement'] ] [ non_hit]
        #svd['prev_error']    = prev_error
        #svd['prev_movement'] = prev_movement

        #svd[coln_ES] = err_sens_non_hit
        # these are varnames related to err sens
        #svd['non_hit'] = non_hit
        #svd['correction'] = correction_non_hit

        if save_result:
            dfpath = fname_full.replace('.npz','.pkl')
            subdf.to_pickle(dfpath )
            svd['dfpath'] = dfpath
            np.savez(fname_full, **svd)
            print(f'Finished and saved results to {fname_full}')

        print(f'Decoding errors summary tmin={tmin_cur}, env={env}:')
        ks = ['decoding_per_var','decoding_per_var_b2b']
        for kk in ks:
            if kk not in svd:
                print(f'!!  {kk} not present')
            else:
                for k,v in svd[kk].items():
                    err = v['dec_error']
                    if err is not None:
                        print(k, v['dec_error'] )

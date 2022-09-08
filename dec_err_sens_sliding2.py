import os
import os.path as op
import numpy as np
import mne
from mne.io import read_raw_fif
from csp_my import SPoC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
import xgboost
#from sklearn.metrics import make_scorer
from scipy.stats import spearmanr
from mne import Epochs
import pandas as pd
import warnings
import sys
from base2 import (getXGBparams,
                   calc_target_coordinates_centered,
                   target_angs, getGPUavail)
#from base2 import B2B_SPoC,
from config2 import n_jobs as n_jobs_def
from config2 import (min_event_duration,genFnSliding,
                     path_data,path_data_tmp,freq_name2freq,
                     stim_channel_name,delay_trig_photodi,
                     stage2evn2event_ids, stage2evn2event_ids_str,
                     paramFileRead, env2envcode, env2subtr)
from error_sensitivity import (enforceTargetTriggerConsistency,
                               computeErrSens2,adjustNonHit)
#from error_sensitivity import enforceTargetTriggerConsistency
from xgboost import XGBRegressor

###########
from datetime import datetime  as dt
from config2 import paramFileRead, genArgParser

parser = genArgParser()
parser.add_argument('--each_SPoC_fit_is_parallel', default=1 )
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

#freq_name = ['broad']
#freqs = [(4, 60)]
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

output_folder = par.get('output_folder',f'corr_spoc_es_sliding2_{hpass}')
ICAstr = par.get('ICAstr','with_ICA'  )  # empty string is allowed
time_locked = par.get('time_locked','target')
control_type = par.get('control_type','movement')
# home position
if par['slide_windows_type'] == 'auto':
    from config2 import stage2time_bounds
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
do_partial_dec            = int( par.get('do_partial_dec',1)            )
est_parallel_across_dims  = int( par.get('est_parallel_across_dims',1)  )
est_parallel_within_dim   = int( par.get('est_parallel_within_dim',0)   )
b2b_each_fit_is_parallel  = int( par.get('b2b_each_fit_is_parallel',0)   )
each_SPoC_fit_is_parallel = int( par.get('each_SPoC_fit_is_parallel',1)   )
classic_dec_verbose       = int( par.get('classic_dec_verbose',3)       )

B2B_SPoC_parallel_type = par.get('B2B_SPoC_parallel_type', 'across_splits')
# 'across_splits_and_dims'

nb_fold                   = int( par.get('nb_fold',6)                   )
decim_epochs              = int( par.get('decim_epochs',2)                   )
n_splits_B2B              = int( par.get('n_splits_B2B',30)             )
SPoC_n_components         = int( par.get('SPoC_n_components',5)         )
safety_time_bound         = float( par.get('safety_time_bound',0.05) )
random_seed               = int( par.get('random_seed',0) )
crop                      = par.get('crop',None) ;
if crop is not None:
    crop = eval(crop)

#analysis_name_from_par = par.get('analysis_name', 'prevmovement_preverrors_errors_prevbelief')
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

fname = op.join(path_data, subject, 'behavdata', f'err_sens_{task}.npz')
f = np.load(fname, allow_pickle=True)['arr_0'][()]
#env2corr      = f['env2corr'][()]
#env2err_sens      = f['env2err_sens'][()]
#env2pre_err_sens  = f['env2pre_err_sens'][()]
env2pre_dec_data  = f['env2pre_dec_data']
env2non_hit  = f['env2non_hit']

# read behav data
fname = op.join(path_data, subject, 'behavdata',
                f'behav_{task}_df.pkl' )
behav_df_full = pd.read_pickle(fname)

targets_codes_full      = np.array(behav_df_full['target_codes'])

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

path_data_tmp_cursubj = op.join(path_data_tmp, subject)
fn_flt_raw = f'raw_{task}_{hpass}_{ICAstr}_{freq_name}.fif'
fn_flt_raw_full = op.join(path_data_tmp_cursubj, fn_flt_raw )

fn_events = f'{task}_{hpass}_{ICAstr}_eve.txt'
fn_events_full = op.join(path_data, subject, fn_events )

from config2 import cleanEvents

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

            events = cleanEvents(events)
        else:
            events = mne.read_events(fn_events_full)
        loaded_flt_raw = True
    except Exception as e:
        print('loading filtered raw failed, got exception ',str(e))


# if raw is not present, re-filter it from zero
if (raw is None) or ( (not loaded_flt_raw) and \
    ( not (use_preloaded_flt_raw and preloaded_flt_raw_is_present) ) ):
    raw = read_raw_fif(op.join(path_data, subject, f'raw_{task}_{hpass}_{ICAstr}.fif' ),
            preload=True)
    if crop is not None:
        raw.crop(crop[0],crop[1])
    events = mne.find_events(raw, stim_channel=stim_channel_name,
                                min_duration=min_event_duration)
    events[:, 0] += delay_trig_photodi
    events = cleanEvents(events)

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

##################################    Start calc
for tmin_cur,tmax_cur in tminmax:
    for env in env_to_run:
        analysis_name = 'err_sens'
        fname_full = genFnSliding(results_folder, env,
                        regression_type,time_locked,
                        analysis_name,freq_name,tmin_cur,tmax_cur)

        # first we need to take all events to ensure consistency
        event_ids_all = stage2evn2event_ids[time_locked]['all']
        bsl = None
        epochs = Epochs(raw, events, event_id = event_ids_all,
                        tmin=tmin_cur, tmax=tmax_cur, preload=True,
                        baseline=bsl, decim=decim_epochs)


        environment_full = np.array(behav_df_full['environment'])
        ## modifies in place
        behav_df_cur = enforceTargetTriggerConsistency(behav_df_full.copy(),
                                        epochs,
                                        environment_full,
                                        save_fname=fname)

        if exit_after == 'enforce_consist':
            sys.exit(0)

        # TODO: maybe I have to reread stuff after ensuring consistency
        environment = np.array(behav_df_cur['environment'])

        #targets      = np.array(behav_df_cur['target'])
        #movement     = np.array(behav_df_cur['org_feedback'])

        if env in env2envcode:
            envcode = env2envcode[env]
            trial_inds = np.where(environment == envcode)[0]
        else:
            assert env == 'all'
            trial_inds = np.arange(len(environment))

        #non_hit = env2non_hit[env]
        #non_hit_not_adj,corr,es,values_for_es,varnames,varnames_def =\
        non_hit_not_adj, df_esv, vndef2vn  =\
            computeErrSens2(behav_df_cur, trial_inds, epochs,
                         enforce_consistency = 0)

        non_hit = adjustNonHit(non_hit_not_adj, env, time_locked)
        #corr = corr[non_hit]   # not used, only saved
        correction_non_hit = df_esv['correction'][non_hit]
        err_sens_non_hit   = df_esv['err_sens'][non_hit]

        ################################  prepare X
        times = epochs.times
        # idk why but its really important to have str values here instead of ints
        ep = epochs[ stage2evn2event_ids_str[time_locked][env] ]
        tmin_safe = tmin_cur + safety_time_bound
        tmax_safe = tmax_cur - safety_time_bound

        X = ep.pick_types(meg=True, ref_meg=False)._data
        assert X.shape[0] == len(non_hit), (X.shape, len(non_hit) )
        wh = (times > tmin_safe) & (times < tmax_safe)
        X = X[non_hit]
        X = X[:, :, wh]

        ##################################  prepare classif

        spoc = SPoC(n_components=5, log=True, reg='oas',
                    rank='full', n_jobs=n_jobs_SPoC,
                    fit_log_level=mne_fit_log_level)
        # Regression for classic decoding
        if regression_type == 'Ridge':
            alphas = np.logspace(-5, 5, 12)
            est = make_pipeline(spoc, RidgeCV())
        elif regression_type == 'xgboost':
            xgb = XGBRegressor(**add_clf_creopts)
            est = make_pipeline(spoc, xgb)
            # Regressions for the B2B

            #param_grid = {
            #    'pca__n_components': [5, 10, 15, 20, 25, 30],
            #    'model__max_depth': [2, 3, 5, 7, 10],
            #    'model__n_estimators': [10, 100, 500],
            #}
            #grid = GridSearchCV(pipeline, param_grid,
            #  cv=5, n_jobs=-1, scoring='roc_auc')
        else:
            raise ValueError('wrong regression value')

        # Cross-validation
        cv = KFold(nb_fold, shuffle=True)

        ##################################
        svd = {'par':par}
        ##################################  run classif

        vars_to_decode = ['err_sens', 'correction', 'prev_error']
        #y = err_sens_non_hit  # already non_hit
        #y_preds = np.zeros(len(y))
        #scores = list()
        #print(f'Starting CV for regression_type={regression_type}')
        #nsplit = 0
        #for train, test in cv.split(X, y):
        #    print(f'Starting split N={nsplit}')
        #    with mne.use_log_level(mne_fit_log_level):
        #        est.fit(X[train], y[train])
        #    y_preds[test] = est.predict(X[test])
        #    score = spearmanr(y_preds[test], y[test])
        #    scores.append(score[0])
        #    nsplit += 1
        #diff = np.abs(y - y_preds)

        #svd['diff_err_sens_pred'] = diff
        #svd['scores_err_sens'] = np.array(scores)
        ###########################
        #addvars = ['error', 'correction']
        addvar_dict = {}
        for addvar in vars_to_decode:
            if addvar in behav_df_cur.columns:
                vals       = np.array(behav_df_cur[addvar])
            elif addvar in df_esv.columns:
                vals       = np.array(df_esv[addvar])
            vals_non_hit = vals[non_hit]  # already non_hit

            res = {}

            y = vals_non_hit
            y_preds = np.zeros(len(y))
            scores = list()
            print(f'{addvar}: Starting CV for regression_type={regression_type}')
            nsplit = 0
            for train, test in cv.split(X, y):
                print(f'Starting split N={nsplit}')
                with mne.use_log_level(mne_fit_log_level):
                    est.fit(X[train], y[train])
                y_preds[test] = est.predict(X[test])
                score = spearmanr(y_preds[test], y[test])
                scores.append(score[0])
                nsplit += 1
            diff = np.abs(y - y_preds)

            res['diff']   = diff
            res['scores'] = scores
            res['vals'] = vals_non_hit  # non_hit

            print(f'{addvar} scores average = {np.mean(scores):.4f}' )

            addvar_dict[addvar] = res

        svd['decoding_per_var'] = addvar_dict
        ##################################  save res

        # 0 -- target angs, 1 -- prev target angs,
        # 2 -- current movemnet, 3 -- prev movements, 4 -- prev error
        #values_for_es = values_for_es[:, non_hit]
        # I'm not sure I really need it, just legacy
        #prev_error    = np.array(values_for_es[4] )
        #prev_movement = np.array(values_for_es[3] )
        #prev_error    = df_esv[vndef2vn['prev_error'] ]    [ non_hit]
        prev_movement = df_esv[vndef2vn['prev_movement'] ] [ non_hit]
        #svd['prev_error']    = prev_error
        svd['prev_movement'] = prev_movement

        #svd['err_sens'] = err_sens_non_hit
        svd['varnames'] = list( vndef2vn.values() )
        svd['varnames_def2varnames'] = vndef2vn
        svd['non_hit'] = non_hit
        #svd['correction'] = correction_non_hit

        np.savez(fname_full, **svd)
        print(f'Finished scores to {fname_full}')

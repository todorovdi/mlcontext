from config2 import n_jobs, freq_name2freq, stage2time_bounds
from config2 import subjects
from config2 import path_code
import numpy as np
import os

script = 'dec_err_sens_sliding2.py'

#rts = ['Ridge', 'xgboost']
#hpasses = ['no_filter', '0.1', '0.05']
rts = ['Ridge']
#hpasses = ['no_filter', '0.1', '0.05']
#hpasses = ['no_filter', '0.1']
hpasses = ['no_filter']

#shift = 0.25
dur   = 0.464
shift = dur / 2
#shift = dur / 4

#dur   = 0.5
#tmins = np.arange(-3,-0.5+shift,shift)
#analysis_name = 'sliding_prevmovement_preverrors_errors_prevbelief'


#epochs_bsl = Epochs(raw, events, event_id=event_ids_tgt,
#                    tmin=-0.46, tmax=-0.05, preload=True,
#                    baseline=None, decim=6)

# rts = rts[:1]
# hpasses = hpasses[:1]
#, decim=6)
# decim=6)

envs = ['stable', 'random']
#envs = ['all']
seeds = [0]
trim_outliers_vs = [0, 1]
trial_group_col_calc_vs = ['trialwe', 'trialwtgt_we']
time_lockeds = ['target', 'feedback']
freq_names = list( freq_name2freq.keys())
dists_rad_from_prevtgt = ['all', '0.00', '0.79', '1.57' ]
dists_trial_from_prevtgt = ['all', 1,2,4]


# just to have less entries
time_lockeds = ['target', 'feedback']
freq_names = ['broad']
dists_rad_from_prevtgt =   ['all', '0.00', '1.57' ]
dists_trial_from_prevtgt = ['all', 1,4]


#ipy = get_ipython()

#pars = []
runstrings = []
# when freq is outside subject I could re-use filterd raws
ind_glob = 0
run      = 'python ' + os.path.join( path_code, script ) + ' '
run_test = 'ipython -i ' + os.path.join( path_code, script ) + ' -- '

from itertools import product as itprod
p = itprod(hpasses, time_lockeds, freq_names, subjects, rts, seeds,
           trim_outliers_vs, trial_group_col_calc_vs, dists_rad_from_prevtgt,
           dists_trial_from_prevtgt)
#for hpass in hpasses:
##    for control_type in ['movement']:
#    #for time_locked in ['target', 'feedback']:
#    #for time_locked in ['target']:
#    for time_locked in ['target', 'feedback']:
#        start, end = stage2time_bounds[time_locked]
#        tmins = np.arange(start,end,shift)
#        tmaxs = dur + tmins
#        #tminmax = list(zip(tmins,tmaxs))
#        for freq_name, freq_limits in freq_name2freq.items():
#            for subject in subjects:
#                #for tmin,tmax in tminmax:
#                for regression_type in rts:
#                    # for env_to_run in envs:
for tpl in p:
    hpass, time_locked, freq_name, subject, regression_type, \
        seed, trim_outliers, trial_group_col_calc, dr, dt  = tpl
    freq_limits = freq_name2freq[freq_name]

    if trial_group_col_calc == 'trialwtgt_we' and ( (dists_rad_from_prevtgt != 'all') \
            or (dists_trial_from_prevtgy != 'all' ) ):
        continue

    s = run
    start, end = stage2time_bounds[time_locked]
    tmins = np.arange(start,end,shift)
    tmaxs = dur + tmins


    s += f' --param_file dec_err_sens_sliding.ini'
    s += f' --random_seed {seed}'
    s += f' --output_folder corr_spoc_es_sliding2_{hpass}'

    s += f' --hpass {hpass}'
    s += f' --slide_window_dur {dur}'
    s += f' --slide_window_shift {shift}'
    s += f' --trial_group_col_calc {trial_group_col_calc}'
    s += f' --trim_outliers {trim_outliers}'
    s += f' --dists_rad_from_prevtgt {dr}'
    s += f' --dists_trial_from_prevtgt {dt}'

    #s += f' --slide_windows_type auto'
    #s += f' --tmin ' + ','.join( map(str,tmins) )
    #s += f' --tmax ' + ','.join( map(str,tmaxs) )
    s += f' --subject {subject}'
    s += f' --time_locked {time_locked}'
    #s += f' --control_type {control_type}'
    s += f' --regression_type ' + ','.join(rts)
    s += f' --freq_name {freq_name}'
    s += f' --freq_limits ' + f'{freq_limits[0]},{freq_limits[1]}'
    s += f' --env_to_run ' + ','.join(envs)

    s += f' --custom_suffix trim{trim_outliers}_{trial_group_col_calc[5:]}_dr{dr}_dt{dt}'
    #s += 'analysis_name'] = analysis_name
    s += '\n'
    runstrings += [s]

    ind_glob += 1

runstrings = [ runstrings[0].replace(run,run_test) ] + runstrings

with open('_runstrings.txt','w') as f:
    f.writelines(runstrings)

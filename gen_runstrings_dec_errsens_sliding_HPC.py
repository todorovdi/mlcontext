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
#shift = dur / 2
shift = dur / 4



envs = ['stable', 'random']
#envs = ['all']
seeds = [0]
#trim_outliers_vs = [0, 1]

trim_outliers_vs = [0]
#trial_group_col_calc_vs = ['trials', 'trialwe', 'trialwtgt_we']
trial_group_col_calc_vs = ['trials']
time_lockeds = ['target', 'feedback']
freq_names = list( freq_name2freq.keys())
dists_rad_from_prevtgt = ['all', '0.00', '0.79', '1.57' ]
dists_trial_from_prevtgt = ['all', 1,2,4]


# just to have less entries
#time_lockeds = ['target', 'feedback']
time_lockeds = ['target'] 
freq_names = ['broad']
#dists_rad_from_prevtgt =   ['all', '0.00', '1.57' ]
dists_rad_from_prevtgt =   ['all' ]
#dists_trial_from_prevtgt = ['all', 1,4]
dists_trial_from_prevtgt = ['all']

control_types = ['movement', 'feedback']

safety_time_bound = 0
discard_hit_twice = 1
#scale_X_robust = 1
#scale_Y_robust = 1
scale_X_robust = 0
scale_Y_robust = 0

windowstr = (f' --slide_window_dur {dur} --slide_window_shift {shift} '
        '--slide_windows_type=auto')
calc_name='slide'

scale_X_robust = 1
scale_Y_robust = 1
discard_hit_twice = 0

##### SPOC_home_ver
#safety_time_bound = 0.05
#windowstr = f' --slide_windows_type explicit --tmin -0.5 --tmax 0 '
#time_lockeds = ['target']
#calc_name='home'
#discard_hit_twice = 0
#
#scale_X_robust = 1
#scale_Y_robust = 1
########

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
           dists_trial_from_prevtgt, control_types)
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
        seed, trim_outliers, trial_group_col_calc, dr, dt, control_type  = tpl
    freq_limits = freq_name2freq[freq_name]

    if trial_group_col_calc == 'trialwtgt_we' and ( (dists_rad_from_prevtgt != 'all') \
            or (dists_trial_from_prevtgy != 'all' ) ):
        continue

    s = run
    #start, end = stage2time_bounds[time_locked]
    #tmins = np.arange(start,end,shift)
    #tmaxs = dur + tmins


    s += f' --param_file dec_err_sens_sliding.ini'
    s += f' --random_seed {seed}'
    s += f' --output_folder corr_spoc_es_sliding2_{hpass}'

    s += f' --hpass {hpass}'
    s += windowstr

    s += f' --safety_time_bound {safety_time_bound}'

    s += f' --trial_group_col_calc {trial_group_col_calc}'
    s += f' --trim_outliers {trim_outliers}'
    s += f' --dists_rad_from_prevtgt {dr}'
    s += f' --dists_trial_from_prevtgt {dt}'

    
    s += f' --each_SPoC_fit_is_parallel 1'
    s += f' --scale_X_robust {scale_X_robust}'
    s += f' --scale_Y_robust {scale_Y_robust}'
    s += f' --discard_hit_twice {discard_hit_twice}'

    #s += f' --slide_windows_type auto'
    #s += f' --tmin ' + ','.join( map(str,tmins) )
    #s += f' --tmax ' + ','.join( map(str,tmaxs) )
    s += f' --subject {subject}'
    s += f' --time_locked {time_locked}'
    s += f' --control_type {control_type}'
    s += f' --regression_type ' + ','.join(rts)
    s += f' --freq_name {freq_name}'
    s += f' --freq_limits ' + f'{freq_limits[0]},{freq_limits[1]}'
    s += f' --env_to_run ' + ','.join(envs)

    s += (f' --custom_suffix {control_type}_CN{calc_name}_trim{trim_outliers}_'
        f'dhittw{discard_hit_twice}'
        f'_{trial_group_col_calc[5:]}_scX{scale_X_robust}Y{scale_Y_robust}_dr{dr}_dt{dt}')
    #s += 'analysis_name'] = analysis_name
    s += '\n'
    runstrings += [s]

    ind_glob += 1

runstrings = [ runstrings[0].replace(run,run_test) ] + runstrings

with open('_runstrings.txt','w') as f:
    f.writelines(runstrings)

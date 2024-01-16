from config2 import n_jobs, freq_name2freq, stage2time_bounds
from config2 import subjects
from config2 import path_code, path_data
import numpy as np
import os
from os.path import join as pjoin

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
#shift = dur / 4  # some 8-ish hours



envs = ['stable', 'random'] # will be joined in one string 
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
#time_lockeds = ['target'] 
freq_names = ['broad']
#dists_rad_from_prevtgt =   ['all', '0.00', '1.57' ]
dists_rad_from_prevtgt =   ['all' ]
#dists_trial_from_prevtgt = ['all', 1,4]
dists_trial_from_prevtgt = ['all']

control_types = ['movement', 'feedback']

safety_time_bound = 0
discard_hit_twice_vs = [1]
#scale_X_robust = 1
#scale_Y_vals = [1]
scale_X_robust = 0
scale_Y_vals = [0]

windowstr = (f' --slide_window_dur {dur} --slide_window_shift {shift} '
        '--slide_windows_type auto')
calc_name='slide'

scale_X_robust = 0
scale_Y_vals = [1]
#discard_hit_twice_vs = 


scale_Y_vals = [2] # this means classical scale (but respecting centering par)
discard_hit_twice_vs = [1]
time_lockeds = ['target', 'feedback']

freq_names = ['broad', 'beta', 'gamma']

centerings = [0]

##### SPOC_home_ver (to compare with Romain output)
# takes 7-ish minutes to compute
#freq_names = ['broad']
#safety_time_bound = 0.05
#windowstr = f' --slide_windows_type explicit --tmin -0.5 --tmax 0 '
#time_lockeds = ['target']
#calc_name='home'
#discard_hit_twice_vs = [1 ]
#scale_X_robust = 0
#scale_Y_vals = [0]
########


##### SPOC_home for model fit decoding
# takes 10-ish minutes to compute
#freq_names = ['broad']
##rts = ['Ridge_noCV']
#freq_names = ['broad'] #, 'beta', 'gamma']
#safety_time_bound = 0.05
#windowstr = f' --slide_windows_type explicit --tmin -0.5 --tmax 0 '
##windowstr = f' --slide_windows_type explicit --tmin 0 --tmax 0.5 '
#time_lockeds = ['target', 'feedback']
#control_types = ['movement']
#calc_name='homeModelES'
#discard_hit_twice_vs = [1]
#scale_X_robust = 0
#scale_Y_vals = [0,1,2 ]
#behav_file = pjoin(path_data, 'row_frame_wmodel_output.pkl.zip' )
#windowstr += f' --behav_file {behav_file}'
#windowstr += ' --do_b2b_dec 1 '
#centerings = [0,1]
#
#colns_classic = ('trials,error_pred_Tan,error_pred_Herz,error_pred_Died,'
#    'prev_error,error')
##,belief,prev_belief # not found
#colns_classic += ',trialwb,trialwtgt_wpertstage_wb,trialwtgt_wb,trialwpert,movement_duration,RT,target_inds'
##colns_classic += ',trialwb,trialwe,trialwpertstage_wb,trialwpert_wb,trialwpert_we,trialwtgt,trialwtgt_wpert_wb,trialwtgt_wpertstage_wb,trialwtgt_wpertstage_we,trialwtgt_wpert_we,trialwtgt_we,trialwtgt_wb,trialwpert,trialwtgt_wpert,movement_duration,RT,target_inds'
#windowstr += f' --behav_file {behav_file} --colns_classic {colns_classic}'
##windowstr += ' --colns_ES err_sens_Tan,err_sens_Herz,err_sens'
#windowstr += ' --colns_ES err_sens'
########

################# SPOC_mini side for model fit decoding
freq_names = ['broad']
#rts = ['Ridge_noCV']
##freq_names = ['broad' , 'beta', 'gamma']
safety_time_bound = 0.00
#dur   = 0.5
#shift = dur * 0.75
windowstr = (f' --slide_window_dur {dur} --slide_window_shift {shift} '
        '--slide_windows_type auto ')
#dur   = 0.5
##shift = dur * 0.75
##windowstr +=  ( ' --time_bounds_slide_target (-1,2) ' 
#            '--time_bounds_slide_feedback  (-1,2) ')
shift = dur * 0.5
windowstr +=  ( ' --time_bounds_slide_target (-5.14,5.068) ' 
            '--time_bounds_slide_feedback  (-5.14,5.068) ')

#centerings = [0,1]
centerings = [1]

time_lockeds = ['target', 'feedback']
control_types = ['movement']
calc_name='slideModelES'
discard_hit_twice_vs = [1]
scale_X_robust = 0
scale_Y_vals = [2 ]
behav_file = pjoin(path_data, 'row_frame_wmodel_output.pkl.zip' )
windowstr += f' --behav_file {behav_file}'
windowstr += ' --do_b2b_dec 1 '

colns_classic = ('trials,error_pred_Tan,error_pred_Herz,error_pred_Died,'
    'prev_error,error')
###,belief,prev_belief # not found
#colns_classic += ',trialwb,trialwe,trialwpertstage_wb,trialwpert_wb,trialwpert_we,trialwtgt,trialwtgt_wpert_wb,trialwtgt_wpertstage_wb,trialwtgt_wpertstage_we,trialwtgt_wpert_we,trialwtgt_we,trialwtgt_wb,trialwpert,trialwtgt_wpert,movement_duration,RT,target_inds'
windowstr += f' --behav_file {behav_file} --colns_classic {colns_classic}'
windowstr += ' --colns_ES err_sens_Tan,err_sens_Herz,err_sens'

###############################
calc_name='homeModelES'
freq_names = ['broad','beta','gamma','theta','alpha']
discard_hit_twice_vs = [0]
time_lockeds = ['target','feedback' ]
windowstr = (' --slide_windows_type explicit '
    '--tmin (-0.5,-0.25,0,0.25) --tmax (0,0.25,0.5,0.75)') # bug :(

#dur = 0.5
#shift=dur/2
#windowstr = (f' --slide_window_dur {dur} --slide_window_shift {shift} '
#        '--slide_windows_type auto ')
#windowstr +=  ( ' --time_bounds_slide_target (-5.14,5.068) ' 
#            '--time_bounds_slide_feedback  (-5.14,5.068) ')


windowstr += ' --do_b2b_dec 1 --recalc_err_sens 0'
windowstr += f' --behav_file {behav_file} '

colnstr = ' --colns_classic {colns_classic}'
colnstr += (' --colns_ES err_sens_Tan,err_sens_Herz,err_sens;'
 'prev_error,error,err_sens_Tan;'
 'prev_error,error,err_sens_Herz;'
 'prev_error,error,err_sens;'
 'error,error_pred_Tan,error_pred_Herz,error_pred_Died;'
 'belief,state_Tan,state_Herz,state_Died;'
 'error,state_Tan,state_Herz,state_Died')
colnstr += (';err_sens_Tan,err_sens_Herz,err_sens_trunc;'
 'prev_error,error,err_sens_trunc;'
 'error,err_sens_Tan,err_sens_Herz,err_sens_trunc;'
 'prev_error,error,err_sens_Tan,err_sens_Herz,err_sens_trunc;'
 'error,err_sens_Tan,err_sens_Herz,err_sens')

########################

windowstr = (' --slide_windows_type explicit '
    '--tmin (-0.5,-0.25,0,0.25) --tmax (0,0.25,0.5,0.75)') # bug :(
windowstr += ' --do_b2b_dec 1 --recalc_err_sens 0'
windowstr += f' --behav_file {behav_file} '

freq_names = ['broad','beta','gamma','theta','alpha']
#freq_names = ['broad','beta','alpha']

colns_classic = ('trials,error_pred_Tan,error_pred_Herz,error_pred_Died,'
    'prev_error,error')
colnstr = f' --colns_classic {colns_classic}'
colnstr += (' --colns_ES trials,error_abs,prev_error_abs;'
'error_abs,err_sens_Tan,err_sens_Herz,err_sens_trunc;'
'trials,state_Died,prev_state_Died;'
'trials,state_Tan,prev_state_Tan;'
'trials,state_Herz,prev_state_Herz;'
'error_mav5,error_mav10,error_std5,error_std10;'
'error_prod;prev_error_prod')

# 'state_Died,prev_state_Died;'
# 'state_Tan,prev_state_Tan;'
# 'state_Herz,prev_state_Herz;'
# 'error_abs,prev_error_abs;'
# 'prev_error_abs,err_sens_Tan,err_sens_Herz,err_sens;'
# 'prev_error_abs,error,err_sens_Tan,err_sens_Herz,err_sens_trunc;'

#'error,state_Tan,state_Herz,state_Died')

script = 'dec_err_sens_sliding2_preopt.py'

###############o
#freq_names = ['broad']
#control_types = ['movement']
#calc_name='slideModelES'
#shift=dur/2
#windowstr = (f' --slide_window_dur {dur} --slide_window_shift {shift} '
#        '--slide_windows_type auto ')
#windowstr +=  ( ' --time_bounds_slide_target (-4.0,4.0) ' 
#            '--time_bounds_slide_feedback    (-4.0,4.0) ')
#windowstr += ' --do_b2b_dec 1 --recalc_err_sens 0'
#windowstr += f' --behav_file {behav_file} --colns_classic {colns_classic}'
#windowstr += (' --colns_ES err_sens_Tan,err_sens_Herz,err_sens_trunc;'
# 'prev_error,error,err_sens_trunc;'
# 'error,err_sens_Tan,err_sens_Herz,err_sens_trunc;'
# 'prev_error,error,err_sens_Tan,err_sens_Herz,err_sens_trunc;'
# 'error,err_sens_Tan,err_sens_Herz,err_sens')
##windowstr += ('err_sens_Tan,err_sens_Herz,err_sens;'
## 'prev_error,error,err_sens_Tan;'
## 'prev_error,error,err_sens_Herz;'
## 'prev_error,error,err_sens;'
## 'error,error_pred_Tan,error_pred_Herz,error_pred_Died;'
## 'belief,state_Tan,state_Herz,state_Died')
#script = 'dec_err_sens_sliding2_preopt.py'


#ipy = get_ipython()

#pars = []
runstrings = []
# when freq is outside subject I could re-use filterd raws
ind_glob = 0
run      = 'python ' + os.path.join( path_code, script ) + ' '
run_test = 'ipython -i ' + os.path.join( path_code, script ) + ' -- '

from itertools import product as itprod
p = itprod(rts, hpasses, freq_names,  seeds,
           trim_outliers_vs, discard_hit_twice_vs, scale_Y_vals, centerings,
           trial_group_col_calc_vs, dists_rad_from_prevtgt,
           dists_trial_from_prevtgt, time_lockeds, control_types, subjects )

for tpl in p:
    regression_type, hpass, freq_name,   \
        seed, trim_outliers, discard_hit_twice, scale_Y, centering,\
        trial_group_col_calc, dr, dt, time_locked, control_type, subject   = tpl
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
    s += f' --XYcentering {centering}'
    s += windowstr
    s += colnstr

    s += f' --safety_time_bound {safety_time_bound}'

    s += f' --trial_group_col_calc {trial_group_col_calc}'
    s += f' --trim_outliers {trim_outliers}'
    s += f' --dists_rad_from_prevtgt {dr}'
    s += f' --dists_trial_from_prevtgt {dt}'

    
    s += f' --each_SPoC_fit_is_parallel 1'
    s += f' --scale_X_robust {scale_X_robust}'
    s += f' --scale_Y_robust {scale_Y}'
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
        f'_{trial_group_col_calc[5:]}_scX{scale_X_robust}Y{scale_Y}c{centering}_dr{dr}_dt{dt}')
    s += f'_{freq_name}'
    if 'slide' in calc_name:
        s += f'_sh{shift:.3f}'
    #s += 'analysis_name'] = analysis_name
    s += '\n'
    runstrings += [s]

    ind_glob += 1

# todo replace custmo prefix
trs = runstrings[0].replace(run,run_test)[:-1] + \
        (' --nskip_trial 6 --nb_fold 2 --n_ridgeCV_alphas 2 '
        '--nb_fold 2 --n_channels_to_use 3 --n_splits_B2B 5'
        ' --dec_error_handling raise') + '\n'
        
runstrings = [ trs ] + runstrings
print(trs )

print(f'Generate len(trs) = {len(runstrings)}')

with open('_runstrings.txt','w') as f:
    f.writelines(runstrings)

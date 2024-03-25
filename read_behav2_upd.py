import os
#data_dir_general = '/home/demitau/data_Quentin'
#data_subdir_mem_err_main = 'full_experiments/data2'
from os.path import join as pjoin
import os,sys; sys.path.append(os.path.expandvars('$CODE_MEMORY_ERRORS'))
import error_sensitivity
#data_dir_input = pjoin(data_dir_general,data_subdir_mem_err_main)
import pandas as pd
import time
from base2 import subAngles


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--subject', required = True, type=str)
parser.add_argument('--n_jobs',  default = 20, type=int )
parser.add_argument('--save_suffix',  default='', type=str )
parser.add_argument('--use_sub_angles',  default=0, type=int )
parser.add_argument('--perturbation_random_recalc',  default=1, type=int )
args = parser.parse_args()

subject = args.subject
n_jobs = args.n_jobs

#data_dir_general = '/home/demitau/data_Quentin'
#data_subdir_mem_err_main = 'full_experiments/data2'
#data_dir_input = pjoin(data_dir_general,data_subdir_mem_err_main)
#scripts_dir = pjoin(data_dir_general,'full_experiments','scripts2')

data_dir_general = os.path.expandvars('$DATA_QUENTIN')
data_dir_input = os.path.expandvars('$DATA_MEMORY_ERRORS_STAB_AND_STOCH')
#scripts_dir = pjoin( os.path.expandvars('$CODE_MEMORY_ERRORS'), 'previous_analyses')
scripts_dir = pjoin( os.path.expandvars('$CODE_MEMORY_ERRORS'))

print(data_dir_input)

use_sub_angles = args.use_sub_angles

#subjects = [f for f in os.listdir(data_dir_input) if f.startswith('sub') ]
#subjects = list(sorted(subjects))
from config2 import path_data

print('----------------' + subject + '-----------------------')
folder = pjoin(path_data, subject, 'behavdata')
files = os.listdir(folder)
fname_behavior = list()
task = 'visuomotor'
fname_behavior.extend([pjoin(folder, f) for f in files if ((task in f) and
                                                             ('.log' in f))])

ts = [(time.time(), 'start')]
def logtime(s):
    global ts
    ts += [(time.time(), s)]
    tdif = ts[-1][0] - ts[-2][0]
    print('Time between ', ts[-2][1], f' and {s} is {tdif:.3f}')

print(fname_behavior)
fnf = fname_behavior[0]
# time is time since start
logcols = ('trials,trigger,target_inds,perturbation,jax1,jax2,'
    'feedbackX_screen,feedbackY_screen,'
    'org_feedbackX_screen,org_feedbackY_screen,'
    'error_distance,environment,time').split(',')
df = pd.read_csv(fnf, names=logcols)
# check that time is increasing
assert (df['time'].diff().iloc[1:] > 0).all()
df['subject'] = subject

logtime('read_csv')


import numpy as np

from config2 import stage2evn2event_ids,envcode2env
from base2 import width,height
from behav_proc import aggRows, checkErrBounds
from base2 import calc_rad_angle_from_coordinates, radius, radius_cursor, radius_target

#from behav_proc import addNonHitCol, addBehavCols
#dfos = addBehavCols(dfcc)

target_angs = (np.array([157.5, 112.5, 67.5, 22.5]) + 90) * \
              (np.pi/180)

subtract_pi = False # in Romain it is False

#####################

phase2trigger = {
    'REST_PHASE': 10,
    'TARGET_PHASE': 20,
    'FEEDBACK_PHASE': 30,
    'ITI_PHASE': 40,
    'BREAK_PHASE': 50
}
trigger2phase = {
    10: 'REST_PHASE',
    20: 'TARGET_PHASE',
    30: 'FEEDBACK_PHASE',
    40: 'ITI_PHASE',
    50: 'BREAK_PHASE'
}

# Trials starts with REST_PHASE, ends with ITI. When break happens, it goes between feedback and ITI

# in degrees
df['perturbation'] = -df['perturbation'] # Romain does so
df['phase'] = df['trigger'].apply(lambda x: trigger2phase[x])

d = {'TARGET_PHASE':'target', 'FEEDBACK_PHASE':'feedback'}
df['env'] = df['environment'].apply(lambda x: envcode2env[x])

df['target_locs'] = df['target_inds'].apply(lambda x: target_angs[x])
if subtract_pi:
    df['target_locs'] = df['target_locs']  - np.pi  ## Romain does not subtract pi
logtime('small operations')

df['trial_index'] = df['trials'] # just for compatibility

#def f(x):
#    ph = x['phase']
#    if ph not in d:
#        return -10000
#    else:
#        #print(ph)
#        inds = stage2evn2event_ids[d[ph] ][x['env'] ]
#        #print(inds, x['tgti'])
#        if len(inds) == 1:
#            r = inds[0]
#        else:
#            r = inds[x['target_inds']] 
#        return r
#    
#df['target_trigger'] = df.apply(f, 1)
#logtime('target_trigger')
                

df['org_feedbackX'] = df['org_feedbackX_screen'] - width /2 
df['org_feedbackY'] = df['org_feedbackY_screen'] - height /2 
if not subtract_pi: # Romain does so  
    df['org_feedbackY'] = -df['org_feedbackY']

df['feedbackX'] = df['feedbackX_screen'] - width /2 
df['feedbackY'] = df['feedbackY_screen'] - height /2 
if not subtract_pi:
    df['feedbackY'] = -df['feedbackY']  # Romain does so 

df = df.drop(columns=['feedbackX_screen','feedbackY_screen',
                      'org_feedbackX_screen','org_feedbackY_screen'] )

#########

# WRONG! I don't have to invalidate any perturbation values because break is a part of a normal trial in Romain code
#dfc_ = df.query('phase == "BREAK_PHASE"')
#grp_ = dfc_.groupby(['trials'])
#dfcpc = aggRows(dfc_ , 
#    'time', 'max', grp_, coltake='corresp').copy()
#dfcpc['perturbation'] = np.nan
#dfcpc[['jax1','jax2','org_feedbackX','org_feedbackY',
#      'feedbackX','feedbackY']] = np.nan

#########

#df['feedbackX'].isna().any()

c = df['feedbackX'].isna()
df.loc[~c, 'feedback'] = calc_rad_angle_from_coordinates(
    df.loc[~c, 'feedbackX'].values, 
    df.loc[~c, 'feedbackY'].values, radius)
if subtract_pi: 
    df['feedback'] = df['feedback'] - np.pi # Romain does not subtract 

c = df['org_feedbackX'].isna()
df.loc[~c, 'org_feedback'] = calc_rad_angle_from_coordinates(
    df.loc[~c, 'org_feedbackX'].values, 
    df.loc[~c, 'org_feedbackY'].values, radius)
if subtract_pi: 
    df['org_feedback'] = df['org_feedback'] - np.pi # Romain does not subtract

if use_sub_angles:
    df['error']  = subAngles(df['feedback'], df['target_locs'])
    df['belief'] = subAngles(df['org_feedback'], df['target_locs']) # what is used for correction
else:
    df['error']  = df['feedback']- df['target_locs']
    df['belief'] = df['org_feedback']- df['target_locs'] # what is used for correction

logtime('small operations')

home_radius = radius_cursor + radius_target

# this is how things were done in Romain script, only 'jax1','jax2' are related to actual traj in the log file and the rest of the columns are mostly for endpoint info even though they are present for every frame
df['traj_dist_from_center'] = np.sqrt((df['jax1'] -  width /2)**2 +\
                                         (df['jax2'] -  height /2)**2)
def f(df__):
    RT = pd.NA
    df_ = df.loc[df__.index]

    subdf0 = df_.query('phase == "TARGET_PHASE"')
    st_time = subdf0.iloc[0]['time']
    subdf1 = subdf0.query('traj_dist_from_center > @home_radius')
    exit_home_time = subdf1.iloc[0]['time']

    RT = exit_home_time -  st_time
    
    reltime_start_tgt_phase = st_time - df_.iloc[0]['time']
    newdf = df_[['trials','time']].copy()
    newdf['reltime_start_tgt_phase'] = reltime_start_tgt_phase
    newdf['RT'] = RT
    return newdf
assert df['time'].diff().min() > 0
r = df.groupby(['trials'], group_keys=False, sort = False).apply(f)
df['RT'] = r['RT']
df['reltime_start_tgt_phase'] = r['reltime_start_tgt_phase']

logtime('RT')

dfc = df
grp = dfc.query('phase == "FEEDBACK_PHASE"').groupby(['trials'])
dfcc = aggRows(dfc, 'time', 'max', grp, coltake = 'corresp')
dfcc['time_diff'] = dfcc['time'].diff()

# for random env redefine perturbation
if args.perturbation_random_recalc:
    inds = dfcc.query('environment == 1').index
    if use_sub_angles:
        dfcc.loc[inds,'perturbation'] = subAngles( dfcc.loc[inds,'feedback'] , dfcc.loc[inds,'org_feedback'] ) * 180 / np.pi 
    else:
        dfcc.loc[inds,'perturbation'] = ( dfcc.loc[inds,'feedback'] - dfcc.loc[inds,'org_feedback'] ) * 180 / np.pi

##########################

FPS = 120
# indices correspond to trials for which AFTER movement break happened 
break_durations = dfc.query('phase == "BREAK_PHASE"').groupby('trials').size() / FPS
break_durations = break_durations.to_frame()
print(break_durations)

###################

# one row = one phase
grp = dfc.groupby(['trials','phase'])
dfcc0 = aggRows(dfc, 'time', 'max', grp, coltake = 'corresp')
dfcc0['time_diff_phase'] = dfcc0['time'].diff() # time difference between phases
dfcc0 = dfcc0.set_index(['trials','phase'])
dfcc0['phase_duration'] = aggRows(dfc, 'time', 'max', grp, coltake = 'corresp').set_index(['trials','phase'])['time'] -\
      aggRows(dfc, 'time', 'min', grp, coltake = 'corresp').set_index(['trials','phase'])['time']

dfcc0 = dfcc0.reset_index()
dfcc0 = dfcc0.sort_values('time')
dfcc0['time_since_last'] = \
    dfcc0.sort_values('time').groupby('phase')['time'].diff()

if args.perturbation_random_recalc:
    inds = dfcc0.query('environment == 1').index
    if use_sub_angles:
        dfcc0.loc[inds,'perturbation'] = subAngles( dfcc0.loc[inds,'feedback'] , dfcc0.loc[inds,'org_feedback'] ) * 180 / np.pi 
    else:
        dfcc0.loc[inds,'perturbation'] = ( dfcc0.loc[inds,'feedback'] - dfcc0.loc[inds,'org_feedback'] ) * 180 / np.pi

logtime('phase_duration')


# I need to always put 'target' here to be consistent with Romain code
tgt = stage2evn2event_ids['target' ]
def f(x):
    ph = x['phase']
    if ph not in d:
        return -10000
    else:
        inds = tgt[x['env'] ]
        if len(inds) == 1:
            r = inds[0]
        else:
            r = inds[x['target_inds']] 
        return r
dfcc0['target_codes'] = dfcc0.apply(f, 1)
logtime('target_codes')


#dfcc0['phase_duration'].describe()

######################


# definte break duration for trials where it happened
#c = ( dfcc0['phase'] == "BREAK_PHASE" )
#trials_with_break = dfcc0[c]['trials'].unique()
#inds_ = dfcc0.query('trials.isin(@trials_with_break)').index
#dfcc0.loc[inds,'pre_break_duration'] = 


dfcc0['pre_break_duration'] = np.nan
dfcc0['was_pre_break'] = False

def f(df):
    dur = 0.
    #print()
    df_ = dfcc0.loc[df.index]
    subdf = df_.query('phase == "BREAK_PHASE"')
    if len(subdf ):
        dur = subdf.iloc[0]['phase_duration']
    #print(dur)
    #pd.DataFrame({'trials'})
    newdf = df_[['trials','phase']]#.copy()
    newdf['pre_break_duration'] = dur
    if dur > 1e-10:
        newdf['was_pre_break'] = True
    else:
        newdf['was_pre_break'] = False
    return newdf
r = dfcc0.groupby(['trials'], group_keys=False).apply(f)

dfcc0.set_index(['trials','phase'])
r.set_index(['trials','phase'])
dfcc0['pre_break_duration'] = r['pre_break_duration']
dfcc0['was_pre_break'] = r['was_pre_break']

dfcc0 = dfcc0.reset_index()

# need to shift because otherwise it is about ongoing trial
assert dfcc0.time.diff().min() > 0
dfcc0.was_pre_break = dfcc0.was_pre_break.shift(1)
dfcc0.loc[0,'was_pre_break'] = False
dfcc0.pre_break_duration = dfcc0.pre_break_duration.shift(1)
dfcc0.loc[0,'pre_break_duration'] = 0.0

logtime('break_duration')

################### take only one phase
dfcc0_ = dfcc0.sort_values('time')
c = dfcc0_['phase'].isin( ["FEEDBACK_PHASE"])
dfcc1 = dfcc0_[c].copy()
print( len(dfcc), len(dfcc1) )
assert np.min(dfcc1['time'].diff().iloc[1:]) > 0
assert not dfcc1.duplicated(['trials']).any()
dfcc1['trial_type'] = ''

c = dfcc1['perturbation'].abs() > 1e-10
dfcc1.loc[c,'trial_type'] = 'perturbation'
dfcc1.loc[~c,'trial_type'] = 'veridical'
dfcc1.loc[dfcc1['phase'] == "BREAK_PHASE",'trial_type'] = 'pause'

###############  add more stuff from dfcc0
#dfcc2 = dfcc2.set_index(['trials','phase'])
qs0 = 'phase == "FEEDBACK_PHASE"'
dfcc1 = dfcc1.reset_index().sort_values('trials')
inds = dfcc1.query(qs0).index
dfcc0_ = dfcc0.reset_index().sort_values('trials')
dfcc1['home_duration'] = np.nan
dfcc1['movement_duration'] = np.nan
dfcc1['trial_duration'] = np.nan

qs = 'phase == "ITI_PHASE"'
dfcc1.loc[inds,'ITI_duration']     = dfcc0_.query(qs)['phase_duration'].values
qs = 'phase == "REST_PHASE"'
dfcc1.loc[inds,'home_duration']     = dfcc0_.query(qs)['phase_duration'].values
qs = 'phase == "TARGET_PHASE"'
dfcc1.loc[inds,'movement_duration'] = dfcc0_.query(qs)['phase_duration'].values

# not counting break part even if present
dfcc1['trial_duration'] = dfcc0.query('phase != "BREAK_PHASE"').groupby(['trials'])['phase_duration'].sum()

# WRONG! I have to just sum all phase durations that are not BREAK_PHASE
#df_ = dfc.query('phase != "BREAK_PHASE"').groupby('trials')
#trial_dur_excbreak = df_['time'].max() - df_['time'].min()
#assert trial_dur_excbreak.to_frame().reset_index()['trials'].diff()[1:].min() > 0
#dfcc1.loc[inds, 'trial_duration'] = trial_dur_excbreak.values

# if I want to compute time distance between target phase starts
# then I'd take dfcc0.query('TARGET_PHASE')['time_since_last']
dfcc1 = dfcc1.sort_values('time')


logtime('dfcc1')

############## creating a "virtual trial" for break part
dfcc0_ = dfcc0.sort_values('time')
c = dfcc0_['phase'].isin( ["FEEDBACK_PHASE","BREAK_PHASE"])
dfcc0_['trial_index_inc_pause'] = c.cumsum() - 1

dfcc2 = dfcc0_[c].copy()
print( len(dfcc), len(dfcc2) )
assert np.min(dfcc2['time'].diff().iloc[1:]) > 0
assert not dfcc2.duplicated(['trial_index_inc_pause']).any()
dfcc2['trial_type'] = ''

c = dfcc2['perturbation'].abs() > 1e-10
dfcc2.loc[c,'trial_type'] = 'perturbation'
dfcc2.loc[~c,'trial_type'] = 'veridical'
dfcc2.loc[dfcc2['phase'] == "BREAK_PHASE",'trial_type'] = 'pause'

###############  add more stuff from dfcc0
#dfcc2 = dfcc2.set_index(['trials','phase'])
qs0 = 'phase == "FEEDBACK_PHASE"'
dfcc2 = dfcc2.reset_index().sort_values('trials')
inds = dfcc2.query(qs0).index
dfcc0_ = dfcc0.reset_index().sort_values('trials')
dfcc2['home_duration'] = np.nan
dfcc2['movement_duration'] = np.nan
dfcc2['trial_duration'] = np.nan

qs = 'phase == "ITI_PHASE"'
dfcc2.loc[inds,'ITI_duration']     = dfcc0_.query(qs)['phase_duration'].values
qs = 'phase == "REST_PHASE"'
dfcc2.loc[inds,'home_duration']     = dfcc0_.query(qs)['phase_duration'].values
qs = 'phase == "TARGET_PHASE"'
dfcc2.loc[inds,'movement_duration'] = dfcc0_.query(qs)['phase_duration'].values

# not counting break part even if present
# WRONG! I have to just sum all phase durations that are not BREAK_PHASE
#df_ = dfc.query('phase != "BREAK_PHASE"').groupby('trials')
#trial_dur_excbreak = df_['time'].max() - df_['time'].min()
#assert trial_dur_excbreak.to_frame().reset_index()['trials'].diff()[1:].min() > 0
#dfcc2.loc[inds, 'trial_duration'] = trial_dur_excbreak.values
dfcc2['trial_duration'] = dfcc0_.query('phase != "BREAK_PHASE"').groupby(['trials'])['phase_duration'].sum()

# if I want to compute time distance between target phase starts
# then I'd take dfcc0.query('TARGET_PHASE')['time_since_last']

qs0 = 'phase == "BREAK_PHASE"'
inds = dfcc2.query(qs0).index.values
qs = 'phase == "BREAK_PHASE"'
dfcc2.loc[inds,'trial_duration'] = dfcc0_.query(qs)['phase_duration'].values

dfcc2 = dfcc2.sort_values('time')

logtime('dfcc2')

#######################

task = 'VisuoMotor'

fname = pjoin(path_data, subject, 'behavdata',
                f'behav_{task}_df_upd{args.save_suffix}.pkl')
print(fname)
dfcc1.to_pickle(fname)


badcols =  checkErrBounds(dfcc1)
print(subject, badcols)


logtime('save dfcc1')

fname = pjoin(path_data, subject, 'behavdata',
                f'behav_{task}_df_upd_seppausetrials{args.save_suffix}.pkl')
print(fname)
dfcc2.to_pickle(fname)

logtime('save dfcc2')




import os
import multiprocessing
import numpy as np

# path_data = '/Volumes/Samsung_T1/MEGdata/MemErrors/'
# path_data = '/Volumes/data/MemErrors/data/'
# path_data = '/data/quentinra/MemErrors/data2/'
path_data     = os.path.expandvars("$DATA_MEMORY_ERRORS_STAB_AND_STOCH")
path_data_tmp = os.path.expandvars("$DATA_TMP_MEMORY_ERRORS_STAB_AND_STOCH")
path_fig      = os.path.expandvars("$FIG_MEMORY_ERRORS_STAB_AND_STOCH")
                                #FIG_MEMORY_ERRORS_STAB_AND_STOCH

subjects_predef = ['sub01_WGPOZPEE', 'sub02_CLTLNQWL', 'sub03_GPVDQMWB',
            'sub04_XNDMUSRS', 'sub05_ZGPBOAQU', 'sub06_DLLYEPVA',
            'sub07_MJWXBESS', 'sub08_TXVPROYY', 'sub09_VFDOXEVC',
            'sub10_BJJWDKEK', 'sub11_ERHGZFPL', 'sub12_ZWFBQSXR',
            'sub13_EALZKBNL', 'sub14_RPEADEJG', 'sub15_TAMMXQQS',
            'sub16_SJILLGUV', 'sub17_SUMYMRAR', 'sub18_BBPOBFOQ',
            'sub19_MVAQVMEL', 'sub20_YOGCJKKB']

# they are same
subjects = [f for f in os.listdir(path_data) if f.startswith('sub') ]
subjects = list(sorted(subjects))

if os.path.expandvars('$USER') == 'demitau':
    n_jobs = multiprocessing.cpu_count() - 2
else:
    n_jobs = multiprocessing.cpu_count()
XGB_tree_method_def = 'gpu_hist'

##########################

event_ids_tgt_stable = [20, 21, 22, 23]
event_ids_tgt_random = [25, 26, 27, 28]
event_ids_tgt = event_ids_tgt_stable + event_ids_tgt_random
#event_ids_tgt = [20, 21, 22, 23, 25, 26, 27, 28]
event_ids_feedback_stable = [30]
event_ids_feedback_random = [35]
event_ids_feedback = event_ids_feedback_stable + event_ids_feedback_random

stage2event_ids = { 'target':event_ids_tgt, 'feedback':event_ids_feedback }
stage2evn2event_ids = { 'target':{'stable': event_ids_tgt_stable,
                                      'random': event_ids_tgt_random},
                            'feedback':{'stable':event_ids_feedback_stable,
                            'random':event_ids_feedback_random} }

env2envcode = dict(stable=0, random=1)
env2subtr   = dict(stable=20, random=25)

freq_names = ['broad', 'theta', 'alpha', 'beta', 'gamma']
freqs = [(4, 60), (4, 7), (8, 12), (13, 30), (31, 60)]
freq_name2freq = dict( list(zip(freq_names,freqs) ) )

target_angs = (np.array([157.5, 112.5, 67.5, 22.5]) + 90) * \
              (np.pi/180)

# target onset time is fixed
stage2time = {'home':(-0.5,0) }
stim_channel_name = 'UPPT001'
delay_trig_photodi = 18  # to account for delay between trig. & photodi.
min_event_duration = 0.02


stage2time_bounds = { 'feedback': (-2,5), 'target':(-5,2) }

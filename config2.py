import os
import multiprocessing
import numpy as np
import socket
from os.path import join as pjoin
import argparse
try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
try:
    import StringIO
except ImportError:
    import io as StringIO



# path_data = '/Volumes/Samsung_T1/MEGdata/MemErrors/'
# path_data = '/Volumes/data/MemErrors/data/'
# path_data = '/data/quentinra/MemErrors/data2/'
path_data     = os.path.expandvars("$DATA_MEMORY_ERRORS_STAB_AND_STOCH")
path_data_tmp = os.path.expandvars("$DATA_TMP_MEMORY_ERRORS_STAB_AND_STOCH")
path_fig      = os.path.expandvars("$FIG_MEMORY_ERRORS_STAB_AND_STOCH")
path_code      = os.path.expandvars("$CODE_MEMORY_ERRORS")
                                #FIG_MEMORY_ERRORS_STAB_AND_STOCH

subjects_predef = ['sub01_WGPOZPEE', 'sub02_CLTLNQWL', 'sub03_GPVDQMWB',
            'sub04_XNDMUSRS', 'sub05_ZGPBOAQU', 'sub06_DLLYEPVA',
            'sub07_MJWXBESS', 'sub08_TXVPROYY', 'sub09_VFDOXEVC',
            'sub10_BJJWDKEK', 'sub11_ERHGZFPL', 'sub12_ZWFBQSXR',
            'sub13_EALZKBNL', 'sub14_RPEADEJG', 'sub15_TAMMXQQS',
            'sub16_SJILLGUV', 'sub17_SUMYMRAR', 'sub18_BBPOBFOQ',
            'sub19_MVAQVMEL', 'sub20_YOGCJKKB']

env2envcode = dict(stable=0, random=1)
env2subtr   = dict(stable=20, random=25)

# this is for stable1, for stable 2 it is inverse
pert_seq = {0: (0,30,0,-30,0), 1: (0,-30,0,30,0)}
block_names = ['stable1','random1','stable2','random2'] # order is important!
pert_stages = np.arange(5, dtype = int)
pertvals = [0, 30, -30]

pert_seq_code_test_trial = 40

control_types_all = ['feedback', 'movement' , 'target', 'belief']
time_lockeds_all = ['feedback', 'target']

a,b = list( zip( *list( env2envcode.items() ) ) )
envcode2env = dict( zip( b,a ) )


####################

hostname = socket.gethostname()
#if hostname.startswith('jsfc'):
#    print('Hostname = ',hostname)
#    # do nothing
if not hostname.startswith('jsfc'):
    try:
        from jupyter_helpers.notifications import Notifications
        #p = '/usr/share/sounds/gnome/default/alerts/'
        sound_file  = pjoin(path_code,'beep-06.mp3')
        sound_file2 = pjoin(path_code,'glitch-in-the-matrix-600.mp3')
        #p1 = p + 'glass.ogg'; p2 = p + 'sonar.ogg';
        p1 = sound_file; p2 = sound_file2
        Notifications(success_audio=p1, time_threshold=2,
            failure_audio=p2)  #    ,integration='GNOME')
        print('Jupyter sounds setting succeded')
    except:
        print('Jupyter sounds setting failed')

####################

# they are same
if os.path.exists(path_data):
    subjects = [f for f in os.listdir(path_data) if f.startswith('sub') ]
    subjects = list(sorted(subjects))
else:
    print(f'data dir {path_data} does not exist, setting default subjects')
    subjects = subjects_predef

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
stage2evn2event_ids = { 'target':
                            {'stable': event_ids_tgt_stable,
                            'random': event_ids_tgt_random,
                            'all':event_ids_tgt },
                        'feedback':
                            {'stable':event_ids_feedback_stable,
                            'random':event_ids_feedback_random,
                            'all':event_ids_feedback} }
stage2evn2event_ids_str = {}
for tl,vs in stage2evn2event_ids.items():
    stage2evn2event_ids_str[tl] = {}
    for env,ids in vs.items():
        stage2evn2event_ids_str[tl][env] = list(map(str,ids))

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
n_trials_in_block = 192

#DEBUG_ntrials       = 20
#DEBUG_nchannels     = 3
DEBUG_ntrials       = 40
DEBUG_nchannels     = 3

#stage2time_bounds = { 'feedback': (-2,5), 'target':(-5,2) }
stage2time_bounds = { 'feedback': (-2,3), 'target':(-2,1.5) }

analysis_name2var_ord = {'movement_errors_next_errors_belief':['movement', 'errors', 'next_errors', 'belief'],
'prevmovement_preverrors_errors_prevbelief':['prevmovement', 'preverrors', 'errors', 'prevbelief'] }


def genFnSliding(results_folder, env,regression_type,
                 time_locked,suffix,freq_name,tmin_cur,tmax_cur,
                 trial_group_col_calc):
    #fn_suffix = (f'{env}_{regression_type}_{time_locked}_'
    #    f'{suffix}_{freq_name}_t={tmin_cur:.2f},{tmax_cur:.2f}')
    fn_suffix = (f'{env}_{trial_group_col_calc}_{regression_type}_{time_locked}_'
        f'{suffix}_{freq_name}_t={tmin_cur:.2f},{tmax_cur:.2f}')
    fn = f'{fn_suffix}.npz'
    fname_full = pjoin(results_folder, fn)
    return fname_full

def parline2par(line):
    tuples = []
    exprs = line.split('; ')
    for expr in exprs:
        if expr.find('=') >= 0:
            lhs,rhs = expr.split('=')
            tuples += [(lhs,rhs)]
    par = dict(tuples)
    return par

def paramFileRead(fname,recursive=True):
    print('--Log: reading paramFile {0}'.format(fname) )

    assert fname is not None, 'recieved None as param fname!'
    param_fname_full = pjoin(path_code, 'params', fname)
    assert os.path.exists( param_fname_full ), param_fname_full
    file = open(param_fname_full, 'r')
    ini_str = '[root]\n' + file.read()
    file.close()
    ini_fp = StringIO.StringIO(ini_str)
    preparams = ConfigParser.RawConfigParser(allow_no_value=True)
    preparams.optionxform = str
    preparams.read_file(ini_fp)
    #preparams.readfp(ini_fp)
    #sect = paramsEnv_pre.sections()
    items= preparams.items('root')
    params = dict(items)

    if(recursive):
        addParamKeys = sorted( [ k for k in params.keys() if 'iniAdd' in k ] )
        lenAddParamKeys = len(addParamKeys)
        if(lenAddParamKeys ):
            print('---Log: found {0} iniAdd\'s, reading them'.format(lenAddParamKeys) )
        for pkey in addParamKeys:
            paramFileName = paramFileRead(params[pkey])
            params.update(paramFileName)

        # we actually want to overwrite some of the params from the added inis
        if(lenAddParamKeys):
            paramsAgain = paramFileRead(fname,recursive=False)
            params.update(paramsAgain)

    return params


class CustomAction(argparse.Action):
    def __init__(self, check_func, *args, **kwargs):
        """
        argparse custom action.
        :param check_func: callable to do the real check.
        """
        self._check_func = check_func
        super(CustomAction, self).__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string):
        if isinstance(values, list):
            values = [self._check_func(parser, v) for v in values]
        else:
            values = self._check_func(parser, values)
        setattr(namespace, self.dest, values)




def genArgParser():
    parser = argparse.ArgumentParser()
    from config2 import n_jobs as n_jobs_def

    # choices
    parser.add_argument('--each_SPoC_fit_is_parallel', default=1 )
    parser.add_argument('--n_jobs',  default = n_jobs_def, type=int )
    parser.add_argument('--subject', required = False)
    #parser.add_argument('--runpar_line_ind', default=None, type=int)
    parser.add_argument('--runpar_line_ind', type=int)
    parser.add_argument('--param_file')
    parser.add_argument('--env_to_run')
    parser.add_argument('--regression_type')
    parser.add_argument('--freq_name', required= False)
    parser.add_argument('--freq_limits')  # narg=*
    parser.add_argument('--hpass') # not float, str!
    parser.add_argument('--output_folder')
    parser.add_argument('--task',default = 'VisuoMotor')

    parser.add_argument('--SLURM_job_id')

    parser.add_argument('--ICAstr', default='with_ICA')
    parser.add_argument('--time_locked', default='target')
    parser.add_argument('--control_type', default='movement')
    parser.add_argument('--time_bounds_slide_target',   type=str )
    parser.add_argument('--time_bounds_slide_feedback', type=str  )
    parser.add_argument('--tmin')
    parser.add_argument('--tmax')
    parser.add_argument('--slide_windows_type', type=str)
    parser.add_argument('--slide_window_dur', type=float)
    parser.add_argument('--slide_window_shift', type=float)
    parser.add_argument('--debug',default = 0, type=int)

    def parse(parser, s):
        vs = s.split(',')
        for vi,v in enumerate(vs):
            if v in ['all', 'None']:
                vs[vi] = None
        return vs
    def parsei(parser, s):
        vs = s.split(',')
        for vi,v in enumerate(vs):
            if v in ['all', 'None']:
                vs[vi] = None
            else:
                vs[vi] = int(v)
        return vs
    parser.add_argument('--dists_trial_from_prevtgt',default = [None],
                        action=CustomAction, check_func=parsei)
    parser.add_argument('--dists_rad_from_prevtgt',  default = [None],
                        action=CustomAction, check_func=parse)
    parser.add_argument('--target_inds_to_use',default = [None],
                        action=CustomAction, check_func=parsei)
    parser.add_argument('--groupcols',default = ['environment'], action=CustomAction,
                        check_func=parse)
    parser.add_argument('--pertvals',default = [None], action=CustomAction,
                        check_func=parse)


    parser.add_argument('--do_classic_dec', default = 1, type = int)
    parser.add_argument('--do_partial_dec', default = 1, type = int)
    parser.add_argument('--est_parallel_across_dims', default = 1, type = int)
    parser.add_argument('--est_parallel_within_dim', default = 0, type = int)
    parser.add_argument('--b2b_each_fit_is_parallel', default = 0, type = int)
    parser.add_argument('--classic_dec_verbose', default = 3)

    parser.add_argument('--B2B_SPoC_parallel_type', default = 'across_splits')

    parser.add_argument('--nb_fold', default = 6, type = int)
    parser.add_argument('--decim_epochs', default = 2, type = int)
    parser.add_argument('--n_splits_B2B', default = 30, type = int)
    parser.add_argument('--SPoC_n_components', default = 5, type = int)
    parser.add_argument('--safety_time_bound', default=0)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--trim_outliers', default=0, type=int)
    parser.add_argument('--crop')
    parser.add_argument('--custom_suffix')

    parser.add_argument('--use_preloaded_raw', default = 0, type = int)
    parser.add_argument('--use_preloaded_flt_raw', default = 0, type = int)
    parser.add_argument('--mne_fit_log_level', default = 'warning')

    parser.add_argument('--load_epochs',  default = 0, type = int)
    parser.add_argument('--save_epochs',  default = 0, type = int)
    parser.add_argument('--load_flt_raw', default = 1, type = int)
    parser.add_argument('--save_flt_raw', default = 1, type = int)

    parser.add_argument('--error_type', default = 'MPE', type = str)
    parser.add_argument('--trial_group_col_calc',
                        default = 'trialwe', type = str)
    parser.add_argument('--block_names_to_use',
                        default = 'all', type = str)

    parser.add_argument('--exit_after', default = 'end')

    parser.add_argument('--nskip_trial', default = 1, type=int)

    #parser.add_argument('--decode_merge_pert', default = 1, type = int)
    #parser.add_argument('--decode_per_pert', default = 1 , type = int)

    parser.add_argument('--save_result', default = 1 , type = int)

    return parser

def cleanEvents(events):
    import warnings
    t = -1
    bad_trials = list()
    bad_events = list()
    for ii, event in enumerate(events):
        if event[2] in event_ids_tgt_stable:
            t += 1
            if events[ii+1, 2] == 100:
                if events[ii+2, 2] != 30:
                    bad_trials.append(t)
                    warnings.warn('Bad sequence of triggers')
                    # Delete bad events until the next beginning of a trial (10)
                    bad_events.append(ii - 1)
                    for iii in range(5):
                        if events[ii + iii, 2] == 10:
                            break
                        else:
                            bad_events.append(ii+iii)
            elif events[ii+1, 2] != 30:
                bad_trials.append(t)
                warnings.warn('Bad sequence of triggers')
                # Delete bad events until the next beginning of a trial (10)
                bad_events.append(ii - 1)
                for iii in range(5):
                    if events[ii + iii, 2] == 10:
                        break
                    else:
                        bad_events.append(ii+iii)
    print('cleanEvents: deleted {len(bad_events)} events')
    events_cleaned = np.delete(events, bad_events, 0)
    return events_cleaned

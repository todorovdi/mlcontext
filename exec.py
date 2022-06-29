#%load_ext autoreload
#%autoreload 2

import os
ipy = get_ipython()
#data_dir_general = '/home/demitau/data_Quentin'
#data_subdir_mem_err_main = 'full_experiments/data2'
from os.path import join as pjoin
#data_dir_input = pjoin(data_dir_general,data_subdir_mem_err_main)
from config2 import n_jobs


#data_dir_general = '/home/demitau/data_Quentin'
#data_subdir_mem_err_main = 'full_experiments/data2'
#data_dir_input = pjoin(data_dir_general,data_subdir_mem_err_main)
#scripts_dir = pjoin(data_dir_general,'full_experiments','scripts2')

data_dir_general = os.path.expandvars('$DATA_QUENTIN')
data_dir_input = os.path.expandvars('$DATA_MEMORY_ERRORS_STAB_AND_STOCH')
#scripts_dir = pjoin( os.path.expandvars('$CODE_MEMORY_ERRORS'), 'previous_analyses')
scripts_dir = pjoin( os.path.expandvars('$CODE_MEMORY_ERRORS'))
scripts_dir_recent = scripts_dir


print(data_dir_input, scripts_dir)

subjects = [f for f in os.listdir(data_dir_input) if f.startswith('sub') ]
subjects = list(sorted(subjects))
print(subjects)

# faster test
#%debug
#runscr_name = sys.argv[1]

#subject = subjects[0]
use_preloaded_raw = False
#hpass = no
script_name = pjoin(scripts_dir_recent,
                    'spoc_home_with_prev_error2.py')

for subject in subjects:
    for hpass in ['no_filter', '0.1', '0.05']:
        for regression_type in  ['Ridge', 'xgboost']:
            ipy.run_line_magic('run', f'-i {script_name}')

            import gc; gc.collect()

##############################################

use_preloaded_raw = False
script_name = pjoin(scripts_dir_recent,
                    'td_long2.py')

for subject in subjects:
    for hpass in ['no_filter', '0.1', '0.05']:
        for regression_type in  ['Ridge']:
            ipy.run_line_magic('run', f'-i {script_name}')

            import gc; gc.collect()

##############################################

#subject = subjects[0]
use_preloaded_raw = False
#hpass = no
script_name = pjoin(scripts_dir_recent,
                    'correlate_spoc_decoding_and_error_sensitivity2.py')

for subject in subjects:
    for hpass in ['no_filter', '0.1', '0.05']:
        for regression_type in  ['Ridge', 'xgboost']:
            ipy.run_line_magic('run', f'-i {script_name}')

            import gc; gc.collect()

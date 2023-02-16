import os
ipy = get_ipython()
from os.path import join as pjoin
from config2 import n_jobs
from config2 import subjects
from config2 import freq_name2freq
import sys

#data_dir_general = os.path.expandvars('$DATA_QUENTIN')
#data_dir_input = os.path.expandvars('$DATA_MEMORY_ERRORS_STAB_AND_STOCH')
scripts_dir = pjoin( os.path.expandvars('$CODE_MEMORY_ERRORS'))

scripts_to_run = ['spoc_home']

#subject = subjects[0]
use_preloaded_raw = False
#hpass = no
script_name = pjoin(scripts_dir,
                    'read_behav2.py')

subj_ind = int(sys.argv[1])
print(subj_ind)
#for subject in subjects[::-1]:
#for subject in subjects:
subject=subjects[subj_ind]
ipy.run_line_magic('run', f'-i {script_name}')

#%load_ext autoreload
#%autoreload 2

import os
#ipy = get_ipython()
#data_subdir_mem_err_main = 'full_experiments/data2'
from os.path import join as pjoin
import sys
from joblib.externals.loky.process_executor import TerminatedWorkerError

data_dir_input = os.path.expandvars('$DATA_MEMORY_ERRORS_STAB_AND_STOCH')
#scripts_dir = pjoin( os.path.expandvars('$CODE_MEMORY_ERRORS'), 'previous_analyses')
scripts_dir = pjoin( os.path.expandvars('$CODE_MEMORY_ERRORS'))

print(data_dir_input, scripts_dir)


with open(pjoin(scripts_dir,'__runpars.txt'), 'r' ) as f:
  lines = f.readlines()

line_ind = int(sys.argv[1])
line = lines[ line_ind ]
from config2 import parline2par
par = parline2par(line)
print(par)
#sys.exit(1)

print(f'final RSID is {line_ind}')
script_name = pjoin(scripts_dir, par['script'])
sn_full = pjoin(scripts_dir,script_name)
print('exists = ',os.path.exists(sn_full))
#try:
#  exec(open(sn_full,'r').read(), globals() )


import subprocess as sp
try:
  ec = sp.call(f'python {script_name} --runpar_line_ind {line_ind}')
  print(f'Exit code = {ec}')
#eval(f'import {script_name}')
#  ec = ipy.run_line_magic('run', f'-i {script_name}')
#  import gc; gc.collect()
#except TerminatedWorkerError as e:
#  print(e)
#  sys.exit(1)
except:
  #print(e)
  #raise e
  #return 1
  sys.exit(1)
finally:
  print('',flush=1)

if ec != 0:
  sys.exit(ec)

# I cannot run import because I cannot define 'par' outside the scope
# I cannot use ipython because of library path errors
# I cannot use exec because it will be difficult to debug
# I cannot use subprocess because this way it is difficult to pass cmd line argumnets

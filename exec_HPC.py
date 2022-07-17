#%load_ext autoreload
#%autoreload 2

import os
ipy = get_ipython()
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

tuples = []
line = lines[ line_ind ]
exprs = line.split('; ')
for expr in exprs:
  if expr.find('=') >= 0:
    lhs,rhs = expr.split('=')
    tuples += [(lhs,rhs)]
par = dict(tuples)
print(par)
#sys.exit(1)



script_name = pjoin(scripts_dir, par['script'])
try:
  ipy.run_line_magic('run', f'-i {script_name}')
  import gc; gc.collect()
except TerminatedWorkerError as e:
  print(e)
  sys.exit(1)

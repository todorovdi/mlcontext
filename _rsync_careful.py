#!/bin/python3
import subprocess as sp
import sys,os
import getopt
import datetime
import numpy as np
from dateutil import parser
from pathlib import Path

# when called from command line I have to supply "*.py" in quotes

#print(sys.argv)

if len(sys.argv) < 3:
    print('Wrong number of arguments! Existing ', sys.argv)
    sys.exit(1)
d1 = sys.argv[-2]
d2 = sys.argv[-1]
excludes = []
dry = False
for arg in sys.argv[1:-2]:
    if arg.startswith('--exclude='):
        excludes += [arg]
    elif arg == '--mode:dry':
        dry = True
    elif arg == '--mode:normal':
        dry = False
excludes_str = ' '.join(excludes)

######################
DEBUG = 0

p = d1  # p is parent dir path - like (except when is not pattern)
starind = d1.find('*')
bracind = d1.find('{')
is_pattern = starind >= 0 or bracind >= 0
if d1[-1] == '/' or d1 == '.':
    p = d1
else:
    if not is_pattern:
        p = d1
    else:
        p = str(Path(d1).parent) + '/'

home = os.path.expandvars('$HOME')
lh = len(home) + 1
print(f'Source = {d1[lh:]} (p={p[lh:]}), dest = {d2[lh:]}')

######################

include_str = ''
#if is_pattern:
#    pattern = Path(d1).name
#    include_str = f'--include={pattern}'
#    print('ptattern = ',pattern)

if DEBUG:
    print(f'dry = {dry}, {include_str}, {excludes_str}')

#first_arg = p
first_arg = d1
if dry:
    s2 = f'rsync -rtvhn --itemize-changes {include_str} {excludes_str} --exclude={__file__} {first_arg} {d2}'
    out = sp.getoutput(s2)
    print(out)
    sys.exit(0)

# from d1 to d2
s = f'rsync -rtvhn --itemize-changes {include_str} {excludes_str} --exclude={__file__} {first_arg} {d2}'
if DEBUG:
    print(s)
out = sp.getoutput(s)
out = out.split('\n')

if DEBUG:
    print(out)
#################
file_contents_changed_dest = []
file_contents_changed_src = []
file_added = []
time_changed = []
changes = out[1:-3]
if DEBUG:
    print(changes)
#################
for change in changes:
    if change.find("No such file or directory") >= 0:
        continue

    spaceloc = change.find(' ')
    code = change[:spaceloc]
    fn = change[spaceloc+1:]

    if DEBUG:
        print(f'change,code = {change,code}')
    if 's' in code:

        #print(change)
        if is_pattern:
            if not os.path.exists( os.path.join(p,fn) ) :
                print(f'!!! does not exist:  {os.path.join(p,fn) }')
                print( change, 'p=', p, 'fn=',fn)
                sys.exit(1)
            tst1 = os.stat( os.path.join(p,fn) ).st_mtime
        else:
            assert '/' in p
            tst1 = os.stat( p ).st_mtime
        tst2 = os.stat( os.path.join(d2,fn) ).st_mtime

        if tst2 > tst1:
            print(f'WARNING: {fn} content has been changed in dest')
            file_contents_changed_dest += [fn]
        else:
            print(f'INFO: {fn} content has been changed in source')
            file_contents_changed_src += [fn]

        #os.stat(d1)
    elif '++' in code:
        file_added += [fn]
    else:
        if 't' in code or 'T' in code:
            time_changed += [fn]


print(f'Num changed contents dest {len(file_contents_changed_dest)}, '
    f'contents src {len(file_contents_changed_src)}, '
    f'time changed {len(time_changed)}')

if len(time_changed):
    print(time_changed)

sync_dest_change_fn = 'sync_dest_changes.log'

if len(file_contents_changed_dest) > 0:
    print('Since we have nonzero changed_dest, we exit')
    print(file_contents_changed_dest)
    with open(sync_dest_change_fn, "a") as sdcf:
        sdcf.writelines(["\n  file_contents_changed_dest = \n"] +\
                        [ os.path.join(p,fn) + '\n' for fn in file_contents_changed_dest] )

    sys.exit(1)
elif (len(time_changed) + len(file_contents_changed_src) + len(file_added) ) > 0:
    s2 = f'rsync -rtvh --itemize-changes {include_str} {excludes_str} --exclude={__file__} {first_arg} {d2}'
    out = sp.getoutput(s2)
    print(out)
print('-----------------')


#>f+++++++++ exec_HPC.py
#>f+++++++++ exec_HPC_old.py
#>f+++++++++ gen_runpars_HPC.py
#>f..t...... spoc_home_with_prev_error2.py
#
#>f+++++++++ exec_HPC.py
#>f+++++++++ exec_HPC_old.py
#>f+++++++++ gen_runpars_HPC.py
#>f.sT...... spoc_home_with_prev_error2.py


# '.' means no change
# + means adding
# o      A s means the size of a regular file is different and will be updated by the file transfer.
#
# o      A  t  means  the  modification  time  is  different and is being updated to the sender’s value (requires
#         --times).  An alternate value of T means that the modification time will be set to  the  transfer  time,
#         which  happens  when  a file/symlink/device is updated without --times and when a symlink is changed and
#         the receiver can’t set its time.  (Note: when using an rsync 3.0.0 client, you might see the s flag com‐
#         bined with t instead of the proper T flag for this time-setting failure.)
#

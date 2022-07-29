import re
import os.path as op
import os
from config2 import path_data, freq_name2freq
import numpy as np
import pandas as pd

def collectResults(subject,folder, freq_name='broad',
                   regression_type='Ridge', keys_to_extract=['par']):
    freqstr = '|'.join( freq_name2freq.keys() )

    dir_full = op.join(path_data, subject, 'results', folder)
    fns = os.listdir ( dir_full )
    tuples = []
    regex = re.compile(r'(stable|random)_(Ridge|xgboost)_(feedback|target)_(.*)_(' +freqstr+ ')_t=(.*),(.*)\.npz')
    for fn in fns:
        r = re.match( regex, fn)
        #print(r )
        if r is None:
            print(f'wrong fn = {fn}')
        grps = r.groups(); #print(grps)
        env_cur,rt_cur,time_locked_cur,analysis_name_cur,\
            freq_name_cur,tmin,tmax = grps

        if regression_type is not None:
            if regression_type != rt_cur:
                continue
        if freq_name is not None:
            if freq_name != freq_name_cur:
                continue

        fn_full = op.join(dir_full,fn)
        f = np.load( fn_full, allow_pickle=1)
        #print( list(f.keys() ), fn_full)
        kvs = []
        for kte in keys_to_extract:
            kv = f[kte][()]
            kvs  += [kv]
        kvs = tuple(kvs)

        tuples += [ (subject, os.stat(fn_full).st_mtime, fn, fn_full,
                    *kvs, *grps )  ]
        #par = f['par'][()]
        #tuples += [ (os.stat(fn_full).st_mtime, fn, fn_full,
        #             par, *grps )  ]
        del f

    cols = ['subject','mtime', 'fn', 'fn_full'] + keys_to_extract + ['env',
                 'rt','time_locked','analysis_name',
                 'freq_name','tmin','tmax' ]
    mtime_ind = cols.index('mtime')
    srt = sorted(tuples, key=lambda x: x[mtime_ind])

    #par = pars[-1]
#    if par['slide_windows_type'] == 'auto':
#        start, end = stage2time_bounds[time_locked]
#        start, end = eval( par.get(f'time_bounds_slide_{time_locked}', (start,end) ) )
#        shift = par.get('slide_window_shift',None)
#        dur = par.get('slide_window_dur', None)
#        tmins = np.arange(start,end,shift)
#        tmaxs = dur + tmins
#
#        tminmax = zip(tmins,tmaxs)

        #tuples_a.shape
    df_collect = pd.DataFrame(tuples,
        columns=cols )

    return df_collect

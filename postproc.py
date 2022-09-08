import re
import os.path as op
import os
from config2 import path_data, freq_name2freq
import numpy as np
import pandas as pd

def collectResults(subject,folder, freq_name='broad',
                   regression_type='Ridge', keys_to_extract=['par'],
                   parent_key = None, time_start=None, time_end = None,
                  fns = None, df=None ):
    # loads from every file in the directory
    if fns is None and df is None:
        freqstr = '|'.join( freq_name2freq.keys() )
        dir_full = op.join(path_data, subject, 'results', folder)
        fns = os.listdir ( dir_full )
        regex = re.compile(r'(all|stable|random)_(Ridge|xgboost)_(feedback|target)_(.*)_('+\
                        freqstr+ ')_t=(.*),(.*)\.npz')
    if df is None:
        print(f'Found {len(fns)} files in total in {dir_full}')
    else:
        print(f'Df len = {len(df)}')
    #tuples = []

    def addRowInfo(f,row, index):
        assert (row is None) or (index is None)
        if parent_key is not None:
            subf = f[parent_key][()]
        else:
            subf = f

        kvs = []
        for kte in keys_to_extract:
            kv0 = subf[kte]
            if parent_key is None:
                kv0 = kv0[()]
                #kvs  += [kv]
                if row is not None:
                    row [kte] = kv0
                else:
                    df.at[index,kte] = kv0
                #key = kte
                #print(kte,len(kv ) )
            else:
                #e.g. f['vars']['varname']['diff']
                for kvn,kvv in kv0.items():
                    #kvs += [ f'{kte}_{kvn}' ]
                    key = f'{kte}_{kvn}'
                    if row is not None:
                        row [key] = kvv
                    #print(key,kvv )
                    else:
                        df.at[ index,key] = kvv

                    #print(df.loc[index,'fn'],kvn, len(kvv) )

    if df is None:
        rows = []
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


            cols = ['subject','mtime', 'fn', 'fn_full', 'env',
                        'rt','time_locked','analysis_name',
                        'freq_name','tmin','tmax' ]
            colvals = [ subject, os.stat(fn_full).st_mtime, fn, fn_full, *grps ]
            row = dict( zip(cols,colvals) )

            from datetime import datetime
            dt = datetime.fromtimestamp(row['mtime'])
            if time_start is not None and dt < time_start:
                continue
            if time_end is not None and dt > time_end:
                continue

            addRowInfo(f,row=row,index=None)

            rows += [row]

            #kvs = tuple(kvs)

            #tuples += [ (subject, os.stat(fn_full).st_mtime, fn, fn_full,
            #            *kvs, *grps )  ]

            #par = f['par'][()]
            #tuples += [ (os.stat(fn_full).st_mtime, fn, fn_full,
            #             par, *grps )  ]
            del f
        print(f'{len(rows)} after filtering by time')
        if len(rows) == 0:
            return None

        df_collect = pd.DataFrame(rows )
        #assert key in list( df_collect.columns )

        df_collect.sort_values(by='mtime', axis=0, inplace=True)
    else:
        # create new columns
        fn_full = df.iloc[0]['fn_full']
        f = np.load( fn_full, allow_pickle=1)
        if parent_key is not None:
            subf = f[parent_key][()]
        else:
            subf = f

        for kte in keys_to_extract:
            kv0 = subf[kte]
            for kvn,kvv in kv0.items():
                key = f'{kte}_{kvn}'
                df[key] = None
        del f

        # fill columns per raw
        for index,row in df.iterrows():
            fn_full = row['fn_full']
            f = np.load( fn_full, allow_pickle=1)
            addRowInfo(f,row=None,index=index)
            del f

        df_collect = df


    #cols = ['subject','mtime', 'fn', 'fn_full'] + keys_to_extract + ['env',
    #             'rt','time_locked','analysis_name',
    #             'freq_name','tmin','tmax' ]
    #mtime_ind = cols.index('mtime')
    #srt = sorted(tuples, key=lambda x: x[mtime_ind])

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
    #df_collect = pd.DataFrame(tuples,
    #    columns=cols )


    return df_collect

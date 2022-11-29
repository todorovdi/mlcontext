import re
import os.path as op
import os
from config2 import path_data, freq_name2freq
import numpy as np
import pandas as pd
import sys

def collectResults(subject,folder, freq_name='broad',
                   regression_type='Ridge', keys_to_extract=['par'],
                   time_locked=['feedback','target'], elem_var=None,
                   env = ['stable','random','all'],
                   parent_key = None, time_start=None, time_end = None,
                  fns = None, df=None, inc_tgc=1  ):

    from pathlib import Path
    from datetime import datetime
    import glob
    from behav_proc import trial_group_cols_all
    # loads from every file in the directory
    # elem_var if a grouping was used (so svd[parent_key][ df[elem_var] val ]
    if isinstance(time_locked, str):
        time_locked = [time_locked]
    if isinstance(env, str):
        env = [env]
    if fns is None and df is None:

        freqstr = '|'.join( freq_name2freq.keys() )
        dir_full = op.join(path_data, subject, 'results', folder)
        #fns = os.listdir ( dir_full )
        fns = glob.glob(dir_full + '/*.npz')

        time_locked_all = ['feedback','target']
        tls = '|'.join(time_locked_all)
        env_all = ['stable','random','all']
        envstr = '|'.join(env_all)
        if inc_tgc:
            tgcc_str = '(' + '|'.join(trial_group_cols_all) + ')_'
        else:
            tgcc_str = ''
        regex = re.compile(f'({envstr})_{tgcc_str}' + r'(Ridge|xgboost)_(' +
                           tls + r')_(.*)_('+\
                        freqstr+ r')_t=(.*),(.*)\.npz')
    if df is None:
        print(f'Found {len(fns)} files in total in {dir_full}')
    else:
        print(f'Df len = {len(df)}')
    #tuples = []

    def addRowInfo(f,row, index, elem_var_val=None, df_to_set=None):
        assert (row is None) or (index is None)
        if parent_key is not None:
            subf = f[parent_key][()]
        else:
            subf = f

        kvs = []
        if elem_var is None:
            for kte in keys_to_extract:
                kv0 = subf[kte]
                if parent_key is None:
                    kv0 = kv0[()]
                    #kvs  += [kv]
                    if row is not None:
                        row [kte] = kv0
                    else:
                        df_to_set.at[index,kte] = kv0
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
                            df_to_set.at[ index,key] = kvv
        else:
            for (kte,elem_var_val_cur),kv0 in subf.items():
                #if elem_var_val == 0.:
                #    print(elem_var_val_cur, elem_var_val)
                if (kte not in keys_to_extract) or (elem_var_val_cur != elem_var_val):
                    #if elem_var_val == 0.:
                    #    print('    skip',kte,elem_var_val_cur)
                    continue
                for kvn,kvv in kv0.items():
                    #kvs += [ f'{kte}_{kvn}' ]
                    key = f'{kte}_{kvn}'
                    if row is not None:
                        row [key] = kvv
                        #row[elem_var] = elem_var_val  # it was already set outside
                    #print(key,kvv )
                    else:
                        df_to_set.at[ index,key  ] = kvv
                        #if elem_var_val == 0.:
                        #    print(index,key, kvv)

                    #if elem_var_val == 0.:
                    #    print(index,key, df_to_set.at[index,elem_var], len(df_to_set.at[ index,key  ] ) )

                    #print(df.loc[index,'fn'],kvn, len(kvv) )

    if df is None:
        rows = []
        for fn_full in fns:
            dt = datetime.fromtimestamp(os.stat(fn_full).st_mtime)
            if time_start is not None and dt < time_start:
                continue
            if time_end is not None and dt > time_end:
                continue

            fn = Path(fn_full).name
            r = re.match( regex, fn)
            #print(r )
            if r is None or r.groups() is None:
                print(f'wrong fn = {fn}, for regex = {regex}')
                raise ValueError('aa')
            grps = r.groups(); #print(grps)
            if inc_tgc:
                env_cur,trial_group_col_calc_cur, rt_cur,time_locked_cur,analysis_name_cur,\
                    freq_name_cur,tmin,tmax = grps
            else:
                env_cur, rt_cur,time_locked_cur,analysis_name_cur,\
                    freq_name_cur,tmin,tmax = grps

            if env_cur not in env:
                continue
            if time_locked_cur not in time_locked:
                continue

            if regression_type is not None:
                if regression_type != rt_cur:
                    continue
            if freq_name is not None:
                if freq_name != freq_name_cur:
                    continue

            #fn_full = op.join(dir_full,fn)
            f = np.load( fn_full, allow_pickle=1)
            #print( list(f.keys() ), fn_full)


            cols = ['subject','mtime', 'fn', 'fn_full', 'env',
                        'trial_group_col_calc',
                        'rt','time_locked','custom_suffix',
                        'freq_name','tmin','tmax',  ]

            colvals = [ subject, dt, fn, fn_full, *grps ]
            row = dict( zip(cols,colvals) )


            #print(fn,dt)

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

        #if elem_var is None:
        #    for kte in keys_to_extract:
        #        kv0 = subf[kte]
        #        for kvn,kvv in kv0.items():
        #            key = f'{kte}_{kvn}'
        #            df[key] = None
        #else:
        #    elem_var_vals_all = []
        #    for (kte,elem_var_val),kv0 in subf.items():
        #        if kte not in keys_to_extract:
        #            continue
        #        elem_var_vals_all += [elem_var_val]

        #        for kvn,kvv in kv0.items():
        #            key = f'{kte}_{kvn}'
        #            df[key] = None
        #    elem_var_vals_all = list(set(elem_var_vals_all))


        #del f

        if keys_to_extract is None:
            keys_to_extract = subf.keys()
        # fill columns per raw
        if elem_var is None:
            # we need to add None keys because we want to have them in rows that we
            # will iter through later
            for kte in keys_to_extract:
                kv0 = subf[kte]
                for kvn,kvv in kv0.items():
                    key = f'{kte}_{kvn}'
                    df[key] = None

            for index,row in df.iterrows():
                fn_full = row['fn_full']
                f = np.load( fn_full, allow_pickle=1)
                addRowInfo(f,row=None,index=index, df_to_set=df)
                del f
            df_collect = df
        else:
            elem_var_vals_all = []
            for (kte,elem_var_val),kv0 in subf.items():
                if kte not in keys_to_extract:
                    continue
                elem_var_vals_all += [elem_var_val]
            elem_var_vals_all = list(set(elem_var_vals_all))

            #####################

            dfs = []
            for elem_var_val in elem_var_vals_all:
                dfc = df.copy()
                dfc[elem_var] = elem_var_val

                for (kte,elem_var_val2),kv0 in subf.items():
                    if kte not in keys_to_extract:
                        continue
                    for kvn,kvv in kv0.items():
                        key = f'{kte}_{kvn}'
                        dfc[key] = None

                #if elem_var_val == 0.:
                #    print(':0 len null', len( dfc[ dfc['err_sens_vals'].isnull() ] ) )
                for index,row in dfc.iterrows():
                    fn_full = row['fn_full']
                    f = np.load( fn_full, allow_pickle=1)
                    #    display(elem_var_val, f['decoding_per_var_and_pert'][()][('err_sens', 0)]['scores'])
                    addRowInfo(f,row=None,index=index,elem_var_val = elem_var_val, df_to_set=dfc)
                    del f
                #if elem_var_val == 0.:
                #    print(':1 len null', len( dfc[ dfc['err_sens_vals'].isnull() ] ) )
                dfs += [dfc]

                #if elem_var_val == 0.:
                #    display(dfc)
                #    sys.exit(0)
                #display(dfc)
            #sys.exit(0)
            df_collect = pd.concat( dfs, ignore_index=1 )




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

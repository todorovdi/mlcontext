import re
import os.path as op
import os
from config2 import path_data, freq_name2freq
import numpy as np
import pandas as pd
import sys

# needed for a dirty hack to save sequences in dataframe cells
class ListWrap:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return 'ListWrap(' + repr(self.data) + ')'

    def __len__(self):
        return len(self.data)

    def isbad(self):
        a = isinstance(self.data, float) 
        b = False
        if a:
            b = np.isna(self.data)
        return a and b

def collectResults0(subjects, folder, regex = None, wildcard='*.npz'):
    # just collect filenames and mtime
    from datetime import datetime
    from pathlib import Path
    import glob
    fns_subj_all = []
    for subject in subjects:
        dir_full = op.join(path_data, subject, 'results', folder)
        fns = glob.glob(dir_full + '/' + wildcard)  # gives full path
        fns_subj_all +=  list( zip([subject] * len(fns) ,fns)  )

    rows = []
    for subject, fnf in fns_subj_all:
        d = {}
        d['fn_full'] = fnf
        fn = Path(fnf).name
        dt = datetime.fromtimestamp(os.stat(fnf).st_mtime)
        d['fn'] = fn
        d['fsz'] = os.path.getsize(fnf)
        d['mtime'] = dt
        d['subject'] = subject
        rows += [ d ]

    #return pd.DataFrame(rows)
    #rows = []
    #for rowi,row in df0.iterrows():
    rows2 = []
    for row in rows:
        pr = parseFn(row, regex=regex)
        rows2 += [ pr ]
    df = pd.DataFrame(rows2)

    return df

def parseFn(row, regex=None, inc_tgc = 1):
    fn = row['fn']
    from behav_proc import trial_group_cols_all

    if regex is None:
        freqstr = '|'.join( freq_name2freq.keys() )
        time_locked_all = ['feedback','target']
        tls = '|'.join(time_locked_all)
        ctrls = '|'.join(['feedback', 'movement'] )
        env_all = ['stable','random','all']
        envstr = '|'.join(['random','stable'] )
        if inc_tgc:
            tgcc_str = '(' + '|'.join(trial_group_cols_all + ['trials']) + ')_'
        else:
            tgcc_str = ''
        regex_str = (f'({envstr})_{tgcc_str}'
                    r'(Ridge|Ridge_noCV|xgboost)_(' + tls + r')_(' + ctrls +  r')_(.*)_('+\
                    freqstr+ r')_t=(.*),(.*)\.npz')
        #print('regex_str = ',regex_str)
        regex = re.compile(regex_str)

    r = re.match( regex, fn)
    #print(r )
    if r is None or r.groups() is None:
        print(f'parseFn: wrong fn = {fn}, for regex = {regex_str}')
        raise ValueError('Did not match')
    grps = r.groups(); #print(grps)
    #print(grps ) 
    env_cur,trial_group_col_calc_cur, rt_cur,time_locked_cur,control_type0, analysis_name_cur,\
            freq_name_cur,tmin_cur,tmax_cur = grps
    cols = ['env','trial_group_col_calc','rt','time_locked','control_type','custom_suffix',
                        'freq_name','tmin','tmax'  ]

    d = dict(zip(cols,   grps ) )
    row = dict(row)
    row.update(d)
    return row

# slow-ish, loads the file
def extractPar(row, parnames = ['SLURM_job_id']):
    if 'par' in row:
        par = row['par']
    else:
        fn_full = row['fn_full']
        print('Loading ',fn_full)
        f = np.load( fn_full, allow_pickle=1)
        #control_type_ = f['par'][()]['control_type']
        #if (control_type is not None) and \
        #        (control_type_ != control_type):
        #    if verbose > 0:
        #        print(f'control_type_: Reject {fn}, {dt}')
        #    continue
        par = f['par'][()]

    d = {}
    for pn in parnames:
        pv = par[pn]
        d[pn] = pv
    return d

def collectResultsPrefilled(df, keys_to_extract=None, use_scratch_when_available = True,
        force_reload=False, verbose=False): #$, parent_key=None):
    rows = []

    #dec_type2varnames = {}
    ctr = 0
    for rowi,row in df.iterrows():
        fnf = None
        if use_scratch_when_available:
            coln = 'fn_full_scratch'
            if coln in row:
                fnf = row[coln]
        if fnf is None:
            print('Scratch not available, loading from $PROJECT')
            coln = 'fn_full'
            fnf = row['fn_full']

        print(f'{ctr}/{len(df)}:  Loading ',fnf)
        f = np.load(fnf,allow_pickle=True)

        par = f['par'][()]
        row['par'] = par
        row['SLURM_job_id'] = par.get('SLURM_job_id',None)
        #df.loc[rowi,'f'] = dict(f)

        try:
            if keys_to_extract is None:
                keys_to_extract = ['alphas','scores','dec_type','Xshape','dec_error']

            for dectype, pk in zip(['classic','b2b'],['decoding_per_var','decoding_per_var_b2b']):
                row1 = row.copy()
                if pk not in f:
                    print(f'WARNING: {pk} not found inf {fnf}')
                    continue
                # take one of decoding dicts
                dpv = f[pk][()]

                # take dec results for one variable
                for varname,d in dpv.items():
                    if 'dec_type' in d:
                        assert dectype == d['dec_type']
        #               print(pk, d['dec_type'])

                    # take info found for this variable
                    for k in d.keys():
                        if k not in keys_to_extract:
                            continue
                        v = d[k]
                        row1[ f'{varname}_{k}' ] = v
                # it's important to use NOT comma as separator becuase b2b already has commas there
                varnames_s = ';'.join( list(sorted( dpv.keys() )) )
                #row[f'{dectype}_varset'] = varnames_s
                row1['varset'] = varnames_s
                row1['dec_type'] = dectype
                    #break
                #break
                rows += [row1]
            ctr += 1
            #del row['f']  # if everything got loaded successfully
            #df.loc[rowi,'f'] = None
        except Exception as e:
            print('ERROR: ',str(e))
            row['err'] = str(e)

    dfr = pd.DataFrame(rows).reset_index(drop=True)
    #dfr['jobind'] = dfr['SLURM_job_id'].apply(lambda x: int(x.split('_')[1] ) )
    return dfr

def collectResults(subject,folder, freq_name='broad',
                   regression_type='Ridge', keys_to_extract=['par'],
                   time_locked=['feedback','target'], elem_var=None,
                   env = ['stable','random','all'],
                   custom_prefix = None,
                   tmin = None,
                   control_type = None,
                   parent_key = None, time_start=None, time_end = None,
                  fns = None, df=None, inc_tgc=1, verbose=0,
                  error_handling='raise'):

    'inc_tgc is whether inc trial group col is searching for'
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
        ctrls = '|'.join(['feedback', 'movement'] )
        env_all = ['stable','random','all']
        envstr = '|'.join(env)
        if inc_tgc:
            tgcc_str = '(' + '|'.join(trial_group_cols_all + ['trials']) + ')_'
        else:
            tgcc_str = ''
        regex_str = (f'({envstr})_{tgcc_str}'
                    r'(Ridge|Ridge_noCV|xgboost)_(' + tls + r')_(' + ctrls +  r')_(.*)_('+\
                    freqstr+ r')_t=(.*),(.*)\.npz')
        print('regex_str = ',regex_str)
        regex = re.compile(regex_str)
    if df is None:
        print(f'Found {len(fns)} files in total in {dir_full}')
    else:
        print(f'Df len = {len(df)}')
    #tuples = []

    def addRowInfo(f,row, index, elem_var_val=None, df_to_set=None,
            error_handling='raise'):
        assert (row is None) or (index is None)
        if parent_key is not None:
            if parent_key not in f.keys():
                if row is not None:
                    print(row['mtime'], row['SLURM_job_id'])
                print( list(f.keys() ) )
            subf = f[parent_key][()]
        else:
            subf = f

        kvs = []
        if elem_var is None:
            ktes_failed = []
            ktes_kvn_failed = []

            for kte in keys_to_extract:
                if kte not in subf:
                    print('addRowInfo: kte not in subf, subf.keys() = ', list(subf.keys() ) )
                    ktes_failed += [kte]
                    continue
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
                            #print(df_to_set.info())
                            #df_to_set.at[ index,key] = None
                            try:
                                # by default df.at pretends to be smart and detects Iterable and raises Exception if I try to set cell with a sequence
                                if isinstance(kvv, (np.ndarray,list,tuple) ):
                                    kvv = ListWrap(kvv) 
                                df_to_set.at[ index,key] = kvv
                            except ValueError as e:
                                print('EXC', e ,key, kvv, 
                                        df_to_set.info() )

                                ktes_kvn_failed += [(kte,kvn)]
                                if error_handling == 'raise':
                                    raise e

            if len(ktes_failed):
                print('addRowInfo: ktes_failed = ', ktes_failed)
            if len(ktes_kvn_failed):
                print('addRowInfo: ktes_kvn_failed = ', ktes_failed)
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
        dts = []
        dts_flt = []
        for fn_full in fns:
            dt = datetime.fromtimestamp(os.stat(fn_full).st_mtime)
            dts += [dt]
            if time_start is not None and dt < time_start:
                continue
            if time_end is not None and dt > time_end:
                continue
        

            fn = Path(fn_full).name
            r = re.match( regex, fn)
            #print(r )
            if r is None or r.groups() is None:
                print(f'collectResults: wrong fn = {fn}, for regex = {regex_str}')
                raise ValueError('Did not match')
            grps = r.groups(); #print(grps)
            #print(grps ) 
            if inc_tgc:
                env_cur,trial_group_col_calc_cur, rt_cur,time_locked_cur,control_type0, analysis_name_cur,\
                    freq_name_cur,tmin_cur,tmax_cur = grps
            else:
                env_cur, rt_cur,time_locked_cur,analysis_name_cur,\
                    freq_name_cur,tmin_cur,tmax_cur = grps

            _,fn = os.path.split(fn_full)

            if (custom_prefix is not None) and (analysis_name_cur != custom_prefix):
                if verbose > 0:
                    print(f'custom prefix: Reject {fn}, {dt}')# ( tmin={tmin}, tmin_cur={tmin_cur} )')
                continue

            if env_cur not in env:
                if verbose > 0:
                    print(f'env: Reject {fn}, {dt}')
                continue
            if time_locked_cur not in time_locked:
                if verbose > 0:
                    print(f'time_locked: Reject {fn}, {dt}')
                continue

            if regression_type is not None:
                if regression_type != rt_cur:
                    if verbose > 0:
                        print(f'regression_type: Reject {fn}, {dt}')
                    continue
            if freq_name is not None:
                if freq_name != freq_name_cur:
                    if verbose > 0:
                        print(f'freq_name: Reject {fn}, {dt}')
                    continue

            if tmin is not None:
                if tmin != tmin_cur:
                    if verbose > 0:
                        print(f'tmin: Reject {fn}, {dt} ( tmin={tmin}, tmin_cur={tmin_cur} )')
                    continue

            #fn_full = op.join(dir_full,fn)
            f = np.load( fn_full, allow_pickle=1)
            #print( list(f.keys() ), fn_full)

            control_type_ = f['par'][()]['control_type']
            if (control_type is not None) and \
                    (control_type_ != control_type):
                if verbose > 0:
                    print(f'control_type_: Reject {fn}, {dt}')
                continue

            SLURM_job_id = f['par'][()]['SLURM_job_id']
            #print(SLURM_job_id)

            dts_flt += [dt]

            if inc_tgc:
                tgc = ['trial_group_col_calc']
            else:
                tgc = []
            cols = ['subject','mtime', 'fn', 'fn_full', 'SLURM_job_id', 'env'] +\
                tgc + ['rt','time_locked','control_type', 'custom_suffix',
                        'freq_name','tmin','tmax'  ]

            colvals = [ subject, dt, fn, fn_full, SLURM_job_id, *grps ]
            row = dict( zip(cols,colvals) )


            #print(fn,dt)

            addRowInfo(f,row=row,index=None, 
                    error_handling=error_handling)

            rows += [row]

            #kvs = tuple(kvs)

            #tuples += [ (subject, os.stat(fn_full).st_mtime, fn, fn_full,
            #            *kvs, *grps )  ]

            #par = f['par'][()]
            #tuples += [ (os.stat(fn_full).st_mtime, fn, fn_full,
            #             par, *grps )  ]
            del f

        if len(dts):
            mi,mx = min(dts), max(dts)
            if len(dts_flt):
                mi_flt,mx_flt = min(dts_flt), max(dts_flt)
            else:
                mi_flt,mx_flt = None,None
            print(f'{len(rows)} after filtering by time.\n  Min time  = {mi_flt}, Max time  = {mx_flt},\n  Min total = {mi}, Max total = {mx}')
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
            ktes_failed = []
            # we need to add None keys because we want to have them in rows that we
            # will iter through later
            for kte in keys_to_extract:
                if kte not in subf:
                    print('collectResults: kte not in subf, subf.keys() = ', list(subf.keys() ) )
                    ktes_failed += [kte]
                    continue
                kv0 = subf[kte]
                if isinstance(kv0, np.ndarray):
                    kv0 = kv0[()]
                for kvn,kvv in kv0.items():
                    key = f'{kte}_{kvn}'
                    # here it gives worning that dataframe is highly fragmented
                    df[key] = None

            for index,row in df.iterrows():
                fn_full = row['fn_full']
                f = np.load( fn_full, allow_pickle=1)
                try:
                    addRowInfo(f,row=None,index=index, df_to_set=df,
                            error_handling=error_handling)
                except (KeyError,ValueError) as e:
                    dt = datetime.fromtimestamp(os.stat(fn_full).st_mtime)
                    print(fn_full, str(dt) )
                    raise e
                del f
            df_collect = df

            if len(ktes_failed):
                print('collectResults: ktes_failed = ', ktes_failed)
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
                    addRowInfo(f,row=None,index=index,elem_var_val = elem_var_val, df_to_set=dfc, error_handling=error_handling)
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

def collectDecodingResults(subjects, subject_inds, output_folder,freq_name, env, control_type, time_locked, 
        custom_prefix, tmin_desired, 
        varnames, varnames_b2b,
        time_start, time_end, verbose=0, 
        debug = 0, error_handling = 'raise'):
    df_persubj=[]
    for subject in np.array(subjects)[subject_inds]:
        print(f'Starting collecting subject {subject}')
        #df_collect = collectResults(subject,hpass)
        #df_collect = collectResults(subject,output_folder,freq_name,
        #        keys_to_extract = ['par','scores_err_sens','diff_err_sens_pred',
        #                           'err_sens', 'prev_error', 'correction'] )
        # , 'non_hit' is not per dec var
        print('###########################')
        print('Collect 1')
        print('###########################')
        df_collect1 = collectResults(subject,output_folder,freq_name,
                keys_to_extract = ['par'], env=env,
                control_type = control_type,
                time_locked = time_locked,
                tmin = tmin_desired, custom_prefix = custom_prefix,
                time_start=time_start, 
                time_end=time_end, verbose=verbose,
                error_handling=error_handling)
        if df_collect1 is None or len(df_collect1) == 0:
            print(f'Collect 1: Found nothing for subject {subject}')
            continue
        fns = list( df_collect1['fn'] )

        print('###########################')
        print('Collect 2')
        print('###########################')
        df_collect2 = collectResults(subject, output_folder, freq_name,
                keys_to_extract = varnames,
                parent_key = 'decoding_per_var',
                df = df_collect1.copy(),
                time_start=time_start, time_end=time_end,
               error_handling=error_handling )
        df_collect2['dec_type'] = 'classic'

        print('###########################')
        print('Collect 3')
        print('###########################')
        try:
            df_collect3 = collectResults(subject, output_folder, 
                freq_name,
                keys_to_extract = varnames_b2b,
                parent_key = 'decoding_per_var_b2b',
                df = df_collect1,
                time_start=time_start, time_end=time_end,
               error_handling=error_handling )
            df_collect3['dec_type'] = 'b2b'

            vnpostfix = '_varnames'
            for ci in np.where(df_collect3.columns.str.endswith(vnpostfix))[0]:
                bn = df_collect3.columns[ci][:-len(vnpostfix)]
                #TODO: problem here: if I have same varname decoded
                # in two different sets of vars for b2b, it will overwrite
                # so maybe I have to duplicate df_collect3 for each set
                order = df_collect3[f'{bn}{vnpostfix}']._values[0]
                from postproc import ListWrap
                if isinstance(order, ListWrap):
                    order = order.data
                if isinstance(order, float) and np.isnan(order):
                    continue
                order_sc = [f'{s}_scores' for s in order]
                def f(x):
                    vns = x[f'{bn}_varnames']
                    scores = x[f'{bn}_scores']
                    if isinstance(scores, ListWrap):
                        scores = scores.data
                    if isinstance(vns, ListWrap):
                        vns = vns.data
                    #correction,error_varnames
                    r = []
                    for vn,sc in zip(vns,scores):
                        #x[ f'{vn}_scores'] = [sc]  # one-el list
                        r += [sc]
                    return tuple(r)
                df_collect3[order_sc ] = df_collect3.apply(f,1, result_type = 'expand')

                order_vals = [f'{s}_vals' for s in order]
                def f(x):
                    vns = x[f'{bn}_varnames']
                    vals = x[f'{bn}_vals']

                    if isinstance(vals, ListWrap):
                        vals = vals.data
                    if isinstance(vns, ListWrap):
                        vns = vns.data
                    #correction,error_varnames
                    r = []
                    for vi in range( vals.shape[1]):
                        #x[ f'{vn}_scores'] = [sc]  # one-el list
                        r += [ vals[:,vi] ]
                    return tuple(r)
                df_collect3[order_vals ] = df_collect3.apply(f,1, result_type = 'expand')
        except KeyError as e:
            print(f'!!!! ERROR: B2B is missing for subj {subject}', str(e) )
            print(' -- args ',freq_name, env, control_type, time_locked, 
        custom_prefix, tmin_desired)
            if error_handling == 'raise':
                raise e
            else:
                df_collect3 = pd.DataFrame( [] )
        # TODO: some info from params
        # trim_outliers
        # trial_group_col_calc
        # freq_name
        # time_locked

        #df_collect2['perturbation'] = np.nan
        #df_collect3 = collectResults(subject, output_folder, freq_name,
        #        keys_to_extract = varnames,
        #                            parent_key = 'decoding_per_var_and_pert',
        #                            df = df_collect2,
        #                            cokey = 'perturbation',
        #                            time_start=time_start, time_end=time_end)

        #for kte in set(df_collect2.columns) - set(df_collect1.columns):
        #    df_collect1[kte] = df_collect2[kte]

        #df_persubj += [df_collect2] #, df_collect3]
        df_persubj += [df_collect2, df_collect3] #, df_collect3]
        if not debug:
            del df_collect1
            del df_collect2
            del df_collect3
        #tmins = df_persubj[df_persubj[env] == 'stable' ]   ['tmin']
    if len(df_persubj):
        df = pd.concat( df_persubj, ignore_index=True )
    else:
        df = None
    return df
    #df.reset_index(inplace=True)
    #df.drop('index',1,inplace=True)

def cleanExtra(dftmp, varnames0):
    p = {'errors':'ignore'}
    dftmp = dftmp.drop(columns=[varname + '_scores' for varname in varnames0], **p )
    dftmp = dftmp.drop(columns=[varname + '_vals' for varname in varnames0], **p )
    dftmp = dftmp.drop(columns=[varname + '_Xshape' for varname in varnames0], **p )
    dftmp = dftmp.drop(columns=[varname + '_mask_valid' for varname in varnames0], **p )
    dftmp = dftmp.drop(columns=[varname + '_mask_diff' for varname in varnames0], **p )
    dftmp = dftmp.drop(columns=[varname + '_non_hit' for varname in varnames0], **p )
    dftmp = dftmp.drop(columns=[varname + '_non_hit_mask' for varname in varnames0], **p )
    return dftmp

def extractClassicCols(dfc, defvarcolns, error_handling='raise' ):
    varnames0 = defvarcolns
    dfsplot = []
    for varname in varnames0:
        print(varname)
        dftmp = dfc.copy()
        def f(x):
            if isinstance(x,ListWrap):
                x = x.data
            if isinstance(x, (np.ndarray, list) ):
                x = np.mean(x)
            return np.array(x)

        try:
            scname = f'{varname}_scores'
            sc = dftmp[scname]        
            sc = sc.apply(f)
        except Exception as e:
            print(f'Error extracting {scname} ', str(e) )
            if error_handling == 'raise':
                raise e
            else:
                sc = None

        dftmp['varname'] = varname
        dftmp['varset'] = ','.join(varnames0)
        dftmp['varseti'] = -1
        #if isinstance(sc, ListWrap):
        #    sc = np.mean(sc.data)
        #if isinstance(sc, np.ndarray):
        #    sc = sc.mean()
        #if isinstance(sc, list):
        #    sc = np.mean(sc)
        #if not isinstance(sc, float):
        #    print(sc)
        #    raise ValueError('ff')
        dftmp['vals_to_plot'] = sc
        try:
            dftmp['Xshape'] =  dftmp[f'{varname}_Xshape']       
        except KeyError as e:
            print('Xshape was not found:', e)
        
        dftmp = cleanExtra(dftmp, varnames0)

        dfsplot += [dftmp]
        del dftmp
    df_class_plot = pd.concat( dfsplot )
    del dfsplot
    import gc; gc.collect()
    return df_class_plot

def extractB2Bcols(dfc,  defvarcolns = ['err_sens,prev_error,error'], varcolns = None, toclean=None  ):
    #ct = "movement"
    #env_cur = 'stable'
    from error_sensitivity import getAnalysisVarnames
    dfc_ = dfc.query('dec_type == "b2b" ')
    dfc_ = dfc_.copy().reset_index()
    print('extractB2Bcols: len(dfc_) = ', len(dfc_))

    #if varcolns is None:
    #    varcolns = ['b2b0_prev_feedback','b2b0_prev_movement', 
    #          'b2b0_prev_error', 'b2b0_error', 'b2b0_prev_belief']
    #
    #dfc_[varcolns] = np.nan

    enums = []  #enums is a list of lists of tuples (varstei, varset string with commas)
    dfsplot = []
    vns_all = []
    for time_locked in dfc['time_locked'].unique():
        for ct in dfc['control_type'].unique():
        #for ct in ['feedback']:
            s=  ','.join( getAnalysisVarnames(time_locked, ct)[0] )
            varnames_b2b = [s] +  defvarcolns  # list of strings with comma-sep varnames

            # it REALLY important to take index.values and not just index
            inds = dfc_.query('control_type == @ct and time_locked == @time_locked').index.values  
            print('extractB2Bcols: len(inds) for control_type {} is {}'.format(ct, len(inds)) )
            if len(inds) == 0:
                continue
            enums += [ list(enumerate(varnames_b2b)) ]
            for vnsi, vns in enumerate(varnames_b2b):
                print(ct,time_locked,vnsi,vns)
                vns_all += [vns]
                if isinstance(vns,str):
                    varnames_cur0 = vns.split(',')
                else:
                    varnames_cur0 = vns
                #varnames_cur = [f'b2b{vnsi}_' + vn for vn in varnames_cur0]
                #print('extractB2Bcols: varnames_cur=',varnames_cur)
                #assert ~dfc_.loc[inds, vns + '_scores'].isna().any()

                ## not working
                ##dfc_.loc[inds, varnames_cur] = dfc_.loc[inds, vns + '_scores'].apply(pd.Series) #, result_type='expand')
                def f(x, vni):
                    if isinstance(x, ListWrap):
                        return x.data[vni]
                    else:
                        if isinstance(x,float):
                            assert np.isnan(x)
                            return x
                        else:
                            return x[vni]

                #for vni,vn in enumerate(varnames_cur):
                #     dfc_.loc[inds, vn] = dfc_.loc[inds, vns + '_scores'].apply(lambda x: f(x,vni), 1)

                for vni,vn in enumerate(varnames_cur0):
                    if vns + '_scores' not in dfc_.columns:
                        print(vns + '_scores', 'missing')
                        continue
                    dftmp = dfc_.loc[inds].copy()

                    dftmp.loc[inds,'vals_to_plot'] = dfc_.loc[inds, vns + '_scores'].apply(lambda x: f(x,vni), 1)
                    #dftmp.loc[inds,'vals_y'] = dfc_.loc[inds, vns + '_vals'].apply(lambda x: f(x,vni), 1)
                    colsdrop = []
                    for csuff in ['Xshape','dec_error','dec_type','scores']:
                        colsdrop += [f'{vn}_{csuff}' for vn in varnames_b2b]
                    dftmp.drop(columns=colsdrop, inplace=True )  

                    dftmp['varname'] = vn
                    dftmp['varset'] = vns
                    dftmp['varseti'] = vnsi
                                                       
                    if toclean is not None:
                        dftmp = cleanExtra(dftmp, toclean)

                    dfsplot += [ dftmp ]
                #d = dict(zip(varnames_cur, dfc_[vns + '_scores'] ) )
                #print(d)

                #break


    #for df in dfsplot:
    #    for csuff in ['Xshape','dec_error','dec_type','scores']:
    #        df.drop(columns = [f'{vn}_{csuff}' for vn in varnames_b2b], inplace=1)
    df_b2b_plot = pd.concat(dfsplot, ignore_index=1)
    #df_b2b_plot = cleanExtra(df_b2b_plot, list(set(vns_all)) ) # I'll need to remove it in classic as well anyway

    #assert not dfc_['b2b0_prev_error'].isna().all()
    
    return dfc_, df_b2b_plot, enums

def runStatTests(df, cols0, coln = 'vals_to_plot', 
        alt='two-sided', verbose = 0 ):
    # takes a long time to finish
    from scipy.stats import ttest_rel, ttest_1samp
    def f(df):
        vsd = {}
        vsl = []
        resd = {'stable_greater_random':np.nan}

        # choosing subject present for both env
        subjs = []
        for env_cur in ['stable', 'random']:        
            ss = df.query('env == @env_cur')['subject'].unique()
            subjs += [ss]
        subj_both = list(set(subjs[0]) & set(subjs[1]))

        for env_cur in ['stable', 'random']:        
            dftmp = df.query('env == @env_cur')            
            if len(dftmp) == 0:
            #print(f'zero len for {ct}: {varname}, skipping')
                continue
            assert len(dftmp) <= 20, len(dftmp)
            vsd[env_cur] = dftmp[coln].values

            dftmp = dftmp.query('subject in @subj_both')        
            vsl += [dftmp[coln].values]
        if len(vsd) == 0:
            return None
        for env_cur in ['stable', 'random']:
            if env_cur not in vsd:
                print(f'No {env_cur} found')
                print(df[cols0].iloc[0])
                resd = pd.DataFrame([resd])
                return resd

            tr = ttest_1samp( vsd[env_cur], 0, alternative=alt )
            s = ' '
            if tr.pvalue < 0.05:
                s = '*'
            #if (tr.pvalue < 0.05) or (not signif_only):
            #    print(s + '{} control={:8}, var={:14}, {}>0, pvalue={:.4f}'.format(tmin, ct, varname, env_cur, tr.pvalue) )
            resd[env_cur] = tr.pvalue
            resd[env_cur + '_len'] = len(vsd[env_cur])
        if len(vsl[0]) != len(vsl[1]):
            print('Not equal!', (len(vsl[0]), len(vsl[1])) )
            print(len(subj_both))
            print(df[cols0].iloc[0])
            resd = pd.DataFrame([resd])
            return resd

        tr = ttest_rel(vsl[0],vsl[1], alternative=alt )

        s = ' '
        if tr.pvalue < 0.05:
            s = '*'
            #tmins_good += [tmin]
        #if (tr.pvalue < 0.05) or (not signif_only):
        #    print(s + '{} control={:8}, var={:14}, stable>random pvalue={:.4f}'.format(tmin, ct, varname, tr.pvalue) )
        #print('')
        if alt == 'greater':
            resd['stable_greater_random'] = tr.pvalue
        else:
            resd['stable_neq_random'] = tr.pvalue
        #resd['tmin'] = tmin
        #resd['varname'] = varname
        #resd['control_type'] = ct
        #resd['custom_suffix'] = suff
        #resd['varseti'] = varseti
        resd = pd.DataFrame([resd])
        #rows += [resd]
        return resd

    df_ttest_res = df.groupby(cols0).apply(f)
    return df_ttest_res


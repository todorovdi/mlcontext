from IPython import get_ipython; ipython = get_ipython()
#%run -i ../exec_HPC.py 0
import os 
import sys
from os.path import join as pjoin
sys.path.append(os.path.expandvars('$CODE_MEMORY_ERRORS'))
import pandas as pd
import time
import numpy as np
from config2 import path_data
import multiprocessing as mp
from config2 import path_fig

import state_space_Tan2014 as tan
from state_space_Tan2014 import *
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import argparse

#from taumodels.models import fitDiedrischenModel,fitTanModel

try :
    print(len(dfcc_all_NIH_))
except NameError:
    fnf_NIH = pjoin(path_data,'df_all_multi_tsz_.pkl.zip')
    dfcc_all_NIH_ = pd.read_pickle(fnf_NIH)
    from datetime import datetime
    print('Data mtime',datetime.fromtimestamp(os.stat(fnf_NIH).st_mtime))

numtrain_CC = 12

parser = argparse.ArgumentParser()
parser.add_argument('--n_jobs',  default = 40, type=int )
parser.add_argument('--save_suffix',  default='', type=str )
parser.add_argument('--n_runs',  default=20, type=int )
parser.add_argument('--do_plot',  default=0, type=int )
parser.add_argument('--exit_after',  default='end', type=str )

parser.add_argument('--subject', required = True, type=str)
parser.add_argument('--reg',  default=0, type=float )
parser.add_argument('--fitmask',  default='no', type=str )
parser.add_argument('--fit_to',  default='errors', type=str )
parser.add_argument('--num_bases',  default=10, type=int )
parser.add_argument('--use_true_errors',  default=0, type=int )
parser.add_argument('--optimize_initial_err_sens',  default=0, type=str )
parser.add_argument('--optimize_initial_state',  default=0, type=int )
parser.add_argument('--thr_val_mult',  default=0, type=float )

parser.add_argument('--ES_thr',  default=None, type=float )

 

 
argscmd = parser.parse_args()

subject = argscmd.subject

assert argscmd.fitmask in ['no','stable','random']


try :
    print(len(dfos) )
    if dfos.iloc[0]['subject'] == argscmd.subject:
        redfos = False
    else:
        redfos = True
except NameError:
    redfos = True

if redfos:
    dfos = dfcc_all_NIH_.query('trial_shift_size == 1 and '
        'trial_group_col_calc == "trials" and retention_factor_s == "0.924"'
        ' and subject == @subject').copy()
    assert not dfos.duplicated(['trials']).any()

    dfos['perturbation_pscadj'] = dfos['perturbation']
    dfos.loc[dfos['pert_seq_code'] == 1, 
        'perturbation_pscadj'] = -dfos['perturbation']
    dfos['trial_index'] = dfos['trials']
    #path_data_cc = '/home/demitau/data_Quentin/full_experiments/context_change_behav/results'

    #dfos['error_deg'] = dfos['error'] * 180 / np.pi
    dfos['error_deg__'] = dfos['error_pscadj'] * 180 / np.pi

# # main run
# nt = 1000
# nsub = 4
# subjects_sub = subjects[0:nsub]
# dfc = df.query('subject in @subjects_sub and trial_index < @nt and trial_index >= @numtrain').copy()
# dfc.loc[dfc.query('error_lh2_ang_deg.abs() > 60').index, 'error_lh2_ang_deg'] = np.nan
# # TODO: maybe I should invalidate too big errors, they are clearly outliers
# do_plot = 0
# #ipython.run_line_magic('run',' -i ' + scr)




timeout = 12
from taumodels.models import simMoE, calcH_min
from taumodels.models import getBestHerzPar

def isSuccess(res, check_q = False):
    (minres_H, pard, seed) = res
    #print('isSuccess: pard={} minres={}'.format(pard,minres_H))
    if pard is None:
        return False
    r = simMoE(-perturb, EC_mask=EC_mask, **pard)
    motor_output,error_pred2,ws,err_sens2, gaussian_centers,gaussian_variances = r 
    
    if check_q:
        q = np.quantile(err_sens2, 0.9) 
    else:
        q = 0
    if minres_H.success and q < 0.98:
        return True
    return False


#def _opt(_x):
#    perturb, error, EC_mask, pre_break_duration, err_sens,\
#        fit_to, optimES0, cap_err_sens, use_true_errors, reg, rnsuff = _x

nruns = argscmd.n_runs
timeout = 50
n_jobs=min(nruns, argscmd.n_jobs )
reg = argscmd.reg

qs = ''
dfname = 'NIH'
t0 = time.time()
if dfname == 'NIH':
    # it is pscadj
    coln_error_ = 'error_deg__'
else:         
    qs += ' and trial_index >= @numtrain'            
    coln_error__ = 'error_lh2_ang_deg'
    coln_error_ = coln_error__



if argscmd.ES_thr is not None:
    ES_thr = argscmd.ES_thr
else:
    # here we need to use all indeed
    dfni = dfcc_all_NIH_[~np.isinf(dfcc_all_NIH_['err_sens'])]
    std_mult = 5
    dfni_d = dfni.groupby(['subject'])['err_sens'].describe().reset_index()
    ES_thr = dfni_d[dfni_d.columns[1:]].mean().to_dict()['std'] * std_mult
    print(f'ES_thr (calc from all) = {ES_thr}')


if redfos:
    #clear_outlier = 120
    clear_outlier = ES_thr * 5
    #dfos = dfos.query(qs).copy()
    inds = dfos.query(
        f'{coln_error_}.abs() > {clear_outlier}').index
    dfos.loc[inds,coln_error_] = np.nan


    ############   here we set arrays tha will be used for calc
    error = dfos[coln_error_].values
    perturb = dfos['perturbation_pscadj'].values
    perturb[perturb > 180] = perturb[perturb > 180] - 360
    perturb[perturb < -180] = perturb[perturb < -180] + 360

    if dfname == 'NIH':
        #EC_mask = np.array( [0] * len(dfos) )
        EC_mask = None
    else:
        EC_mask = (dfos['trial_type'] == "error_clamp").values

    pre_break_duration = dfos['pre_break_duration'].values
    err_sens = dfos['err_sens'].values

    from behav_proc import truncateNIHDfFromErr
    dfos_thr = truncateNIHDfFromErr(dfos)
    #err_sens_thr = dfos_thr['err_sens_trunc']



if argscmd.fitmask == 'stable':
    fitmask_vals = (dfos['env'] == 'stable').values
elif argscmd.fitmask == 'random':
    fitmask_vals = (dfos['env'] == 'random').values
else:
    fitmask_vals = None


def _opt(**kwargs):
    rnsuff = kwargs['rnsuff']; del kwargs['rnsuff'];
    subject = kwargs['subject']; del kwargs['subject'];
    fitmask = kwargs['fitmask']; 
    kwargs['fitmask'] = kwargs['fitmask_vals'];
    del kwargs['fitmask_vals']; 

    s = int(np.ceil(time.time() * 1e4) ) % int( 1e5 )
    kwargs['seed'] = s

    # run minimization
    # pard is the parameter dictionary
    minres_H, pard = calcH_min(**kwargs)

    del kwargs['pert']
    del kwargs['errors']
    del kwargs['EC_mask']
    del kwargs['pre_break_duration']
    del kwargs['err_sens']
    #kwargs['fitmask'] = argscmd.fitmask # if I do so, I will always have the same fitmask when run from jupyter
    kwargs['rnsuff'] = rnsuff
    kwargs['fitmask'] = fitmask
    return (minres_H, pard, kwargs, subject)


optimES0 = 'WRONG'; capES = False; 
if argscmd.optimize_initial_state:
    initial_state_bounds = (-45,45)
else:
    initial_state_bounds = (0,0)

arg0 = dict(pert=-perturb, errors = error,  EC_mask =EC_mask, 
            pre_break_duration = pre_break_duration,
            err_sens = err_sens, fit_to = argscmd.fit_to, 
            optimize_initial_err_sens=optimES0,
            reg=reg, timeout = timeout,
            fitmask_vals = fitmask_vals,
            fitmask = argscmd.fitmask,
            cap_err_sens = capES,
            initial_state_bounds = initial_state_bounds,
             use_true_errors = argscmd.use_true_errors, 
            num_bases = argscmd.num_bases,
           err_handling = 'raise', subject=subject)

thr_val = dfos_thr['error_initstd'].values[0]*180/np.pi

if argscmd.exit_after == "init":
    sys.exit(0)

small_err_thr = thr_val * argscmd.thr_val_mult
optimize_initial_err_sens = argscmd.optimize_initial_err_sens
assert optimize_initial_err_sens in ['optimize','from_data','custom']

arg = arg0.copy()
arg['fit_to'] = argscmd.fit_to
arg['cap_err_sens']    = False
arg['use_true_errors'] = ute
arg['optimize_initial_err_sens'] = oes
arg['small_err_thr'] = small_err_thr

s = f'_sethr=mestd*{small_err_thr / thr_val:.1f}'
arg['rnsuff'] =  argscmd.save_suffix + s 
args = nruns * [ arg ]; 


if argscmd.exit_after == "args_prep":
    sys.exit(0)

######################################################
######################################################


from behav_proc import aggRows
plr = pd.DataFrame(plr, columns=['minres','pard','arg','subject']) 
plrg = plr[~plr.pard.isnull()].copy()

for k in plrg.pard.values[0]:
    plrg[k] = plrg.pard.apply(lambda x: x.get(k,None))

for k in plrg.arg.values[0]:
    plrg[k] = plrg.arg.apply(lambda x: x.get(k,None))

plrg['fun'] = plrg.apply(lambda x: x['minres'].fun, 1)                   

lbd = lambda x: (f'fit_to={x["fit_to"]}; optimES0={x["optimize_initial_err_sens"]};'
    f' capES={x["cap_err_sens"]}; useTrueErr={x["use_true_errors"]};'
    f' sethr={x["small_err_thr"]:.2f}; {x["rnsuff"]}'     )
plrg['runname'] = plrg.apply(lbd,1)

bestCalcs = aggRows(plrg, 'fun', 'min',
     grp = plrg.groupby(['subject','runname'] ) )

#if argscmd.do_save:
#bestCalcs.to_pickle(pjoin(path_data,'herz_param_calced2.pkl.zip')
#    ,compression='zip')
assert not bestCalcs.duplicated(['subject','runname']).any()

if argscmd.do_plot:
    dfcc_all_NIH_['perturbation_pscadj'] = dfcc_all_NIH_['perturbation']
    dfcc_all_NIH_.loc[dfcc_all_NIH_['pert_seq_code'] == 1, 
    'perturbation_pscadj'] = -dfcc_all_NIH_['perturbation']

    dfr =[]
    for rowi,row in bestCalcs.iterrows():
        runname = row['runname']
        pard_ = row['pard']
        if pard_ is None:
            print('empty',runname)
            continue
        pard = pard_.copy()
        if 'fun' in pard:
            del pard['fun']
            del pard['nit']
            del pard['nfev']        
        
        subj = row["subject"]
        df0 = dfcc_all_NIH_.query('trial_shift_size == 1 and '
            'trial_group_col_calc == "trials" and '
            'retention_factor_s == "0.924" and '
            'subject == @subj').sort_values(['trials'])
        pre_break_duration = df0['pre_break_duration'].values
        error = df0['error_pscadj'].values * 180 / np.pi
        perturb = df0['perturbation_pscadj'].values

        use_true_errors = row['use_true_errors']
        cap_err_sens = row['cap_err_sens']    
        r = simMoE(-perturb,EC_mask=EC_mask,true_errors=error,
                    pre_break_duration=pre_break_duration,
                    **pard, cap_err_sens=cap_err_sens,
                   small_err_thr = row['small_err_thr'],
                   num_bases = row['num_bases'],
            use_true_errors=use_true_errors)
        
        (motor_output,error_pred2,ws,err_sens2, 
            gaussian_centers,gaussian_variances) = r 
        _dat = np.array((motor_output,error_pred2,
            err_sens2, np.arange(len(motor_output), dtype=int) ) )
        df_ = pd.DataFrame(data=_dat.T, 
            columns = ['motor_output','error_pred2',
                    'err_sens2','trial_index'])
                
        df_['err_sens'] = df0['err_sens'].values
        df_['error_pscadj_deg'] = df0['error_pscadj'].values * 180 / np.pi

        for kn,kv in row.items():
            df_[kn] = kv
        dfr += [df_ ]
    dfr = pd.concat(dfr,ignore_index=1)

    runnames = dfr['runname'].unique()
    print(runnames)


    #%debug
    import warnings

    for rn in runnames:
        df_ = dfr.query('runname == @rn and fit_to == "errors"')
        if len(df_) == 0:
            print('Emtpy for ',rn)
            continue
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("ignore",category=FutureWarning)
            warnings.filterwarnings("ignore",category=UserWarning)
            fg,df__ = relplot_multi(kind='line',data=df_,
                    x='trial_index',
                ys=[['error_pred2','error_pscadj_deg'],
                   ['err_sens','err_sens2']], 
                ylabel = ['error','ES'],
                ylim=[(-100,100), (-2,2)], aspect = 2,
                                   facet_kws = {'sharex':True,
                                               'sharey':False})

            fg.refline(y=0)    
            fg.refline(x=192)
            fg.refline(x=192*2)
            fg.refline(x=192*3)
            plt.suptitle(rn)    

            plt.savefig(pjoin(path_fig, rn + '.pdf'))
            plt.close()



# this is to handle optim that hangs in the middle. But it makes things slower and not parallel
#for i in range(nruns*max_attempts):
#    # Create a queue to store the result
#    queue = mp.Queue()
#    # Create a process that runs the optimize function and puts the result in the queue
#    p = mp.Process(target=lambda q, x: 
#        q.put(_opt(x)), 
#            args=(queue, (perturb, error, EC_mask, pre_break_duration, reg)))
#    p.start()
#    p.join(timeout)
#
#    # Check if the process is still alive
#    if p.is_alive():
#        p.terminate()
#        p.join()
#        print(f"------ {subj} Terminating optimization: time limit reached")                
#        res = None,None,None
#    else:
#        res = queue.get()
#
#    if isSuccess(res):
#        dfnamesubj2res_Herz[(dfname,subj)].append(res)
#        plr.append(res)
#        print('-------------------')
#        print('get output but with success!'
#                ' new len(plr) = {len(plr)} ( of needed {nruns})')
#        print('-------------------')
#    else:
#        print('-------------------')
#        print('get output but w/ot success')
#        print('-------------------')
#    
#    if len(plr) >= nruns:
#        print(f'Finishing for {subj} after {i+1} runs')
#        break
#    else:
#        print(f'---- {subj} Len plr = ',len(plr))

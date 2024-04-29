from joblib import Parallel, delayed

import os
#data_dir_general = '/home/demitau/data_Quentin'
#data_subdir_mem_err_main = 'full_experiments/data2'
from os.path import join as pjoin
import os,sys; sys.path.append(os.path.expandvars('$CODE_MEMORY_ERRORS'))
import error_sensitivity
#data_dir_input = pjoin(data_dir_general,data_subdir_mem_err_main)
import pandas as pd
import seaborn as sns
import sys, traceback

from base2 import (width, height, radius, calc_target_coordinates_centered,
                  calc_rad_angle_from_coordinates)

import numpy as np

#data_dir_general = '/home/demitau/data_Quentin'
#data_subdir_mem_err_main = 'full_experiments/data2'
#data_dir_input = pjoin(data_dir_general,data_subdir_mem_err_main)
#scripts_dir = pjoin(data_dir_general,'full_experiments','scripts2')

data_dir_general = os.path.expandvars('$DATA_QUENTIN')
data_dir_input = os.path.expandvars('$DATA_MEMORY_ERRORS_STAB_AND_STOCH')
#scripts_dir = pjoin( os.path.expandvars('$CODE_MEMORY_ERRORS'), 'previous_analyses')
scripts_dir = pjoin( os.path.expandvars('$CODE_MEMORY_ERRORS'))


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_jobs',  default = 20, type=int )
parser.add_argument('--save_suffix',  default='_test', type=str )
parser.add_argument('--use_sub_angles',  default=0, type=int )
parser.add_argument('--read_behav_version',  default='v2', type=str )
parser.add_argument('--n_subjects',  default=20, type=int )
parser.add_argument('--coln_error',  default='error', type=str )
parser.add_argument('--coln_correction_calc',  default=None, type=str )
 
# script flow params
parser.add_argument('--do_read',  default=1, type=int )
parser.add_argument('--do_collect',  default=1, type=int )
parser.add_argument('--do_add_cols',  default=1, type=int )
parser.add_argument('--do_calc_ES',  default=1, type=int )
parser.add_argument('--do_plot',  default=1, type=int )
parser.add_argument('--do_save',  default=1, type=int )
parser.add_argument('--save_owncloud',  default=0, type=int )
parser.add_argument('--perturbation_random_recalc',  default=1, type=int )
# ES calc params
parser.add_argument('--trial_shift_size_max',  default=1, type=int )
parser.add_argument('--do_per_tgt',  default=0, type=int )
parser.add_argument('--do_per_env',  default=0, type=int )
parser.add_argument('--retention_factor',  default='1', type=str )
parser.add_argument('--reref_target_locs',  default=0, type=int )

 
args = parser.parse_args()

retention_factor = None
if ',' in args.retention_factor:
    retention_factor = args.retention_factor.split(',')
else:
    retention_factor = [args.retention_factor]
    #retention_factor = map(float, retention_factor)


print(data_dir_input, scripts_dir)
 
use_sub_angles = args.use_sub_angles

subjects = [f for f in os.listdir(data_dir_input) if f.startswith('sub') ]
subjects = list(sorted(subjects))
print(subjects)


###########################################################

if args.do_read:
    print('Start reading raw .csv files')
    import subprocess as sp
    #for subject in subjects:
    n_jobs = args.n_jobs
    def f(subject):
        #ipy = get_ipython()
        #ipy.run_line_magic('run', f'-i {script_name} -- --subject {subject}')
        if args.read_behav_version == 'v2':
            script_name = pjoin(scripts_dir,'read_behav2_upd.py')
            p = sp.Popen(f"python {script_name} --subject {subject} --use_sub_angles {use_sub_angles} --save_suffix '{args.save_suffix}' --perturbation_random_recalc {args.perturbation_random_recalc}".split() )
        elif args.read_behav_version == 'v1':
            script_name = pjoin(scripts_dir,'read_behav2.py')
            p = sp.Popen(f"python {script_name} --subject {subject} --save_suffix {args.save_suffix}".split() )
        p.wait()

    r = Parallel(n_jobs=n_jobs,
         backend='multiprocessing')( (delayed(f)\
            ( subject) for subject in subjects[:args.n_subjects]) )
    #for subject in subjects:
    #    ipy.run_line_magic('run', f'-i {script_name}')
    


###########################################################
target_angs = (np.array([157.5, 112.5, 67.5, 22.5]) + 90) * \
              (np.pi/180)
target_coords = calc_target_coordinates_centered(target_angs)     

import datetime

from behav_proc import *
if args.do_collect:
    behav_df_all = []
    #or subj in subjects[:4]:#[si]
    for subj in subjects[:args.n_subjects]:
        behav_data_dir = pjoin(data_dir_input,'behavdata')
        #behavdata
        task = 'VisuoMotor'
        if args.read_behav_version == 'v2':
            updstr = '_upd'
            fname = pjoin(path_data, subj, 'behavdata',
                        f'behav_{task}_df{updstr}{args.save_suffix}.pkl' )
        else:
            updstr = ''
            fname = pjoin(path_data, subj, 'behavdata',
                        f'behav_{task}_dfv1{updstr}{args.save_suffix}.pkl' )
        # where we would save cleaned df
        #fname_EC = pjoin(path_data, subj, 'behavdata',
        #                f'behav_{task}_df_EC.pkl' )
        behav_df_full = pd.read_pickle(fname)
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(fname))

        behav_df = pd.read_pickle(fname)
        behav_df['subject'] = subj
        behav_df['mtime'] = mtime
        
        behav_df_all += [behav_df]
        
    behav_df_all = pd.concat(behav_df_all)
    # behav_df_all['correction'] = None
    # behav_df_all['err_sens'] = None
    #df_all = behav_df_all.reset_index()
    bc = ['index','level_0']
    bc = list( set(bc) & set( behav_df_all.columns) )
    df_all = behav_df_all.drop(columns=bc).sort_values(['subject','trials']).reset_index(drop=True)

    assert len( df_all['subject'].unique() ) == args.n_subjects



    fn = f'df_all{args.save_suffix}.pkl.zip'
    fnf = pjoin(path_data,fn)
    if args.do_save:
        behav_df_all.to_pickle(fnf , compression='zip')
        print(fnf)
        if args.save_owncloud:
            tstr = str( datetime.datetime.now() )[:10] 
            behav_df_all.to_pickle(pjoin('/home/demitau/current/merr_data',fn + '_' + tstr) , compression='zip')

badcols =  checkErrBounds(df_all)
#assert len(badcols) == 0

#%debug
if args.do_add_cols:
    addBehavCols(df_all)

from base2 import subAngles
df_all['vals_for_corr'] = subAngles(df_all['target_locs'], df_all['org_feedback']) # movement 
vars_to_pscadj = ['vals_for_corr']
for varn in vars_to_pscadj:
    df_all[f'{varn}_pscadj'] = df_all[varn]
    df_all.loc[df_all['pert_seq_code'] == 1, f'{varn}_pscadj']= -df_all[varn]

#try:
#    addBehavCols(df_all)
#except Exception as e:
#    exc_info = sys.exc_info()
#    exc = traceback.TracebackException(*exc_info, capture_locals=True)
#    
#    
#    stackframe = exc_info[2].tb_next.tb_frame
#    local_vars_in_fun = stackframe.f_locals
#    display(exc_info, stackframe)
#    print('line num = ',exc_info[2].tb_next.tb_lineno)


#pertvals = list( df_all['perturbation'].unique() ) + [None] 

envs = ['stable','random','all']

#tgt_inds_all = list( df_all['target_inds'].unique() )  + [None]

tgt_inds_all =  [None]
if args.do_per_tgt:
    tgt_inds_all += list(df_all['target_inds'].unique() )

#from figure.plot_behav2 import getSubDf, getColn, computeErrSensVersions
envs_cur = [ 'all']
if args.do_per_env:
    envs_cur += ['stable', 'random']
block_names_cur = ['all']
pertvals_cur = [None]
gseqcs_cur = [ (0,1) ]
tgt_inds_cur = tgt_inds_all
dists_rad_from_prevtgt_cur = [None]
dists_trial_from_prevtgt_cur = [None]
error_type = 'MPE'  # observed - goal, motor performance error

#%debug
#try:
if args.do_calc_ES:
    df_all_multi_tsz, ndf2vn = computeErrSensVersions(df_all, envs_cur,
        block_names_cur,pertvals_cur,gseqcs_cur,tgt_inds_cur,
        dists_rad_from_prevtgt_cur,dists_trial_from_prevtgt_cur,
        coln_nh = 'non_hit_not_adj',
        coln_nh_out = 'non_hit_shifted',
        computation_ver='computeErrSens3',
        subj_list = subjects[:args.n_subjects], error_type=error_type,
        trial_shift_sizes = np.arange(1, args.trial_shift_size_max + 1),
                 addvars=[], use_sub_angles = use_sub_angles, 
            retention_factor = retention_factor,
            reref_target_locs = args.reref_target_locs, 
        coln_error=args.coln_error, coln_correction_calc = args.coln_correction_calc)

    #assert not df_all_multi_tsz.duplicated().any() 
    assert not df_all_multi_tsz.duplicated(['subject','trials','trial_group_col_calc','trial_shift_size','retention_factor_s']).any()
#except Exception as e:
#    exc_info = sys.exc_info()
#    exc = traceback.TracebackException(*exc_info, capture_locals=True)
#    
#    ei = exc_info[2]    
#    #display(ei.tb_frame)
#    psf = ei
#
#    lfprev = None; lf = None
#    lfs = []
#    di = 0
#    while psf.tb_frame is not None:
#        stackframe = psf.tb_frame        
#
#        psf = psf.tb_next  
#        if ('conda' not in stackframe.f_code.co_filename) and\
#            ('mamba' not in stackframe.f_code.co_filename):        
#            display(di, stackframe)
#            lfprev = lf
#            lf = stackframe.f_locals
#            lfs += [lf]
#            di += 1
#        if psf is None:
#            break
#        
#    excsfmt = traceback.format_exc()
#    print(excsfmt)


    # dirty hack
    df_all_multi_tsz['err_sens'] = -df_all_multi_tsz['err_sens']
    df_all_multi_tsz['prev_err_sens'] = -df_all_multi_tsz['prev_err_sens']

    fn = f'df_all_multi_tsz_{args.save_suffix}.pkl.zip'
    fnf = pjoin(path_data,fn)
    print(fnf)
    if args.do_save:
        df_all_multi_tsz.to_pickle(fnf, compression='zip')

        df_all_multi_tsz.query('subject == @subjects[0]').\
            to_pickle(pjoin(path_data,'df_ext_onesubj.pkl.zip'),
                  compression='zip')

##############################

df_ = df_all_multi_tsz.query('trial_shift_size == 1 and trial_group_col_calc == "trials" and retention_factor_s == "1.000"')
assert not df_.duplicated(['subject','trials']).any()


##############################

if args.do_plot:
    df_ = truncateDf(df_, q=0,infnan_handling='discard',coln='err_sens' )
    #plt.figure()
    me = df_.groupby(['subject','environment'], observed=True).\
        mean(numeric_only=1).reset_index()
    #me.groupby(['env','thr']).size()
    sns.set(font_scale=1.3)
    fg = sns.catplot(data = me, kind='violin', y='err_sens', 
        hue = 'environment', x='environment',  palette = ['tab:orange', 'tab:grey'], legend=None)
    #order=[0,1],  

    #addTitleInfo(fg.axes.flatten()[0])
    for ax in fg.axes.flatten():
        ax.axhline(y=0, c='r', ls=':'); #ax.set_ylim(-5,5)

    from figure.mystatann import plotSigAll
    from config2 import path_fig

    plotSigAll(ax, 0.8, 0.05, ticklen=0.02,
           df=me, coln='err_sens', colpair = 'environment')


    from behav_proc import comparePairs 
    ttrssig, ttrs = comparePairs(df_, 'err_sens', 'environment')
    assert ttrssig.query('ttstr == "0.0 > 1.0" and not pooled')['pval'].iloc[0] < 0.05

    ##############################

    fnfig = pjoin(path_fig, f'test_ES_mean.pdf')
    plt.savefig(fnfig)

from config2 import n_jobs, freq_name2freq, stage2time_bounds
from config2 import subjects
from config2 import path_code
import numpy as np
import os

script = 'spoc_slide2.py'

#rts = ['Ridge', 'xgboost']
#hpasses = ['no_filter', '0.1', '0.05']
rts = ['Ridge']
#hpasses = ['no_filter', '0.1', '0.05'] 
hpasses = ['no_filter'] 

#shift = 0.25
dur   = 0.464
#shift = dur / 2
shift = dur / 4

#dur   = 0.5
#tmins = np.arange(-3,-0.5+shift,shift)
#analysis_name = 'sliding_prevmovement_preverrors_errors_prevbelief'


#epochs_bsl = Epochs(raw, events, event_id=event_ids_tgt,
#                    tmin=-0.46, tmax=-0.05, preload=True,
#                    baseline=None, decim=6)

# rts = rts[:1]
# hpasses = hpasses[:1]
#, decim=6)
# decim=6)

envs = ['stable', 'random']
seed = 0

#ipy = get_ipython()

#pars = []
runstrings = []
# when freq is outside subject I could re-use filterd raws
ind_glob = 0
run      = 'python ' + os.path.join( path_code, script ) + ' ' 
run_test = 'ipython -i ' + os.path.join( path_code, script ) + ' -- ' 

for hpass in hpasses:
    for control_type in ['movement']:
        for time_locked in ['target', 'feedback']:
            start, end = stage2time_bounds[time_locked]
            tmins = np.arange(start,end,shift)
            tmaxs = dur + tmins
            #tminmax = list(zip(tmins,tmaxs))
            for freq_name, freq_limits in freq_name2freq.items():
                ind_loc_start = ind_glob
                for subject in subjects:
                    s = run
                    #for tmin,tmax in tminmax:
                    #for regression_type in rts:
                    # for env_to_run in envs:

                    s += f' --param_file spoc_sliding.ini'
                    s += f' --random_seed {seed}'
                    s += f' --output_folder spoc_sliding_{hpass}'
                    s += f' --hpass {hpass}'

                    s += f' --slide_window_shift {shift}'
                    s += f' --slide_window_dur {dur}'
                    s += f' --slide_windows_type auto'
                    #s += f' --tmin ' + ','.join( map(str,tmins) )
                    #s += f' --tmax ' + ','.join( map(str,tmaxs) )
                    s += f' --subject {subject}'
                    s += f' --time_locked {time_locked}'
                    s += f' --control_type {control_type}'
                    s += f' --regression_type ' + ','.join(rts)
                    s += f' --freq_name {freq_name}'
                    s += f' --freq_limits ' + f'{freq_limits[0]},{freq_limits[1]}'
                    s += f' --env_to_run ' + ','.join(envs)
                    #s += 'analysis_name'] = analysis_name
                    s += '\n'
                    runstrings += [s]

                    #par = {}
                    #par['script']=script
                    #par['decim_epochs']=2
                    #par['n_jobs']=n_jobs
                    #par['use_preloaded_raw'] = 0


                    #par['hpass'] =hpass
                    #par['output_folder'] = f'spoc_sliding_{hpass}'

                    #par['tmin']=','.join( map(str,tmins) )
                    #par['tmax']=','.join( map(str,tmaxs) )
                    #par['subject']=subject
                    #par['time_locked'] = time_locked
                    #par['control_type'] = control_type
                    #par['regression_type'] = ','.join(rts)
                    #par['freq_name'] =freq_name
                    #par['freq_limits'] =freq_limits
                    #par['env_to_run'] = ','.join(envs)
                    ##par['analysis_name'] = analysis_name
                    #par['safety_time_bound'] = 0
                    #pars += [par]
                    ind_glob += 1


#def par2runstring(par):
#    from config2 import path_code
#    import os
#    s = 'python '
#    s += os.path.join( path_code, par['script'] ) + ' ' 
#    for pn,pv in par.items():
#        if pn == 'script':
#            continue
#        s = s + f'--{pn} {pv}; '


#            # then we want to clear (after the tasks are finished of course)
#            # so it would be some delayed execution... idk
#            par = {'_action':'delete'}
#            par['freq_name'] =freq_name
#            par['hpass'] =hpass
#            par['ind_range'] = ind_loc_start,ind_glob  # [a,b)
#            pars += [par]


#ss = []
#for par in pars:
#    s = ''
#    for pn,pv in par.items():
#      s = s + f'{pn}={pv}; '
#    s += '\n'
#    ss += [s]
#
#with open('__runpars.txt','w') as f:
#    f.writelines(ss)

runstrings = [ runstrings[0].replace(run,run_test) ] + runstrings

with open('_runstrings.txt','w') as f:
    f.writelines(runstrings)

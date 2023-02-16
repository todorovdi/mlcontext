from config2 import n_jobs,freq_name2freq
from config2 import subjects

script = 'spoc_home_with_prev_error2.py'

rts = ['Ridge', 'xgboost']
hpasses = ['no_filter', '0.1', '0.05']

#rts = rts[:1]
#hpasses = hpasses[:1]

pars = []
# when freq is outside subject I could re-use filterd raws
ind_glob = 0
for hpass in hpasses:
    for freq_name,freq_limits in freq_name2freq.items():
        ind_loc_start = ind_glob
        for subject in subjects:
            for regression_type in  rts:
                for env_to_run in ['stable', 'random']:
                    par = {}
                    par['script']=script
                    par['subject']=subject
                    par['n_jobs']=n_jobs
                    par['hpass'] =hpass
                    par['regression_type'] =regression_type
                    par['freq_name'] =freq_name
                    par['freq_limits'] =freq_limits
                    par['env_to_run'] =env_to_run
                    pars += [par]
                    ind_glob += 1
        # then we want to clear (after the tasks are finished of course)
        # so it would be some delayed execution... idk
        #par = {'_action':'delete'}
        #par['freq_name'] =freq_name
        #par['hpass'] =hpass
        #par['ind_range'] = ind_loc_start,ind_glob  # [a,b)
        #pars += [par]


ss = []
for par in pars:
    s = ''
    for pn,pv in par.items():
      s = s + f'{pn}={pv}; '
    s += '\n'
    ss += [s]

with open('__runpars.txt','w') as f:
    f.writelines(ss)


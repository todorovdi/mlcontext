import numpy as np
import sys
from scipy.optimize import minimize
import os
from joblib import Parallel, delayed



def calcH_min(pert, errors, EC_mask, pre_break_duration, err_sens = None,
            pause_mask = None, reg = 0,
             target_loc= 0, timeout = 30, 
             optimize_initial_err_sens = True,
             fit_to = 'errors', fitmask = None,
             online_feedback = False, verbose=0, cap_err_sens = True,
             use_true_errors = False, num_bases = 10, 
              custom_initial_err_sens = 0.,
              small_err_thr = 0., err_handling = 'raise', seed=0,
              optim_pars = None):
    # for one IC

    if online_feedback:
        raise ValueError('not implemented yet')

    # weight_forgetting_exp_pause is starting
    weight_retention_pause0 = 0.9

    error_region_size = np.abs(pert).max() * 1.2
    assert error_region_size > 1e-10
    error_region = [ -error_region_size, error_region_size]

    if fit_to == 'err_sens': 
        assert err_sens is not None

    from collections import OrderedDict as odict

    assert optimize_initial_err_sens in ['optimize', 'from_data', 'custom']

    args_opt = odict ([])
    if optimize_initial_err_sens == 'optimize':
        args_opt['initial_sensitivity'] = (0.0001,0.5)

    args_opt.update( odict ([
        #('initial_sensitivity', (0.0001,0.5)),
        ('eta', (0.001,0.2)),  # weight update speed
        ('alpha', (0.5,1)) ,  # retention
        ('alpha_w',   (0.3,1 ) ),
    ]) )
    if (pause_mask is not None) or (np.max(pre_break_duration) > 1e-3 ) :
        print('set pause')
        args_opt['weight_forgetting_exp_pause'] = (-2,  2)

    #   ('gaussian_variance', (0.1,5)), # of gaussians
    #   ('execution_noise_variance', (0, error_region_size / 2 )),
    #   ('sensory_noise_variance',   (0, error_region_size / 2))

    #print(args_opt)
    pnames,pbounds = zip(*args_opt.items())
    #print(pnames)
    #print(pbounds)

    #x0 = np.array([0.1, 1, 0.01, 0.9, 0.01, 0.01]) # You can change these values as you like
    #x0 += np.array([1,1,0,1,1,1] ) * np.random.uniform(-1,1,size=len(x0) ) * x0 / 5

    x0 = []
    for lower, upper in pbounds:
        x = np.random.uniform(lower, upper)
        x0.append(x)
    x0 = np.array(x0)

    from pprint import pprint
    #print('x0 = ')
    #pprint(list( (zip(pnames, x0, pbounds ) ) ) )

    #from config2 import FormatPrinter
    #fmt = FormatPrinter({float:'%.2f'})
    #fmt.pprint(list( (zip(pnames, x0, pbounds ) ) ) )

    from scipy.linalg import LinAlgError

    errors_ = interpnan(errors.copy())

    import time
    global start
    start = time.time()

    #use_true_errors = fit_to == 'err_sens'
    if use_true_errors:
        print('Use true errors ON')
        assert errors is not None, 'true_errors is None when it should not be'

    if optimize_initial_err_sens == 'from_data':
        es = err_sens[ list(range(5,12)) + list(range(192, 192+24)) ]
        es = es[~np.isinf(es)]
        ES0_ = np.mean(es)
        if cap_err_sens:
            ES0_ = max(0, min(1,ES0_) )
    elif optimize_initial_err_sens == 'custom':
        ES0_ = custom_initial_err_sens

    def _minim_H(pars):
        #if pause_mask is not None:
        if optimize_initial_err_sens == 'optimize':
            if 'weight_forgetting_exp_pause' in args_opt:
                (ES0, eta, alpha, alpha_w, weight_forgetting_exp_pause ) = pars
                pause_mask_ = pause_mask
            else:
                (ES0, eta, alpha, alpha_w ) = pars
                weight_forgetting_exp_pause = weight_retention_pause0
                pause_mask_ = np.zeros_like(pert, dtype=bool)
        else:
            if 'weight_forgetting_exp_pause' in args_opt:
                (eta, alpha, alpha_w, weight_forgetting_exp_pause ) = pars
                pause_mask_ = pause_mask
            else:
                (eta, alpha, alpha_w ) = pars
                weight_forgetting_exp_pause = weight_retention_pause0
                pause_mask_ = np.zeros_like(pert, dtype=bool)

            ES0 = ES0_
        sensory_noise_variance = 0
        execution_noise_variance = 0
 
        try:
            # pert is true pert
            # compare errors (meas in the end) with current errors on same positions
            # on each trial predict motor output on next trial
            # error pred N is then computed from observed pert at N at predicted on prev trial motor output
            # one can assume that update of ES happens in the end of the trial
            # has execution noise (to get error) and sensory noise (to get motor output pred). Yes, not inversely
            r = simMoE( pert, alpha=alpha,
                           eta=eta, initial_sensitivity=ES0,
                sensory_noise_variance =sensory_noise_variance,
                execution_noise_variance = execution_noise_variance,
                gaussian_variance = None, alpha_w = alpha_w,
                weight_forgetting_exp_pause = weight_forgetting_exp_pause,
                pre_break_duration = pre_break_duration,
                EC_mask = EC_mask, error_region = error_region,
                           target_loc = target_loc, pause_mask = pause_mask_,
                          use_true_errors = use_true_errors, cap_err_sens = cap_err_sens,
                            true_errors=errors, num_bases = num_bases,
                       small_err_thr = small_err_thr, seed=seed)
            #np.ascontiguousarray(pertur
            #print('aaa')
            (motor_output,error_pred,ws,err_sens_pred,
            gaussian_centers,gaussian_variances) = r

            #assert not np.isnan(motor_output).any() 
            #assert not np.isnan(error_pred).any() 

            if np.isnan(error_pred).any() :
                return np.nan

            if fitmask is None:
                mse1 = np.mean( ( errors_ - error_pred ) **2 )
                ges = ~np.isinf(err_sens)
                mse2 = np.mean( ( err_sens_pred[ges] - err_sens[ges] ) **2 )
            else:
                assert len(fitmask) == len(errors_)
                mse1 = np.mean( ( errors_[fitmask] - error_pred[fitmask]) **2 )

                ges = ~np.isinf(err_sens[fitmask])
                mse2 = np.mean( ( err_sens_pred[fitmask][ges] - err_sens[fitmask][ges] ) **2 )

            if verbose > 0:
                print(f'mse1 = {mse1},  mse2 = {mse2}')
            if fit_to == 'errors':
                B = err_sens_pred
                pen = np.mean( np.sum(B[~np.isnan(B)]**2 ) ) * reg
                r = mse1 + pen
            elif fit_to == 'err_sens':
                r = mse2            
            elif fit_to == 'err_sens_smooth':
                r = mse2_sm            
            elif fit_to == 'errors_and_err_sens':
                r = mse1 + mse2            
            elif fit_to == 'errors_and_err_sens_smooth':
                r = mse1 + mse2_sm            
            else:
                raise ValueError(f'wrong fit target {fit_to}')
            #print('loss = {:.3f}  (mse={:.3f}   pen={:.3f}'.format(r, mse, pen))
        except LinAlgError as e:
            print('Linalg error')
            r = np.nan

        global start # Use the global variable for the start time
        elapsed = time.time() - start # Elapsed time in seconds
        if elapsed > timeout:
            print('time is up')
            return np.nan
            #raise ValueError('time is up')

        return r


    if optim_pars is None:
        #tol = 1e-6
        #step = 1e-5
        tol = 1e-7
        step = 1e-6
        minim_disp =  1
        opts = {'method': 'SLSQP', 'tol':tol,
                'options':{'maxiter':3000, 'eps':step,
                    'disp':minim_disp, 'ftol':tol}}
    else:
        opts = optim_pars

    #opts = {'method': 'Nelder-Mead', 'tol':tol,
    #        'options':{'maxiter':3000, 'disp':1}}

    # Define a simple function for the callback
    #def stopper(xk=None, max_sec=40):
    #    global start # Use the global variable for the start time
    #    elapsed = time.time() - start # Elapsed time in seconds
    #    if elapsed > max_sec:
    #        raise ValueError('time is up')
    #        warnings.warn("Terminating optimization: time limit reached", UserWarning) # Raise a warning if the time limit is reached
    #    #else:
    #    #    print("Elapsed: %.3f sec" % elapsed) #

    result = minimize(_minim_H, x0, bounds=pbounds, **opts)
      #                callback = stopper)
                        #  args=(errors, args))
    if result.success:
        d = dict(zip(pnames,result.x) )
        print('opt finished with:', d )

        if optimize_initial_err_sens != 'optimize':
            d['initial_sensitivity'] = ES0_
    else:
        d = None
        print('opt finished None :(')
    return result, d

def simMoE(pert, alpha=1. , eta=0.01, initial_sensitivity= 0.1,
              sensory_noise_variance = 0.,
              execution_noise_variance = 0.,
               gaussian_variance = None, alpha_w = 1.,
               EC_mask = None, error_region = None,
            weight_forgetting_exp_pause = 1e-3,
              num_bases = 10, target_loc = 0, pause_mask = None,
              pre_break_duration = None, true_errors = None,
               use_true_errors = False, cap_err_sens = True,
           small_err_thr = 0., seed= 0):
    '''
    this is a proxy function needed to bridge normal python and numba code
    '''
    np.random.seed(seed)

    if error_region is None:
        error_region_ = np.abs(pert).max() * 1.2
        error_region = [ -error_region_, error_region_]
        print('error_region = ' ,error_region)

    if use_true_errors: 
        assert true_errors is not None, 'true_errors is None when it should not be'

    if gaussian_variance is None:
        gaussian_variance = (error_region[1] - error_region[0]) / num_bases

    if num_bases == -1:  # auto select depending on max error size
        if error_region[1] > 90:
            num_bases = 20
        else:
            num_bases = 10

    args = { 'num_bases': num_bases,
        'error_region': np.array(error_region),
        'initial_sensitivity': initial_sensitivity,
        'gaussian_variance': gaussian_variance,
        #'variance': gaussian_variance, # of gaussians
        'eta': eta,  # weight update speed
        'alpha': alpha,  # retention
        'alpha_w' :alpha_w, # retention of weights
        'execution_noise_variance': execution_noise_variance,
        'sensory_noise_variance': sensory_noise_variance,
        'target_loc': target_loc,
        'weight_forgetting_exp_pause':weight_forgetting_exp_pause,
        'use_true_errors':use_true_errors,
        'cap_err_sens':cap_err_sens,
        'perturbations':pert,
            'small_err_thr':small_err_thr}

    if pre_break_duration is not None:
        args['pre_break_duration'] = np.ascontiguousarray( pre_break_duration, dtype=np.int32)                  
    else:
        args['pre_break_duration'] = np.ascontiguousarray( np.zeros(len(pert)), dtype=np.float64)

    if EC_mask is not None:
        args['channel_mask'] =  EC_mask
    else:
        args['channel_mask'] = np.ascontiguousarray( np.zeros(len(pert)), dtype=np.int32)

    if pause_mask is not None:
        args['pause_mask'] =  pause_mask
    else:
        args['pause_mask'] = np.ascontiguousarray( np.zeros(len(pert)), dtype=np.int32)

    if true_errors is not None:
        args['true_errors'] = np.ascontiguousarray( true_errors, dtype=np.float64)
    else:
        args['true_errors'] = np.ascontiguousarray( np.zeros(len(pert)), dtype=np.float64)

    ## for debug
    #for an,av in args.items():
    #    print(an,type(av))

    # 'zeta': 0.9, # for markov model
    #'perturbations': [],

    #    motor_output[i+1] = args['alpha'] * motor_output[i] -\
    #        s * (errors[i] + np.random.randn() * \
    #             args['sensory_noise_variance'])

    #r = Hcalc(args)

    #from taumodels.herzfeld_model import calc as Hcalc
    try:
        from herzfeld_model import calc as Hcalc
    except ModuleNotFoundError:
        from taumodels.herzfeld_model import calc as Hcalc
    r = Hcalc(**args)

    #(motor_output,error_pred,ws,err_sens,
    # gaussian_centers,gaussian_variances) = r
    return r

#def MSE_H_orig(errors, pert, alpha, eta, ES0 ):
#    r = calcH_orig( pert)
#    (motor_output,error_pred,ws,err_sens,
#     gaussian_centers,gaussian_variances) = r
#    return np.mean( ( errors - error_pred ) **2 )

def _minim_T(nbT,startIdx, a0,c0,b0,x00, state_retention_pause0, optbounds, opts,
           perturb,error, pause_mask, bclip,  modelBeforeSwitch , reg,
             inds_state_reset, stats_powers, use_true_error, use_true_error_stats, motor_var):
    # this function is ran inside parallel  that is why it is not internal
    # Define a lambda function to pass to minimize function
    # optimize first 4 params: A,c,Bstart,X0.
    # retention, coef, start lr, start state
    import taumodels.state_space_Tan2014.error_model_tanSS as tan

    p = os.path.join( os.path.expandvars('$CODE_MEMORY_ERRORS'), 'state-space-adaptation' )
    sys.path.append(p)

    #print('x0 = ', dict(zip(parnames0, x0 ) ) )

    #print('_minim {} {};'.format(nbT,startIdx) )
    f = lambda P: tan.MSE_error_model_tan(P,nbT,startIdx,
            modelBeforeSwitch,perturb,error, pause_mask, clip=bclip, reg=reg, inds_state_reset=inds_state_reset, stats_powers=stats_powers,
        use_true_error=use_true_error,
        use_true_error_stats =use_true_error_stats,
                                        motor_var = motor_var  )
    if pause_mask is None:
        IC = [a0,c0,b0,x00]
    else:
        IC = [a0,c0,b0,x00,state_retention_pause0],

    res = minimize(f,IC, bounds=optbounds,**opts)

    if np.isnan(res.fun ):
        return nbT,startIdx, None
    return nbT,startIdx, res

def interpnan(arr):
    nas = np.isnan(arr); _fnz = np.flatnonzero
    arr[np.isnan(arr)] = np.interp(_fnz(nas),_fnz(~nas),arr[~nas])
    return arr

def fitTanModel(dfos, coln_error, n_jobs, bclip = (-0.5,0.5),
               modelBeforeSwitch = 'UnkAdapt', pause_mask = None,
                nruns = 3, reg = 0, neg_error = False,
                maxNbTrials = 30, maxStartIdx = 12,
                minNbTrials = 3, minStartIdx = 3,
                cmin = 1e-6, cmax=0.9,
                amin = 0.2, amax = 1. - 1e-6,
                online_feedback=True,
                use_true_error=False, use_true_error_stats=True,
               inds_state_reset = [], stats_powers=(2,-2),
                motor_var = 0.):
    # modelBeforeSwitch = 'NoAdapt'  # then ignore Bstart



    import taumodels.state_space_Tan2014.error_model_tanSS as tan
    subjs = dfos['subject'].unique()
    assert len(subjs) == 1
    subj = subjs[0]
    # Parameters initialization
    #minNbTrials = 3 # minimum number of "retained" trials
    #maxNbTrials # maxumum number of "retained" trials

    #minStartIdx = 3 # minimum index of first trial where previous trials are taken into account
    #maxStartIdx = 12 # maximum index of first trial where previous trials are taken into account
    #a0 = 0.95 # initial retention factor
    #b0 = 0.01 # initial adaptation factor (for trials before startIdx)
    #x00 = 0 # initial initial state

    #c0 = 0.001 # initial c coefficient (that is multiplied by the ratio of mean error and error variance)
    #amin = 0.3 # minimum retention factor
    #amax = 1 # maximum retention factor
    bmin = 0 # minimum starting adaptation rate
    bmax = bclip[1] # maximum starting adaptation rate
    x0min = -30 # minimum initial state
    x0max = 30 # maximum initial state
    state_retention_pause0 = 0.9

    optbounds = [(amin,amax),(cmin,cmax),(bmin,bmax),(x0min,x0max)]
    if pause_mask is not None:
        optbounds += [ (0.3 , 1. ) ]

    if pause_mask is not None:
        parnames0 = 'A,c,Bstart,X0,state_retention_pause'.split(',')
    else:
        parnames0 = 'A,c,Bstart,X0'.split(',')


    # Set the optimization options as a dictionary

    step = 1e-8
    opts = {'method': 'SLSQP', 'tol':1e-10,
            'options':{'maxiter':3000, 'eps':step, 'disp':0} }

    # A, c, b_start, nbT, x0 and error (eventually add startIdx)
    # nbT = NbPreviousTrials
    #subj2out_Tan = {}
    #Q: interpolate NaN trials -- is it good?
    # measured_error has to have shape  trials x subjects
    # loop on the number of subjects
    # for each subject optimization of the model is made separtely
    #for s in dfc['subject'].unique():
    out_cursubj = {}
    #subj2out_Tan[s] = out_cursubj
    error = dfos.query('subject == @subj')[coln_error].to_numpy()

    if online_feedback:
        perturb = dfos.query('subject == @subj')['perturbation'].to_numpy()
        #perturb_cursubj        = dfos['perturbation'].values
    else:
        perturb = dfos.query('subject == @subj')['perturbation'].shift(1).to_numpy()
        #perturb_cursubj        = dfos['perturbation'].shift(1).values
        perturb[0] = 0

    if neg_error:
        error = -error
    assert not np.isnan(perturb).any()
    print('Num NaNs in errors  = ' , np.sum( np.isnan(error) ) )
    # Interpolate NaNs using numpy indexing and numpy.interp
    #nas = np.isnan(error); _fnz = np.flatnonzero
    #error[np.isnan(error)] = np.interp(_fnz(nas),_fnz(~nas),error[~nas])
    error = interpnan(error)
    print('error.shape = ',error.shape)
    assert not np.isnan(error).any()
    minfval = np.inf
    minres = None
    idx = 0
    args = []
    for nbT in range(minNbTrials, maxNbTrials+1):
        for startIdx in range(minStartIdx, maxStartIdx+1):
            x0 = []
            for lower, upper in optbounds:
                x = np.random.uniform(lower, upper)
                x0.append(x)

            if pause_mask is None:
                x0.append(state_retention_pause0)

            x0 = np.array(x0)
            args += [(nbT,startIdx,*x0) ]

    args = nruns * args
    print(f'Start parallel over nbTrials and startIdx, {len(args) } args over {n_jobs} workers')
    #a0,c0,b0,x00,state_retention_pause0
    if n_jobs > 1:
        plr = Parallel(n_jobs=n_jobs, backend='multiprocessing',
                       )(delayed(_minim_T)(*arg,
                        optbounds,opts,
                        perturb,error, pause_mask, bclip,
                        modelBeforeSwitch, reg,
                       inds_state_reset, stats_powers,
                        use_true_error, use_true_error_stats,
                                           motor_var) \
                            for arg in args)
    else:
        plr = [ _minim_T(*arg, optbounds,opts,
                        perturb,error, pause_mask, bclip,
                        modelBeforeSwitch, reg,
                       inds_state_reset, stats_powers,
                        use_true_error, use_true_error_stats,
                        motor_var ) for arg in args  ]


    for r in plr:
        nbT,startIdx,res = r
        #print(nbT,startIdx, res.fun)
        if res.fun < minfval:
            minres = res
            minfval = res.fun
            # only defined if come inside this if
            parvec = res.x
            nbT_opt = nbT
            startIdx_opt = startIdx

    pars = [*parvec, nbT_opt, startIdx_opt]


    parnames = parnames0 + 'NbPreviousTrials,StartIdx'.split(',')
    nbOptParams = len(parnames)
    pars_  = dict(zip(parnames,pars) )
    out_cursubj['params'] = pars_

    # run simulation
    #print(pars)
    r = tan.error_model_tan(*pars,
                         modelBeforeSwitch,perturb,error,
                            motor_var=motor_var)
    output_Tan, state_Tan, adaptationRate_Tan = r
    out_cursubj['states'] = state_Tan
    out_cursubj['y_pred'] = output_Tan
    out_cursubj['optres'] = minres
    out_cursubj['adaptation_rate'] = adaptationRate_Tan

    print(minres)
    #print('adaptationRate_Tan.shape = ',adaptationRate_Tan.shape)

    out_cursubj['y_pred'] = output_Tan
    out_cursubj['adaptation_rate'] = adaptationRate_Tan

    sigma2 = np.var(output_Tan - error)

    # to make formula shorter
    N = len(error)
    log_2pi = np.log(2*np.pi)
    nops = nbOptParams

    likelihood = -1/(2*sigma2)*minfval - (N/2)*np.log(sigma2) - (N/2)*log_2pi
    AIC = 2*nops - 2*likelihood + (2*nops*(nops+1))/(N - nops - 1)

    out_cursubj['mismatch_var'] = sigma2
    out_cursubj['minfval'] = minfval
    out_cursubj['AIC'] = AIC
    out_cursubj['reg'] = reg
    out_cursubj['bclip'] = bclip
    out_cursubj['pause_mask'] = pause_mask
    out_cursubj['online_feedback'] = online_feedback
    out_cursubj['use_true_error'] = use_true_error
    out_cursubj['use_true_error_stats'] = use_true_error_stats
    out_cursubj['stats_powers'] = stats_powers
    out_cursubj['nruns'] = nruns
    out_cursubj['maxNbTrials'] =  maxNbTrials
    out_cursubj['minNbTrials'] =  minNbTrials
    out_cursubj['maxStartIdx'] =  maxStartIdx
    out_cursubj['minStartIdx'] =  minStartIdx
    out_cursubj['cmin'] =         cmin
    out_cursubj['cmax'] =         cmax
    out_cursubj['amin'] =         amin
    out_cursubj['amax'] =         amax
    out_cursubj['motor_var'] =    motor_var

    return out_cursubj

def _minim_D():
    return 0

def fitDiedrischenModel(dfos, coln_error, linalg_error_handling = 'ignore', nruns=1, online_feedback = True):
    # n_jobs == 1 always here

    from taumodels.state_space_fit_toolbox.state_space_fit import state_space_fit
    from scipy.optimize import minimize # Import the minimize function from scipy.optimize module

    # we'll shift it later if offline feedback but we need to calc beh with unshifted
    perturb_cursubj0        = dfos['perturbation'].values
    measured_error         = dfos[coln_error].values
    perturb_cursubj        = dfos['perturbation'].values
    #if online_feedback:
    #    perturb_cursubj        = dfos['perturbation'].values
    #else:
    #    perturb_cursubj        = dfos['perturbation'].shift(1).values
    #    perturb_cursubj[0] = 0
    behavior_cursubj = perturb_cursubj0 - measured_error
    #behavior_cursubj = perturb_cursubj - measured_error; behavior_cursubj[0] = 0

    #                              a0   b0   x00
    #params_init = np.array([   1,  0.1,  0])[None,:]
    #                             amin amax bmin bmax x0min x0max
    #search_space  = np.array([   1,   1,   0, 0.5,  -15, 15])[None,:]
    search_space  = np.array([   0.3,   1,   0, 0.8,  -45, 45])[None,:]

    parnames0 = 'A,B,X0'.split(',')
    optbounds = list(zip( search_space[0,::2], search_space[0,1::2])  )


    # optimization options

    #opts = {'method': 'SLSQP', 'tol':1e-10, 'options':{'maxiter':3000}} # Set the optimization options as a dictionary
    #nbOptParams = len(parnames0)

    #modeltype2res = {}
    #behavior[:,s] = measured_error[:,s] # Use numpy indexing to get a column of an array
    #nas = np.isnan(behavior_cursubj)
    #_fnz = np.flatnonzero
    #One-dimensional linear interpolation for monotonically increasing sample points.
    # np.interp 1st s args
    #xarray_like
    #The x-coordinates at which to evaluate the interpolated values.

    #xp1-D sequence of floats
    #The x-coordinates of the data points, must be increasing if argument period is not specified. Otherwise, xp is internally sorted after normalizing the periodic boundaries with xp = xp % period.

    #fp1-D sequence of float or complex
    #The y-coordinates of the data points, same length as xp.
    #behavior_cursubj[nas] =\
    #    np.interp(_fnz(nas),_fnz(~nas),behavior_cursubj[~nas]) # Interpolate NaNs using numpy indexing and numpy.interp

    behavior_cursubj = interpnan(behavior_cursubj)

    assert not np.isnan(behavior_cursubj).any()

    #print(perturb_cursubj[190:200])
    # Fit a one-state model without retention factor (Diedrieschen, 2003),
    # using a deterministic lmse approach
    #print( params_init_Diedrischen, search_space_Diedrischen )
    plr = []
    for i in range(nruns):
        x0 = []
        for lower, upper in optbounds:
            x = np.random.uniform(lower, upper)
            x0.append(x)
        x0 = np.array(x0)
        #print('x0 = ', dict(zip(parnames0, x0 ) ) )
        params_init = x0[None,:]
        #print(params_init.shape)

        r = state_space_fit(behavior_cursubj, perturb_cursubj,
                params_init, search_space,'lmse', 'norm',
                linalg_error_handling = linalg_error_handling,
                sep_err_begend = not online_feedback)
        plr += [r]

    #plr = Parallel(n_jobs=n_jobs, backend='multiprocessing',
    #               )(delayed(_minim)(*arg, *x0,
    #                optbounds,opts,
    #                perturb,error, pause_mask, bclip, modelBeforeSwitch, reg) for arg in args)

    minfval = np.inf
    # when using lmse noises_var = np.var(output - behavior)
    for r in plr:
        #nbT,startIdx,res = r
        #print(nbT,startIdx, res.fun)
        output, params, asymptote, noises_var, states, performances  = r
        AICc, mse = performances
        if mse < minfval:
            minfval = mse
            # only defined if come inside this if
            ropt = r

    #pars = [*parvec, nbT_opt, startIdx_opt]




    #outs_ output, params, asymptote, noises_var, states, performances
    # asymptote is not very useful for me
    output, params, asymptote, noises_var, states, performances  = ropt
    d = {'output': output,
     'params': params,
     'asymptote': asymptote,
     'noises_var': noises_var,
     'states': states,
     'performances': performances,
         'nruns':nruns,
         'online_feedback':online_feedback}

    dp = dict(list(zip( 'A,B,X0'.split(','), params[:,0] )))
    print(dp)

    return d

def fitAlbertModel(dfos, coln_error, linalg_error_handling = 'ignore',
                  use_mex = 0  ):
    from taumodels.state_space_fit_toolbox.state_space_fit import state_space_fit
    from scipy.optimize import minimize # Import the minimize function from scipy.optimize module
    perturb_cursubj        = dfos['perturbation'].values
    measured_error = dfos[coln_error].values
    behavior_cursubj = perturb_cursubj - measured_error

    # Fit a two-state model with retention factor using stochastic EM approach
    # (Albert, 2018) optimizing parameters in log-space
    params_init =       np.array( [[0.95, 0.05,  0], # Use nested lists to create a 2D array in Python
                                [0.7,  0.3,  0]] )
    #                             amin amax bmin bmax x0min x0max
    search_space =       np.array( [[ 0.2,   1,   0, 0.5,    0,  0], # Use nested lists to create a 2D array in Python
                                [ 0.2,   1,   0, 0.5,  -15, 15]] )

    r = state_space_fit(behavior_cursubj, perturb_cursubj,
            params_init, search_space, 'em', 'log',
                        linalg_error_handling = linalg_error_handling,
                        use_mex = use_mex)

    output, params, asymptote, noises_var, states, performances  = r
    d = {'output': output,
     'params': params,
     'asymptote': asymptote,
     'noises_var': noises_var,
     'states': states,
     'performances': performances}

    d['y_pred'] = d['output']  # for compatibility

    #performances = [AICc, mse_opt] # mse_opt is the MSE for best params

    #modeltype2res['Albert'] = d

    return d

def getBestHerzPar(plr2):
    import pandas as pd
    rows = []
    nbads = 0
    for tpl in plr2:
        if tpl[1] is None:
            nbads += 1
            continue
        d = tpl[1].copy()
        d['fun'] = tpl[0].fun
        d['nit'] = tpl[0].nit
        d['nfev'] = tpl[0].nfev
        d['seed'] = tpl[2]
        rows += [d]

    print(f'getBestHerzPar: nbads={nbads} out of {len(plr2)}')
    dfrespar = pd.DataFrame(rows)#.describe()
    if len(dfrespar):
        #display( dfrespar.describe() )

        ind =  dfrespar['fun'].idxmin()
        #print(ind)
        best_par = dfrespar.loc[ind].to_dict()
        return best_par
    return None
        #print(best_par)

def _minim_S(nbT,startIdx, a0,c0,b0,x00, state_retention_pause0, optbounds, opts,
           perturb,error, pause_mask, bclip,  modelBeforeSwitch , reg ):
    # this function is ran inside parallel  that is why it is not internal
    # Define a lambda function to pass to minimize function
    # optimize first 4 params: A,c,Bstart,X0.
    # retention, coef, start lr, start state
    from taumodels.sugiyama import MSE as MSE_S

    p = os.path.join( os.path.expandvars('$CODE_MEMORY_ERRORS'), 'state-space-adaptation' )
    sys.path.append(p)

    #print('x0 = ', dict(zip(parnames0, x0 ) ) )

    #print('_minim {} {};'.format(nbT,startIdx) )
    f = lambda X: MSE_S(X,error, perturb, pause_mask,
            clip=bclip, reg=reg)
    if pause_mask is None:
        IC = [a0,c0,b0,x00]
    else:
        IC = [a0,c0,b0,x00,state_retention_pause0],

    res = minimize(f,IC, bounds=optbounds,**opts)

    if np.isnan(res.fun ):
        return nbT,startIdx, None
    return nbT,startIdx, res

def fiSugiyamaModel(dfos, coln_error,
    linalg_error_handling = 'ignore', nruns=1):
    # n_jobs == 1 always here

    from scipy.optimize import minimize # Import the minimize function from scipy.optimize module

    perturb_cursubj        = dfos['perturbation'].values
    measured_error = dfos[coln_error].values
    behavior_cursubj = perturb_cursubj - measured_error

    args = nruns * args
    print(f'Start parallel over nbTrials and startIdx, {len(args) } args over {n_jobs} workers')
    #a0,c0,b0,x00,state_retention_pause0
    plr = Parallel(n_jobs=n_jobs, backend='multiprocessing',
                   )(delayed(_minim_S)(*arg,
                    optbounds,opts,
                    perturb,error, pause_mask, bclip, modelBeforeSwitch, reg) for arg in args)

    for r in plr:
        nbT,startIdx,res = r
        #print(nbT,startIdx, res.fun)
        if res.fun < minfval:
            minres = res
            minfval = res.fun
            # only defined if come inside this if
            parvec = res.x
            nbT_opt = nbT
            startIdx_opt = startIdx

    pars = [*parvec, nbT_opt, startIdx_opt]


    parnames = parnames0 + 'NbPreviousTrials,StartIdx'.split(',')
    pars_  = dict(zip(parnames,pars) )
    out_cursubj['params'] = pars_

    r = tan.error_model_tan(*pars,
                         modelBeforeSwitch,perturb,error)
    output_Tan, state_Tan, adaptationRate_Tan = r
    out_cursubj['states'] = state_Tan
    out_cursubj['y_pred'] = output_Tan
    out_cursubj['optres'] = minres
    out_cursubj['adaptation_rate'] = adaptationRate_Tan

    print(minres)
    #print('adaptationRate_Tan.shape = ',adaptationRate_Tan.shape)

    out_cursubj['y_pred'] = output_Tan
    out_cursubj['adaptation_rate'] = adaptationRate_Tan

    sigma2 = np.var(output_Tan - error)

    # to make formula shorter
    N = len(error)
    log_2pi = np.log(2*np.pi)

    likelihood = -1/(2*sigma2)*minfval - (N/2)*np.log(sigma2) - (N/2)*log_2pi
    AIC = 2*nops - 2*likelihood + (2*nops*(nops+1))/(N - nops - 1)

    out_cursubj['minfval'] = minfval
    out_cursubj['AIC'] = AIC

    return out_cursubj


    return d

if __name__ == "__main__":
    import sys,os
    import numpy as np
    #sys.path.append(os.path.expandvars('$CODE_MEMORY_ERRORS'))
    #from taumodels.models import simMoE, calcH_min
    perturb = np.array([0,0,30])
    error = perturb
    EC_mask = np.array([0,0,0])
    pre_break_duration = np.array([0,0,0])
    reg = 0

    #%debug
    #_opt((perturb, error, EC_mask, pre_break_duration, reg))

    calcH_min(-perturb, error, EC_mask, pre_break_duration,
                                    reg=reg, timeout = 12)

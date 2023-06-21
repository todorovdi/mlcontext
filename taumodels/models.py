import numpy as np
import sys
from scipy.optimize import minimize
import os
from joblib import Parallel, delayed

from taumodels.herzfeld_model import calc as Hcalc


def calcH_min(pert, errors, EC_mask, pause_mask = None, reg = 0,
             target_loc= 0, timeout = 30 ):
    #reg = 1e-2
    # weight_retention_pause is starting       
    weight_retention_pause0 = 0.9
    parbounds = {}
    #pr = prod( listlists)

    error_region = np.abs(pert).max() * 1.2

    from collections import OrderedDict as odict

    args_opt = odict ([
        ('initial_sensitivity', (-0.5,0.5)),
        ('gaussian_variance', (0.1,5)), # of gaussians
        ('eta', (0.001,0.2)),  # weight update speed
        ('alpha', (0.5,1)) ,  # retention
        ('execution_noise_variance', (0, error_region / 2 )),
        ('sensory_noise_variance',   (0, error_region / 2)),
        ('alpha_w',   (0.3,1 ) )
    ])
    if pause_mask is not None:
        args_opt['weight_retention_pause'] = (0.2,1)

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

    def _minim_H(pars):
        if pause_mask is not None:
            (ES0, gaussian_variance, eta, alpha, execution_noise_variance,
             sensory_noise_variance, alpha_w, weight_retention_pause ) = pars
        else:
            (ES0, gaussian_variance, eta, alpha, execution_noise_variance,
             sensory_noise_variance, alpha_w ) = pars
            weight_retention_pause = weight_retention_pause0

        try:
            r = calcH_orig( pert, alpha=alpha, 
                           eta=eta, initial_sensitivity=ES0,
                sensory_noise_variance =sensory_noise_variance, 
                execution_noise_variance = execution_noise_variance,
                gaussian_variance = gaussian_variance, alpha_w = alpha_w, 
                weight_retention_pause = weight_retention_pause,
                EC_mask = EC_mask, error_region = error_region,
                           target_loc = target_loc, pause_mask = pause_mask)
            (motor_output,error_pred,ws,err_sens,
            gaussian_centers,gaussian_variances) = r
            mse = np.mean( ( errors_ - error_pred ) **2 )


            B = err_sens
            pen = np.mean( np.sum(B[~np.isnan(B)]**2 ) ) * reg
            r = mse + pen
            #print('loss = {:.3f}  (mse={:.3f}   pen={:.3f}'.format(r, mse, pen))
        except LinAlgError as e:
            r = np.nan

        global start # Use the global variable for the start time
        elapsed = time.time() - start # Elapsed time in seconds
        if elapsed > timeout:
            print('time is up')
            return np.nan
            #raise ValueError('time is up')

        return r


    tol = 1e-6
    #tol = 1e-7
    step = 1e-5
    minim_disp =  1
    opts = {'method': 'SLSQP', 'tol':tol,
            'options':{'maxiter':3000, 'eps':step, 
                'disp':minim_disp, 'ftol':tol}}

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
    else:
        d = None
        print('opt finished None :(')
    return result, d

def calcH_orig(pert, alpha=1. , eta=0.01, initial_sensitivity= 0.1,
              sensory_noise_variance = 0.,
              execution_noise_variance = 0.,
               gaussian_variance = 1., alpha_w = 1.,
               EC_mask = None, error_region = None,
            weight_retention_pause = 0.9,
              num_bases = 20, target_loc = 0, pause_mask = None ):
    if error_region is None:
        error_region = np.abs(pert).max() * 1.2
        print('error_region = ' ,error_region)
    args = { 'num_bases': num_bases,
        'error_region': error_region,
        'initial_sensitivity': initial_sensitivity,
        'variance': gaussian_variance, # of gaussians
        'eta': eta,  # weight update speed
        'alpha': alpha,  # retention
        'alpha_w' :alpha_w, # retention of weights
        'execution_noise_variance': execution_noise_variance,
        'sensory_noise_variance': sensory_noise_variance,
        'target_loc': target_loc,
        'pause_mask':pause_mask,
        'weight_retention_pause':weight_retention_pause,
        'perturbation':pert }
    if EC_mask is not None:
        args['channel_mask'] =  EC_mask

    # 'zeta': 0.9, # for markov model
    #'perturbations': [],

    #    motor_output[i+1] = args['alpha'] * motor_output[i] -\
    #        s * (errors[i] + np.random.randn() * \
    #             args['sensory_noise_variance'])

    args['perturbation']  = pert
    r = Hcalc(args)
    #(motor_output,error_pred,ws,err_sens,
    # gaussian_centers,gaussian_variances) = r
    return r

#def MSE_H_orig(errors, pert, alpha, eta, ES0 ):
#    r = calcH_orig( pert)
#    (motor_output,error_pred,ws,err_sens,
#     gaussian_centers,gaussian_variances) = r
#    return np.mean( ( errors - error_pred ) **2 )

class Herzfeld_model:
    # this is my implementation
    def __init__(self,ntrials,min_err = -10, max_err = 10, basis_len = 10,
                beta = 0.05, sigma = None, decay=1., err_sens_def=0.1,
                weights_def = None):
        #weights_def = 0.05
        # here everything is relative to the target.
        # So we assume that it does not depend on the target
        self.basis_err = np.linspace(min_err,max_err,basis_len)
        self.basis_len = len(self.basis_err)
        self.ntrials = ntrials

        self.err_sens = np.zeros(ntrials)
        #self.feedback = np.zeros((ntrials,2))
        #self.errors   = np.zeros((ntrials,2))
        self.pert_est = np.zeros(ntrials)
        self.feedback = np.zeros(ntrials)
        self.errors   = np.zeros(ntrials)
        self.weights  = np.zeros((ntrials,self.basis_len))

        # pert_est = pert_est env
        self.feedback_orig_prev = 0  # was participant _expects_ without perturbation
                                     # 0 because we did not move right before
        self.feedback_prev      = np.nan  # known to particpant
        self.pert_est_prev      = 0  # pert_est about pert, initially expect precise movement
        self.err_prev           = 0  # konwn to participant
        self.err_sens_prev      = err_sens_def

        # params from Herzfield supp material v2 p7 (for fig 3B)
        # width or bell curve of error around basis error
        if sigma is None:
            self.sigma = (self.basis_err[1]-self.basis_err[0]) / basis_len
        else:
            self.sigma = sigma
        # another possible value from Herzfeld is 0.001 for Fig. 3C
        self.beta  = beta # speed of update of the weights to compute err sens
        self.decay = decay
        self.prev_trial_ind = -1  # last actualized


        weights_init_adjust_step = 0.001
        if weights_def is None:
            w = np.zeros(self.basis_len);
            while self.err2err_sens(w,0) < err_sens_def:
                w += weights_init_adjust_step; #% get sensitivities as close to initial sensitivity as possible
            self.weights_prev =  w
        else:
            self.weights_prev  = np.ones(self.basis_len) * weights_def

        print(f'weights_prev = {self.weights_prev}')


    # bold g (Herzfeld eq 3)
    def transfer_fun_vec(self,err):
        r = np.zeros(self.basis_len)
        for i in range(self.basis_len):
            r[i] = self.basis_fun(i,err)
        return r

    def basis_fun(self,i,err):
        exparg = -(err - self.basis_err[i])**2
        exparg /= (2 * (self.sigma**2) )
        r  = np.exp( exparg )
        return r

    def err2err_sens(self,weights,err):
        assert len(weights) == self.basis_len
        r = 0
        for i in range(self.basis_len):
            r += weights[i] * self.basis_fun(i,err)
        return r

    def update(self, feedback_new):
        # here error is more like discrepancy, not the reaching error
        # self.feedback_orig_prev is unknown, to be fitted
        if self.prev_trial_ind == -1:
            #feedback_predict_prev = feedback_new
            feedback_predict_prev = 0
        else:
            feedback_predict_prev = self.feedback_orig_prev + self.pert_est_prev
        # error = difference between truth and prediction
        err_new = feedback_new - feedback_predict_prev
        # read_behave2: 'belief' in Romain == orig_feedback (unk to particpant) - target
        # Herzfeld sm p4 bottom pert_est = \hat{y} = u + \hat{x}

        # update weights (Herzfeld eq 3)
        for i in range(self.basis_len):
            bold_g = self.transfer_fun_vec(self.err_prev)  # G
            factor = bold_g / np.linalg.norm(bold_g,2)
            assert factor.shape == self.basis_err.shape
            weights_cur = self.weights_prev + \
                self.beta * np.sign(err_new * self.err_prev) * factor

        # (Herzfeld eq 2)
        err_sens_cur = self.err2err_sens(weights_cur, err_new)

        # (Herzfeld eq 1)
        pert_est_cur = self.decay * self.pert_est_prev + \
            self.err_sens_prev * err_new

        # what I'll see on next trial
        self.prev_trial_ind += 1

        assert self.prev_trial_ind >= 0
        self.feedback[ self.prev_trial_ind ] = feedback_new
        self.errors[ self.prev_trial_ind ]   = err_new
        self.weights[ self.prev_trial_ind ]  = weights_cur
        self.err_sens[ self.prev_trial_ind ] = err_sens_cur
        self.pert_est[ self.prev_trial_ind ] = pert_est_cur

        self.feedback_prev = feedback_new
        self.weights_prev = weights_cur
        self.err_prev = err_new
        self.pert_est_prev = pert_est_cur
        self.err_sens_prev = err_sens_cur

        #self.err_prev_prev = self.err_prev
        #self.feedback_orig_prev = feedback - predict
        #self.feedback_predict_prev = feedback - self.pert_est_cur

        # next move (feedback_orig) should be like feedback_predict_prev - pert_est_prev

        #err_cur = feedback_cur - feedback_predict_cur
    def print_basic(self):
        me,mx = np.mean(self.weights_prev),np.max(self.weights_prev)
        print((f'prev: fb={self.feedback_prev:.3f}, err={self.err_prev:.4f},'
              f' pert_est={self.pert_est_prev:.4f}, err_sens={self.err_sens_prev:.8f}'
              f' wmax={mx:.4f}') )



def _minim_T(nbT,startIdx, a0,c0,b0,x00, state_retention_pause0, optbounds, opts,
           perturb,error, pause_mask, bclip,  modelBeforeSwitch , reg ):
    # this function is ran inside parallel  that is why it is not internal
    # Define a lambda function to pass to minimize function
    # optimize first 4 params: A,c,Bstart,X0.
    # retention, coef, start lr, start state
    import taumodels.state_space_Tan2014.error_model_tanSS as tan

    p = os.path.join( os.path.expandvars('$CODE_MEMORY_ERRORS'), 'state-space-adaptation' )
    sys.path.append(p)
    
    #print('x0 = ', dict(zip(parnames0, x0 ) ) )

    #print('_minim {} {};'.format(nbT,startIdx) )
    f = lambda Z: tan.MSE_error_model_tan(Z,nbT,startIdx,
            modelBeforeSwitch,perturb,error, pause_mask, clip=bclip, reg=reg)
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
                nruns = 3, reg = 0, neg_error = False, maxNbTrials = 30, maxStartIdx = 12,
               minNbTrials = 3, minStartIdx = 3 ):
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
    a0 = 0.95 # initial retention factor
    b0 = 0.01 # initial adaptation factor (for trials before startIdx)
    x00 = 0 # initial initial state
    
    c0 = 0.001 # initial c coefficient (that is multiplied by the ratio of mean error and error variance)
    amin = 0.3 # minimum retention factor
    amax = 1 # maximum retention factor
    cmin = 0 # minimum c
    cmax = 0.8 # maximum c
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
    opts = {'method': 'SLSQP', 'tol':1e-10, 'options':{'maxiter':3000,
                                                      'eps':step, 'disp':0} }

    # A, c, b_start, nbT, x0 and error (eventually add startIdx)
    # nbT = NbPreviousTrials
    nbOptParams = 6
    #subj2out_Tan = {}
    #Q: interpolate NaN trials -- is it good?
    # measured_error has to have shape  trials x subjects
    # loop on the number of subjects
    # for each subject optimization of the model is made separtely
    #for s in dfc['subject'].unique():
    out_cursubj = {}
    #subj2out_Tan[s] = out_cursubj
    error = dfos.query('subject == @subj')[coln_error].to_numpy()
    perturb = dfos.query('subject == @subj')['perturbation'].to_numpy()
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
    plr = Parallel(n_jobs=n_jobs, backend='multiprocessing',
                   )(delayed(_minim_T)(*arg,
                    optbounds,opts,
                    perturb,error, pause_mask, bclip, modelBeforeSwitch, reg) for arg in args)

    for r in plr:
        nbT,startIdx,res = r
        #print(nbT,startIdx, res.fun)
        if res.fun < minfval:
            minres = res
            minfval = res.fun
            # only defined if come inside this if
            Z = res.x
            nbT_opt = nbT
            startIdx_opt = startIdx

    pars = [*Z, nbT_opt, startIdx_opt]


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
    nops = nbOptParams

    likelihood = -1/(2*sigma2)*minfval - (N/2)*np.log(sigma2) - (N/2)*log_2pi
    AIC = 2*nops - 2*likelihood + (2*nops*(nops+1))/(N - nops - 1)

    out_cursubj['minfval'] = minfval
    out_cursubj['AIC'] = AIC

    return out_cursubj

def _minim_D():
    return 0

def fitDiedrischenModel(dfos, coln_error, linalg_error_handling = 'ignore', nruns=1):
    # n_jobs == 1 always here

    from taumodels.state_space_fit_toolbox.state_space_fit import state_space_fit
    from scipy.optimize import minimize # Import the minimize function from scipy.optimize module

    perturb_cursubj        = dfos['perturbation'].values
    measured_error = dfos[coln_error].values
    behavior_cursubj = perturb_cursubj - measured_error

    #                              a0   b0   x00
    #params_init = np.array([   1,  0.1,  0])[None,:]
    #                             amin amax bmin bmax x0min x0max
    #search_space  = np.array([   1,   1,   0, 0.5,  -15, 15])[None,:]
    search_space  = np.array([   0.3,   1,   0, 0.8,  -45, 45])[None,:]

    parnames0 = 'A,B,X0'.split(',')
    optbounds = list(zip( search_space[0,::2], search_space[0,1::2])  )


    # optimization options

    opts = {'method': 'SLSQP', 'tol':1e-10, 'options':{'maxiter':3000}} # Set the optimization options as a dictionary
    nbOptParams = 6 # A, c, b_start, nbT, x0 and error (eventually add startIdx)

    modeltype2res = {}
    #behavior[:,s] = measured_error[:,s] # Use numpy indexing to get a column of an array
    nas = np.isnan(behavior_cursubj)
    _fnz = np.flatnonzero
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
        print('x0 = ', dict(zip(parnames0, x0 ) ) )
        params_init = x0[None,:]
        print(params_init.shape)

        r = state_space_fit(behavior_cursubj, perturb_cursubj,
                params_init, search_space,'lmse', 'norm',
                linalg_error_handling = linalg_error_handling)
        plr += [r]

    #plr = Parallel(n_jobs=n_jobs, backend='multiprocessing',
    #               )(delayed(_minim)(*arg, *x0,
    #                optbounds,opts,
    #                perturb,error, pause_mask, bclip, modelBeforeSwitch, reg) for arg in args)

    minfval = np.inf
    for r in plr:
        #nbT,startIdx,res = r
        #print(nbT,startIdx, res.fun)
        output, params, asymptote, noises_var, states, performances  = r
        AICc, mse = performances 
        if mse < minfval:
            minfval = mse
            # only defined if come inside this if
            ropt = r

    #pars = [*Z, nbT_opt, startIdx_opt]




    #outs_ output, params, asymptote, noises_var, states, performances
    output, params, asymptote, noises_var, states, performances  = ropt
    d = {'output': output,
     'params': params,
     'asymptote': asymptote,
     'noises_var': noises_var,
     'states': states,
     'performances': performances}

    dp = dict(list(zip( 'A,B,X0'.split(','), params[:,0] )))
    display(dp)

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
    for tpl in plr2:
        d = tpl[1].copy()
        d['fun'] = tpl[0].fun
        d['nit'] = tpl[0].nit
        d['nfev'] = tpl[0].nfev
        d['seed'] = tpl[2]
        rows += [d]
    dfrespar = pd.DataFrame(rows)#.describe()
    if len(dfrespar):
        #display( dfrespar.describe() )

        ind =  dfrespar['fun'].idxmin()
        #print(ind)
        best_par = dfrespar.loc[ind].to_dict()
        return best_par
    return None
        #print(best_par)

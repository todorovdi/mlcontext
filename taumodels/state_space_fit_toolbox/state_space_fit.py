# Import numpy and os for vector operations and file handling
import numpy as np
import os
from state_space_fit_toolbox.pyEMtoolbox import generalized_expectation_maximization
from numba import jit

# Author: Lucas Struber
# Email: lucas.struber@univ-grenoble-alpes.fr
# Institution: University Grenoble Alpes
# Lab: TIMC Laboratory
# Advisor: Fabien Cignetti
# Date: June 28, 2021
# Version: 1.0
#
# Summary: fit a state-space model to experimental data
#
# Input description:
#    behavior: the motor output on each trial.
#       size = nbTrials x 1
#    perturb: the perturbation on each trial.
#       size = nbTrials x 1
#    params_init: an initial guess of the parameters that seed the
#       optimization algorithm, and that also define the number of states
#       of the model - 3 parameters for a one-state model [A,b,x0] and
#       6 for a two-state model [As, bs, x0s; Af, bf, x0f].
#       size = nbStates x 3
#    params_search_ranges: a matrix containing upper and lower bounds for
#       the model parameters  - 6 bounds for a one-state model [Amin, Amax,
#       bmin, bmax, x0min, x0max] and 12 for a two-states model [Asmin,
#       Asmax, bsmin, bsmax, x0smin, x0smax; Afmin, Afmax, bfmin, bfmax,
#       x0fmin, x0fmax].
#       size = nbStates x 6
#    fit_method: a string that define the optimization algorithm that is
#       used (and the stochastic/deterministic nature of the modeling).
#       possible values: 'lmse' or 'em' (i.e. least mean square error
#       estimator or expectation-maximization algorithm)
#       Credits: provided EM toolbox has been adapted from Scott Albert
#       toolbox avalaible at http://shadmehrlab.org/tools
#    search_method: a string that define the form of the search space for
#       optimization algorithm.
#       possible values: 'norm' or 'log'. If set to "log", it considers
#       that provided parameters must be converted in the 0-1 log space.
#
# Output description:
#    output: model output (sum of states)
#       size = nbTrials x 1
#    params: optimized set of parameters
#       size = nbStates x 3
#    asymptote: predicted asymptote of the model
#       asymptote = ones(1,nbStates)*((eye(nbStates)-(diag(A)-b'*...
#       ones(1,nbStates)))^(-1))*b'*mean(perturb);
#       size = 1 x 1
#    noises_var: extracted noises of the model. If fit_method is 'lmse',
#       only one noise is considered, between model and measurements
#       (sigmam), and if fit_method is em, three noises are optimized
#       through Kalman filter and MLE procedure, sigmau, sigmax (motor
#       noise and state noise variances)
#       size = nbStates x 3
#    states: states' vector. In one-state model, states = output.
#       size = nbTrials x nbStates
#    performances: model performances, characterized by two classic
#       indicators: AICc (corrected Akaike information criteria) and MSE
#       (mean square error)
#       size = 1 x 2

def state_space_fit(behavior, perturb, params_init=None,
    params_search_ranges=None, fit_method='lmse', search_method='norm',
                    linalg_error_handling='raise', use_mex = 1,
                    inds_state_reset = [],
                    sep_err_begend = 0):
    inds_state_reset = np.array(inds_state_reset)
    # Add EM-toolbox to path using os module functions
    # behavior is perturb - measured_err
    em_toolbox_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EM-toolbox')
    os.sys.path.append(em_toolbox_path)

    if params_init is None:
        params_init = np.array([[0.9, 0.1, 0]])

    nbStates = params_init.shape[0]
    if nbStates >= 3:
        raise ValueError("error: this function only deals with state-spaces models with one or two states")

    if params_init.shape[1] < 2:
        raise ValueError("error: params_init must contain at least A and b initial guess for each state")
    elif params_init.shape[1] < 3:
        for s in range(nbStates):
            params_init[s, 2] = 0

    if params_search_ranges is None:
        params_search_ranges = np.hstack((np.zeros((nbStates, 1)),
            np.ones((nbStates, 1)),  # Amin - Amax
            np.zeros((nbStates, 1)), np.ones((nbStates, 1)),  # Bmin - Bmax
            np.zeros((nbStates, 1)), np.zeros((nbStates, 1))))  # x0min - x0max

    if params_search_ranges.shape[0] != nbStates:
        raise ValueError("error: the size of params_search_ranges is not compatible with the number of states (defined from params_init)")

    if params_search_ranges.shape[1] < 4:
        raise ValueError("error: params_search_ranges must "
            "contain at least Amin, Amax, bmin and bmax for each state")
    elif params_search_ranges.shape[1] < 6:
        for s in range(nbStates):
            params_search_ranges[s, 4] = 0
            params_search_ranges[s, 5] = 0

    if fit_method is None:
        fit_method = 'lmse'

    if search_method is None:
        search_method = 'norm'

    if fit_method not in ['lmse', 'em']:
        #print('unrecognized fit method: set fit method to lmse')
        #fit_method = 'lmse'
        raise ValueError(f'unrecognized fit method: {fit_method}')

    if search_method not in ['norm', 'log']:
        #print('unrecognized search method: set search method to norm')
        #search_method = 'norm'
        raise ValueError(f'unrecognized search method: {search_method}')

    if search_method == 'log':
        params_init[:, 0:2] = -np.log(1 / params_init[:, 0:2] - 1)
        params_search_ranges[:, 0:4] = -np.log(1 / params_search_ranges[:, 0:4] - 1)

        params_init[np.isinf(params_init)] = np.sign(params_init[np.isinf(params_init)]) * 10
        params_search_ranges[np.isinf(params_search_ranges)] = np.sign(params_search_ranges[np.isinf(params_search_ranges)]) * 10

    from scipy.optimize import minimize

    # Local function to be used by minimize that computes the mean square error
    # of a deterministic state-space given a set of parameters

    # Number of parameters optimized (for AIC computation) + constraints
    # Initialization
    nbOptParams = 0

    if nbStates == 2:
        A_con = np.zeros((2, 6))
        b_con = np.zeros((2, 1))

    for s in range(nbStates):
        if params_search_ranges[s, 0] != params_search_ranges[s, 1]:
            nbOptParams += 1
            if s == 1:
                # As >= Af + step
                A_con[0, :] = [-1, 1, 0, 0, 0, 0]
                if fit_method == 'lmse':
                    b_con[0, 0] = -0.001
                else:
                    b_con[0, 0] = -0.01

        if params_search_ranges[s, 2] != params_search_ranges[s, 3]:
            nbOptParams += 1
            if s == 1:
                # Bf >= Bs + step
                A_con[1, :] = [0, 0, 1, -1, 0, 0]
                if fit_method == 'lmse':
                    b_con[1, 0] = -0.001
                else:
                    b_con[1, 0] = -0.01

        if params_search_ranges[s, 4] != params_search_ranges[s, 5]:
            nbOptParams += 1

    if fit_method == 'lmse':
        nbOptParams += 1  # sigma m
    else:
        nbOptParams += 2  # sigma u and sigma x

    ##################################
    # actual fit happens here
    ##################################
    N = len(behavior)

    if fit_method == 'lmse':
        if nbStates == 1:
            #print(params_init.flatten())
            bds = list(zip(params_search_ranges[0, ::2],
                    params_search_ranges[0, 1::2]))
            #print(bds )
            if sep_err_begend:
                f = deterministic_ss_mse2
            else:
                f = deterministic_ss_mse
            res = minimize(f, params_init.flatten(),
                args=(perturb, behavior, search_method, inds_state_reset),
                bounds=bds)

        elif nbStates == 2:
            if sep_err_begend:
                f = deterministic_ss_mse2
            else:
                f = deterministic_ss_mse
            res = minimize(f, params_init.flatten(), args=(perturb, behavior, search_method, inds_state_reset),
                bounds=list(zip(params_search_ranges[:, [0, 2, 4]].flatten(), params_search_ranges[:, [1, 3, 5]].flatten())),
                constraints={'type': 'ineq', 'fun':
                    lambda x: A_con @ x - b_con.flatten()})

        S_opt = res.x
        output, states, err_pred = simulate_deterministic_state_space(S_opt, perturb, behavior, search_method, inds_state_reset=inds_state_reset)

        noises_var = np.var(output - behavior)
        mse_opt = np.mean((output - behavior)**2)
        lik_opt = (N / 2) * (-mse_opt / noises_var - np.log(noises_var) - np.log(2 * np.pi))

    #from EM_toolbox.emtb import generalized_expectation_maximization
    # Assuming generalized_expectation_maximization and simulate_deterministic_state_space functions are implemented in Python
    if fit_method == 'em':
        num_iter = 100
        if nbStates == 1:
            S_opt, likelihoods = generalized_expectation_maximization(
                np.hstack((params_init.flatten(), 2, 2, 5)),  # Initial guess
                behavior, perturb, np.zeros(N), np.nan * np.ones(N), np.ones(nbStates),  # Paradigm
                np.vstack((params_search_ranges[:, [0, 2, 4]].flatten(), params_search_ranges[:, [1, 3, 5]].flatten(), [0.0001, 10000], [0.0001, 10000], [0.0000001, 10])).T,  # Search-space
                None, None,  # constraints
                num_iter, use_mex, search_method)
        elif nbStates == 2:
            bds0 = np.vstack((params_search_ranges[:, [0, 2, 4]].flatten(),
                 params_search_ranges[:, [1, 3, 5]].flatten() ) ).T
            bds_add = np.array([[0.0001, 10000], [0.0001, 10000], [0.0000001, 10]])
            print(bds0.shape, bds_add.shape)
            #print(bds.T.shape)

            bds = np.vstack([bds0,bds_add])

            # 'orig' code
            # np.vstack((params_search_ranges[:, [0, 2, 4]].flatten(),
            #        params_search_ranges[:, [1, 3, 5]].flatten(),
            #        [0.0001, 10000], [0.0001, 10000], [0.0000001, 10])).T

            b_con_ = b_con[:,0] # Dmitry: otherwise Linear constraint cannot broadcast
            c = np.ones(nbStates) # param for Kalman smoother and m_step
            EC = np.zeros(N) # mask whether it is error clamp
            EC_value = np.nan * np.ones(N) # vals of EC is there
            S_opt, likelihoods = generalized_expectation_maximization(
                np.hstack((params_init.flatten(), 2, 2, 5)),  # Initial guess
                behavior, perturb,
                EC, EC_value, c,
                bds,  # Search-space
                A_con, b_con_,  # constraints
                num_iter, use_mex, search_method)

        # for the optimium param values
        # output is basically summed stated
        output, states, err_pred = simulate_deterministic_state_space(S_opt[:3 * nbStates], perturb, behavior, search_method, inds_state_reset=inds_state_reset)

        mse_opt = np.mean((output - behavior) ** 2)
        lik_opt = np.max(likelihoods)

        noises_var = S_opt[3 * nbStates: 3 * nbStates + 2]

    AICc = 2 * nbOptParams - 2 * lik_opt + \
        (2 * nbOptParams * (nbOptParams + 1)) / \
        (N - nbOptParams - 1)
    performances = [AICc, mse_opt]

    A = S_opt[:nbStates]
    b = S_opt[nbStates: 2 * nbStates]
    x0 = S_opt[2 * nbStates: 3 * nbStates]

    if search_method == 'log':
        A = 1 / (1 + np.exp(-A))
        b = 1 / (1 + np.exp(-b))

    from scipy.linalg import LinAlgError
    try:
        invarg = np.eye(nbStates) -\
                (np.diag(A) - b[:, np.newaxis] @ np.ones((1, nbStates)))
        inv = np.linalg.inv(invarg)
        asymptote = np.ones((1, nbStates)) @ inv @ \
            (b[:, np.newaxis] * np.mean(perturb))
    except LinAlgError as e:
        asymptote = None
        if linalg_error_handling == 'raise':
            raise e
        elif linalg_error_handling == 'ignore':
            print('state_space_fit: Ignore linalg err during asymptote cale: ',e)
        else:
            raise ValueError('unk err handl')


    params = np.vstack((A, b, x0))

    return output, params, asymptote, noises_var, states, err_pred, performances


def deterministic_ss_mse(S, perturb, behavior, search_method, inds_state_reset):
    # S: set of parameters [a1, ..., as, b1, ... bs, x01, ..., x0s]
    #print('deterministic_ss_mse: S = ',S)
    #print('deterministic_ss_mse: S shape = ',S.shape)

    # this simulation does not really uses behavior even though it takes it
    # as a parameter
    # search_method is either norm or log
    model_estimate,state_estim, err_pred = \
        simulate_deterministic_state_space(S, perturb, behavior,
        search_method, inds_state_reset=inds_state_reset)
    #print(model_estimate.shape, behavior.shape)
    mse = np.mean((model_estimate - behavior)**2)
    return mse

def deterministic_ss_mse2(S, perturb, behavior, search_method, inds_state_reset):
    #beh = pert - meas_err => err = pert - beh
    model_estimate,state_estim,err_pred = \
        simulate_deterministic_state_space2(S, perturb, behavior,
        search_method, inds_state_reset=inds_state_reset)
    #obs_error = perturb - behavior
    #mse = np.mean((obs_error - err_pred)**2)

    mse = np.mean((model_estimate - behavior)**2)
    return mse

@jit(nopython=True)
def simulate_deterministic_state_space(S, perturb, behavior, search_space,
                                       inds_state_reset ):
    # Summary: simulate a state-space model ignoring noises from a set of
    # parameters
    #
    # Input description:
    #    S: set of parameters [a1, ..., as, b1, ... bs, x01, ..., x0s]
    #       where s is the number of states
    #    b_i is constant learning rate for certain state
    #    behavior: the motor output on each trial (not really used)
    #    perturb: the perturbation on each trial
    #    search_space: a string "norm" or "log". If set to "log", it considers
    #       that provided parameters are in the 0-1 log space
    #       only needed to unpack params right
    #
    # Output description:
    #    y_pred: model output (sum of states)
    #    state: states' vector

    if len(S) % 3 != 0:
        print('S shape = ',S.shape)
        print('S = ',S)
        raise ValueError("S must contain A, B and X0")

    nbStates = len(S) // 3 # Use integer division to get the number of states

    # unpack params
    A = S[:nbStates] # Use slicing to get the first nbStates elements of S
    b = S[nbStates:2*nbStates] # Use slicing to get the next nbStates elements of S
    X0 = S[2*nbStates:] # Use slicing to get the remaining elements of S

    #A = S[::3]
    #b = S[1::3]
    #X0 = S[2::3]

    if search_space == 'log':
        A = 1 / (1 + np.exp(-A)) # Use numpy.exp for element-wise
        b = 1 / (1 + np.exp(-b))

    y_pred = np.zeros_like(behavior) # Use numpy.zeros_like to create an array
    # of zeros with the same shape and type as behavior

    state = np.zeros((len(behavior), nbStates)) # Use numpy.zeros to create a
    # 2D array of zeros with the given shape

    for s in range(nbStates):
        state[0,s] = X0[s] # Use indexing to access and assign elements of an array
    y_pred[0] = np.sum(state[0]) # Use numpy.sum to calculate the sum of an array

    err_pred = np.zeros(len(behavior)) # Use numpy.zeros to create a
    for t in range(1, len(behavior)):
        yprev = y_pred[t-1]
        state_prev = state[t-1,:]
        if t in inds_state_reset:
            yprev = 0
            state_prev[0] = 0 # first is fast

        pert_to_use = perturb[t]
        for s in range(nbStates):
            state[t,s] = A[s]*state_prev[s] +\
                b[s]*(pert_to_use - yprev) # pred err

        # for offline feedback it should be y_pred[t+1] = state[t]
        # y_pred[t] will be compared in the end with behavior[t]
        y_pred[t] = np.sum(state[t])

        err_pred[t] = perturb[t] - y_pred[t]
        # if offline fb
        # in the end of trial t-1 we will have info about perturb[t-1]
        # we update y_pred[t] (before making movement) 
        # based on prev state and discrepancy between reality and this state

    # later y_pred will be compared with behavior. It is not error pred, it is just sum of states
    return y_pred, state, err_pred

@jit(nopython=True)
def simulate_deterministic_state_space2(S, perturb, behavior, search_space, 
                                       inds_state_reset, verbose=0):
    # 2nd ver is for offline feedback

    # Summary: simulate a state-space model ignoring noises from a set of
    # parameters
    #
    # Input description:
    #    S: set of parameters [a1, ..., as, b1, ... bs, x01, ..., x0s]
    #       where s is the number of states
    #    b_i is constant learning rate for certain state
    #    behavior: the motor output on each trial (not really used)
    #    perturb: the perturbation on each trial
    #    search_space: a string "norm" or "log". If set to "log", it considers
    #       that provided parameters are in the 0-1 log space
    #       only needed to unpack params right
    #
    # Output description:
    #    y_pred: model output (sum of states)
    #    state: states' vector

    if len(S) % 3 != 0:
        print('S shape = ',S.shape)
        print('S = ',S)
        raise ValueError("S must contain A, B and X0")

    nbStates = len(S) // 3 # Use integer division to get the number of states

    # unpack params
    A = S[:nbStates] # Use slicing to get the first nbStates elements of S
    b = S[nbStates:2*nbStates] # Use slicing to get the next nbStates elements of S
    X0 = S[2*nbStates:] # Use slicing to get the remaining elements of S

    assert nbStates == 1

    if search_space == 'log':
        A = 1 / (1 + np.exp(-A)) # Use numpy.exp for element-wise
        b = 1 / (1 + np.exp(-b))

    # it is sum of states, not endpoint and not error
    y_pred = np.zeros_like(behavior) # Use numpy.zeros_like to create an array
    # of zeros with the same shape and type as behavior

    state = np.zeros((len(behavior), nbStates)) # Use numpy.zeros to create a
    # 2D array of zeros with the given shape

    err_pred = np.zeros(len(behavior)) # Use numpy.zeros to create a

    for s in range(nbStates):
        state[0,s] = X0[s] # Use indexing to access and assign elements of an array
    y_pred[0] = np.sum(state[0]) # Use numpy.sum to calculate the sum of an array

    # not shifted pert. I.e. pert[n] pertubation indeed happening on trial n, even if not percieved until the end of trial
    for t in range(1, len(behavior)):
        # before I had ... - y_pred[t]
        err_est_before_mvt = perturb[t-1] - y_pred[t-1] 
        state_prev = state[t-1,:]
        if t in inds_state_reset:
            err_est_before_mvt = 0.
            state_prev[0] = 0 # first is fast
        err_observed = perturb[t] - y_pred[t]


        #pert_to_use = perturb[t]
        # stat[t] is that state in the beginning of trial t
        for s in range(nbStates):
            state[t,s] = A[s]*state_prev[s] +\
                b[s]*err_est_before_mvt 

        # for offline feedback it should be y_pred[t+1] = state[t]
        # y_pred[t] will be compared in the end with behavior[t]
        y_pred[t] = np.sum(state[t])
        err_pred[t] = err_est_before_mvt # err as predicted before mvt

        if verbose:
            print('Need to reimplem')
            #print(f'{t}: p[t-1]={perturb[t-1]:.3f} p[t]={perturb[t]:.3f} y_pred[t-1]={y_pred[t-1]:.3f} ep={err_est_before_mvt:.3f} eo={err_observed:.3f} y_pred[t]={y_pred[t]:.3f}')

        # if offline fb
        # in the end of trial t-1 we will have info about perturb[t-1]
        # we update y_pred[t] (before making movement) 
        # based on prev state and discrepancy between reality and this state

    # later y_pred will be compared with behavior
    return y_pred, state, err_pred

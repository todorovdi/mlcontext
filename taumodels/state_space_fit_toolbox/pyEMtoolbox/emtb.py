import numpy as np
from scipy.optimize import LinearConstraint


def incomplete_log_likelihood(y, e, c, parameters, search_method):
    if (len(parameters) - 3) % 3 != 0:
        raise ValueError("parameters must contain A, B and X0")
    nbStates = (len(parameters) - 3) // 3

    if search_method == 'log':
        parameters[:nbStates] = 1 / (1 + np.exp(-parameters[:nbStates]))
        parameters[nbStates:2 * nbStates] = \
            1 / (1 + np.exp(-parameters[nbStates:2 * nbStates]))

    A = np.diag(parameters[:nbStates])
    b = parameters[nbStates:2 * nbStates]
    x1 = parameters[2 * nbStates:3 * nbStates]

    sigmax2 = parameters[3 * nbStates]
    sigmau2 = parameters[3 * nbStates + 1]
    sigma12 = parameters[3 * nbStates + 2]

    V1 = sigma12 * np.eye(nbStates)

    Q = sigmax2 * np.eye(nbStates)

    N = len(y)

    xnnm1 = [None] * N
    Vnnm1 = [None] * N
    xnn = [None] * N
    Vnn = [None] * N

    xnnm1[0] = x1
    Vnnm1[0] = V1

    for n in range(N):
        k = (Vnnm1[n] @ c) / (c.T @ Vnnm1[n] @ c + sigmau2)

        y_error = y[n] - (c.T @ xnnm1[n])

        xnn[n] = xnnm1[n] + k * y_error

        Vnn[n] = (np.eye(nbStates) - k @ c.T) @ Vnnm1[n]

        if n < N - 1:
            xnnm1[n + 1] = A @ xnn[n] + b * e[n]
            Vnnm1[n + 1] = A @ Vnn[n] @ A.T + Q

    likelihood = -(N / 2) * np.log(2 * np.pi)
    for n in range(N):
        SIGMA = c.T @ Vnnm1[n] @ c + sigmau2
        MU = c.T @ xnnm1[n]

        likelihood = likelihood - (1 / 2) * np.log(SIGMA) -\
            (1 / 2) * ((y[n] - MU) ** 2) / SIGMA

    return likelihood


def expected_complete_log_likelihood(parameters, y, e, c, xnN, VnN, Vnp1nN):
    # Input description:
    #    parameters_0: the current estimate of the two state model parameters
    #    y: the motor output on each trial
    #    e: the error experienced by the subject on each trial
    #    c: a model parameter that is assumed invariant
    #    xnN: This is shorthand for the quantity x(n|N). It is the smoothed
    #        Kalman state expectation  E[x(n)|y(1),y(2),...,y(N)].
    #    VnN: This is shorthand for the quantity V(n|N). It is the smoothed
    #        Kalman state variance  var(x(n)|y(1),y(2),...,y(N)).
    #    Vnp1nN: This is shorthand for the quantity V(n+1,n|N). It is the
    #        smoothed Kalman covariance of consecutive states, also written as
    #        cov(x(n+1),x(n)|y(1),y(2),...,y(N)).

    # stores input variables using descriptive names
    if (len(parameters) - 3) % 3 != 0:
        raise ValueError("parameters must contain A, B, and X0")
    nbStates = (len(parameters) - 3) // 3

    A = np.diag(parameters[:nbStates])
    b = np.array(parameters[nbStates:2 * nbStates])
    x1 = np.array(parameters[2 * nbStates:3 * nbStates])
    sigmax2 = parameters[3 * nbStates]
    sigmau2 = parameters[3 * nbStates + 1]
    sigma12 = parameters[3 * nbStates + 2]

    # sets the means and variances for the initial states
    V1 = sigma12 * np.eye(nbStates)

    # sets matrices and vectors for the update of the fast and slow states
    Q = sigmax2 * np.eye(nbStates)

    # stores the number of trials
    N = len(y)

    # precomputes important quantities referenced in the expected complete
    # log-likelihood function
    QinvA = np.linalg.solve(Q, A)
    Qinvb = np.linalg.solve(Q, b)
    AtQinv = np.linalg.solve(Q, A.T)
    AtQinvA = AtQinv @ A
    AtQinvb = A.T @ Qinvb
    btQinv = np.linalg.solve(Q, b.T)
    btQinvA = btQinv @ A
    btQinvb = btQinv @ b

    # compute the likelihood
    # In this section the expected complete log-likelihood is computed in
    # different parts.

    # TERM 1: one of the parts of the expected complete log-likelihood function
    # derived from the likelihood of observing the motor output given the
    # states
    term1 = sum(y[n] ** 2 + c.T @ (VnN[n] + xnN[n] @ xnN[n].T) @ c -
                2 * y[n] * c.T @ xnN[n] for n in range(N))
    term1 = -term1 / (2 * sigmau2)

    # TERM 2: one of the parts of the expected complete log-likelihood function
    # derived from the likelihood of observing the state on trial n+1 given the
    # state on trial n
    term2 = sum(xnN[n + 1].T @ np.linalg.solve(Q, xnN[n + 1]) +
                np.trace(np.linalg.solve(Q, VnN[n + 1])) -
                xnN[n + 1].T @ QinvA @ xnN[n] -
                np.trace(np.linalg.solve(Q, Vnp1nN[n].T)) -
                xnN[n + 1].T @ Qinvb * e[n] -
                xnN[n].T @ AtQinv @ xnN[n + 1] -
                np.trace(np.linalg.solve(AtQinv, Vnp1nN[n]))
                + xnN[n].T @ AtQinvA @ xnN[n] +
                np.trace(AtQinvA @ VnN[n]) + xnN[n].T @ AtQinvb * e[n] -
                e[n] * btQinv @ xnN[n + 1] +
                e[n] * btQinvA @ xnN[n] + e[n] * btQinvb * e[n]
                for n in range(N - 1))
    term2 = -term2 / 2

    # TERM 3: one of the parts of the expected complete log-likelihood function
    # derived from the likelihood of observing the initial state
    term3 = xnN[0].T @ np.linalg.solve(V1, xnN[0]) +\
        np.trace(np.linalg.solve(V1, VnN[0])) - \
        xnN[0].T @ np.linalg.solve(V1, x1) - \
        x1.T @ np.linalg.solve(V1, xnN[0]) +\
        x1.T @ np.linalg.solve(V1, x1)
    term3 = -term3 / 2

    # TERM 4: one of the parts of the expected complete log-likelihood function
    # derived from the pre-exponential factors
    term4 = -(1 / 2) * np.log(np.linalg.det(V1)) -\
        (N / 2) * np.log(sigmau2) - (3 / 2) * N * np.log(2 * np.pi)

    # TERM 5: one of the parts of the expected complete log-likelihood function
    # derived from the pre-exponential factors of x(n+1) given x(n)
    term5 = -(N - 1) * np.log(np.linalg.det(Q)) / 2

    # computes the likelihood from the sum of all terms
    likelihood = term1 + term2 + term3 + term4 + term5

    return likelihood


def generalized_expectation_maximization(parameters, y, r,
        EC, EC_value, c,
        search_space, A_con, b_con,
        num_iterations, use_mex, search_method):
    '''
% Summary: This function coordinates the EM algorithm. It successively
%     calls the Kalman Smoother to specify the E-Step, and then performs
%     the M-step to update the parameter set.
%
% Notes: For more information about this package see README.pdf.
%
% Input description:
%    parameters: an initial guess of the two state model parameters that
%        seed the EM algorithm
%    y: the motor output on each trial
%    r: the perturbation on each trial
%    EC: an array that indicates if a trial is an error-clamp trial
%        If the n-th entry is non-zero, this indicates that trial n is an
%        error-clamp trial
%        If the n-th entry is zero, this indicates that trial n is not an
%        error-clamp trial
%    EC_value: an array that indicates the value of the clamped error on
%        each error-clamp trial.
%    c: a model parameter that is assumed invariant
%    search_space: a matrix containing upper and lower bounds for the model
%        parameters
%    constraints: an array that specifies linear inequality constraints
%        between the fast and slow retention factors, and error
%        sensitivities
%    num_iterations: the number of EM iterations
%    use_mex: determines if a mex function will be used in the M-step of
%        the algorithm, or a regular MATLAB function (m-file)
%    search_method: a string that define the form of the search space for
%       optimization algorithm.
%       possible values: "norm" or "log". If set to "log", it considers
%       that provided parameters must be converted in the 0-1 log space.
%
% Output description:
%    parameters: the final parameters obtained at the conclusion of all the
%        EM iterations
%    likelihoods: an array containing the value of the incomplete
%        log-likelihood function on each iteration of the EM algorithm
'''

    # computes the error on each trial
    N = len(y)  # the number of trials
    e = np.zeros(N)
    # allocates space for the errors
    for n in range(N):  # iterate through each trial
        if EC[n] == 0:
            # this is not an error-clamp trial
            e[n] = r[n] - y[n]
        else:
            # this is an error-clamp trial
            e[n] = EC_value[n]

    # creates an array that stores the value of the incomplete log-likelihood
    # function on each iteration of the EM algorithm
    likelihoods = np.zeros(num_iterations)

    # the EM algorithm
    for n in range(num_iterations):
        # E-step: get the smoothed Kalman estimates of states, variances, and
        # covariances using a Kalman smoother
        xnN, VnN, Vnp1nN = kalman_smoother(parameters, y, e, c, search_method)

        # M-step: perform maximum likelihood estimation in a contrained
        # parameter space
        parameters = m_step(parameters, xnN, VnN, Vnp1nN, y, e, c,
                            search_space, A_con, b_con, use_mex, search_method)

        # computes the incomplete log-likelihood for this parameter set
        likelihoods[n] = incomplete_log_likelihood(y, e, c, parameters,
                                                   search_method)

        # checks to make sure that the likelihood function has increased
        if (n > 0) and (likelihoods[n] < likelihoods[n - 1]):
            # the likelihood has not increased, warn the modeler
            print('Warning: The expected complete log-likelihood '
                  'function has stopped increasing')

    return parameters, likelihoods

# Author: Scott Albert
# Email: salbert8@jhu.edu
# Institution: Johns Hopkins University
# Lab: Laboratory for Computational Motor Control
# Advisor: Reza Shadmehr
# Date: July 25, 2017
# Location: Baltimore, MD 21211
# % Version: 1.1
def m_step(parameters_0, xnN, VnN, Vnp1nN, y, e, c,
           search_space, A_con, b_con, use_mex, search_method):
    """
    Summary: This function uses fmincon to perform the M-step of the EM
        algorithm. The parameter set that maximizes the expected complete
        log-likelihood function in a constrained parameter space is identified
        using fmincon.

    Notes: For more information about this package see README.pdf.

    Input description:
        parameters_0: the current estimate of the two state model parameters
        xnN: This is shorthand for the quantity x(n|N). It is the smoothed
            Kalman state expectation  E[x(n)|y(1),y(2),...,y(N)].
        VnN: This is shorthand for the quantity V(n|N). It is the smoothed
            Kalman state variance  var(x(n)|y(1),y(2),...,y(N)).
        Vnp1nN: This is shorthand for the quantity V(n+1,n|N). It is the
            smoothed Kalman covariance of consecutive states, also written as
            cov(x(n+1),x(n)|y(1),y(2),...,y(N)).
        y: the motor output on each trial
        e: the error experienced by the subject on each trial
        c: a model parameter that is assumed invariant
        search_space: a matrix containing upper and lower bounds for the model
            parameters
        constraints: an array that specifies linear inequality constraints
            between the fast and slow retention factors, and error
            sensitivities
        use_mex: determines if an mex function will be used or a regular
            MATLAB function (m-file)

    Output description:
        parameters_final: the parameter set identified by fmincon that
            maximizes the expected complete log-likelihood function

    Modification: Lucas Struber
    Email: lucas.struber@univ-grenoble-alpes.fr
    Institution: University Grenoble Alpes
    Lab: TIMC Laboratory
    Advisor: Fabien Cignetti
    Date: June 28, 2021
    % Version: 1.1b
    Summary of modifications:
    1. parameters extraction from parameters vector has been modified to
    handle multi-states models (instead of only two-state model)
    2. search_method parameters has been added to allow a logarithmic form
    for the search space
    """
    from scipy.optimize import minimize

    if (len(parameters_0) - 3) % 3 != 0:
        raise ValueError("parameters must contains A, B and X0")
    nbStates = (len(parameters_0) - 3) // 3
    if nbStates == 1:
        use_mex = 0

    def likelihood_function(x):
        """
        creates a local function to be used by minimize that computes the negated
        value of the expected complete log-likelihood function
        """
        if search_method == 'log':
            x[0:nbStates] = 1. / (1 + np.exp(-x[0:nbStates]))
            x[nbStates: 2 * nbStates] = 1. / (1 + np.exp(-x[nbStates: 2 * nbStates]))

        if use_mex:
            # calls an executable mex function to compute the likelihood
            likelihood = expected_complete_log_likelihood_mex(x, y, e, c, xnN, VnN, Vnp1nN)
        else:
            # calls a Python function to compute the likelihood
            likelihood = expected_complete_log_likelihood(x, y, e, c, xnN, VnN, Vnp1nN)

        # negates the likelihood
        negated_likelihood = -likelihood
        return negated_likelihood

    # specifies the lower and upper bounds for the minimize search
    # search space is 2D array shape nparams x 2
    lb = search_space[:, 0]
    ub = search_space[:, 1]
    bounds = list(zip(lb, ub))

    if nbStates == 1:
        result = minimize(likelihood_function, parameters_0, bounds=bounds)
        parameters_final = result.x
    elif nbStates == 2:
        # add sigmax2, sigmau2 and sigma12 to the linear constraint
        A_con = np.concatenate((A_con, np.zeros((2, 3))), axis=1)
        linear_constraint = LinearConstraint(A_con, b_con, b_con)

        # uses minimize to maximize the expected complete log-likelihood function in
        # a constrained parameter space
        result = minimize(likelihood_function, parameters_0,
                constraints=linear_constraint, bounds=bounds)
        parameters_final = result.x

    return parameters_final

def kalman_smoother(parameters, y, e, c, search_method):
    if (len(parameters) - 3) % 3 != 0:
        raise ValueError("parameters must contain A, B, and X0")
    nbStates = (len(parameters) - 3) // 3

    if search_method == 'log':
        parameters[:nbStates] = 1. / (1. + np.exp(-parameters[:nbStates]))
        parameters[nbStates:2 * nbStates] = 1. / (1. + np.exp(-parameters[nbStates:2 * nbStates]))

    A = np.diag(parameters[:nbStates])
    b = parameters[nbStates:2 * nbStates].T
    x1 = parameters[2 * nbStates:3 * nbStates].T

    sigmax2 = parameters[3 * nbStates]
    sigmau2 = parameters[3 * nbStates + 1]
    sigma12 = parameters[3 * nbStates + 2]

    V1 = sigma12 * np.eye(nbStates)

    Q = sigmax2 * np.eye(nbStates)

    N = len(y)

    xnnm1 = [None] * N
    Vnnm1 = [None] * N
    xnn = [None] * N
    Vnn = [None] * N

    xnnm1[0] = x1
    Vnnm1[0] = V1

    for n in range(N):
        k = (Vnnm1[n] @ c) / (c.T @ Vnnm1[n] @ c + sigmau2)

        y_error = y[n] - (c.T @ xnnm1[n])

        xnn[n] = xnnm1[n] + k * y_error

        Vnn[n] = (np.eye(nbStates) - k @ c.T) @ Vnnm1[n]

        if n < N - 1:
            xnnm1[n + 1] = A @ xnn[n] + b * e[n]

            Vnnm1[n + 1] = A @ Vnn[n] @ A.T + Q

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%% Kalman smoother %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # In this section, a Kalman smoother is used to compute the smoothed
    # expectations and covariances of the hidden states.
    #
    # Notation:
    #    N it the total number of trials
    #    the smoothed expectation E(x(n)|y(1),...,y(N)) is denoted xnN
    #    the smoothed variance var(x(n)|y(1),...,y(N)) is denoted VnN
    #    the smoothed covariance cov(x(n+1),x(n)|y(1),...,y(N)) is denoted
    #       Vnp1nN

    # allocate space for the smoothed expectations and variances
    xnN = [None] * N
    VnN = [None] * N

    # instantiate the expectation and variance of the final trial as the
    # posteriors obtained at the end of the forward Kalman filter
    xnN[-1] = xnn[-1]
    VnN[-1] = Vnn[-1]

    # allocate space for the J parameter
    Jn = [None] * N

    # backwards recursions for Kalman smoothing
    for n in range(N - 2, -1, -1):
        # computes J
        Jn[n] = Vnn[n] @ (A.T / Vnnm1[n + 1])
        # computes the smoothed variance
        VnN[n] = Vnn[n] + Jn[n] @ (VnN[n + 1] - Vnnm1[n + 1]) @ Jn[n].T
        # computes the smoothed expectation
        xnN[n] = xnn[n] + Jn[n] @ (xnN[n + 1] - xnnm1[n + 1])

    # computes the smoothed covariances
    Vnp1nN = [None] * (N - 1)
    for n in range(N - 1):
        Vnp1nN[n] = VnN[n + 1] @ Jn[n].T

    return xnN, VnN, Vnp1nN

#def expected_complete_log_likelihood_mex()
def expected_complete_log_likelihood_mex(parameters, y, e,
                                         c, xnN, VnN, Vnp1nN):
    # I either need matlab engine to call function from .so library that
    # uses mxArray or I have to rewrite the cpp function
    raise ValueError('not implemented')
    return 0

# Import numpy and matplotlib for vector operations and plotting
import numpy as np
import matplotlib.pyplot as plt

# Author: Lucas Struber
# Email: lucas.struber@univ-grenoble-alpes.fr
# Institution: University Grenoble Alpes
# Lab: TIMC Laboratory
# Advisor: Fabien Cignetti
# Date: June 28, 2021
# Version: 1.0
#
# Summary: example use of state_space_fit function
from state_space_fit import state_space_fit

# load behavioral data
print("clear all, clc;") # This is not needed in Python
print("load('example_behavior') # 80 trials x 3 subjects") # This depends on the file format, but you can use numpy.load for .npy files
# need def
behavior = perturb - measured_error

linalg_error_handling = 'ignore'

# Parameters initialization
#                              a0   b0   x00
params_init_Diedrischen   = np.array([   1,  0.1,  0])[None,:]
params_init_Albert =       np.array( [[0.95, 0.05,  0], # Use nested lists to create a 2D array in Python
                             [0.7,  0.3,  0]] )
#                             amin amax bmin bmax x0min x0max
search_space_Diedrischen   = np.array([   1,   1,   0, 0.5,  -15, 15])[None,:]
search_space_Albert =       np.array( [[ 0.2,   1,   0, 0.5,    0,  0], # Use nested lists to create a 2D array in Python
                              [ 0.2,   1,   0, 0.5,  -15, 15]] )

# optimization options
from scipy.optimize import minimize # Import the minimize function from scipy.optimize module
opts = {'method': 'SLSQP', 'tol':1e-10, 'options':{'maxiter':3000}} # Set the optimization options as a dictionary

nbOptParams = 6 # A, c, b_start, nbT, x0 and error (eventually add startIdx)

# init fit outputs
nbSub = measured_error.shape[1] # Use numpy.shape to get the dimensions of an array
nbTrials = measured_error.shape[0]

output_Diedrischen = np.zeros((nbTrials,nbSub))
params_Diedrischen = np.zeros((3,nbSub)) # 3 parameters, A, n, x0
asymptote_Diedrischen= np.zeros(nbSub)
noise_Diedrischen = np.zeros(nbSub) # 1 noise (LMSE approach), sigma m
perf_Diedrischen = np.zeros((2,nbSub)) # 2 performances indicators, AICc & MSE

output_Albert = np.zeros((nbTrials,nbSub))
params_Albert = np.zeros((6,nbSub)) # 6 parameters, As, Af, bs, bf, x0s, x0f
asymptote_Albert = np.zeros(nbSub)
noise_Albert = np.zeros((2,nbSub)) # 2 noises (EM algorithm), sigmau and sigmax
internalStates_Albert = np.zeros((nbTrials,2,nbSub)) # 2 states model
perf_Albert = np.zeros((2,nbSub)) # 2 performances indicators, AICc & MSE

# loop on the number of subjects
for s in range(nbSub):
    print('---------- Subj',s)
    modeltype2res = {}
    #behavior[:,s] = measured_error[:,s] # Use numpy indexing to get a column of an array
    behavior_cursubj = behavior[:,s]
    perturb_cursubj = perturb[:,s]
    nas = np.isnan(behavior_cursubj)
    behavior_cursubj[nas] =\
        np.interp(np.flatnonzero(nas),
        np.flatnonzero(~nas),
        behavior_cursubj[~nas]) # Interpolate NaNs using numpy indexing and numpy.interp
    #One-dimensional linear interpolation for monotonically increasing sample points.
    # np.interp 1st s args
    #xarray_like
    #The x-coordinates at which to evaluate the interpolated values.

    #xp1-D sequence of floats
    #The x-coordinates of the data points, must be increasing if argument period is not specified. Otherwise, xp is internally sorted after normalizing the periodic boundaries with xp = xp % period.

    #fp1-D sequence of float or complex
    #The y-coordinates of the data points, same length as xp.

    assert not np.isnan(behavior_cursubj).any()

    # Fit a one-state model without retention factor (Diedrieschen, 2003),
    # using a deterministic lmse approach
    #print( params_init_Diedrischen, search_space_Diedrischen )
    r = state_space_fit(behavior_cursubj, perturb_cursubj,
            params_init_Diedrischen, search_space_Diedrischen,
            'lmse', 'norm', linalg_error_handling = linalg_error_handling)
    #(output_Diedrischen[:,s], params_Diedrischen[:,s],
    # asymptote_Diedrischen[s], noise_Diedrischen[s], _,
    # perf_Diedrischen[:,s]) = r

    #outs_ output, params, asymptote, noises_var, states, performances
    output, params, asymptote, noises_var, states, performances  = r
    d = {'output': output,
     'params': params,
     'asymptote': asymptote,
     'noises_var': noises_var,
     'states': states,
     'performances': performances}
    modeltype2res['Diedrischen'] = d

    # Fit a two-state model with retention factor using stochastic EM approach
    # (Albert, 2018) optimizing parameters in log-space
    use_mex = 0
    r = state_space_fit(behavior_cursubj, perturb_cursubj,
            params_init_Albert, search_space_Albert, 'em', 'log',
                        linalg_error_handling = linalg_error_handling,
                        use_mex = use_mex)
    #(output_Albert[:,s], params_Albert[:,s], asymptote_Albert[s],
    # noise_Albert[:,s], internalStates_Albert[:,:,s],
    # perf_Albert[:,s]  ) = r

    output, params, asymptote, noises_var, states, performances  = r
    d = {'output': output,
     'params': params,
     'asymptote': asymptote,
     'noises_var': noises_var,
     'states': states,
     'performances': performances}
    modeltype2res['Albert'] = d


    plt.figure() # Use matplotlib.pyplot to plot figures
    plt.title('Subject ' + str(s+1) + ' - LMSE algorithm') # Use string concatenation to make titles
    plt.plot(behavior_cursubj, linewidth=1, label='Observed error') # Use label argument to add legends
    plt.plot(modeltype2res['Diedrischen']['output'], linewidth=2, label='Model output')
    plt.ylabel('Angle (°)')
    plt.xlabel('Trials')
    plt.legend() # Call legend function to show legends
    plt.show() # Call show function to display the figure

    plt.figure() # Use matplotlib.pyplot to plot figures
    plt.title('Subject ' + str(s+1) + ' - EM algorithm') # Use string concatenation to make titles
    plt.plot(behavior_cursubj, linewidth=1, label='Observed error') # Use label argument to add legends
    plt.plot(output_Albert[:,s], linewidth=2, label='Model output')
    plt.plot(modeltype2res['Albert']['states'][:,0], color=[0.8500, 0.3250, 0.0980], linestyle='--', linewidth=1, label='Slow-state')
    plt.plot(modeltype2res['Albert']['states'][:,1], color=[0.8500, 0.3250, 0.0980], linestyle='-.', linewidth=1, label='Fast-state')
    plt.ylabel('Angle (°)')
    plt.xlabel('Trials')
    plt.legend() # Call legend function to show legends
    plt.show() # Call show function to display the figure


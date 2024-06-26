# Import numpy and matplotlib for vector operations and plotting
import numpy as np
import matplotlib.pyplot as plt
#from error_model_tanSS import *
import state_space_Tan2014.error_model_tanSS as tan

# has to be defined outside of the script
#print(len(measured_error) )

# Loading files and constant initialization
print("clear all, clc;") # This is not needed in Python
#print("load('example_behavior') # 80 trials x 3 subjects") # This depends on the file format, but you can use numpy.load for .npy files

# Parameters initialization
minNbTrials = 3 # minimum number of "retained" trials
maxNbTrials = 40 # maxumum number of "retained" trials
minStartIdx = 6 # minimum index of first trial where previous trials are taken into account
maxStartIdx = 6 # maximum index of first trial where previous trials are taken into account
a0 = 0.95 # initial retention factor
b0 = 0.01 # initial adaptation factor (for trials before startIdx)
x00 = 0 # initial initial state
c0 = 0.001 # initial c coefficient (that is multiplied by the ratio of mean error and error variance)
amin = 0.5 # minimum retention factor
amax = 1 # maximum retention factor
cmin = 0 # minimum c
cmax = 0.1 # maximum c
bmin = 0 # minimum adaptation rate
bmax = 0.5 # maximum adaptation rate
x0min = -30 # minimum initial state
x0max = 30 # maximum initial state

optbounds = [(amin,amax),(cmin,cmax),(bmin,bmax),(x0min,x0max)]

# optimization options
from scipy.optimize import minimize
# Set the optimization options as a dictionary
opts = {'method': 'SLSQP', 'tol':1e-10, 'options':{'maxiter':3000}}

# A, c, b_start, nbT, x0 and error (eventually add startIdx)
# nbT = NbPreviousTrials
nbOptParams = 6

# init fit outputs
#nbSub = measured_error.shape[1] # Use numpy.shape to get the dimensions of an array
#nbSub = len(subjects) # Use numpy.shape to get the dimensions of an array
#nbTrials = measured_error.shape[0]

#output_Tan = np.zeros((nbTrials,nbSub))
#params_Tan = np.zeros((6,nbSub)) # 3 parameters, A, c, b_start, x0, nbTrials, startIdx
subj2out_Tan = {}
#adaptationRate_Tan = np.zeros((nbTrials,nbSub))
#perf_Tan = np.zeros((2,nbSub)) # 2 performances indicators, AICc & MSE

#Q: interpolate NaN trials -- is it good?

# measured_error has to have shape  trials x subjects
# loop on the number of subjects
# for each subject optimization of the model is made separtely
for s in dfc['subject'].unique():
    out_cursubj = {}
    subj2out_Tan[s] = out_cursubj
    error = dfc.query('subject == @s')[coln_error].to_numpy()
    perturb = dfc.query('subject == @s')['perturbation'].to_numpy()
    assert not np.isnan(perturb).any()
    print('Num NaNs in errors  = ' , np.sum( np.isnan(error) ) )
    #error = measured_error[:,s] # Use numpy indexing to get a column of an array
    # Interpolate NaNs using numpy indexing and numpy.interp
    error[np.isnan(error)] = np.interp(np.flatnonzero(np.isnan(error)),
                                       np.flatnonzero(~np.isnan(error)),
                                       error[~np.isnan(error)])
    assert not np.isnan(error).any()

    # with least square minimization
    minfval = np.inf
    idx = 0
    #fval_all = np.zeros(maxNbTrials-minNbTrials+1)
    args = []


    for nbT in range(minNbTrials, maxNbTrials+1):
        #idx += 1
        for startIdx in range(minStartIdx, maxStartIdx+1):
            args += [(nbT,startIdx) ]

    from joblib import Parallel, delayed

    def _minim(nbT,startIdx  ):
        # Define a lambda function to pass to minimize function
        # optimize first 4 params: A,c,Bstart,X0.
        # retention, coef, start lr, start state
        print('_minim',nbT,startIdx)
        f = lambda Z: tan.MSE_error_model_tan(Z,nbT,startIdx,
                'NoAdapt',perturb,error)
        res = minimize(f,[a0,c0,b0,x00],
            bounds=optbounds,**opts) # Call the minimize function with the objective function, initial guess, bounds and options
        if np.isnan(res.fun ):
            sys.exit(1)
        return nbT,startIdx, res

    print('Start parallel')
    plr = Parallel(n_jobs=n_jobs, backend='multiprocessing',
                   )(delayed(_minim)(*arg) for arg in args)

    #fval_all[idx-1] = res.fun
    for r in plr:
        nbT,startIdx,res = r
        #print(nbT,startIdx, res.fun)
        if res.fun < minfval:
            minfval = res.fun
            # only defined if come inside this if
            Z = res.x
            nbT_opt = nbT
            startIdx_opt = startIdx

#         FigName = strcat('NbTrialsOpt - Condition :',CondNames{c},' - Sub : ', num2str(s));
#         figure('Name',FigName);
#         plot(MinNbTrials:MaxNbTrials,fval_all);

    pars = [Z[0], Z[1], Z[2], Z[3], nbT_opt, startIdx_opt]

    parnames = 'A,c,Bstart,X0,NbPreviousTrials,StartIdx'.split(',')
    pars_  = dict(zip(parnames,pars) )
    out_cursubj['params'] = pars_

    #params_Tan[0,s],params_Tan[1,s],
    #                     params_Tan[2,s],params_Tan[3,s],
    #                     params_Tan[4,s],params_Tan[5,s]
    r = tan.error_model_tan(*pars,
                         'NoAdapt',perturb,error)
    output_Tan, state_Tan, adaptationRate_Tan = r
    out_cursubj['states'] = state_Tan
    out_cursubj['y_pred'] = output_Tan
    out_cursubj['adaptation_rate'] = adaptationRate_Tan

    sigma2 = np.var(output_Tan - error)


    # Store the length of error in a variable
    N = len(error)
    log_2pi = np.log(2*np.pi)
    nops = nbOptParams

    likelihood = -1/(2*sigma2)*minfval - (N/2)*np.log(sigma2) - (N/2)*log_2pi
    AIC = 2*nops - 2*likelihood + (2*nops*(nops+1))/(N- nops - 1)

    out_cursubj['minfval'] = minfval
    out_cursubj['AIC'] = AIC

    if do_plot:
        plt.figure() # Use matplotlib.pyplot to plot figures
        plt.title('Subject ' + s) # Use string concatenation to make titles
        plt.plot(error, linewidth=1, label='Observed error') # Use label argument to add legends
        plt.plot(output_Tan, linewidth=2, label='Model output')
        plt.ylabel('Angle (°)')
        plt.xlabel('Trials')
        plt.twinx() # Use twinx function to create a second y-axis
        plt.plot(adaptationRate_Tan, linewidth=2, label='Adaptation rate', color='green') # Use color argument to change the line color
        plt.legend() # Call legend function to show legends
        plt.show() # Call show function to display the figure

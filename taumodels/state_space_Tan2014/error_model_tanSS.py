import numpy as np
from numba import jit

@jit(nopython=True)
def error_model_tan(A,c,Bstart,X0,
        NbPreviousTrials,StartIdx,ModelBeforeSwitch,
        Perturb,error,  clip=np.array([-0.5,0.5]), 
       state_retention_pause = 0.9,
        pause_mask = np.array([]),  motor_var = 0.,
        use_true_error = False, use_true_error_stats = True,  
                    inds_state_reset=np.array([]), 
                    stats_powers=np.array([2,-2]) ):
    # c is the coefficient in front of the variation
    # last argument is long array
    # X0 is init state
    # A is retention
    # NbPreviousTrials is number of trials taken to comput variablity (in Tan 20)
    # Perturb, error -- true experimental observations
    # ModelBeforeSwitch either "NoAdapt" or something else (in this case learning rate IC
    # taken to be Bstart

    # Initialize y_pred, state and B as zero vectors with the same size as error
    if len(inds_state_reset) > 0: 
        assert np.max(inds_state_reset) < len(error)

    no_adapt_before_switch = (ModelBeforeSwitch == "NoAdapt")

    y_pred = np.zeros_like(error)
    state = np.zeros_like(error)
    B = np.zeros_like(error)

    if len(pause_mask) == 0:
        pause_mask = np.zeros_like(error)

    # Set the initial state to X0
    state[0] = X0

    noise = np.random.normal(loc=0, scale=motor_var, size=len(error))

    # Calculate the first prediction error as the difference between Perturb[0] and X0
    #if online_feedback:
    #    y_pred[0] = Perturb[0] - X0
    #else:
    #    y_pred[0] = 0
    y_pred[0] = Perturb[0] - X0

    # Loop over the rest of the elements in error
    for t in np.arange(1,len(error)):
        # If t is less than StartIdx
        errorStd = 0.
        suberr   = 0.

        if pause_mask[t]:
            assert np.isnan( Perturb[t]  )
            B[t] = B[t - 1]
            state[t]  = state[t-1] * state_retention_pause
            y_pred[t] = np.nan
        else:
            if t < StartIdx:
                # Check the value of ModelBeforeSwitch
                if no_adapt_before_switch:
                    # Set B[t] to zero
                    B[t] = 0
                else:
                    # Set B[t] to Bstart
                    B[t] = Bstart
            else:
                # Calculate the mean and standard deviation of the previous NbPreviousTrials errors
                if use_true_error_stats:
                    err = error
                else:
                    err = y_pred

                #suberr = error[max(0,t-NbPreviousTrials):t]
                suberr = err[max(0,t-NbPreviousTrials):t]

                errorMean = np.abs( np.mean(suberr) )
                errorStd  = np.std( suberr)
    #             errorMean = np.mean(y_pred[max(0,t-NbPreviousTrials):t])
    #             errorVar = np.std(y_pred[max(0,t-NbPreviousTrials):t])
                # Calculate B[t] as a function of c, errorMean and errorVar

                if abs(errorStd) < 1e-10:
                    # dirty hack but in my data I can have errorVaer = 0
                    # for small NbPreviousTrials and or due to "easy" trials
                    B[t] = B[t-1]
                else:
                    pm,ps=stats_powers
                    B[t] = c*(errorMean**pm)*(errorStd**ps)
                # Clip B[t] to the range [-0.5, 0.5]
                B[t] = max( B[t], clip[0] )  # clip from below
                B[t] = min( B[t], clip[1] )  # clip from above
                #if(B[t] >  0.5): B[t] =  0.5
                #if(B[t] < -0.5): B[t] = -0.5

            # Update the state as a function of A, state[t-1], B[t] and y_pred[t-1]
            #print(t, state[t-1], y_pred[t-1], B[t], errorVar, error[t-1], suberr  )

            if t in inds_state_reset:
                st = 0
                yp = 0
            else:
                st = state[t-1]
                if use_true_error:
                    #yp = error[t-1] # nonsense
                    yp = -error[t-1] # okay
                    #yp = Perturb[t-1] - error[t-1]  # state middle 
                    #yp = -( Perturb[t-1] - error[t-1])  # nonsesne
                    #yp = ( Perturb[t-1] + error[t-1]  ) # state middle
                    #yp = -( Perturb[t-1] + error[t-1]) # complete nonsesne
                else:
                    yp = y_pred[t-1] 

            state[t] = A*st + B[t]*yp
            # Calculate the prediction error as the difference between Perturb[t] and state[t]
            y_pred[t] = Perturb[t] - state[t]
            y_pred[t] += noise[t]
            #if online_feedback:
            #    y_pred[t] = Perturb[t] - state[t]
            #else:
            #    y_pred[t] = Perturb[t-1] - state[t]

    return y_pred, state, B

# Define a function named MSE_TanSS that accepts 6 inputs and returns 3 outputs
def MSE_error_model_tan(parvec,NbTrials,StartIdx,ModelBeforeSwitch,
                Perturb,error, pause_mask, clip, reg = 0.,
                    inds_state_reset=[], stats_powers =(2,-2),
                    use_true_error=False, use_true_error_stats=True,
                        motor_var = 0., c_scale=100.):

    # Unpack parvec into four variables
    if pause_mask is None:
        A,c,Bstart,X0 = parvec
        state_retention_pause = 0.9
        use_pause_mask = False
        pause_mask = np.zeros_like(error)
    else:
        A,c,Bstart,X0,state_retention_pause = parvec
        use_pause_mask = True

    inds_state_reset = np.array(inds_state_reset)


    #for k,v in locals().items():
    #    print(k, type(v) )

    # Call the errorModel_TanSS function with the unpacked variables and the other inputs
    y_pred, state, B = error_model_tan(A,c,Bstart,X0,
        NbTrials,StartIdx,ModelBeforeSwitch,Perturb,error, 
        clip=np.array(clip),
        state_retention_pause=state_retention_pause, 
        pause_mask = pause_mask, inds_state_reset=inds_state_reset, 
        stats_powers= np.array(stats_powers),
        motor_var = motor_var,
        use_true_error=use_true_error, use_true_error_stats =use_true_error_stats)

    # Calculate the mean squared error between y_pred and error
    #print(y_pred, error)
    nna = np.isnan(y_pred).sum()
    if nna > 0:
        print(*parvec,NbTrials,StartIdx, nna)
        print(y_pred)
        raise ValueError()
    #std = np.std(error)
    std = 1
    mse0 = np.mean((  (y_pred - error) / std )**2)

    #pen = np.quantile(B[~np.isnan(B)], 0.9)**2 * reg
    pen = np.mean( np.sum(B[~np.isnan(B)]**2 ) ) * reg
    mse =  mse0 + pen
    #print('mse0 = {}, pen = {}'.format( mse0,    pen ) )

    return mse


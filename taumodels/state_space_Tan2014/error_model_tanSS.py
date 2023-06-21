import numpy as np
def error_model_tan(A,c,Bstart,X0,
        NbPreviousTrials,StartIdx,ModelBeforeSwitch,
        Perturb,Error, clip=(-0.5,0.5), pause_mask = None,
                   state_retention_pause = 0.9 ):
    # c is the coefficient in front of the variation
    # last argument is long array
    # X0 is init state
    # A is retention
    # NbPreviousTrials is number of trials taken to comput variablity (in Tan 20)
    # Perturb, Error -- true experimental observations
    # ModelBeforeSwitch either "NoAdapt" or something else (in this case learning rate IC
    # taken to be Bstart

    # Initialize y_pred, state and B as zero vectors with the same size as Error
    y_pred = np.zeros_like(Error)
    state = np.zeros_like(Error)
    B = np.zeros_like(Error)

    if pause_mask is None:
        pause_mask = np.zeros_like(Error)

    # Set the initial state to X0
    state[0] = X0

    # Calculate the first prediction error as the difference between Perturb[0] and X0
    y_pred[0] = Perturb[0] - X0

    # Loop over the rest of the elements in Error
    for t in range(1,len(Error)):
        # If t is less than StartIdx
        ErrorVar = None
        suberr = None

        if pause_mask[t]:
            assert np.isnan( Perturb[t]  )
            B[t] = B[t - 1]
            state[t] = state[t-1] * state_retention_pause
            y_pred[t] = Perturb[t] - state[t]
        else:
            if t < StartIdx:
                # Check the value of ModelBeforeSwitch
                if(ModelBeforeSwitch == "NoAdapt"):
                    # Set B[t] to zero
                    B[t] = 0
                else:
                    # Set B[t] to Bstart
                    B[t] = Bstart
            else:
                # Calculate the mean and standard deviation of the previous NbPreviousTrials errors
                suberr = Error[max(0,t-NbPreviousTrials):t]
                ErrorMean = np.mean(suberr)
                ErrorVar  = np.std( suberr)
    #             ErrorMean = np.mean(y_pred[max(0,t-NbPreviousTrials):t])
    #             ErrorVar = np.std(y_pred[max(0,t-NbPreviousTrials):t])
                # Calculate B[t] as a function of c, ErrorMean and ErrorVar

                if abs(ErrorVar) < 1e-10:
                    # dirty hack but in my data I can have ErrorVaer = 0
                    # for small NbPreviousTrials and or due to "easy" trials
                    B[t] = B[t-1]
                else:
                    B[t] = c*(ErrorMean**2)/(ErrorVar**2)
                # Clip B[t] to the range [-0.5, 0.5]
                B[t] = max( B[t], clip[0] )  # clip from below
                B[t] = min( B[t], clip[1] )  # clip from above
                #if(B[t] >  0.5): B[t] =  0.5
                #if(B[t] < -0.5): B[t] = -0.5

            # Update the state as a function of A, state[t-1], B[t] and y_pred[t-1]
            #print(t, state[t-1], y_pred[t-1], B[t], ErrorVar, Error[t-1], suberr  )
            state[t] = A*state[t-1] + B[t]*y_pred[t-1]
            # Calculate the prediction error as the difference between Perturb[t] and state[t]
            y_pred[t] = Perturb[t] - state[t]

    return y_pred, state, B

# Define a function named MSE_TanSS that accepts 6 inputs and returns 3 outputs
def MSE_error_model_tan(Z,NbTrials,StartIdx,ModelBeforeSwitch,
                        Perturb,Error, pause_mask, clip, reg = 0. ):

    # Unpack Z into four variables
    if pause_mask is None:
        A,c,Bstart,X0 = Z
        state_retention_pause = 0.9
    else:
        A,c,Bstart,X0,state_retention_pause = Z


    # Call the ErrorModel_TanSS function with the unpacked variables and the other inputs
    y_pred, state, B = error_model_tan(A,c,Bstart,X0,
        NbTrials,StartIdx,ModelBeforeSwitch,Perturb,Error, clip=clip,
        state_retention_pause=state_retention_pause, pause_mask = pause_mask)

    # Calculate the mean squared error between y_pred and Error
    #print(y_pred, Error)
    nna = np.isnan(y_pred).sum()
    if nna > 0:
        print(*Z,NbTrials,StartIdx, nna)
        print(y_pred)
        raise ValueError()
    #std = np.std(Error)
    std = 1
    mse0 = np.mean((  (y_pred - Error) / std )**2) 

    #pen = np.quantile(B[~np.isnan(B)], 0.9)**2 * reg
    pen = np.mean( np.sum(B[~np.isnan(B)]**2 ) ) * reg
    mse =  mse0 + pen
    #print('mse0 = {}, pen = {}'.format( mse0,    pen ) )

    return mse


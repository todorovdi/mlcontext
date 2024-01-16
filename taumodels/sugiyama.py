# adapted from Sugiyama 2023

# This version clips the motor learning parameters to specific ranges to prevent diverging.
# While simple clipping with "if" is used here, it can be also implemented in other ways,
# such as conversion by a logistic function and the like.
import numpy as np
import pandas as pd
import scipy

def clip(alpha,beta):
    # clipping
    if alpha > 0: # clip alpha to be negative or zero
        alpha = 0
    elif alpha < -1: # clip alpha to be no less than -1
        alpha = -1

    if beta < 0: # clip beta to be positive or zero
        beta = 0
    elif beta > 1: # clip beta to be no more than 1
        beta = 1

    return alpha,beta

def MSE(errors, reg, **kwargs):
    df = sim_behavior(**kwargs)
    ep = df['error_pred'].values
    pen = reg * SMTH
    mse = scipy.linalg.norm( ep - errors ) + pen
    return mse

# Set simulation code
def sim_behavior(num_tri_train, init_alpha, init_alpha_sd,
                      init_beta, init_beta_sd, show_score, pert, difficulty,
                      score_func, search_noise_sd, eta_beta, eta_alpha,
                      inc_baseline, block_ind,
                 alpha_pause_retention, beta_pause_retention,
                 pause_mask = None,
                    sub_id = None, meta_lrn = 'auto', nspg=None ):
    # sub_id: subject ID
    # nspg: number of subjects per group (only used to deside if meta_lrn is on or off)
    # num_tri_train: number of trials in training phase
    # init_alpha: initial value of alpha parameter (decay rate)
    # init_alpha_sd: standard deviation of noise added to initial alpha
    # init_beta: initial value of beta parameter (learning rate)
    # init_beta_sd: standard deviation of noise added to initial beta
    # show_score: a list indicating whether feedback is provided for each trial
    # pert: a list of rotation angles for each trial (perturabtion sizes/directions)
    # difficulty: difficulty level of the task
    # score_func: a function that calculates the performance score based on motor memory, rotation angle, difficulty level, and meta learning group
    # search_noise_sd: standard deviation of exploration noise
    # eta_beta: learning rate for beta parameter
    # eta_alpha: learning rate for alpha parameter
    # inc_baseline: a boolean indicating whether to include baseline block in the simulation
    # block_ind: a list of block numbers (Q numbers or IDs?) for each trial

    if isinstance(meta_lrn,str) and meta_lrn == 'auto':
        assert  nspg is not None
        meta_lrn = 1 if sub_id <= nspg else 0 # just assing half of subjects to Lrn and the other to NLrn

    spe =          [0] * num_tri_train # prediction error
    score =        [0] * num_tri_train # performance score

    # motor memory (latent), but, since observation is defined deterministically (y=x),
    # this is considered observable in this simulation
    x =            [0] * num_tri_train
    policy =       [0] * num_tri_train # action policy
    search_noise = [0] * num_tri_train # exploration noise (just realization of normal noise, zero autocorrel)
    alpha =        [0] * num_tri_train # decay rate parameter
    beta =         [0] * num_tri_train # learning rate parameter
    error_expectation = [0] * num_tri_train # in the very beg don't expect an error

    assert not (pause_mask is not None) and (show_score is not None)
    if pause_mask is None:
        pause_mask = [0] * num_tri_train
    else:
        show_score = pause_mask

    # set initial param
    # both alpha and beta change during
    alpha[0] = init_alpha + np.random.normal(0, init_alpha_sd) # initialize alpha with some noise
    beta[0]  = init_beta  + np.random.normal(0, init_beta_sd) # initialize beta with some noise

    alpha[0],beta[0] = clip(alpha[0],beta[0])

    for ti in range(num_tri_train): # loop over trials

        ### Behavior

        # M Trial
        if show_score[ti] == 1: # if feedback is provided
            score[ti] = score_func(x[ti],pert[ti - 1], difficulty, meta_lrn) # scoring for lrn and nlrn is defined inside this function.
            spe[ti] = 0 # no prediction error

        # E or Null
        else: # if feedback is not provided
            score[ti] = np.nan # no score
            spe[ti] = pert[ti] - x[ti] # prediction error is the difference between rotation and motor memory


        ### Learning
        # ES (and retention) is updated only if score is provided

        ## memory update
        search_noise[ti] = np.random.normal(0, search_noise_sd) # generate exploration noise from normal distribution
        # calculate action policy as a function of motor memory, prediction error, and exploration noise
        # retain a bit of prev motor state, add ES*SPE
        policy[ti] = alpha[ti]*x[ti] + beta[ti]*spe[ti] + search_noise[ti]

        if ti != num_tri_train - 1: # if not the last trial
        # recall that alpha is negative
            x[ti+1] = x[ti] + policy[ti] # update motor memory by adding action policy

        error_expectation[ti+1] = x[ti+1] + pert[ti+1]

        ## Meta-learn
        # if feedback is provided and
        # not the first or last trial and
        # not in baseline block (if inc_baseline is True)
        # [probably] if include baseline then expect block_ind[ti] = 0 corresp
        # to baseline
        if show_score[ti] == 1 and ti > 0 and ti != num_tri_train - 1 and\
            (not inc_baseline or block_ind[ti] > 1):
            # update beta by adding a term proportional to the product of previous exploration noise, prediction error, and score
            beta[ti+1]  = beta[ti]  + eta_beta*search_noise[ti-1] *spe[ti-1]*score[ti]
            # update alpha by adding a term proportional to the product of previous exploration noise, motor memory, and score
            alpha[ti+1] = alpha[ti] + eta_alpha*search_noise[ti-1]*x[ti-1]  *score[ti]

            # Q: why don't we do clipping here?
        else: # otherwise
            # Q: why don't we decay anything here?
            if ti != num_tri_train - 1: # if not the last trial
                if pause_mask[ti]:
                    alpha[ti+1] = alpha[ti] * alpha_pause_retention
                    beta[ti+1]  = beta[ti]  * beta_pause_retention
                else:
                    beta[ti+1]  = beta[ti] # keep beta unchanged
                    alpha[ti+1] = alpha[ti] # keep alpha unchanged

                    # clipping
                    alpha[ti+1],beta[ti+1] = clip(alpha[ti+1],beta[ti+1])

    df_res = pd.DataFrame({'sub_id': sub_id, 'meta_lrn': meta_lrn,
        'perturbation': pert,
        'score': score, 'x': x,
        'policy': policy, 'search_noise':search_noise,
        'error_pred':error_expectation,
            'alpha': alpha, 'beta': beta}) # create a data frame with the results
    return df_res


# for Herzfeld and Galea they took reward = -|TaskError|

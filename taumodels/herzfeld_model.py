import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True, cache=False)
def calc(error_region, initial_sensitivity,
        gaussian_variance, # of gaussians
        eta,  # weight update speed
        alpha,  # retention
        alpha_w, # retention of weights
        weight_forgetting_exp_pause,
        execution_noise_variance,
        sensory_noise_variance,
        target_loc,        
        pause_mask,
        pre_break_duration,
        perturbations, # only needed to compute errors
        channel_mask,
        true_errors,
        num_bases = 10 ,
        small_err_thr = 0.,
        use_true_errors = False,
        cap_err_sens = True,
        verbose=0):
    #for k,v in args.items():
    #    if k != 'perturbation':
    #        print(k,v)
    #print('use_true_errors =',use_true_errors)
    # Where are the gaussian centers located?
    assert num_bases > 0
    gaussian_centers = np.ascontiguousarray( np.linspace(error_region[0],
                    error_region[1], num_bases) )

    # in this model ES cannot get negative 
    if cap_err_sens:
        assert initial_sensitivity >= 0, initial_sensitivity

    # What are the variances?
    gaussian_variances = np.ascontiguousarray( np.ones(num_bases) * gaussian_variance )

    # Initialize weights for the Gaussian bases
    w = np.ascontiguousarray( np.zeros(num_bases) )

    # I think this is appropriate for original Herzfeld 2014 where initial ES is not added to transient ES explicitly
    #while get_sensitivity(0, w, gaussian_centers, gaussian_variances) < initial_sensitivity:
    #    w = w + 0.001
    #    #print(w)
    #w_initial = w

    #w = w_initial

    assert len(pre_break_duration) == len(perturbations)


    # Where is the target located (cm)?
    #from collections.abc import Iterable
    #target_reach = target_loc
    #if not isinstance(target_reach, Iterable):
    #    target_reach = [0] * len(perturbations)
    target_reach = np.repeat(target_loc, len(perturbations)  )
    #if not isinstance(target_reach, Iterable):
    #    target_reach = [0] * len(perturbations)

    # What is the initial reach (cm)?
    x_0 = 0

    # Loop through each of perturbations
    # motor output is org_feedback - target_loc
    motor_output = np.zeros(len(perturbations))
    motor_output[0] = x_0
    error_pred = np.ascontiguousarray( np.zeros(len(perturbations), dtype=np.float64) ) # predicted errors
    err_sens = np.repeat(np.nan, len(perturbations) ) 
    ws = np.full( (len(perturbations), num_bases), np.nan )


    error_for_update = np.nan
    error_for_update_w = np.nan

    rng = np.arange(len(perturbations) - 1, dtype=np.int64)
    #print(rng)
    for i in rng:

        if pause_mask[i]:
            print(i,'paaause')
            assert i > 0
            assert np.isnan( perturbations[i]  )
            w = w * np.exp( - weight_forgetting_exp_pause )
            error_pred[i] = np.nan
            s = np.nan # in this case we don't want to save ES
            motor_output[i] = np.nan # on the prev trial we assigned it to not-nan assuming that it was a regular trial

            if use_true_errors:
                error_for_update = true_errors[i-1]
            else:
                error_for_update = error_pred[i-1]

            # here unlike in the main equations we update weights first
            # here we use i-1, not i like in main equations
            s_ = get_sensitivity(error_for_update, w, gaussian_centers, gaussian_variances)
            #assert isinstance(s_,float), (type(s_), s_)
            if (s_ > 1) and cap_err_sens:
                if verbose:
                    print(f"Warning: {i} Sensitivity > 1 {s}")
                s_ = 1
            motor_output[i+1] = alpha * motor_output[i-1] -\
                s_ * (error_for_update + np.random.randn() * \
                     sensory_noise_variance)
        else:
            # know it in the end of the trial
            error_pred[i] = motor_output[i] - (target_reach[i] + perturbations[i]) + \
                    np.random.randn() * execution_noise_variance
            if channel_mask[i]:
                error_pred[i] = 0

            if use_true_errors:
                error_for_update = true_errors[i]
                if i > 0:
                    error_for_update_w = true_errors[i-1]
            else:
                error_for_update = error_pred[i]
                if i > 0:
                    error_for_update_w = error_pred[i-1]

            if pre_break_duration[i] > 1e-10:
                # decay weights in pause
                w = w * np.exp( - weight_forgetting_exp_pause * pre_break_duration[i] ) 

            # Now that we have the error on the reach, we can determine the next
            # motor output
            #print('i = ',i)
            #print(w.shape, gaussian_centers.shape, gaussian_variances.shape)
            s = initial_sensitivity + get_sensitivity(error_for_update, w, gaussian_centers, gaussian_variances)
            #assert isinstance(s,float), (type(s), s)
            if (s > 1) and cap_err_sens:
                if verbose:
                    print(f"Warning: {i} Sensitivity > 1 {s}")
                s = 1

            #print(i,'kddc')
            # Create the motor output for the next trial
            motor_output[i+1] = alpha * motor_output[i] -\
                s * (error_for_update + np.random.randn() * \
                     sensory_noise_variance)
            #print(i,'post kddc')

            # Update the weights
            if i > 0:
                signc =  np.sign(error_for_update_w * error_for_update)
                # if both are larger than thr
                if np.min(np.abs(np.array([error_for_update_w, error_for_update]) ))  >= small_err_thr:
                    w = update_weights(error_for_update_w, w,
                        gaussian_centers, gaussian_variances,
                        signc, eta, alpha_w )
            #print(i,'post uw')

        err_sens[i+1] = s
        ws[i] = w
        #print(i,err_sens[i+1])
        #print(i,w)

    #print(err_sens[-1])


    #err_sens = np.array(err_sens)
    return (motor_output,error_pred,ws,err_sens,
            gaussian_centers,gaussian_variances)

@jit(nopython=True)
def get_sensitivity(error, weights, centers, variances):
    #sensitivity = 0
    #print(weights.shape)
    #print(error.shape)
    #print(centers.shape)
    #for i in range(len(weights)):
    #    sensitivity += weights[i] * np.exp(-(error - centers[i])**2 / (2 * variances[i]**2))
    
    sensitivity = weights * np.exp(-(error - centers)**2 / (2 * variances**2))
    sensitivity = np.sum(sensitivity)
        #print(sensitivity)
    return sensitivity

@jit(nopython=True)
def update_weights(error, weights, centers, variances, sign_consist, eta,
                   alpha_w):
    # sign_consist is sign consistency
    g_x = np.exp(-(error - centers)**2 / (2 * variances**2))
    #invarg = g_x * g_x.T
    invarg = g_x[:,None] @ g_x[:,None].T
    weights_upd = alpha_w * weights + eta * sign_consist * np.linalg.pinv(invarg) @ g_x
    return weights_upd

def plot_bases(weights, centers, variances):
    x_axis = np.linspace(np.min(centers), np.max(centers), 100)
    for i in range(len(weights)):
        plt.plot(x_axis, weights[i] * np.exp(-(x_axis - centers[i])**2 / (2 * variances[i]**2)), linewidth=2.0)
    plt.xlabel('Error')
    plt.ylabel('Basis Set')
    plt.show()

def plot_sensitivity(weights, centers, variances):
    x_axis = np.linspace(np.min(centers), np.max(centers), 100)
    y_values = np.zeros(len(x_axis))
    for i in range(len(weights)):
        y_values += weights[i] * np.exp(-(x_axis - centers[i])**2 / (2 * variances[i]**2))
    plt.plot(x_axis, -y_values, linewidth=2.0)
    plt.xlabel('Error')
    plt.ylabel('Sensitivity (unitless)')
    plt.show()

def plot_adaptation(weights, centers, variances):
    x_axis = np.linspace(np.min(centers), np.max(centers), 100)
    y_values = np.zeros(len(x_axis))
    for i in range(len(weights)):
        y_values += weights[i] * np.exp(-(x_axis - centers[i])**2 / (2 * variances[i]**2))
    y_values = y_values * x_axis
    plt.plot(x_axis, -y_values, linewidth=2.0)
    plt.xlabel('Error')
    plt.ylabel('Adaptation')
    plt.show()

def plot_motor_output(perturbations, target_reach, motor_output):
    plt.plot(range(1, len(perturbations) + 1), perturbations + target_reach, linewidth=2)
    plt.plot(range(1, len(perturbations) + 1), motor_output, linewidth=2)
    plt.legend(['Reach Goal', 'Motor Output'])
    plt.xlabel('Trial #')
    plt.ylabel('Distance (cm)')
    plt.show()

def plot_binned_adaptation(errors, motor_output):
    errors = errors[:-1]
    bins = np.linspace(np.min(errors), np.max(errors), 15)
    bin_width = (np.max(errors) - np.min(errors)) / 15
    adaptation = motor_output[1:] - motor_output[:-1]
    binned_adaptation = np.zeros(len(bins))
    for i in range(len(binned_adaptation)):
        binned_adaptation[i] = np.mean(adaptation[(errors > bins[i] - bin_width / 2) & (errors < bins[i] + bin_width / 2)])
    plt.plot(bins, binned_adaptation, linewidth=2.0)
    plt.xlabel('Error')
    plt.ylabel('Adaptation')
    plt.show()


def gen_markov_perturbation(n, zeta):
    perturbations = np.zeros(n)
    perturbations[0] = (np.random.rand() < 0.5) - 0.5
    num_transitions = 0
    while (np.sum(perturbations) != 0 and num_transitions != np.ceil(n * zeta)):
        for i in range(1, n):
            if np.random.rand() < zeta:
                perturbations[i] = -perturbations[i-1]
                num_transitions += 1
            else:
                perturbations[i] = perturbations[i-1]
    return perturbations

#if __name__ == '__main__':
#    # this should be subject to some sort of optimization
#    args = {
#        'num_bases': 20,
#        'error_region': 10,
#        'initial_sensitivity': 0.1,
#        'perturbations': [],
#        'variance': 1,
#        'eta': 0.01,
#        'alpha': 1,
#        'execution_noise_variance': 0.0,
#        'sensory_noise_variance': 0.0,
#        'zeta': 0.9,
#        'channels': [],
#        'plot': False
#    }
#
#    r = calc(args)
#
#    if args['plot']:
#        # Figure 1
#        plt.figure(1)
#        plt.clf()
#
#        # Subplot 1
#        plt.subplot(2, 1, 1)
#        plot_adaptation(w, gaussian_centers, gaussian_variances)
#
#        # Subplot 2
#        plt.subplot(2, 1, 2)
#        plot_motor_output(perturbations, target_reach, motor_output)
#
#        # Figure 2
#        plt.figure(2)
#        plot_adaptation(w, gaussian_centers, gaussian_variances)
#
#        # Figure 3
#        plt.figure(3)
#        plot_adaptation(w - w_initial, gaussian_centers, gaussian_variances)
#
#        # Show all plots
#        plt.show()


if __name__ == "__main__":
    from herzfeld_model import calc
    import numpy as np
    calc( np.array([-30,30]), 0,
    1,             #gaussian_variance, # of gaussians 
    0,             #eta,  # weight update speed       
    0,             #alpha,  # retention               
    0,             #alpha_w, # retention of weights   
    0,             #weight_forgetting_exp_pause,      
    0,             #execution_noise_variance,         
    0,             #sensory_noise_variance,           
    0,             # target_loc,          
    np.array([0,0,0]),       # pause_mask,                      
    np.array([0,1.5,0]),     # pre_break_duration,              
    np.array([0,0,30]),      # perturbations,                   
    np.array([0,0,0]),       # channel_mask,                    
    true_errors = np.array([0,0,0]) )       # true_errors,                     
                   # num_bases = 10 ,                 
                   # verbose=0):                      
                                

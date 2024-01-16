import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def calc(error_region, initial_sensitivity,
        gaussian_variance, # of gaussians
        eta,  # weight update speed
        alpha,  # retention
        alpha_w, # retention of weights
        execution_noise_variance,
        sensory_noise_variance,
        target_loc,
        pause_mask,
        weight_retention_pause,
        perturbation,
        channel_mask,
        true_errors = None,
        num_bases = 10 ,
         verbose=0):
    #for k,v in args.items():
    #    if k != 'perturbation':
    #        print(k,v)
    # Where are the gaussian centers located?
    gaussian_centers = np.linspace(-args['error_region'],
                    args['error_region'], args['num_bases'])

    # What are the variances?
    gaussian_variances = np.ones(args['num_bases']) * args['variance']

    # Initialize weights for the Gaussian bases
    w = np.zeros(args['num_bases'])
    while get_sensitivity(0, w, gaussian_centers, gaussian_variances) < args['initial_sensitivity']:
        w = w + 0.001
    w_initial = w

    # Generate the perturbations
    if 'perturbation' not in args:
        print(f'Using Markov pert with z={args["zeta"]}')
        perturbations = gen_markov_perturbation(100, args['zeta'])
        perturbations[perturbations < 0] = 0
        perturbations[perturbations > 0] = 1
    else:
        perturbations = args['perturbation']

    pauses = args.get('pause_mask', None)
    if pauses is None:
        pauses = np.zeros_like(perturbations)

    # Check to see if any of these are channel trials
    if 'channel_mask' not in args:
        channels = np.zeros(perturbations.shape)
    else:
        channels = args['channel_mask']

    # Where is the target located (cm)?
    from collections.abc import Iterable
    target_reach = args['target_loc']
    if not isinstance(target_reach, Iterable):
        target_reach = [0] * len(perturbations)

    # What is the initial reach (cm)?
    x_0 = 0

    # Loop through each of perturbations
    # motor output is org_feedback - target_loc
    motor_output = np.zeros(len(perturbations))
    motor_output[0] = x_0
    errors = np.zeros(len(perturbations)) # predicted errors
    err_sens = [np.nan]
    ws = []
    for i in range(len(perturbations) - 1):
        if pauses[i]:
            assert np.isnan( perturbations[i]  )
            w = w * args['weight_retention_pause']
            errors[i] = np.nan
            s = np.nan
            motor_output[i] = np.nan # on the prev trial we assigned it to not-nan assuming
            # that it is a regular trial

            # here unlike in the main equations we update weights first
            # here we use i-1, not i like in main equations
            s_ = get_sensitivity(errors[i-1], w, gaussian_centers, gaussian_variances)
            assert isinstance(s,float), (type(s), s)
            if (s > 1).any():
                if verbose:
                    print(f"Warning: {i} Sensitivity > 1 {s}")
                s = 1
            motor_output[i+1] = args['alpha'] * motor_output[i-1] -\
                s_ * (errors[i-1] + np.random.randn() * \
                     args['sensory_noise_variance'])
        else:
            # know it in the end of the trial
            errors[i] = motor_output[i] - (target_reach[i] + perturbations[i]) + \
                    np.random.randn() * args['execution_noise_variance']
            if channels[i]:
                errors[i] = 0

            # Now that we have the error on the reach, we can determine the next
            # motor output
            #print('i = ',i)
            #print(w.shape, gaussian_centers.shape, gaussian_variances.shape)
            s = get_sensitivity(errors[i], w, gaussian_centers, gaussian_variances)
            assert isinstance(s,float), (type(s), s)
            if (s > 1).any():
                if verbose:
                    print(f"Warning: {i} Sensitivity > 1 {s}")
                s = 1

            # Create the motor output for the next trial
            motor_output[i+1] = args['alpha'] * motor_output[i] -\
                s * (errors[i] + np.random.randn() * \
                     args['sensory_noise_variance'])

            # Update the weights
            if i > 0:
                w = update_weights(errors[i-1], w,
                    gaussian_centers, gaussian_variances,
                    np.sign(errors[i-1] * errors[i]), args['eta'],
                                   args['alpha_w'] )

        err_sens += [s]
        ws += [w]


    err_sens = np.array(err_sens)
    return (motor_output,errors,ws,err_sens,
            gaussian_centers,gaussian_variances)

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

@jit(nopython=True)
def get_sensitivity(error, weights, centers, variances):
    sensitivity = 0
    #print(weights.shape)
    #print(error.shape)
    #print(centers.shape)
    for i in range(len(weights)):
        sensitivity += weights[i] * np.exp(-(error - centers[i])**2 / (2 * variances[i]**2))
        #print(sensitivity)
    return sensitivity

@jit(nopython=True)
def update_weights(error, weights, centers, variances, tde, eta,
                   alpha_w):
    g_x = np.exp(-(error - centers)**2 / (2 * variances**2))
    #invarg = g_x * g_x.T
    invarg = g_x[:,None] @ g_x[:,None].T
    weights = alpha_w * weights + eta * tde * np.linalg.pinv(invarg) @ g_x
    return weights

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

if __name__ == '__main__':
    # this should be subject to some sort of optimization
    args = {
        'num_bases': 20,
        'error_region': 10,
        'initial_sensitivity': 0.1,
        'perturbations': [],
        'variance': 1,
        'eta': 0.01,
        'alpha': 1,
        'execution_noise_variance': 0.0,
        'sensory_noise_variance': 0.0,
        'zeta': 0.9,
        'channels': [],
        'plot': False
    }

    r = calc(args)

    if args['plot']:
        # Figure 1
        plt.figure(1)
        plt.clf()

        # Subplot 1
        plt.subplot(2, 1, 1)
        plot_adaptation(w, gaussian_centers, gaussian_variances)

        # Subplot 2
        plt.subplot(2, 1, 2)
        plot_motor_output(perturbations, target_reach, motor_output)

        # Figure 2
        plt.figure(2)
        plot_adaptation(w, gaussian_centers, gaussian_variances)

        # Figure 3
        plt.figure(3)
        plot_adaptation(w - w_initial, gaussian_centers, gaussian_variances)

        # Show all plots
        plt.show()

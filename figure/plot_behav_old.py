import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pandas
from pandas import DataFrame as df
from config2 import subjects, path_data, path_fig
from base2 import (width, height, radius, calc_target_coordinates_centered,
                   calc_rad_angle_from_coordinates)
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel

sns.set_palette('colorblind')
plt.style.use('seaborn')
color_palette = sns.color_palette("colorblind", 8).as_hex()
colors = [color_palette[1], color_palette[7]]

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma']
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Calculate the X and Y coordinates of the targets (centered)
# Here we add 90 to start at the bottom vertical (as feedback)
target_angs = (np.array([157.5, 112.5, 67.5, 22.5]) + 90) * \
              (np.pi/180)
target_coords = calc_target_coordinates_centered(target_angs)


# Function to fit (b0 is baseline, b1 is starting point, b2 is learning rate)
def func(x, b0, b1, b2, b3):
    return b0 + b1*np.exp(-b2*(x-b3))


# Find blocks of consecutive values (for perturbations blocks)
def ranges(where_perturb):
    from itertools import groupby
    from operator import itemgetter
    blocks = list()
    for k, g in groupby(enumerate(where_perturb), lambda ix: ix[0] - ix[1]):
        blocks.append(np.array(list((map(itemgetter(1), g)))))
    return blocks

task = 'VisuoMotor'

# Boundaries of the function
bounds = ([0, 0, 0, 0.01],
          [50*np.pi/180, 50*np.pi/180, 50*np.pi/180, 1])

all_errors = list()
all_perturbations = list()
all_trajectoryX = list()
all_trajectoryY = list()
for subject in subjects:
    folder = op.join(path_data, subject, 'behavdata')
    fname = op.join(folder,f'behav_{task}_df.pkl')
    behav_df = pandas.read_pickle(fname)
    errors = np.array(behav_df['error'])
    perturbations = np.array(behav_df['perturbation'])
    trajectoryX = np.array(behav_df['trajectoryX'])
    trajectoryY = np.array(behav_df['trajectoryY'])
    environment = np.array(behav_df['environment'])
    feedback = np.array(behav_df['feedback'])
    org_feedback = np.array(behav_df['org_feedback'])
    target = np.array(behav_df['target'])

    # Convert all_perturbations in radians
    perturbations = perturbations * (np.pi/180.)
    # calculate the resulting perturbation during random environment
    for ii in range(len(perturbations)):
        if environment[ii]:
            perturbations[ii] = feedback[ii] - org_feedback[ii]
    # Participants randomly started with 30 or -30 degree perturb. To plot the
    # errors, we invert the perturbation and error to always start with 30.
    if perturbations[24] != 30 * (np.pi/180.):
        perturbations = -perturbations
        errors = -errors
    all_perturbations.append(perturbations)
    all_errors.append(errors)
    all_trajectoryX.append(trajectoryX)
    all_trajectoryY.append(trajectoryY)

all_perturbations = np.array(all_perturbations)
all_errors = np.array(all_errors)

# Plot perturbations and errors
plt.figure(figsize=(2.6, 2))
plt.plot(range(0, 192), all_perturbations[1].T[0:192],
         color=color_palette[0], linewidth=0.5,
         label='Stable Perturbations')
plt.plot(range(192, 384), all_perturbations[1].T[192:384],
         color=color_palette[0], linewidth=0.2,
         label='Random Perturbations', alpha=1)
plt.plot(range(384, 576), all_perturbations[1].T[384:576],
         color=color_palette[0], linewidth=0.5)
plt.plot(range(576, 768), all_perturbations[1].T[576:768],
         color=color_palette[0], linewidth=0.2, alpha=1)

plt.plot(range(0, 192), all_errors.T[0:192], color=colors[0], alpha=0.01)
# plt.plot(range(192, 384), all_errors.T[192:384], color=colors[1], alpha=0.01)
plt.plot(range(384, 576), all_errors.T[384:576], color=colors[0], alpha=0.01)
# plt.plot(range(576, 768), all_errors.T[576:768], color=colors[1], alpha=0.01)
plt.plot(range(0, 192), all_errors.T[0:192].mean(axis=1),
         color=colors[0], linewidth=0.3,
         label='Errors (stable environment)')
plt.plot(range(192, 384), all_errors[1].T[192:384],
         color=colors[0], linewidth=0.2,
         label='Errors (random environment)', alpha=1)
plt.plot(range(384, 576), all_errors.T[384:576].mean(axis=1),
         color=colors[0], linewidth=0.3)
plt.plot(range(576, 768), all_errors[1].T[576:768],
         color=colors[0], linewidth=0.2, alpha=1)
plt.ylim(-np.pi/4., np.pi/4.)
plt.yticks([-np.pi/6., 0, np.pi/6.], [])
plt.xticks([0, 100, 200, 300, 400, 500, 600, 700])
plt.xlabel('Trial Number')
plt.tight_layout()
# plt.legend()
fname_fig = op.join( path_fig, 'fig1_behav_Visuo_exp2.png')
plt.savefig(fname_fig, dpi=400)
plt.close()

# Fit learning function
perturb_blocks = ranges(np.where(np.abs(all_perturbations[0]) == 30*np.pi/180.)[0])
first_perturb = all_errors[:, perturb_blocks[0]]
second_perturb = -all_errors[:, perturb_blocks[1]]
third_perturb = -all_errors[:, perturb_blocks[2]]
fourth_perturb = all_errors[:, perturb_blocks[3]]
all_b1s = list()
all_b2s = list()
x = np.arange(first_perturb.shape[1])
for lcs in [first_perturb, second_perturb, third_perturb, fourth_perturb]:
    b1s = list()
    b2s = list()
    for lc in lcs:
        popt, pcov = curve_fit(func, xdata=x, ydata=lc, bounds=bounds)
        b1s.append(popt[1])
        b2s.append(popt[2])
    b1s = np.array(b1s)
    b2s = np.array(b2s)
    all_b1s.append(b1s)
    all_b2s.append(b2s)
all_b1s = np.array(all_b1s)
all_b2s = np.array(all_b2s)
ttest_rel(all_b2s[2], all_b2s[3])
print(f'Plotting finished, saved to {fname_fig}')

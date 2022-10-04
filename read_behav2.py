import os
import os.path as op
import numpy as np
from pandas import DataFrame as df
from config2 import *
from base2 import (width, height, radius, calc_target_coordinates_centered,
                  calc_rad_angle_from_coordinates)
from error_sensitivity import computeErrSens
home_radius = 8 + 12  # radius_cursor + radius_target

all_trials_duration = list()
all_reaction_time = list()
all_movement_duration = list()
# this is task for the behav filename, it is not the same as in processed filename
task = 'visuomotor'

#sl = slice(None,2)
#sl = slice(None,None,None)
#for subject in subjects[sl]:
folder = op.join(path_data, subject, 'behavdata')
files = os.listdir(folder)
fname_behavior = list()
fname_behavior.extend([op.join(folder, f) for f in files if ((task in f) and
                                                             ('.log' in f))])
log = np.genfromtxt(fname_behavior[0], dtype=float,
                    delimiter=',')

# ---------------- Print infos about the session
# print number of trials
print('----------------' + subject + '-----------------------')
nb_trials = int(log[-1, 0]) + 1
print('Total number of trials: ' + str(nb_trials))
# print session duration
total_duration = str((log[-1, 12] - log[0, 12])/60.)
print('Total duration (min): ' + total_duration)
# print average trial duration (columns 11 is time in log)
# print the movement duration (from target onset to feedback)
trial_duration = list()
reaction_time = list()
home_duration = list()
movement_duration = list()
feedback_duration = list()
ITI_duration = list()
for trial in range(nb_trials):
    i = np.where(log[:, 0] == trial)[0]
    home_phase = np.where(log[i, 1] == 10)[0]
    target_phase = np.where(log[i, 1] == 20)[0]
    feedback_phase = np.where(log[i, 1] == 30)[0]
    ITI_phase = np.where(log[i, 1] == 40)[0]
    if task == 'visuomotor':
        trial_duration.append(log[i[-1], 12] - log[i[0], 12])
        home_duration.append(log[i[home_phase[-1]], 12] -
                             log[i[home_phase[0]], 12])
        movement_duration.append(log[i[target_phase[-1]], 12] -
                                 log[i[target_phase[0]], 12])
        feedback_duration.append(log[i[feedback_phase[-1]], 12] -
                                 log[i[feedback_phase[0]], 12])
        ITI_duration.append(log[i[ITI_phase[-1]], 12] -
                            log[i[ITI_phase[0]], 12])
        # if trial != nb_trials:
        #     reach_to_netxtrial_time
    elif task == 'locaerror':
        trial_duration.append(log[i[-1], 8] - log[i[0], 8])
        movement_duration.append(log[i[target_phase[-1]], 8] -
                                 log[i[target_phase[0]], 8])
    for ii, (x, y) in enumerate(zip(log[i[target_phase[:]], 4],
                                    log[i[target_phase[:]], 5])):
        dist = ((x-width/2.)**2 + (y-height/2.)**2)**(1/2.)
        if dist > home_radius:
            reaction_time.append(log[i[target_phase[ii]], 12] -
                                 log[i[target_phase[0]], 12])
            break
print('Average trial duration (sec): ' + str(np.mean(trial_duration)))
print('Home Duration (sec): ' + str(np.mean(home_duration)))
print('Reaction Time (sec): ' + str(np.mean(reaction_time)))
print('Average movement duration (sec): ' +
      str(np.mean(movement_duration)))
print('Feedback duration (sec): ' + str(np.mean(feedback_duration)))
print('ITI duration (sec): ' + str(np.mean(ITI_duration)))

# Calculate the X and Y coordinates of the targets (centered)
# Here we add 90 to start at the bottom vertical (as feedback)
target_angs = (np.array([157.5, 112.5, 67.5, 22.5]) + 90) * \
              (np.pi/180)
target_coords = calc_target_coordinates_centered(target_angs)


# ----------------- Output the errors, perturbation and trajectory
err = list()
feedbackX = list()
feedbackY = list()
targets = list()
target_locs = list()
target_inds = list()
target_codes = list()
perturbations = list()
trajectoryX_ = list()
trajectoryY_ = list()
org_feedbackX = list()
org_feedbackY = list()
environment = list()
# old
#for trial in range(nb_trials):
#    trial_inds = np.where(log[:, 0] == trial)[0]
#    target_phase = np.where(log[trial_inds, 1] == 20)[0]
#    feedback_phase = np.where(log[trial_inds, 1] == 30)[0]
#    err.append(log[trial_inds[feedback_phase[10]], 10])
#    # clockwise perturbation is positive in the task. Here we apply minus to
#    # use trigonometry convention (counterclockwise is positive)
#    perturbations.append(-log[trial_inds[feedback_phase[10]], 3])
#    trajectoryX_.append(log[trial_inds[target_phase], 4] - width/2)
#    trajectoryY_.append(-(log[trial_inds[target_phase], 5] - height/2))
#    feedbackX.append(log[trial_inds[feedback_phase[10]], 6] - width/2)
#    feedbackY.append(-(log[trial_inds[feedback_phase[10]], 7] - height/2))
#    org_feedbackX.append(log[trial_inds[feedback_phase[10]], 8] - width/2)
#    org_feedbackY.append(-(log[trial_inds[feedback_phase[10]], 9] - height/2))
#    targets.append(int(log[trial_inds[feedback_phase[10]], 2]))
#    environment.append(int(log[trial_inds[feedback_phase[10]], 11]))


ab = env2envcode.items()
a,b = zip(*ab)
envcode2env = dict( zip(b,a) )
# log is an array of shape numtrials x numvars
# width -- size of the screen
TARGET_PHASE = 20
FEEDBACK_PHASE = 30
col_trigger = 1
col_tgt    = 2
col_pert   = 3
col_traj_X = 4
col_traj_Y = 5

col_feedback_X = 6
col_feedback_Y = 7

col_org_feedback_X = 8
col_org_feedback_Y = 9


# This variable saves the unperturbed, unprojected feedback
# self.org_feedbackX, self.org_feedbackY = self.joyX, self.joyY
# self.error_distance = np.sqrt((self.feedbackX - self.target_types[self.target_to_show][0])**2 +
#                                              (self.feedbackY - self.target_types[self.target_to_show][1])**2)

col_err = 10            # err is saved during experiment
col_env = 11
spec_timeind = 10
# Q: why alwasy 10th index?
for trial in range(nb_trials):
    trial_inds = np.where(log[:, 0] == trial)[0]
    log_cur = log[trial_inds,:]
    timeinds_tgt = np.where(log_cur[:,col_trigger] == TARGET_PHASE)[0]
    timeinds_fb  = np.where(log_cur[:,col_trigger] == FEEDBACK_PHASE)[0]

    err.append(log_cur[timeinds_fb[spec_timeind], col_err])
    # clockwise perturbation is positive in the task. Here we apply minus to
    # use trigonometry convention (counterclockwise is positive)
    perturbations.append(-log_cur[timeinds_fb[spec_timeind], col_pert])
    trajectoryX_.append(  log_cur[timeinds_tgt, col_traj_X] - width/2)
    trajectoryY_.append(-(log_cur[timeinds_tgt, col_traj_Y] - height/2))

    feedbackX.append(  log_cur[timeinds_fb[spec_timeind], col_feedback_X] - width/2)
    feedbackY.append(-(log_cur[timeinds_fb[spec_timeind], col_feedback_Y] - height/2))

    org_feedbackX.append(  log_cur[timeinds_fb[spec_timeind], col_org_feedback_X] - width/2)
    org_feedbackY.append(-(log_cur[timeinds_fb[spec_timeind], col_org_feedback_Y] - height/2))

    envcode = int(log_cur[timeinds_fb[spec_timeind], col_env])
    env = envcode2env[envcode]
    environment.append(envcode)
    # it is in int from 0 to 3
    target_ind = int(log_cur[timeinds_fb[spec_timeind], col_tgt])
    target_inds.append(target_ind)
    if env == 'stable':
        target_codes.append( event_ids_tgt_stable[target_ind] )
    else:
        target_codes.append( event_ids_tgt_random[target_ind] )

    target_locs.append( target_angs[target_inds[trial]] )



err = np.array(err)
feedbackX = np.array(feedbackX)  # x coordinate of the feedback
feedbackY = np.array(feedbackY)  # y coordinate of the feedback
org_feedbackX = np.array(org_feedbackX)  # x coordinate of the org_feedback
org_feedbackY = np.array(org_feedbackY)  # y coordinate of the org_feedback
target_codes = np.array(target_codes)  # trial type (from 0 to 3)
target_inds = np.array(target_inds)  # trial type (from 0 to 3)
target_locs = np.array(target_locs)  # trial type (from 0 to 3)
environment = np.array(environment)
perturbations = np.array(perturbations)
trajectoryX = np.array(trajectoryX_, dtype=object)
trajectoryY = np.array(trajectoryY_, dtype=object)
# get feedback in rad (with 0 being the bottom vertical)
feedback = calc_rad_angle_from_coordinates(feedbackX, feedbackY)
# get org_feedback in rad (with 0 being the bottom vertical)
org_feedback = calc_rad_angle_from_coordinates(org_feedbackX,
                                               org_feedbackY)
# get error in rad
errors = list()
for trial in range(nb_trials):
    errors.append(feedback[trial] - target_locs[trial])
errors = np.array(errors)
# get the belief in rad
belief = list()
for trial in range(nb_trials):
    belief.append(org_feedback[trial] - target_locs[trial])
belief = np.array(belief)

# # plot errors and perturbations
# plt.plot(perturbations)
# plt.plot(errors)
# # plot targets and trajectory
# for targ in target_coords:
#     plt.plot(targ[0], targ[1], 'ro', markersize=15, color = 'C7')
# for i in range(len(trajectoryX)):
#     if targets[i] == 0:
#         plt.plot(trajectoryX[i], trajectoryY[i], color='C1')
#     if targets[i] == 1:
#         plt.plot(trajectoryX[i], trajectoryY[i], color='C2')
#     if targets[i] == 2:
#         plt.plot(trajectoryX[i], trajectoryY[i], color='C3')
#     if targets[i] == 3:
#         plt.plot(trajectoryX[i], trajectoryY[i], color='C4')


task = 'VisuoMotor'
behav_df = df({'trials': range(nb_trials),
               'perturbation': perturbations,
               'error': errors,
               'err': err,
               'trajectoryX': trajectoryX, 'trajectoryY': trajectoryY,
               'feedbackX': feedbackX,
               'feedbackY': feedbackY,
               'feedback': feedback,
               'org_feedback': org_feedback,
               'belief': belief,
               'target_inds': target_inds,
               'target_codes': target_codes,
               'target_locs': target_locs,
               'RT': reaction_time,
               'trial_duration': trial_duration,
               'movement_duration': movement_duration,
               'environment': environment})

fname = op.join(path_data, subject, 'behavdata',
                f'err_sens_{task}.npz')
# save inside
r = computeErrSens(behav_df, subject, fname=fname)

fname = op.join(path_data, subject, 'behavdata',
                f'behav_{task}_df.pkl')
behav_df.to_pickle(fname)

#         all_trials_duration.append(np.mean(trial_duration))
#         all_reaction_time.append(np.mean(reaction_time))
#         all_movement_duration.append(np.mean(movement_duration))
#
# all_trials_duration = np.array(all_trials_duration)
# all_reaction_time = np.array(all_reaction_time)
# all_movement_duration = np.array(all_movement_duration)

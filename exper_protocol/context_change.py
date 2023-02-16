# -*- coding: utf-8 -*-
# needed for joystick
import pygame
# from pygame.locals import *
from psychopy import gui, core
import time
import os
# import logging
# from win32api import GetSystemMetrics
import sys
import numpy as np
import math
from os.path import join as pjoin
from itertools import product,repeat
# import pylink
# import EyeLink

def fnuniquify(path):
    'uniqify filename'
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1
    return path



def trial_info2trgtpl(ti, phase):
    tpl = (ti['trial_type'], ti['vis_feedback_type'], ti['tgti'], phase)
    return tpl

def get_target_angles(num_targets, target_location_pattern):
    if num_targets == 3:
        targetAngs = np.array([-15,0,15])
    elif num_targets == 2:
        targetAngs = np.array([-30,30])
    elif num_targets == 1:
        targetAngs = np.array([0])
    elif num_targets == 4:
        if target_location_pattern == 'diamond':
            mult = 90 # in deg
            targetAngs = np.arange(num_targets) * mult
        elif target_location_pattern == 'fan':
            mult = 15 # in deg
            m = num_targets * mult
            targetAngs = np.arange(num_targets) * mult -  m / 2
        else:
            raise ValueError(f'target_location_pattern = {target_location_pattern} not implemented')
    elif num_targets >= 4:
        mult = 15 # in deg
        m = num_targets * mult
        targetAngs = np.arange(num_targets) * mult -  m / 2
    targetAngs = targetAngs + (180 + 90)
    return targetAngs

class VisuoMotor:

    def add_param(self, name, value):
        self.params.update({name: value})
        self.paramfile.write(name + ' = ' + str(value) + '\n')

    def add_param_comment(self, comment):
        self.paramfile.write(comment + '\n')

    def initialize_parameters(self, info):
        # self.debug = False
        self.params = {}
        self.task_id = 'context_change'
        self.subject_id = info['participant']
        self.session_id = info['session']
        self.timestr = time.strftime("%Y%m%d_%H%M%S")

        subdir = 'data'
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        if self.debug:
            self.filename = pjoin(subdir, self.subject_id + '_' + self.task_id)
        else:
            self.filename = pjoin(subdir, self.subject_id + '_' + self.task_id +
                             '_' + self.timestr)
        self.paramfile = open(self.filename + '.param', 'w')
        self.logfile = open(self.filename + '.log', 'w')
        # for debug mostly
        self.trigger_logfile = open(self.filename + '_trigger.log', 'w')


        self.add_param('controller_type', 'joystick')
        #self.add_param('controller_type', 'mouse')

        #self.add_param('joystick_angle2cursor_control_type', 'angle_scaling')
        self.add_param('joystick_angle2cursor_control_type', 'velocity')

        self.add_param_comment('# Width of screen')      # updates self.param dictionary
        #self.add_param('width', 800)
        self.add_param('width', 1000)
        self.add_param_comment('# Height of screen')
        #self.add_param('height', 800)
        self.add_param('height', 1000)
        self.add_param_comment('# DESIRED Frames per second for plotting')
        # obesrved FPS can be slighlty different
        #self.add_param('FPS', 120)
        #self.add_param('FPS', 120)
        self.add_param('FPS', 60)
        self.add_param_comment('# Radius of the cursor')
        self.add_param('radius_cursor', 10)

        self.add_param('radius_home', self.params['radius_cursor'] * 3.5)
        self.add_param('radius_home_strict_inside',
                       self.params['radius_home'] - self.params['radius_cursor'])

        self.add_param('radius_return', self.params['radius_cursor'] + \
                self.params['radius_home'])

        self.add_param_comment('# Radius of the target')
        self.add_param('radius_target', 19)
        # distance from the start location to the target center
        self.add_param_comment('# Radius of the invisible boundary')
        self.add_param('dist_tgt_from_home',
                int(round(self.params['height']*0.5*0.7)))

        self.add_param('show_text', 0)
        self.add_param_comment('# Use eye tracker?')
        self.add_param('use_eye_tracker', 1)
        #self.add_param_comment('# Use triggers?')
        #self.add_param('use_true_triggers', 1)

        # TODO: discuss with Romain
        self.add_param('max_EUR_reward', 20)



        #################  durations
        self.add_param_comment('# Time inside the home position'
                               'before trial starts (seconds)')
        self.add_param('time_at_home', 0.5)
        self.add_param_comment('# Time for feedback (seconds)')
        #self.add_param('time_feedback', 0.25)
        self.add_param_comment('# if online feedback, duration'
                               'of reach  (seconds)')
        if self.debug:
            self.add_param('time_feedback', 2.)
        else:
            # shorter time for mouse
            if self.params['controller_type'] == 'mouse':
                self.add_param('time_feedback', 0.95)
            else:
                self.add_param('time_feedback', 1.15)

        # see  Haith 2015 JNeuro (used 0.3s and 1.5s) and Bracco 2018 (used
        # 1.5s)
        # but for behav task we'll make it shorter (though not destroy
        # completely otherwise similarity will be less)
        self.add_param_comment('# time between the target appearance (w/o cursor visible) and the movement start (seconds)')
        #self.add_param('motor_prep_duration', 1.5)
        self.add_param('motor_prep_duration', 0.5)

        self.add_param_comment('# Time for intertrial interval (seconds)')
        #self.add_param('ITI_duration', 1.5)
        self.add_param('ITI_duration', 2)
        self.add_param_comment('# Max jitter during ITI (seconds)')
        self.add_param('ITI_jitter', 0.1)
        self.add_param_comment('# Show text?')
        self.add_param('explosion_duration', 0.7)

        self.add_param('pause_duration', 60)
        if self.debug:
            self.add_param('pause_duration', 10)

        # in seconds
        self.add_param('trigger_duration',     75   / 1000)  # 50 was in Romain, 100 in Marine
        self.add_param('MEG_trigger_duration', 1000 / 1000)


        self.add_param('training_text_show_duration', 3)


        # not used for now
        if self.debug:
            self.add_param('return_duration', 60)
        else:
            self.add_param('return_duration', 10)

        ######################  Categorical params
        # whether feedback is shown always on circle with fixed radius or
        # normally
        self.add_param('feedback_fixed_distance', 0)
        self.add_param('send_startstop_MEG_triggers', 0)

        self.add_param('rest_return_home_indication', 'return_circle') # ot 'text'

        self.params['diode_width']  = 50 # maybe 100?
        self.params['diode_height'] = 50

        w = 1920
        h = 1080

        # enables or disable return phase
        #self.add_param('rest_after_return',1)
        self.add_param('rest_after_return',0)
        self.add_param('prep_after_rest',1)

        self.add_param('return_home_after_pause',1)
        self.add_param('return_home_after_ITI',1)
        self.add_param('hit_when_passing',0)

        self.add_param('notify_early_move', 0)

        # can be also offline though less tested
        self.add_param('feedback_type', 'online')

        # 'no' or 'explosion'
        self.add_param('hit_notif', 'no')
        #   'target_explode', 'cursor_explode', 'text', 'no'
        self.add_param('fail_notif', 'no')

        # how we determine that the reach is finished before the time elapsed
        #self.reach_end_event = 'target_reached'
        #self.reach_end_event = 'distance_reached'
        self.add_param('reach_end_event', 'stopped')
        self.add_param_comment('# How long should we stay at one point before the movement is judged finished?')
        self.add_param('stopping_min_dur', 0.13)
        self.add_param('stop_rad_px', 3)



        ###########################################
        ################### Task seq params
        ###########################################
        self.add_param('scale_pos',1.25)
        self.add_param('scale_neg',0.75)

        many_contexts = 0
        if many_contexts:
            self.add_param('pert_block_types',
                'rot-15,rot15,rot30,rot60,rot90,scale-,scale+,reverse_x')
        else:
            #self.add_param('pert_block_types','rot-15,rot15,rot30,rot60')
            self.add_param('pert_block_types','rot-15,rot15,rot30')

        self.add_param('spec_trial_modN',8)
        self.add_param('allow_context_conseq_repetition', 0)
        # one can also use combinations of scale and rot
        # 'scale-&rot-15','scale-&rot30'

        #self.add_param('num_targets',3)
        self.add_param('num_targets',4)
        #self.add_param('target_location_pattern', 'diamond')
        self.add_param('target_location_pattern', 'fan')
        targetAngs = get_target_angles(self.params['num_targets'],
                    self.params['target_location_pattern'])
        tas = map(lambda x: f'{x:.0f}', targetAngs)
        print('targetAngs = ',targetAngs)
        self.add_param('target_angles',tas)

        #self.add_param('block_len_min',5)
        #self.add_param('block_len_max',9)
        #self.add_param('n_context_appearences',3)

        self.add_param('block_len_min',6)
        self.add_param('block_len_max',10)
        self.add_param('n_context_appearences',3)

        if self.debug:
            self.add_param('block_len_min',2)
            self.add_param('block_len_max',5)

        if self.debug:
            self.add_param('num_initial_veridical',2)
        else:
            #self.add_param('num_initial_veridical',6)
            self.add_param('num_initial_veridical',20)

        self.add_param('randomize_tgt_initial_veridical', 1)

    def __init__(self, info, task_id='',
                 use_true_triggers = 1, debug=False, seed= 1):
        self.debug = debug # affects fullscreen or not and other things
        self.initialize_parameters(info)
        #self.use_true_triggers = self.params['use_true_triggers']
        self.use_true_triggers = use_true_triggers
        if (self.use_true_triggers):
            print("We will try to send actual trigger")
        else:
            print("Triggers will be saved to logfile")


        #self.eye_tracker = pylink.EyeLink(None)
        self.size = self.params['width'], self.params['height']
        #self.trigger_countdown = -1
        self.trigger_value = 0
        self.MEG_rec_started = 0
        # inc in the end of 'PAUSE' and 'ITI'
        self.trial_index = 0  # index of current trials
        self.counter_hit_trials = 0
        self.reward_accrued = 0
        self.reward = 0
        self.last_trial_full_success = 0

        # currently not used
        #self.frame_counters = {'at_home':0, 'feedback_shown':0, 'pause':0,
        #                       'return':0, 'ITI':0}


        self.last_reach_too_slow = 0
        self.last_reach_stopped_away = 0
        self.last_reach_not_full_rad = 0
        self.left_home_during_prep = 0
        self.current_phase_trigger = None
        self.current_phase = None

        #self.first_phase_after_start = 'REST'
        self.first_phase_after_start = 'TRAINING_START'
        # ITI, then RETURN then REST
        w = 'JOYSTICK_CALIBRATION_'
        sides = ['left','right','up','down']
        self.calib_seq = [ w + side.upper() for side in sides ]
        self.phase2dir = dict( zip( self.calib_seq, sides ) )

        self.phase2trigger = {'JOYSTICK_CALIBRATION_LEFT':3,
                             'JOYSTICK_CALIBRATION_RIGHT':4,
                             'JOYSTICK_CALIBRATION_UP':5,
                             'JOYSTICK_CALIBRATION_DOWN':6,
                             'JOYSTICK_CALIBRATION_CENTER':7,
                             'TRAINING_START':8,
                             'TRAINING_END':9,
                              'REST':10, 'RETURN':15,
                              'GO_CUE_WAIT_AND_SHOW':25,
                              'TARGET':20, 'FEEDBACK': 35,
                              'TARGET_AND_FEEDBACK':30,
                           'ITI':40, 'BREAK':50, 'PAUSE':60 }
        self.trigger2phase = dict((v, k) for k, v in self.phase2trigger.items())
        #ct = time.time()
        self.phase_start_times = dict( zip(self.phase2trigger.keys(), len(self.phase2trigger) * [-1.] ) )
        self.phase_start_times['at_home'] = 0.
        self.phase_start_times['trigger'] = 0.
        self.phase_start_times['current_trial'] = time.time()
        self.init_target_positions()  # does not draw anything, only calc

        self.phase2text = { 'TRAINING_START':
            r"Étape d'entrenement: " f'faites {self.params["num_initial_veridical"]} '
            'mouvements,\n la récompense ne sera pas calculée',
            'TRAINING_END':
            'La tache principale commence'}


        ########################  graphics-related

        if self.params['controller_type'] == 'joystick':
            self.phase_shift_event_type = pygame.JOYBUTTONDOWN
        else:
            self.phase_shift_event_type = pygame.MOUSEBUTTONDOWN

        pygame.init()
        self.clock = None

        if self.params['controller_type'] == 'joystick':
            from pygame import error as pygerr
            try:
                self.HID_controller = pygame.joystick.Joystick(0)
            except pygerr as e:
                print(e)
                raise ValueError('Problem with joystick')
            self.HID_controller.init()
            print(self.HID_controller)
        else:
            self.HID_controller = pygame.mouse

        # fonts
        self.foruser_font_size = 24
        self.foruser_center_font_size = 30
        self.debug_font_size = 14
        self.font_face = 'Calibri'
        self.myfont_debug = pygame.font.SysFont(self.font_face, self.debug_font_size)
        self.myfont_popup = pygame.font.SysFont(self.font_face, self.foruser_center_font_size)

        if self.params['controller_type'] == 'joystick':
            retpos_str = '\nAfter the end, return to home position.\n'
        else:
            retpos_str = ''
        #self.instuctions_str = 'We will start soon. Please wait for instructions'
        self.instuctions_str = (f'You will see targets that you have to reach using {self.params["controller_type"]}\n\n'
        'Start moving only after you see BOTH the bright green target and the cursor.\n'
        'If you leave too early, you will see a red circle helping you to return back.\n'
        'Remember: you have to keep your hand steady at the end to complete the reach.\n'
        f'{retpos_str}\n'
        'After you finish you will receive Eur reward\n proportional to your performance :)')
        self.instuctions_str += '\n\npress "c" to calibrate joystick'
        self.instuctions_str += '\npress "q","w" to control cursor size'
        self.instuctions_str += '\npress "ESCAPE" to exit'

        if self.params['controller_type'] == 'mouse':
            self.instuctions_str += '\n\nClicking mouse button returns you to center (only to be used in emergency)'

        self.instuctions_str += f'\n\nNow press any {self.params["controller_type"]} button start the task.\n\n'

        self.break_start_str = 'BREAK'
        # color to which target changes when it is touched by the feedback
        self.color_bg = [100, 100, 100]
        #self.color_hit = [255, 0, 0]   # red
        self.color_hit = [0 , 255,  0]   # green
        self.color_miss = [255, 0, 0]   # red
        self.color_feedback_def = [255, 255, 255]
        self.color_cursor_orig = [255, 255, 255]
        self.color_cursor_orig_debug = [200, 100, 100] # reddish
        self.color_traj_orig_debug = [100, 50, 60] # reddish
        self.color_traj_feedback_debug = [200, 200, 200]  #whitish
        self.color_diode = [255, 255, 255]
        self.color_diode_off = [0,0,0]
        self.color_text = [255,255,255]
        self.color_wait_for_go_cue = [120,120,0] # yellow
        self.color_feedback = self.color_feedback_def
        self.color_target_def = [0, 255, 0]  # green
        # current color of the target, this is not fixed in time
        self.color_target = self.color_target_def

        if self.params['rest_return_home_indication'] == 'return_circle':
            #self.color_return_circle = [240, 60, 60]  # reddish
            self.color_return_circle = [130, 90, 90]  # reddish
        else:
            self.color_return_circle = [240, 215, 40]  # greenish


        # it changes value to 1 only once, when we start the experiment
        self.task_started = 0
        self.ctr_endmessage_show_def = 1000
        self.ctr_endmessage_show = self.ctr_endmessage_show_def

        self.color_photodiode = self.color_diode_off
        self.home_position = (int(round(self.params['width']/2.0)),
                              int(round(self.params['height']/2.0)))

        ##################################
        self.jaxlims_d = {}
        # precalib on workstation
        self.jaxlims_d = {'left': 0.788848876953125, 'right': 0.6519775390625,
                          'up': 0.724334716796875, 'down': 0.618743896484375}
        self.jaxcenter =  {'ax1':0.055, 'ax2':-0.06} #{'ax1' :0}
        self.jaxcenter =  {'ax1':-0.04, 'ax2':-0.013} #{'ax1' :0}

        # when set to true, we take as center the axes values
        # at the start of the app. So at the start the joystick has to be
        # untouched
        self.joystick_center_autocalib = True

        #self.jaxlims = -1,1

        self.tgti_to_show = -1
        self.cursorX = 0
        self.cursorY = 0
        self.feedbackX = 0
        self.feedbackY = 0
        self.unpert_feedbackX = 0
        self.unpert_feedbackY = 0
        self.just_moved_home = 0
        self.just_finished_pretraining = 0

        self.feedbackX_when_crossing = 0
        self.feedbackY_when_crossing = 0

        self.error_distance = 0 # distance between current feedback and current target

        self.block_stack = []
        #self.counter_random_block = 0

        self.free_from_break = 0
        # trajectory starting from last target phase
        self.reset_traj()

        ###########################

        self.scale_params = {'scale-':self.params['scale_neg'],
                             'scale+':self.params['scale_pos']}
        self.pert_block_types = self.params['pert_block_types'].split(',')
        #self.pert_block_types = ['rot-15', 'rot15', 'rot30', 'rot60', 'rot90',
        #               'scale-', 'scale+', 'reverse_x'  ]
        # 'scale-&rot-15','scale-&rot30'
        self.vis_feedback_types = self.pert_block_types + ['veridical']
        special_trial_block_types = ['error_clamp_sandwich', 'error_clamp_pair']
        trial_type = ['veridical', 'perturbation', 'error_clamp', 'pause']

        target_inds = np.arange(len(self.target_coords) )
        vfti_seq0 = list( product( self.vis_feedback_types, target_inds ) )
        n_contexts = len(vfti_seq0)
        vfti_seq_noperm = list(vfti_seq0) * self.params['n_context_appearences']

        # TODO: manage seed here, make it participant or date depenent explicitly
        if seed is None:
            self.seed = 1
        else:
            self.seed = seed
        np.random.seed(self.seed)

        was_repet = True
        while was_repet:
            ct_inds = np.random.permutation(np.arange(len(vfti_seq_noperm) ) )
            if self.params['allow_context_conseq_repetition']:
                break
            else:
                # check for repetitions
                was_repet = False
                for i in range(len(ct_inds) - 1):
                    vfti1 = vfti_seq_noperm[ct_inds[i] ]
                    vfti2 = vfti_seq_noperm[ct_inds[i+1] ]
                    if vfti1 == vfti2:
                        was_repet = True
                        # break from this and return to outer while
                        print(f'Found repeating context {i},{i+1}:    {vfti1} == {vfti2}')
                        break
                vfti0 = vfti_seq_noperm[ct_inds[0] ]
                if self.params['num_initial_veridical'] > 1 and vfti0[0] == 'veridical':
                    print('First was veridical')
                    was_repet = True

        #print('ct_inds',ct_inds)
        self.vfti_seq = [vfti_seq_noperm[i] for i in ct_inds] # I don't want to convert to numpy here
        n_blocks = len(self.vfti_seq)
        ns_context_repeat = np.random.randint( self.params['block_len_min'], self.params['block_len_max'],
                                              size=n_blocks )

        #n_blocks = n_contexts * self.params['n_context_appearences']
        #seq0 = np.tile( np.arange(n_contexts), self.params['n_context_appearences'])
        #context_seq = np.random.permutation(seq0)

        def genSpecTrialBlock(pause_trial,
                              error_clamp_pair, error_clamp_sandwich,
                              d):
            trial_infos = []
            # inplace
            if pause_trial:
                dspec = {'vis_feedback_type':'veridical', 'tgti':tgti,
                     'trial_type': 'pause', 'special_block_type': None }
                trial_infos += [dspec]
            if error_clamp_sandwich:
                dspec = {'vis_feedback_type':'veridical', 'tgti':tgti,
                     'trial_type': 'error_clamp',
                     'special_block_type': 'error_clamp_sandwich' }
                trial_infos += [dspec]
                #
                dspec = d.copy()
                dspec['special_block_type'] = 'error_clamp_sandwich'
                trial_infos += [dspec]
                #
                dspec = {'vis_feedback_type':'veridical', 'tgti':tgti,
                     'trial_type': 'error_clamp',
                     'special_block_type': 'error_clamp_sandwich' }
                trial_infos += [dspec]
            if error_clamp_pair:
                dspec = {'vis_feedback_type':'veridical', 'tgti':tgti,
                     'trial_type': 'error_clamp',
                     'special_block_type': 'error_clamp_pair' }
                trial_infos += [dspec]
                #
                dspec = {'vis_feedback_type':'veridical', 'tgti':tgti,
                     'trial_type': 'error_clamp',
                     'special_block_type': 'error_clamp_pair' }
                trial_infos += [dspec]
            return trial_infos

        ###############################
        # Add pretraining
        ###############################
        self.trial_infos = [] # this is the sequence of trial types
        for i in range(self.params['num_initial_veridical']  ):
            if self.params['randomize_tgt_initial_veridical']:
                tgti_cur = np.random.randint( self.params['num_targets'] )
            else:
                tgti_cur = min( self.params['num_targets'] - 1, 1)
                #print('tgti_cur',tgti_cur)

            d = {'vis_feedback_type':'veridical', 'tgti':tgti_cur,
                    'trial_type': 'veridical',
                 'special_block_type': 'pretraining' }
            self.trial_infos += [d]


        # debug scale-
        #d = {'vis_feedback_type':'scale-', 'tgti':tgti_cur,
        #        'trial_type': 'perturbation',
        #     'special_block_type': None }
        #self.trial_infos += [d] * 3


        ## debug reverse
        #d = {'vis_feedback_type':'reverse_x', 'tgti':tgti_cur,
        #        'trial_type': 'perturbation',
        #     'special_block_type': None }
        #self.trial_infos += [d] * 3
        ###################



        # TODO: care about whether pauses and clamps are parts of the block or
        # not
        block_start_inds = []
        for (bi, num_context_repeats), (vis_feedback_type, tgti) in\
                zip( enumerate(ns_context_repeat), self.vfti_seq):
            #r += [ (vis_feedback_type, tgti) ] * ncr
            # NOTE: if I insert pauses later, it will be perturbed
            block_start_inds += [ len( self.trial_infos ) ]
            if vis_feedback_type == 'veridical':
                ttype = vis_feedback_type
            else:
                ttype = 'perturbation'
            d = {'vis_feedback_type':vis_feedback_type, 'tgti':tgti,
                 'trial_type': ttype, 'special_block_type': None }

            pause_trial_middle, error_clamp_pair_middle,\
                    error_clamp_sandwich_middle = 0,0,0
            pause_trial_end, error_clamp_pair_end,\
                    error_clamp_sandwich_end = 0,0,0


            #if self.debug:
            #    pause_trial_middle, error_clamp_pair_middle,\
            #            error_clamp_sandwich_middle = 0,1,0

            if bi > 2: #or self.debug:
                N = self.params['spec_trial_modN']
                if bi % N == 0:
                    pause_trial_middle       = 1
                elif bi % N == 4:
                    pause_trial_end          = 1
                elif bi % N == 2:
                    error_clamp_pair_end     = 1
                elif bi % N == 6:
                    error_clamp_sandwich_end = 1

            hd = num_context_repeats // 2
            rhd = num_context_repeats - hd
            self.trial_infos += [d] * hd

            # insert in the middle
            self.trial_infos += genSpecTrialBlock(pause_trial_middle,
                error_clamp_pair_middle, error_clamp_sandwich_middle,d)

            self.trial_infos += [d] * rhd

            self.trial_infos += genSpecTrialBlock(pause_trial_end,
                error_clamp_pair_end, error_clamp_sandwich_end,d)


        n_pauses = (num_context_repeats *  2/8)
        expected_max_reward = len( self.trial_infos ) * 0.6
        self.reward2EUR_coef = self.params['max_EUR_reward'] / expected_max_reward



        # I want list of tuples -- target id, visual feedback type, phase
        # (needed for trigger)
        self.CONTEXT_TRIGGER_DICT = {}
        CTD_to_export = {}
        trigger = self.phase2trigger['PAUSE'] + 40   # to get 100
        for ti in self.trial_infos:
            for phase in self.phase2trigger.keys():
                tpl = trial_info2trgtpl(ti, phase)
                if tpl not in self.CONTEXT_TRIGGER_DICT:
                    self.CONTEXT_TRIGGER_DICT[ tpl ] = trigger
                    s = f'{tpl[0]},{tpl[1]},{tpl[2]},{tpl[3]}'
                    CTD_to_export[s  ] = trigger
                    trigger += 1


        import json
        s = json.dumps(CTD_to_export, indent=4)
        self.paramfile.write( '# trial param and phase 2 trigger values \n' )
        self.paramfile.write( '# trial_type, vis feedback, tgti, phase \n' )
        self.paramfile.write( s + '\n' )

        self.paramfile.write( '# trial_infos = \n' )
        for tc in range( len(self.trial_infos) ):
            ti = self.trial_infos[tc]
            s = '{} = {}, {}, {}, {}'.format(tc, ti['trial_type'], ti['tgti'],
                  ti['vis_feedback_type'], ti['special_block_type'] )
            print(s)
            self.paramfile.write( f'# {s}\n' )
        #s = json.dumps( {'trial_infos':list(enumerate(self.trial_infos) ) }
        #               , indent=4)
        #self.paramfile.write( s + '\n' )

        self.paramfile.close()


        # print first trial infos
        #if self.debug:
        print(f'In total we have {len(self.trial_infos)} trials ')
        for tc in range(30):
            ti = self.trial_infos[tc]
            print(tc, ti['trial_type'], ti['tgti'],
                  ti['vis_feedback_type'], ti['special_block_type'] )


        # to debug end
        #self.trial_infos = self.trial_infos[:2]


        self.MEG_start_trigger = 252
        self.MEG_stop_trigger = 253
        ## Start MEG recordings
        #port.setData(0)
        #time.sleep(0.1)
        #port.setData(252)
        ## Stop MEG recordings
        #time.sleep(1)
        #port.setData(253)
        #time.sleep(1)
        #port.setData(0)

        if (self.use_true_triggers):
            if self.params['trigger_device'] == 'inpout32':
                assert sys.platform.startswith('win32')
                from ctypes import windll
                self.trigger_port = 0x378
                self.trigger = windll.inpout32
            elif self.params['trigger_device'] == 'parallel':
                import parallel
                self.trigger_port = '0x3FE8'
                self.trigger = parallel.ParallelPort(address=self.trigger_port)


            self.send_trigger(0)


    def init_target_positions(self):
        '''
        called in class constructor
        '''
        #targetAngs = [22.5+180, 67.5+180, 112.5+180, 157.5+180]
        #targetAngs = get_target_angles(self.params['num_targets'])
        targetAngs = self.params['target_angles']
        self.target_angles = targetAngs  # in deg

        # list of 2-tuples
        self.target_coords = []
        for tgti,tgtAngDeg in enumerate(targetAngs):
            tgtAngRad = float(tgtAngDeg)*(np.pi/180)
            # this will be given to pygame.draw.circle as 3rd arg
            # half screen width + cos * radius
            # half screen hight + sin * radius
            self.target_coords.append((int(round(self.params['width']/2.0 +
                                          np.cos(tgtAngRad) * self.params['dist_tgt_from_home'])),
                                      int(round(self.params['height']/2.0 +
                                          np.sin(tgtAngRad) * self.params['dist_tgt_from_home']))))

    def send_trigger_cur_trial_info(self):
        ti = self.trial_infos[self.trial_index]
        tpl = trial_info2trgtpl(ti, self.current_phase)
        self.send_trigger( self.CONTEXT_TRIGGER_DICT[tpl], tpl )

    def send_trigger(self, value, add_info = None):
        # We block it if it has sent something already
        #if (self.trigger_countdown == -1):
        if (self.trigger_value == 0):
            if (self.use_true_triggers):
                if self.params['trigger_device'] == 'inpout32':
                    self.trigger.Out32(self.trigger_port, value)
                elif self.params['trigger_device'] == 'parallel':
                    self.trigger.setData(value)
            self.trigger_value = value

            # always write to trigger logfile
            if add_info is not None:
                sa = '   ' + str(add_info)
            else:
                sa = ''
            s = str(value) + sa + '\n'
            self.trigger_logfile.write(s )
        # print("Sent trigger " + str(value))
        # For how long the trigger is gonna be raised? (in ms)
        if (value != 0):
            td = self.params['trigger_duration']
            #self.trigger_countdown = int(round(self.params['FPS']*(td/1000.0)))
            self.phase_start_times['trigger'] = time.time()

    def on_init(self):
        pygame.init()
        if self.debug:
            self._display_surf = pygame.display.set_mode(self.size)
        else:
            #self._display_surf = pygame.display.set_mode(self.size, pygame.FULLSCREEN)
            self._display_surf = pygame.display.set_mode(self.size)
        self._running = True

    def save_scr(self, trial_info = None, prefix=None):
        dir_scr = './screenshots'
        if trial_info is None:
            vft = None
        else:
            vft = trial_info['vis_feedback_type']
        if not os.path.exists(dir_scr):
            os.makedirs(dir_scr)
        if prefix is None:
            prefix = f'end_{self.current_phase}'

        fn_scr0 = f"{dir_scr}/{prefix}_tind={self.trial_index}_vft={vft}.jpg"
        fn_scr = fnuniquify(fn_scr0)
        pygame.image.save(pygame.display.get_surface(),fn_scr)
        print('Screenshot saved to ',fn_scr)

    def on_event(self, event):
        if event.type not in [pygame.MOUSEMOTION, pygame.JOYBALLMOTION,
                              pygame.JOYHATMOTION, pygame.JOYAXISMOTION,
                              pygame.WINDOWMOVED, pygame.AUDIODEVICEADDED,
                              pygame.WINDOWSHOWN, pygame.WINDOWTAKEFOCUS,
                              pygame.KEYUP]:
            if pygame.mouse.get_focused():
                print('on_event: ',event)

        if (event.type == pygame.MOUSEMOTION) and self.just_moved_home:
            self.just_moved_home = 0

        if event.type == pygame.QUIT:
            self._running = False
        if event.type == self.phase_shift_event_type:
            print(self.cursorX, self.cursorY, self.current_phase,
                  self.trial_infos[self.trial_index] )

            # if task was not started and a button was pressed, start the task
            if self.task_started == 0:
                self.current_phase = self.first_phase_after_start
                ct = time.time()
                self.phase_start_times[self.current_phase] = ct
                if self.current_phase == 'REST':
                    # TODO: are you sure we need to set it here?
                    self.phase_start_times['at_home'] = ct
                    self.send_trigger_cur_trial_info()
                self.moveHome()
                pygame.mouse.set_visible(0)
                self.reset_traj()

                self.send_trigger(self.MEG_start_trigger)
                self.MEG_rec_started = 1

                self.task_started = 1

            # if task was started and a button was pressed, release break
            else:
                self.moveHome()
                self.free_from_break = 1

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.on_cleanup(-1)
            if event.key == pygame.K_F11 and pygame.mouse.get_focused():
                if (self._display_surf.get_flags() & pygame.FULLSCREEN):
                    self._display_surf = pygame.display.set_mode(self.size)
                else:
                    self._display_surf = pygame.display.set_mode(self.size,
                        pygame.FULLSCREEN)
            if event.key == pygame.K_q:
                self.params['radius_cursor'] = self.params['radius_cursor']-1
            if event.key == pygame.K_w:
                self.params['radius_cursor'] = self.params['radius_cursor']+1
            if event.key == pygame.K_r:
                self.moveHome()
            if event.key == pygame.K_s:
                self.save_scr()
            if (event.key == pygame.K_c) and (not self.task_started == 1) and\
                    (self.params['controller_type'] == 'joystick' ):
                self.current_phase = self.calib_seq[0]
                self.reset_traj()
                pygame.mouse.set_visible(0)

                print('Start calibration')
            #if event.key == pygame.K_c:
            #    if (self.params['use_eye_tracker']):
            #        EyeLink.tracker(self.params['width'], self.params['height'])

    def moveHome(self):
        ja2cct = self.params['joystick_angle2cursor_control_type']
        if self.params['controller_type'] == 'joystick' and ja2cct == 'angle_scaling':
            #self.HID_controller.set_axis( [0,0]  )
            print(f'moveHome: with joystick {ja2cct} we cannot reset position')
        else:
            # get_pos will not return new position until a MOUSEMOTION event is
            # executed
            # https://stackoverflow.com/questions/36995302/python-pygame-mouse-position-not-updating
            #print('before set', self.HID_controller.get_pos()  )
            if self.params['controller_type'] == 'mouse':
                self.HID_controller.set_pos(self.home_position)
            #print('after set',self.HID_controller.get_pos()  )

            self.cursorX,self.cursorY = self.home_position
            self.cursor_pos_update()
            print(f'moveHome: Moved home cursor={self.cursorX,self.cursorY}')
            self.just_moved_home = 1

    def drawTgt(self, radmult = 1.):
        pygame.draw.circle(self._display_surf, self.color_target,
                           self.target_coords[self.tgti_to_show],
                           self.params['radius_target'] * radmult, 0)

    def drawTextMultiline(self, lines, pos_label = 'lower_left', font = None, voffset_glob = 0):
        if font is None:
            font = self.myfont_debug
        if pos_label == 'lower_left':
            voffset = voffset_glob
            for line in lines[::-1]:
                text_render = font.render(line, True, self.color_text)
                ldt = font.size(line)
                self._display_surf.blit(text_render, (5, self.params['height']-ldt[1] - voffset))
                voffset += ldt[1]
        elif pos_label == 'center':
            voffset = voffset_glob
            for line in lines[::-1]:
                text_render = font.render(line, True, self.color_text)
                ldt = font.size(line)

                pos = (int(round(((self.params['width'] - ldt[0]) / 2.0)) ),
                    int(round(((self.params['height'] - ldt[1]) / 2.0)) +\
                        - voffset ) )

                self._display_surf.blit(text_render, pos)
                voffset += ldt[1]
        elif pos_label == 'upper_right':
            #longest = 0
            #for line in lines[::-1]:
            #    text_render = font.render(line, True, (255, 255, 255))
            #    ldt = font.size(line)
            #    longest = max(ldt[0], longest)

            voffset = voffset_glob
            for line in lines[::-1]:
                text_render = font.render(line, True, self.color_text)
                ldt = font.size(line)

                pos = (int(round(((self.params['width'] - ldt[0]) )) ),
                    int(round((( ldt[1]) / 2.0)) + voffset ) )

                self._display_surf.blit(text_render, pos)
                voffset += ldt[1]

        else:
            raise ValueError(f'not implemented {pos_label}')

    def drawPopupText(self, text, pos = 'center', font_size = None,
                      length_info = None, text_render = None):
        #print(f'drawing text : {text}')
        #, font_size=30
        if font_size is not None:
            font = pygame.font.SysFont(self.font_face, font_size)
        else:
            font = self.myfont_popup

        if text_render is None:
            text_render = font.render(text,
                    True, self.color_text)

        #self.myfont_popup = pygame.font.SysFont('Calibri', 30)
        if length_info is None:
            ldt = font.size(text)
        else:
            ldt = length_info
        if isinstance(pos,tuple):
            if isinstance(pos[0],str):
                pos_label  = pos[0]
                pos_add_coords = pos[1]
            else:
                assert isinstance(pos[0],(int,float )), pos
                pos_label  = None
                pos_add_coords = pos
        if isinstance(pos,str):
            pos_label  = pos
            pos_add_coords = 0,0

        if pos_label is None:
            pos = pos_add_coords
        elif pos_label == 'center':
            pos = (int(round(((self.params['width'] - ldt[0]) / 2.0)) + \
                       pos_add_coords[0] ),
                int(round(((self.params['height'] - ldt[1]) / 2.0))) +\
                   pos_add_coords[1] )
        elif pos_label == 'upper_left':
            pos = pos_add_coords
        elif pos_label == 'upper_right':
            pos = (int(round(self.params['width'] - ldt[0] + pos_add_coords[0] )),
                   pos_add_coords[1])

        self._display_surf.blit(text_render, pos)
        return ldt

    def drawReturnCircle(self):
        dist = np.sqrt((self.cursorX - self.home_position[0])**2  +
                    (self.cursorY - self.home_position[0])**2)
        #thickness = 3
        thickness = 2

        # if the last was scale then we wante to scale return radius as well
        # we need to subtract 1 becasue RETURN goes after ITI, where it was
        # increased
        trial_info = self.trial_infos[self.trial_index - 1]
        ctxtpl = (trial_info['vis_feedback_type'], trial_info['tgti'] )
        vft, tgti = ctxtpl
        if vft.startswith('scale'):
            scale = self.scale_params[vft]
            dist *= scale
        pygame.draw.circle(self._display_surf, self.color_return_circle,
                           self.home_position,
                           dist, thickness)

    def drawHome(self):
        pygame.draw.circle(self._display_surf, [0, 0, 0],
                           self.home_position,
                           int(self.params['radius_home']), 2)

    def drawCursorFeedback(self, radmult = 1.):
        #trial_info = self.trial_infos[self.trial_index]
        #ttype = trial_info['trial_type']
        #if ttype == 'error_clamp':
        #    print('EEECCCC')
        #    #take distance vector, plot it mutiplied
        #    vec_ideal = np.array(self.target_coords[self.tgti_to_show]) - \
        #        np.array(self.home_position)
        #    vec_feedback = np.array( [self.feedbackX, self.feedbackY ] ) - \
        #        np.array(self.home_position)
        #    lvf = np.linalg.norm(vec_feedback)
        #    lvi = np.linalg.norm(vec_ideal)
        #    vec = (float(lvf) / float(lvi)) * vec_ideal
        #    vec = vec.astype(np.int)
        #    pygame.draw.circle(self._display_surf, self.color_feedback,
        #                   (vec[0],vec[1]),
        #                   self.params['radius_cursor'], 0)
        #else:
        radius = self.params['radius_cursor']
        radius *= radmult
        pygame.draw.circle(self._display_surf, self.color_feedback,
                        (self.feedbackX, self.feedbackY),
                        radius, 0)

    def drawCursorOrig(self, debug=0, verbose=0):
        r = self.params['radius_cursor']
        c = self.color_cursor_orig
        if debug:
            c = self.color_cursor_orig_debug
            r /= 2


        if self.just_moved_home:
            #return True
            x,y = self.home_position
        else:
            x,y = (self.cursorX, self.cursorY)

        pygame.draw.circle(self._display_surf, c, (x,y) , r, 0)
        if verbose:
            print(f'Draw orig cursor {self.cursorX, self.cursorY}')


    def drawTraj(self, pert=0):
        #print('drawTraj beg')
        if (not pert) and (len(self.trajX) < 2):
            return
        if pert and (len(self.trajfbX) < 2):
            return

        if pert:
            thickness = 6
            tpls = list(zip(self.trajfbX, self.trajfbY ))
            c = self.color_traj_feedback_debug  # whitish for feedback
        else:
            thickness = 2
            tpls = list(zip(self.trajX, self.trajY ))
            c = self.color_traj_orig_debug  # redish for true movement

        pygame.draw.lines(self._display_surf, c, False,
                          tpls, thickness )

    def drawPerfInfo(self, reward_type = ['money', 'hit']):
        monetary_value_last = self.reward * self.reward2EUR_coef
        monetary_value_tot = self.reward_accrued * self.reward2EUR_coef

        #ldt = self.drawPopupText(
        #    f'Trial N={self.trial_index}/{len(self.trial_infos)}',
        #                   pos='upper_right', font_size = 30)
        #ldt1 = self.drawPopupText( f'Nhits = {self.counter_hit_trials}',
        #                   pos=('upper_right',(0,ldt[1]) ),
        #                   font_size = 30)
        #if self.debug:
        #    ldt2 = self.drawPopupText( f'Reward total = {self.reward_accrued:.2f}',
        #                       pos=('upper_right',(0,ldt[1]+ldt1[1]) ),
        #                       font_size = 30)
        #    ldt3 = self.drawPopupText( f'Last reward = {self.reward:.2f}',
        #                       pos=('upper_right',(0,ldt[1]+ldt1[1] + ldt2[1]) ),
        #                       font_size = 30)

        perfinfo = []
        perfinfo += [ f'Trial N={self.trial_index}/{len(self.trial_infos)}'  ]
        if 'hit' in reward_type:
            perfinfo += [ f'Nhits        = {self.counter_hit_trials}' ]
            perfinfo += [ f'Reward total = {self.reward_accrued:.2f}' ]
            perfinfo += [ f'Last reward  = {self.reward:.2f}' ]
        if 'money' in reward_type:
            perfinfo += [ f'Reward total = {monetary_value_tot:.2f} Eur' ]
            perfinfo += [ f'Last reward  = {monetary_value_last:.2f} Eur' ]

        return perfinfo

    def on_render(self):
        '''
        called from on_execute
        '''
        if (self.task_started == 2):
            self._display_surf.fill(self.color_bg)
            endstrs = [ 'Task finished, well done!' ]
            delay = self.ctr_endmessage_show / self.params['FPS']
            endstrs += [f'the window will close in {delay:.0f} seconds']
            #self.drawPopupText(pause_str)
            perfstrs = self.drawPerfInfo(reward_type = ['money'] )
            if self.ctr_endmessage_show == self.ctr_endmessage_show_def:
                print('Participant results ',perfstrs)
                #subdir = 'data'
                #fnf = pjoin(subdir,
                #    f'final_perf_subj={self.subject_id}_sess={self.session_id}.txt')
                #with open(fnf, 'w') as f:
                #    f.writelines(perfstrs )

                self.logfile.write('\n\n' )
                self.logfile.write(";".join(perfstrs) )
            self.drawTextMultiline( endstrs + [''] + perfstrs,
                                   font = self.myfont_popup,
                                    pos_label= 'center')


        elif (self.task_started == 1):
            # Clear screen
            self._display_surf.fill(self.color_bg)
            if self.debug:
                perfstrs = self.drawPerfInfo()
                self.drawTextMultiline( perfstrs, font = self.myfont_popup,
                                       pos_label = 'upper_right')

            if self.current_phase in ['TRAINING_START', 'TRAINING_END']:
                txt = self.phase2text[self.current_phase]
                self.drawTextMultiline(txt.split('\n'), pos_label = 'center', font = self.myfont_popup )
            if self.current_phase == 'ITI':
                # draw nothing except maybe explosion of tgt
                if self.reward < (1. - 1e-6):
                    if self.params['fail_notif'] == 'text':
                        if self.last_reach_not_full_rad:
                            s = 'Reach did not even arrive to target distance in required time'
                            self.drawPopupText(s)
                        else:
                            if self.last_reach_too_slow == 1:
                                s = 'Reach was too slow'
                                self.drawPopupText(s)
                            elif self.last_reach_too_slow == 2:
                                s = 'Have not kept cursor at the target for enough time'
                                self.drawPopupText(s)
                    elif self.params['fail_notif'] == 'cursor_explode':
                        self.color_feedback = self.color_miss
                        tdif = time.time() - \
                            self.phase_start_times[self.current_phase]
                        self.drawCursorFeedback( 1 + 1.2 * np.sin( (tdif/self.params['explosion_duration']) * np.pi  ) )
                    elif self.params['fail_notif'] == 'target_explode':
                        self.color_target = self.color_miss
                        tdif = time.time() - \
                            self.phase_start_times[self.current_phase]
                        self.drawTgt( 1 + 1 * np.sin( (tdif/self.params['explosion_duration']) * np.pi  ) )
                    # if 'no' then just do nothing
                    elif self.params['fail_notif'] != 'no':
                        raise ValueError(f'wrong fail notif {self.params["fail_notif"]}')

                # we could have gotten good reward also when we hit but don't
                # stay at target long enough
                if abs( self.reward - 1.) < 1e-6 and self.last_trial_full_success:
                    if self.params['hit_notif'] == 'explosion':
                        self.color_target = self.color_hit
                        tdif = time.time() - self.phase_start_times[self.current_phase]
                        self.drawTgt( 1 + 1.2 * np.sin( (tdif/self.params['explosion_duration']) * np.pi  ) )
                    elif self.params['hit_notif'] != 'no':
                        raise ValueError(f'wrong hit notif {self.params["hit_notif"]}')
                #elif self.last_reach_stopped_away:
                #    s = 'Reach stopped in some strange place'
                #    self.drawPopupText(s)

            if self.current_phase == 'REST':
                # if not very far from home, draw cursor

                at_home_ext = self.is_home('unpert_cursor', 'radius_home_strict_inside', 2)
                at_home = self.is_home('unpert_cursor', 'radius_home_strict_inside', 1)
                if at_home_ext:
                    self.drawCursorOrig(verbose=0)
                if not at_home:
                    if self.params['rest_return_home_indication'] == 'text':
                        s = 'Return to the center'
                        if self.left_home_during_prep and \
                                self.params['notify_early_move']:
                            s = 'You moved too early. ' + s
                        self.drawPopupText(s, pos = ('center', (0,-70) ) )
                    elif self.params['rest_return_home_indication'] == 'return_circle':
                        self.drawReturnCircle()
                    else:
                        raise ValueError(f"wrong indication value {self.params['rest_return_home_indication']}")
                    #print(f'---draw popup {s}: {self.cursorX}, {self.cursorY}')
                self.drawHome()

            if self.current_phase == 'PAUSE':
                self.drawHome()

                # time left
                #R = ( ( int(self.params['FPS'] * self.params['pause_duration']  )) -\
                #        self.frame_counters["pause"] ) / (self.params['FPS'] )
                #R = int(R)

                timedif = time.time() - self.phase_start_times[self.current_phase]
                R = int(self.params['pause_duration'] - timedif)

                pause_str = f'Pause, time left={R} seconds'
                #self.drawPopupText(pause_str)
                perfstrs = self.drawPerfInfo(reward_type = ['money'] )
                self.drawTextMultiline( [pause_str] + [''] + perfstrs, font = self.myfont_popup,
                                       pos_label= 'center')

            if self.params['feedback_type'] == 'offline':
                if self.current_phase == 'TARGET':
                    self.drawTgt()
                    self.drawHome()

                if (self.current_phase == 'FEEDBACK'):
                    self.drawTgt()
                    self.drawHome()
                    self.drawCursorFeedback()
            elif self.params['feedback_type'] == 'online':
                if self.current_phase == 'TARGET_AND_FEEDBACK':
                    self.drawTgt()
                    self.drawHome()
                    self.drawCursorFeedback()

            if self.current_phase == 'GO_CUE_WAIT_AND_SHOW':
                self.color_target = self.color_wait_for_go_cue
                self.drawTgt()
                self.drawHome()
                #self.drawPopupText('WAIT', 'center')

            if self.current_phase == 'RETURN':
                self.drawReturnCircle()

                thickness = 2
                pygame.draw.circle(self._display_surf, self.color_target_def,
                                   self.home_position,
                                   self.params['radius_return'], thickness)

            show_diode = 1
            diode_width  = self.params['diode_width']
            diode_height = self.params['diode_height']

            if show_diode:
                pygame.draw.rect(self._display_surf, self.color_photodiode,
                                (0, 0, diode_width, diode_height), 0)
                #from psychopy import visual
                #Pixel = visual.Rect(
                #    win=win, name='topleftpixel', units='pix',
                #    pos=(-window_size[1], window_size[1]/2),
                #    size=(window_size[0]*2/5, 200),
                #    fillColor=[-1, -1, -1],
                #    lineColor=[-1, -1, -1])

            # maybe show info on participant hitting performance
            #if (self.params['show_text']):
            #    self._display_surf.blit(self.currentText, (0, 0))

            if (self.current_phase == 'BREAK'):
                #self.drawTextCenter(self.break_text, self.length_text)
                self.drawPopupText(self.break_start_str, font_size = self.foruser_font_size)
        # if not task_started
        else:
            self._display_surf.fill(self.color_bg)

            if self.params['controller_type'] == 'joystick':
                ax1 = self.HID_controller.get_axis(0)
                ax2 = self.HID_controller.get_axis(1)
            else:
                ax1,ax2 = None,None
            val = None
            isgood = False
            if self.current_phase == 'JOYSTICK_CALIBRATION_LEFT':
                isgood = ax1 <= -0.3
                val = ax1
            elif self.current_phase == 'JOYSTICK_CALIBRATION_RIGHT':
                isgood = ax1 >= 0.3
                val = ax1
            elif self.current_phase == 'JOYSTICK_CALIBRATION_UP':
                isgood = ax2 <= -0.3
                val = ax2
            elif self.current_phase == 'JOYSTICK_CALIBRATION_DOWN':
                isgood = ax2 >= 0.3
                val = ax2
            isbad = not isgood
            if self.current_phase in self.phase2dir:
                self.cursor_pos_update()
                self.trajX += [ self.cursorX ]
                self.trajY += [ self.cursorY ]
                self.traj_jax1 += [ax1]
                self.traj_jax2 += [ax2]

                side = self.phase2dir[self.current_phase]
                stopped = self.test_stopped_jax()
                if stopped and isbad:
                    self.drawPopupText(f'please move {side}')
                else:
                    self.drawPopupText(f'Calibrating joystick: please move maximum {side}')
                    self.drawCursorOrig() # with diff color and size

                if stopped and (not isbad):
                    self.jaxlims_d[side] = abs( val )
                    print(f'   {self.current_phase}   {isbad}  {ax1},{ax2}'  )
                    print(f'joystick calibration: set max {side} = {val:.2f}')

                    i = self.calib_seq.index(self.current_phase)
                    if (i + 1) < len(self.calib_seq):
                        self.current_phase = self.calib_seq[i+1]
                        self.reset_traj()
                    else:
                        print('joystick caliration finished ',self.jaxlims_d)
                        self.current_phase = None
            else:
                instr = self.instuctions_str.split('\n')
                #self.drawPopupText(instr,
                #                   font_size = self.foruser_font_size)
                self.drawTextMultiline(instr, font = self.myfont_popup,
                                       pos_label= 'center', voffset_glob = -100 )

                #ax2 = self.HID_controller.get_axis(1)
                #max_ax1 = ax1
                #max_ax2 = ax2


        if self.debug:
            trial_info = self.trial_infos[self.trial_index]
            if self.current_phase is None:
                phase_str = 'None'
            else:
                phase_str = self.current_phase

            m = min(self.trial_index + 40, len(self.trial_infos) )
            next_spec_trial_ind = None
            next_spec_trial_type = None
            for tc in range(self.trial_index + 1, m ):
                ti = self.trial_infos[tc]
                if ti['trial_type'] not in ['veridical', 'perturbation']:
                    next_spec_trial_ind = tc
                    next_spec_trial_type = ti['trial_type']
                    break
            trd = next_spec_trial_ind - self.trial_index
            if self.clock is not None:
                fps = self.clock.get_fps()
            else:
                fps = None

            debugstrs = []
            if self.params['controller_type'] == 'joystick':
                ax1 = self.HID_controller.get_axis(0)
                ax2 = self.HID_controller.get_axis(1)
                axstr = f'({ax1:.3f},{ax2:.3f})'
            else:
                axstr = ''
            debugstr = (f'Phase={phase_str:40}; X,Y={self.feedbackX},{self.feedbackY} (Xt,Yt={self.cursorX},{self.cursorY}){axstr}, '
                        f'FPS={fps:4.0f}')
            debugstrs += [debugstr]

            debugstr = f'tgti_to_show={self.tgti_to_show};  {list(trial_info.values())};  '
            tdif = time.time() - self.phase_start_times["at_home"]
            if tdif < 1000:
                debugstr += f' counter_inside={tdif:.1f}'
            debugstrs += [debugstr]

            debugstr = f'next special trial {next_spec_trial_type} in {trd} trials'
            debugstrs += [debugstr]


            # show prev trials
            m = max(self.trial_index - 4, 0 )
            for tc in range(m, self.trial_index ):
                ti = self.trial_infos[tc]
                debugstr = f'trial_infos[{tc}]={ti}'
                debugstrs += [debugstr]

            # show next trials
            m = min(self.trial_index + 4, len(self.trial_infos) )
            for tc in range(self.trial_index, m ):
                ti = self.trial_infos[tc]
                debugstr = f'trial_infos[{tc}]={ti}'
                if tc == self.trial_index:
                    debugstr = '**' + debugstr
                debugstrs += [debugstr]
            #
            self.drawTextMultiline(debugstrs)


            self.drawCursorOrig(debug=1) # with diff color and size
            self.drawTraj(pert=1)
            self.drawTraj()

        pygame.display.update()

    def test_reach_finished(self, ret_ext = False):
        #at_home = self.point_in_circle(self.home_position, (self.cursorX, self.cursorY),
        #                        self.params['radius_cursor'], verbose=0)
        at_home = self.is_home('unpert_cursor', 'radius_home_strict_inside')

        stop_rad_px = self.params['stop_rad_px']
        stopped = self.test_stopped(stop_rad_px = stop_rad_px)
        radius_reached = self.test_radius_reached()
        r = at_home,stopped,radius_reached
        if self.params['reach_end_event'] == 'stopped':
            b = stopped and (not at_home)
        elif self.params['reach_end_event'] == 'distance_reached':
            b = radius_reached
        if self.debug and b:
            print('test_reach_finished: ', self.params['reach_end_event'])
        if ret_ext:
            return b, r
        else:
            return b

    #def test_target_reached(self):
    #    '''
    #    called from vars_update
    #    says if the cursor have reached the tgt pos
    #    '''
    #    self.point_in_circle( tgtctuple ,  (self.feedbackX, self.feedbackX),
    #        self.params['radius_target'] +\
    #                self.params['radius_cursor'], verbose=0)

    #    centeredX = self.cursorX - self.home_position[0]
    #    centeredY = self.cursorY - self.home_position[1]
    #    return (centeredX**2+centeredY**2)**(1/2.) >= self.params['dist_tgt_from_home']

    def test_stopped(self, stop_rad_px = 3):
        '''
        called from vars_update
        '''
        fps = self.clock.get_fps()
        if self.clock is not None:
            fps = self.clock.get_fps()
        else:
            return None

        ntimebins = int( fps * self.params['stopping_min_dur'] )
        if (len(self.trajX) < ntimebins) or \
                (self.get_dist_from_home() < self.params['radius_home'] ):
            return False
        #else:
        #    print('dist_from_home=' , self.get_dist_from_home() )

        #stdX = np.std( self.trajfbX[-ntimebins:] )
        #stdY = np.std( self.trajfbY[-ntimebins:] )
        # if I use trajfb then for error clamp it works bad
        stdX = np.std( self.trajX[-ntimebins:] )
        stdY = np.std( self.trajY[-ntimebins:] )

        return max(stdX , stdY) <= stop_rad_px

    def test_stopped_jax(self, stop_rad_px = 0.05):
        ntimebins = int( self.params['FPS'] * self.params['stopping_min_dur'] )
        if (len(self.traj_jax1) < ntimebins) or \
                (self.get_dist_from_home() < 0.3):
            return False
        std1 = np.std( self.traj_jax1[-ntimebins:] )
        std2 = np.std( self.traj_jax2[-ntimebins:] )
        #print(f'{std1:.3f},{std2:.3f}')
        return max(std1 , std2) <= stop_rad_px

    def get_dist_from_home(self):
        centeredX = self.cursorX - self.home_position[0]
        centeredY = self.cursorY - self.home_position[1]
        return (centeredX**2+centeredY**2)**(1/2.)

    def test_radius_reached(self):
        '''
        called from vars_update
        says if the cursor have reached the tgt pos
        '''
        return self.get_dist_from_home() >= ( self.params['dist_tgt_from_home'] -\
                                             self.params['radius_target']-
                                             self.params['radius_cursor'] )

    def alter_feedback(self, coordinates, perturbation, alteration_type):
        if alteration_type in ['veridical', 'perturbation']:
            return self.apply_visuomotor_pert(coordinates,
                                              perturbation)
        elif alteration_type == 'error_clamp':
            vec_ideal = np.array(self.target_coords[self.tgti_to_show]) - \
                np.array(self.home_position)
            vec_feedback = np.array( coordinates ) - \
                np.array(self.home_position)
            lvf = np.linalg.norm(vec_feedback)
            lvi = np.linalg.norm(vec_ideal)
            vec = (float(lvf) / float(lvi)) * vec_ideal
            vec = vec.astype(np.int) + np.array(self.home_position)
            #print('EC ',coordinates, vec)
            return tuple(vec)
        else:
            raise ValueError(f'wrong {alteration_type}')

    def apply_visuomotor_pert(self, coordinates, perturbation):
        '''
        called in vars_update
        rotates coordinates
        '''
        assert perturbation in self.vis_feedback_types

        scale = 1.
        rotang_deg = 0.
        sign_x = 1.
        # whether we have combined pert of a regular one
        ampersandi = perturbation.find('&')
        if ampersandi < 0:
            if perturbation.startswith('rot'):
                rotang_deg = int(perturbation[3:])
            elif perturbation.startswith('scale'):
                scale = self.scale_params[perturbation]
            elif perturbation.startswith('reverse'):
                sign_x = -1.
        else:
            # scale should go first
            assert perturbation.startswith('scale')
            si = perturbation.find('scale')
            assert si == 0
            sc = perturbation[:ampersandi]
            scale = self.scale_params[ sc ]

            ri = perturbation.find('rot')
            rotang_deg = int(perturbation[ri + 3:])


        rotang_rad = rotang_deg*(np.pi/180)
        #perturbAngle = perturbation*(np.pi/180)
        my_coords = [-1, -1]
        # subtract center
        # home position but float
        my_coords[0] = coordinates[0] - self.params['width']/2.0
        my_coords[1] = coordinates[1] - self.params['height']/2.0
        # rotate
        if self.params['feedback_fixed_distance']:
            mult = self.params['dist_tgt_from_home']
        else:
            mult = (my_coords[0]**2 + my_coords[1]**2)**(1/2.)

        cursorReachX = np.cos(np.arctan2(my_coords[1], my_coords[0]) +
                              rotang_rad) * mult
        cursorReachY = np.sin(np.arctan2(my_coords[1], my_coords[0]) +
                              rotang_rad) * mult
        cursorReachX *= scale
        cursorReachY *= scale

        cursorReachX *= sign_x

        # translate back
        cursorReachX = cursorReachX + self.params['width']/2.0
        cursorReachY = cursorReachY + self.params['height']/2.0
        cursorReachX = int(round(cursorReachX))
        cursorReachY = int(round(cursorReachY))
        return cursorReachX, cursorReachY


    def point_in_circle(self, circle_center, point, circle_radius, verbose=False):
        d = math.sqrt(math.pow(point[0]-circle_center[0], 2) +
                      math.pow(point[1]-circle_center[1], 2))
        if verbose:
            print(d, circle_radius)
        return d < circle_radius

    def timer_check(self, phase, parname, thr = None, use_frame_counter = False, verbose=0):
        # time.time() is time in second
        if use_frame_counter:
            raise ValueError('not impl')
        else:
            if thr is None:
                assert parname is not None
                thr = self.params[parname]
            else:
                assert parname is None
            timedif = time.time() - self.phase_start_times[phase]
            r =  ( timedif >= thr )
            if verbose:
                print('r = ',r,'timedif = ',timedif, '  strt=', self.phase_start_times[phase], 'thr=',thr)
        return r

    def cursor_pos_update(self):
        if self.params['controller_type'] == 'joystick':
            # from -1 to 1 -- not really. But below 1 in mod
            ax1 = self.HID_controller.get_axis(0)
            ax2 = self.HID_controller.get_axis(1)
            ax1orig = ax1 + 0  # copy
            ax2orig = ax2 + 0  # copy

            noise1 = abs( self.jaxlims_d['right'] - self.jaxlims_d['left'] )
            noise2 = abs( self.jaxlims_d['up'] - self.jaxlims_d['down'] )

            if len(self.jaxlims_d) < 4:
                jaxlims_h = -1,1
                jaxlims_v = -1,1
            else:
                #jaxlims_h = self.jaxlims_d['left'],  self.jaxlims_d.get['right']
                #jaxlims_v = self.jaxlims_d['bottom'],self.jaxlims_d.get['top']

                # joystick away -- neg, to me -- pos
                ax1 -= self.jaxcenter['ax1']
                ax2 -= self.jaxcenter['ax2']

                ax1orig = ax1 + 0  # copy
                ax2orig = ax2 + 0  # copy

                # recall that jaxlims_d are all positive and jaxcenter is not
                # jaxlims_d were computed without subtracting jaxcenter
                if ax1 > 0:
                    ax1 /= (self.jaxlims_d['right'] - self.jaxcenter['ax1'] )
                else:
                    ax1 /= (self.jaxlims_d['left'] + self.jaxcenter['ax1'] )

                if ax2 < 0: # it we are above the center
                    # if jaxcenter is positive so to go full up (which means
                    # strongly negative ax2) we'd have to travel more, we have
                    # to add something positive to scale
                    # if it is negative then similartly we have to add negative
                    ax2 /= (self.jaxlims_d['up']   + self.jaxcenter['ax2'] )
                else:
                    ax2 /= (self.jaxlims_d['down'] - self.jaxcenter['ax2'] )
            #jaxrng_h = np.abs( np.diff(jaxlims_h)[0] )
            #jaxrng_v = np.abs( np.diff(jaxlims_v)[0] )

             #/ (jaxrng_h / 2)
             #/ (jaxrng_v / 2)

            if self.params['joystick_angle2cursor_control_type'] == 'angle_scaling':
                # set cursor as a multiple of the total height/width
                self.cursorX = self.home_position[0] + ax1 * (self.params['width'] / 2)
                self.cursorY = self.home_position[1] + ax2 * (self.params['height'] / 2)
            elif self.params['joystick_angle2cursor_control_type'] == 'velocity':
                coefX = 0.05
                coefY = 0.05
                ml = 10
                #  self.current_phase == 'TARGET_AND_FEEDBACK'
                noisesc_start = 1.1
                noisesc_cont  = 0.5
                if abs(ax1orig) > noise1 * noisesc_start or ( len(self.trajX) > ml and abs(ax1orig) > noise1 * noisesc_cont ) :
                    self.cursorX += ax1 * (self.params['width'] / 2)   * coefX
                if abs(ax2orig) > noise2 * noisesc_start or ( len(self.trajY) > ml and abs(ax1orig) > noise1 * noisesc_cont ):
                    self.cursorY += ax2 * (self.params['height'] / 2)  * coefY
            else:
                raise ValueError(self.params['joystick_angle2cursor_control_type'])

            self.cursorX = int(  self.cursorX  )
            self.cursorY = int(  self.cursorY  )
            # old
            #self.cursorX = int(round(((ax1 - jaxlims[0]) / jaxrng) *
            #                    (self.params['width'] - 0) + 0))
            #self.cursorY = int(round(((ax2 - jaxlims[0]) / jaxrng) *
            #                    (self.params['height'] - 0) + 0))
        else:
            self.cursorX, self.cursorY = self.HID_controller.get_pos()

    def reset_traj(self):
        self.trajX     = [] # true traj
        self.trajY     = []
        self.trajfbX   = []
        self.trajfbY   = []
        self.traj_jax1 = []  # joystick angles raw
        self.traj_jax2 = []

    def reset_main_vars(self):
        self.feedbackX = 0
        self.feedbackY = 0
        self.unpert_feedbackX = 0
        self.unpert_feedbackY = 0
        self.feedbackX_when_crossing = 0
        self.feedbackY_when_crossing = 0
        self.error_distance = 0
        self.color_target = self.color_target_def

    def vars_update(self):
        '''
        is called if the task is running
        it govers phase change
        '''
        prev_phase = self.current_phase
        self.cursor_pos_update()
        # only if task is running (i.e. when task_start is True)


        #print('alall')

        prev_trial_index = self.trial_index
        trial_info = self.trial_infos[self.trial_index]
        ctxtpl = (trial_info['vis_feedback_type'], trial_info['tgti'] )
        vft, tgti = ctxtpl

        #print ( self.trial_infos[0]['tgti'] )

        if (self.current_phase == 'REST'):
            # if at home, increase counter
            at_home = self.is_home('unpert_cursor', 'radius_home_strict_inside')
            if not at_home:
                # if we leave center, reset to zero
                self.phase_start_times["at_home"] = time.time()

            # if we spent inside time at home than show target
            #at_home_enough = (self.frame_counters["at_home"] == self.params['FPS']*\
            #        self.params['time_at_home'])
            at_home_enough = self.timer_check("at_home", 'time_at_home')
            if at_home_enough:
                if trial_info['trial_type'] == 'pause':
                    self.current_phase = 'PAUSE'
                else:
                    if self.params['prep_after_rest']:
                        self.current_phase = 'GO_CUE_WAIT_AND_SHOW'
                    else:
                        if self.params['feedback_type'] == 'online':
                            self.current_phase = 'TARGET_AND_FEEDBACK'
                        else:
                            self.current_phase = 'TARGET'

                        print(f'at_home_enough so start {self.current_phase} display {trial_info}')

                        self.reset_traj()
                        #print(self.frame_counters["at_home"], self.params['FPS']*self.params['time_at_home'])

                        # self.trial_infos[self.trial_index, 0]
                        # depending on whether random or stable change target
                        self.tgti_to_show = tgti
                        # I don't need to put send trigger here because phase changes
                        # so it will be detected below
                        #self.send_trigger_cur_trial_info()   # send trigger after spending enough at home
                        self.color_photodiode = self.color_diode

        elif (self.current_phase == 'GO_CUE_WAIT_AND_SHOW'):
            self.tgti_to_show = tgti
            wait_finished = self.timer_check(self.current_phase,
                                    'motor_prep_duration')
            if wait_finished:
                at_home = self.is_home('unpert_cursor', 'radius_home_strict_inside', 1)
                self.reset_traj()
                if not at_home:
                    self.left_home_during_prep = 1
                    self.current_phase = 'REST'
                else:
                    self.left_home_during_prep = 0
                    if self.params['feedback_type'] == 'online':
                        self.current_phase = 'TARGET_AND_FEEDBACK'
                    else:
                        self.current_phase = 'TARGET'
                    #print(self.frame_counters["at_home"], self.params['FPS']*self.params['time_at_home'])
                    print(f'Start target and feedback display {trial_info}')

                    # self.trial_infos[self.trial_index, 0]
                    # depending on whether random or stable change target
                    self.tgti_to_show = tgti
                    # I don't need to put send trigger here because phase changes
                    # so it will be detected below
                    #self.send_trigger_cur_trial_info()   # send trigger after spending enough at home
                    self.color_photodiode = self.color_diode

        elif self.current_phase in ['TRAINING_START', 'TRAINING_END']:
            text_show_finished = self.timer_check(self.current_phase,
                                    'training_text_show_duration')
            if text_show_finished:

                ct = time.time()
                self.phase_start_times['at_home'] = ct
                self.moveHome()
                self.send_trigger_cur_trial_info()
                self.reset_traj()
                self.color_feedback = self.color_feedback_def

                if self.params['return_home_after_ITI']:
                    self.moveHome()

                if self.current_phase != 'TRAINING_START':
                    self.trial_index += 1
                    print ('Start Trial: ' + str(self.trial_index))

                self.current_phase = 'REST'

        elif self.current_phase == 'TARGET_AND_FEEDBACK':
            self.trajX += [ self.cursorX ]
            self.trajY += [ self.cursorY ]

            self.color_target = self.color_target_def

            #reach_time_finished = (self.frame_counters["feedback_shown"] == \
            #        self.params['FPS'] * self.params['time_feedback'])

            hit_cond = self.point_in_circle(self.target_coords[self.tgti_to_show],
                                    (self.feedbackX, self.feedbackY),
                                    self.params['radius_target'] +
                                    self.params['radius_cursor'])

            reach_time_finished = self.timer_check(self.current_phase,
                                    'time_feedback')
            if reach_time_finished:
                self.timer_check(self.current_phase,
                                    'time_feedback', verbose=1)
            # if time is up we switch to ITI, else
            reach_finished, extinfo = self.test_reach_finished(ret_ext = 1)
            at_home,stopped,radius_reached = extinfo
            if reach_time_finished or reach_finished:
                print(f'reach_time_finished={reach_time_finished}, '
                      f'at_home={at_home}, stopped={stopped}, '
                      f'radius_reached={radius_reached}, hit_cond={hit_cond}')

                if self.debug:
                    self.save_scr(trial_info, prefix='keypress')

                if reach_finished and hit_cond:
                    full_success = 1
                else:
                    full_success = 0

                self.last_trial_full_success = full_success
                if full_success:
                    self.last_reach_too_slow     = 0
                    self.last_reach_stopped_away = 0
                    self.last_reach_not_full_rad = 0
                else:
                    if reach_time_finished: # and not (reach_finished):
                        if hit_cond:
                            self.last_reach_too_slow = 2
                        else:
                            self.last_reach_too_slow = 1
                        tdif = time.time() - self.phase_start_times[self.current_phase]
                        print(f'Reach too slow: {tdif:.2f} sec')
                    else:
                        self.last_reach_too_slow = 0

                    if stopped and (not reach_time_finished):
                        self.last_reach_stopped_away = 1

                    if stopped and (not radius_reached) and (not hit_cond):
                        self.last_reach_not_full_rad = 1
                    else:
                        self.last_reach_not_full_rad = 0

                # for the coming ITI update its duration
                ITI_jitter_low = 0.
                ITI_jitter_high = 1.
                self.ITI_jittered = self.params['ITI_duration'] +\
                        self.params['ITI_jitter'] * \
                        np.random.uniform(ITI_jitter_low, ITI_jitter_high)
                print(f'Trial {self.trial_index}: {self.current_phase} finish condition met')

                if self.get_dist_from_home() < self.params['radius_target']:
                    print('Too long staying at the target')
                self.current_phase = 'ITI'
                self.color_photodiode = self.color_diode_off

                # check pretraining
                if full_success:
                    self.counter_hit_trials += 1
                    self.reward = 1
                else:
                    # regarless whether we hit or not if to slow then no reward
                    #d = (self.error_distance - (self.params['radius_target'] + self.params['radius_cursor'] ))
                    d = self.error_distance / self.params['radius_target']
                    d = max(d , 1.)
                    # control decay of reward larger means stronger punishment
                    # for errror
                    self.reward = np.power( 1 / d, 1.2)

                if trial_info['special_block_type'] != 'pretraining':
                    self.reward_accrued += self.reward

                print(f'Reward = {self.reward:.2f}, accrued = {self.reward_accrued:.2f}')


            # else draw feedback and check hit
            else:
                #self.frame_counters["feedback_shown"] += 1

                #self.feedbackX, self.feedbackY = \
                #        self.apply_visuomotor_pert((self.cursorX, self.cursorY), vft)
                self.feedbackX, self.feedbackY = \
                    self.alter_feedback(
                    (self.cursorX, self.cursorY), vft, trial_info['trial_type'])


                self.trajfbX += [ self.feedbackX ]
                self.trajfbY += [ self.feedbackY ]

                # This variable saves the unperturbed, unprojected feedback
                self.unpert_feedbackX, self.unpert_feedbackY = self.cursorX, self.cursorY
                self.error_distance = np.sqrt((self.feedbackX - self.target_coords[self.tgti_to_show][0])**2 +
                                              (self.feedbackY - self.target_coords[self.tgti_to_show][1])**2)

                if self.params['hit_when_passing'] and hit_cond:
                    # hit color
                    self.color_target = self.color_hit

                #circle crossing check
                vec_feedback = np.array( [self.feedbackX, self.feedbackY]  ) - \
                    np.array(self.home_position)
                dist_from_home = np.linalg.norm(vec_feedback)
                if np.abs(dist_from_home - self.params['dist_tgt_from_home']) < \
                        self.params['radius_cursor']:
                    self.feedbackX_when_crossing = self.feedbackX
                    self.feedbackY_when_crossing = self.feedbackY

            #self.currentText = self.myfont.render(
            #    'Hits: ' + str(self.counter_hit_trials) + '/' +\
            #    str(self.trial_index+1), True, self.color_text)

        elif (self.current_phase == 'TARGET'):
            if self.params['feedback_type'] == 'online':
                raise ValueError('nooo!')
            self.trajX += [ self.cursorX ]
            self.trajY += [ self.cursorY ]
            print(f'len traj = {len(self.trajX) }')
            # if we have reached radius, set to FEEDBACK
            if (self.test_reach_finished()):
                #self.feedbackX, self.feedbackY = \
                #        self.apply_visuomotor_pert((self.cursorX, self.cursorY), vft)
                self.feedbackX, self.feedbackY = \
                    self.alter_feedback(
                    (self.cursorX, self.cursorY), vft, trial_info['trial_type'])

                # This variable saves the unperturbed, unprojected feedback
                self.unpert_feedbackX, self.unpert_feedbackY = self.cursorX, self.cursorY
                self.trajfbX += [ self.feedbackX ]
                self.trajfbY += [ self.feedbackY ]

                self.error_distance = np.sqrt((self.feedbackX - self.target_coords[self.tgti_to_show][0])**2 +
                                              (self.feedbackY - self.target_coords[self.tgti_to_show][1])**2)

                if self.point_in_circle(self.target_coords[self.tgti_to_show],
                                        (self.feedbackX, self.feedbackY),
                                        self.params['radius_target'] +
                                        self.params['radius_cursor']):
                    self.color_target = self.color_hit
                    self.counter_hit_trials += 1
                self.current_phase = 'FEEDBACK'
                #self.current_phase_trigger = self.trigger2phase[self.current_phase]
                #self.send_trigger(self.current_phase_trigger)
                self.color_photodiode = self.color_diode_off

        elif (self.current_phase == 'FEEDBACK'):
            if self.params['feedback_type'] == 'online':
                raise ValueError('nooo!')
            #self.frame_counters["feedback_shown"] += 1
            tdb = time.time() - self.phase_start_times['FEEDBACK'] > self.params['time_feedback']
            if tdb:
                if ((self.trial_index+1) != len(self.trial_infos)):
                    # if change between RANDOM and STABLE, give a break
                    break_condition_met = False
                    if break_condition_met:
                        self.current_phase = 'BREAK'
                        self.free_from_break = 0
                    else:
                        self.current_phase = 'ITI'
                else:
                    self.current_phase = 'ITI'

                self.ITI_jittered = self.params['ITI_duration'] +\
                        self.params['ITI_jitter'] * np.random.random_sample()


        elif (self.current_phase == 'BREAK'):
            if (self.free_from_break):
                self.current_phase = 'ITI'


        elif (self.current_phase == 'PAUSE'):
            #self.frame_counters["pause"] = self.frame_counters["pause"] + 1
            #pause_finished = self.frame_counters["pause"] == int(self.params['FPS'] *\
            #                              self.params['pause_duration'] )
            pause_finished = self.timer_check(self.current_phase, "pause_duration")
            if pause_finished:
                self.current_phase = 'REST'
                #for ctr in self.frame_counters:
                #    self.frame_counters[ctr] = 0
                self.reset_main_vars()
                self.trial_index += 1
                print(f'{self.current_phase}: trial_index inc, now it is {self.trial_index}')

                if self.params['return_home_after_pause']:
                    self.moveHome()
                print ('Start Trial: ' + str(self.trial_index))

        elif (self.current_phase == 'ITI'):
            #self.frame_counters["ITI"] += 1
            #ITI_finished =(self.frame_counters["ITI"] == \
            #        int(self.params['FPS'] * self.ITI_jittered))
            ITI_finished = self.timer_check(self.current_phase,
                                            parname=None,
                                            thr = self.ITI_jittered)
            if ITI_finished:
                #for ctr in self.frame_counters:
                #    self.frame_counters[ctr] = 0
                self.reset_main_vars()
                self.trial_index += 1
                print(f'{self.current_phase}: trial_index inc, now it is {self.trial_index}')

                if self.params['rest_after_return']:
                    self.current_phase = 'RETURN'
                else:
                    self.current_phase = 'REST'
                    self.color_feedback = self.color_feedback_def

                if self.params['return_home_after_ITI']:
                    self.moveHome()
                print ('Start trial # ' + str(self.trial_index))

                if self.trial_index < len(self.trial_infos):
                    trial_info2 = self.trial_infos[self.trial_index]
                    if trial_info['special_block_type'] == 'pretraining' and \
                        trial_info2['special_block_type'] != 'pretraining':
                        self.just_finished_pretraining = 1
                        print('Just finished pretraining')
                        self.current_phase = 'TRAINING_END'
                        self.reset_main_vars()

        elif (self.current_phase == 'RETURN'):
            at_home = self.is_home('unpert_cursor', 'radius_return')
            #time_is_up = ( self.frame_counters["return"] == int(self.params['FPS'] *\
            #        self.return_max_duration) )
            time_is_up = self.timer_check(self.current_phase,'return_duration')

            if at_home or time_is_up:
                self.current_phase = 'REST'
                #self.frame_counters["return"] = 0
                # even if we have not returned in time, we forcibly put on
                # there
                # TODO: maybe here one needs to punish participant otherwise
                # they won't return at all
                self.moveHome()
            #else:
            #    self.frame_counters["return"] += 1
        else:
            print("Error")
            raise ValueError('wrong phase')
            self.on_cleanup(-1)

        if self.trial_index == len(self.trial_infos):
            self.task_started = 2

        self.current_phase_trigger = self.phase2trigger[self.current_phase]
        if self.current_phase != prev_phase:
            self.phase_start_times[self.current_phase] = time.time()
            # do I need it?
            if self.current_phase == 'REST':
                self.phase_start_times["at_home"] = time.time()
            #self.send_trigger(self.current_phase_trigger)
            if self.task_started == 1:
                self.send_trigger_cur_trial_info()

            tdif = time.time() - self.phase_start_times[prev_phase]

            print(f'Phase change {prev_phase} -> {self.current_phase};  {prev_trial_index} -> {self.trial_index}')
            print(f"             {prev_phase} finished after {tdif:.3f} s")
            #print(f'   tgt = {self.tgti_to_show}')

        if prev_trial_index != self.trial_index and self.trial_index < len(self.trial_infos):
            trial_info2 = self.trial_infos[self.trial_index]
            tc = time.time()
            tdif = tc - self.phase_start_times['current_trial']
            self.phase_start_times['current_trial'] = tc

            print(f'Trial index change! {prev_trial_index} -> {self.trial_index}')
            print(f'TIME: trial completed in {tdif:.2f} sec')
            print(f'  prev trial = {trial_info}')
            print(f'  new trial  = {trial_info2}')




        ## needed because apparently mouse position is not reset immediately
        #if self.just_moved_home:
        #    self.just_moved_home = 0



    def update_log(self):
        # logging
        '''
        saves everything to the log
        called from on_execute, so on every frame
        '''
        if (self.task_started == 2):
            return
        ti = self.trial_infos[self.trial_index]
        # prepare one log line
        self.current_log = []
        self.current_log.append(self.trial_index)
        self.current_log.append(self.current_phase_trigger)
        self.current_log.append(self.tgti_to_show) #target index
        self.current_log.append(ti['vis_feedback_type']) #perturbation
        self.current_log.append(ti['trial_type']) #perturbation
        self.current_log.append(ti['special_block_type']) #perturbation
        #self.current_log.append(self.cursorX)
        #self.current_log.append(self.cursorY)
        self.current_log.append(self.feedbackX)
        self.current_log.append(self.feedbackY)
        # this is the same as cursorX
        self.current_log.append(self.unpert_feedbackX) # before was called 'org_feedback'
        self.current_log.append(self.unpert_feedbackY)
        self.current_log.append(self.error_distance)  # Euclidean distance
        # TODO: save area difference of traj

        # maybe I will add it several times if I cross more than once
        # or stay around dist. But it is okay, I can select later first or last
        # crossing during data processing
        self.current_log.append(self.feedbackX_when_crossing)  # Euclidean distance
        self.current_log.append(self.feedbackY_when_crossing)  # Euclidean distance

        self.current_log.append(self.current_time - self.initial_time)
        self.logfile.write(",".join(str(x) for x in self.current_log) + '\n')

    def is_home(self, coords_label = 'unpert_cursor',
                param_name = 'radius_return', mult=1.):
        if self.just_moved_home:
            return True
        if coords_label == 'unpert_cursor':
            pos = (self.cursorX, self.cursorY)
        elif coords_label == 'feedback':
            pos = (self.feedbackX, self.feedbackY)
        else:
            raise ValueError('wrong coords label')

        at_home = self.point_in_circle(self.home_position, pos,
                                self.params[param_name] * mult, verbose=0)
        return at_home

    def on_cleanup(self, exit_type):
        '''
        at the end of on_execute
        '''
        self.send_trigger(self.MEG_stop_trigger)
        time.sleep(0.05)
        self.MEG_rec_started = 0

        self.send_trigger(0)
        if (exit_type == 0):
            print("Yay")
        elif (exit_type == -1):
            print("Ouch")
        self.logfile.close()
        self.trigger_logfile.close()

        pygame.quit()
        exit()


    def on_execute(self):
        if self.on_init() is False:
            self._running = False
        self.clock = pygame.time.Clock()
        #clock = pygame.time.Clock()
        self.initial_time = time.time()
        #if (self.params['use_eye_tracker']):
        #    EyeLink.tracker(self.params['width'], self.params['height'])
        self.initial_time = time.time()

        if self.joystick_center_autocalib and self.params['controller_type'] == 'joystick':
            ax1 = self.HID_controller.get_axis(0)
            ax2 = self.HID_controller.get_axis(1)
            if max(abs(ax1),abs(ax2) ) > 0.15:
                raise ValueError(('You have to start the app with joystick'
                    ' in the center position'))
            self.jaxcenter =  {'ax1':ax1, 'ax2':ax2} #{'ax1' :0}

        # MAIN LOOP
        while(self._running):
            #self.clock.tick()
            self.current_time = time.time()
            # process events
            for event in pygame.event.get():
                self.on_event(event)

            # when task is running
            if (self.task_started == 1):
                self.vars_update()

                # count time since last trigger (to send trigger reset signal
                # later)
                if self.trigger_value in [self.MEG_start_trigger, self.MEG_stop_trigger]:
                    td = self.params['MEG_trigger_duration']
                else:
                    td = self.params['trigger_duration']
                if self.trigger_value > 0 and (time.time() - self.phase_start_times['trigger']) > td:
                    self.trigger_value = 0
                    self.send_trigger(self.trigger_value)
                #if (self.trigger_countdown > 0):
                #    self.trigger_countdown -= 1
                #elif (self.trigger_countdown == 0):
                #    self.trigger_countdown = -1
                #    # trigger reset
                #    self.send_trigger(0)
                self.update_log()
            elif (self.task_started == 2):
                if self.trial_index == len(self.trial_infos):
                    if self.ctr_endmessage_show == self.ctr_endmessage_show_def:
                        print("Success ! Experiment finished ")
                    if self.ctr_endmessage_show == 0:
                        # end program
                        break
                    else:
                        self.ctr_endmessage_show -= 1

            #print(clock)


            self.on_render()
            # print(time.time()-self.current_time)
            msElapsed = self.clock.tick_busy_loop(self.params['FPS'])

        self.on_cleanup(0)


if __name__ == "__main__":

    #app = VisuoMotor()

    info = {}

    info['participant'] = ''
    info['session'] = ''

    show_dialog = 1
    if show_dialog:
        dlg = gui.DlgFromDict(info)
        if not dlg.OK:
            core.quit()
    else:
        info['participant'] = 'Dmitrii'
        info['session'] = 'session1'


    seed = 8
    app = VisuoMotor(info, use_true_triggers=0, debug=1, seed=seed)
    #app = VisuoMotor(info, use_true_triggers=0, debug=0, seed=seed)
    app.on_execute()

    # it starts a loop in which it check for events using pygame.event.get()
    # then it executes on_event for all dected events
    # after than if we are running a task it runs vars_update

    # keys:   ESC -- exit, f11 -- fullscreen,  q,w -- change cursor radius
    # pressing joystick button can change experiment phase if task_started == 0


    #task_started is more like "task is running now"

    # Q: what is trigger countdown? And how trigger sending works in general?
    # Q: what is rest_phase + 5
    # more generally, what does if event.type == pygame.JOYBUTTONDOWN in
    #   on_event do?

    # photodiode color is set to white when change from REST to TARGET
    # to black when change from TARGET to FEEDBACK

    # top left = (0,0), top right (max, right)
    # so first is X (goes right), second is Y (goes down)

    # REST is used to settle at home

    # REST is both during ITI and before task. In it we plot orig feedback at
    # home
    # free_from_break asks to transition from BREAK to ITI

    # after start task we go to REST

    # ITI phase after time_feedback
    # ITI then REST

    # during ITI show nothing, during REST show orig, wait a bit and then
    # switch to TARGET_PHASE

    # show feedback only in FEEDBACK_PHASE


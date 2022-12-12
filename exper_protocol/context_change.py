from __future__ import print_function
# needed for joystick
import pygame
from pygame.locals import *
import time
import random
import math
import logging
import os
#from win32api import GetSystemMetrics
import sys
if sys.platform.startswith('win32'):
    from ctypes import windll
from random import shuffle
import numpy as np
import math
#import pylink
#import EyeLink


class VisuoMotor:

    def add_param(self, name, value):
        self.params.update({name: value})
        self.paramfile.write(name + ' = ' + str(value) + '\n')


    def add_param_comment(self, comment):
        self.paramfile.write(comment + '\n')


    def initialize_parameters(self, info):
        #self.debug = False
        self.debug = True # affects fullscreen or not
        self.params = {}
        self.task_id = 'visuomotor'
        self.subject_id = info['participant']
        self.session_id = info['session']
        self.timestr = time.strftime("%Y%m%d_%H%M%S")
        self.filename = ('data/' + self.subject_id + '_' + self.task_id +
                         '_' + self.timestr)
        self.paramfile = open(self.filename + '.param', 'w')
        self.logfile = open(self.filename + '.log', 'w')
        self.add_param_comment('# Width of screen')      # updates self.param dictionary
        self.add_param('width', 800)
        self.add_param_comment('# Height of screen')
        self.add_param('height', 800)
        self.add_param_comment('# Frames per second for plotting')
        self.add_param('FPS', 120)
        self.add_param_comment('# Radius of the cursor')
        self.add_param('radius_cursor', 10)
        self.add_param_comment('# Radius of the target')
        self.add_param('radius_target', 14)
        # distance from the start location to the target center
        self.add_param_comment('# Radius of the invisible boundary')
        self.add_param('radius', int(round(self.params['height']*0.5*0.8)))
        self.add_param_comment('# Time inside the home position before trial \
                        starts (seconds)')
        self.add_param('time_inside', 0.5)
        self.add_param_comment('# Time for feedback (seconds)')
        self.add_param('time_feedback', 0.25)
        self.add_param_comment('# Time for intertrial interval (seconds)')
        self.add_param('time_ITI', 1.5)
        self.add_param_comment('# Max jitter during ITI (seconds)')
        self.add_param('jitter', 0.1)
        self.add_param_comment('# Show text?')
        self.add_param('show_text', 0)
        self.add_param_comment('# Use eye tracker?')
        self.add_param('use_eye_tracker', 1)
        self.add_param_comment('# Use triggers?')
        self.add_param('use_triggers', 1)
        self.paramfile.close()


    def __init__(self, info, task_id='',
                 use_triggers = 1, use_joystick=1):
        self.initialize_parameters(info)
        self.params['use_triggers'] = use_triggers
        self.use_triggers = self.params['use_triggers']
        if (self.use_triggers):
            print("Using triggers")
        else:
            print("NOT using triggers")

        #self.eye_tracker = pylink.EyeLink(None)
        self.size = self.params['width'], self.params['height']
        self.trigger_port = 0x378
        self.trigger_countdown = -1
        self.trial_counter = 0
        self.num_trials_task = 80
        self.counter_inside = 0
        self.counter_feedback = 0
        self.counter_ITI = 0
        self.counter_PAUSE = 0
        self.counter_hit_trials = 0

        self.REST_PHASE = 10
        self.TARGET_PHASE = 20
        self.FEEDBACK_PHASE = 30
        self.ITI_PHASE = 40
        self.BREAK_PHASE = 50
        self.PAUSE_1MIN_PHASE = 60
        #self.ERROR_CLAMP_PHASE = 70  # NO, THIS IS NOT A PHASE! It is feedback type
        #self.STABLE_CONDITION_ID = 0
        #self.RANDOM_CONDITION_ID = 1
        self.current_phase = None
        self.CONTEXT_TRIGGER_DICT = {}

        self.use_joystick = use_joystick
        if use_joystick:
            self.phase_shift_event_type = pygame.JOYBUTTONDOWN
        else:
            self.phase_shift_event_type = pygame.MOUSEBUTTONDOWN
        self.init_target_positions()
        pygame.init()
        self.myfont = pygame.font.SysFont('Calibri', 24)
        self.string_instructions = 'We will start soon. Please wait for instructions'
        self.length_text = self.myfont.size(self.string_instructions)

        self.string_break = 'BREAK'


        self.currentText = self.myfont.render('Hits: 0/0', True,
                                              (255, 255, 255))
        self.instructions_text = self.myfont.render(self.string_instructions,
                                                    True, (255, 255, 255))


        self.break_text = self.myfont.render(self.string_break,
                                                    True, (255, 255, 255))

        # it changes value to 1 only once, when we start the experiment
        self.start_task = 0

        self.color_photodiode = [0, 0, 0]
        self.home_position = (int(round(self.params['width']/2.0)),
                              int(round(self.params['height']/2.0)))
        if use_joystick:
            self.my_joystick = pygame.joystick.Joystick(0)
            self.my_joystick.init()
        else:
            self.my_joystick = pygame.mouse
        self.cursorX = 0
        self.cursorY = 0
        self.feedbackX = 0
        self.feedbackY = 0
        self.unpert_feedbackX = 0
        self.unpert_feedbackY = 0
        self.error_distance = 0
        self.colorTarget = [0, 255, 0]

        self.block_stack = []
        #self.counter_random_block = 0

        self.free_from_break = 0
        # trajectory starting from last target phase
        self.trajX = []
        self.trajY = []

        a = np.random.sample()
        trials_vector = [6, 15, 6, 15, 6]
        if a >= 0.5:
            perturb_vector = [0, 30, 0, -30, 0]
            #perturb_vector.append([0, -30, 0, 30, 0])
        else:
            perturb_vector = [0, -30, 0, 30, 0]
            #perturb_vector.append([0, 30, 0, -30, 0])


        self.scale_params = {'scale-':0.75, 'scale+':1.25}
        self.pert_block_types = ['rot-15', 'rot15', 'rot30', 'rot60', 'rot90',
                       'scale-', 'scale+', 'reverse_x'  ]
        self.vis_feedback_types = self.pert_block_types + ['veridical']
        special_trial_block_types = ['error_clamp_sandwich', 'error_clamp_pair']
        trial_type = ['veridical', 'perturbation', 'error_clamp', 'pause']
        block_len_min = 7
        block_len_max = 13
        n_context_appearences = 3

        target_inds = np.arange(len(self.target_coords) )
        from itertools import product,repeat
        vfti_seq0 = product( self.vis_feedback_types, target_inds )
        n_contexts = len(vfti_seq0)
        vfti_seq_noperm = list(vfti_seq0) * n_context_appearences
        # TODO: manage seed here, make it participant or date depenent explicitly
        ct_inds = np.random.permutation(np.arange(len(vfti_seq_noperm) ) )
        vfti_seq = [vfti_seq0[i] for i in ct_inds] # I don't want to convert to numpy here
        n_blocks = len(vfti_seq)
        ns_context_repeat = np.random.randint( n_contexts, size=n_blocks )

        #n_blocks = n_contexts * n_context_appearences
        #seq0 = np.tile( np.arange(n_contexts), n_context_appearences)
        #context_seq = np.random.permutation(seq0)

        self.trial_infos = [] # this is the sequence of trial types
        # I want list of tuples -- target id, visual feedback type, phase
        # (needed for trigger)
        trigger = self.ERROR_CLAMP_PHASE + 10
        # TODO: care about whether pauses and clamps are parts of the block or
        # not
        block_start_inds = []
        for (bi, num_context_repeats), (vis_feedback_type, tgti) in\
                zip( enumerate(ns_context_repeat), vfti_seq):
            #r += [ (vis_feedback_type, tgti) ] * ncr
            # NOTE: if I insert pauses later, it will be perturbed
            block_start_inds += [ len( self.trial_infos ) ]
            if vis_feedback_type == 'veridical':
                ttype = vis_feedback_type
            else:
                ttype = 'perturbation'
            d = {'vis_feedback_type':vis_feedback_type, 'tgti':tgti}
            self.trial_infos += [d] * num_context_repeats

            tpl = (vis_feedback_type, tgti)
            if tpl not in self.CONTEXT_TRIGGER_DICT:
                self.CONTEXT_TRIGGER_DICT[ tpl ] = trigger
                trigger += 1
            #if bi % 4 == 0:
            # INSERT clamp or pause

        # TODO: 1min pause
        # TODO:
        # define block sequence (generated randomly)

        # prepare trial info
        # self.trial_infos = np.empty([1, 3])  # Initialize trial_infos
        # NOTE! this assumes fixed number of trials
        #for block in ['stable1', 'random', 'stable2', 'random']:
        #    for nbTrial, perturbation in zip(trials_vector, perturb_vector):
        #        x = np.zeros(nbTrial)
        #        x1 = np.concatenate((x, x + 1, x + 2, x + 3)) # horizontal
        #        x1 = np.random.permutation(x1)  # target id
        #        if 'stable' in block:
        #            if block == 'stable1':
        #                x2 = np.ones(nbTrial*4) * perturbation
        #            elif block == 'stable2':
        #                x2 = np.ones(nbTrial*4) * perturbation * (-1)   # pert if deterministic
        #            x3 = np.zeros(nbTrial*4) + self.STABLE_CONDITION_ID  # phase
        #        elif block == 'random':
        #            x2 = np.zeros(nbTrial*4)
        #            x3 = np.zeros(nbTrial*4) + self.RANDOM_CONDITION_ID  # phase
        #        x1x2 = np.column_stack((x1, x2, x3))
        #        self.trial_infos = np.vstack((self.trial_infos, x1x2))


        # Q: why delete zeroth?
        #self.trial_infos = np.delete(self.trial_infos, (0), axis=0)
        print(self.trial_infos)
        self.tgti_to_show = 0
        if (self.use_triggers):
            self.trigger = windll.inpout32
            self.send_trigger(0)


    def init_target_positions(self):
        '''
        called in class constructor
        '''
        #targetAngs = [22.5+180, 67.5+180, 112.5+180, 157.5+180]
        targetAngs = np.array([-15,0,15]) + (180 + 90)

        # list of 2-tuples
        self.target_coords = []
        for tgti,tgtAngDeg in enumerate(targetAngs):
            tgtAngRad = tgtAngDeg*(np.pi/180)
            # this will be given to pygame.draw.circle as 3rd arg
            # half screen width + cos * radius
            # half screen hight + sin * radius
            self.target_coords.append((int(round(self.params['width']/2.0 +
                                          np.cos(tgtAngRad) * self.params['radius'])),
                                      int(round(self.params['height']/2.0 +
                                          np.sin(tgtAngRad) * self.params['radius']))))


    def send_trigger(self, value):
        if (self.use_triggers):
            # We block it if it has sent something already
            if (self.trigger_countdown == -1):
                self.trigger.Out32(self.trigger_port, value)
            # print("Sent trigger " + str(value))
            # For how long the trigger is gonna be raised? (in ms)
            if (value != 0):
                self.trigger_countdown = int(round(self.params['FPS']*(50/1000.0)))


    def on_init(self):
        pygame.init()
        if self.debug:
            self._display_surf = pygame.display.set_mode(self.size)
        else:
            self._display_surf = pygame.display.set_mode(self.size, pygame.FULLSCREEN)
        self._running = True


    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        if event.type == self.phase_shift_event_type:
            print(self.my_joystick.get_pos(), self.current_phase,
                  self.trial_infos[self.trial_counter] )

            # if task was not started and a button was pressed, start the task
            if self.start_task == 0:
                self.current_phase = self.REST_PHASE
                # TODO: send env info?
                #if self.trial_infos[self.trial_counter, 2] == 0:
                #    self.send_trigger(self.current_phase)
                #elif self.trial_infos[self.trial_counter, 2] == 1:
                #    self.send_trigger(self.current_phase+5)
                self.start_task = 1

            # if task was started and a button was pressed, release break
            else:
                self.free_from_break = 1

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.on_cleanup(-1)
            if event.key == pygame.K_f:
                if (self._display_surf.get_flags() & pygame.FULLSCREEN):
                    self._display_surf = pygame.display.set_mode(self.size)
                else:
                    self._display_surf = pygame.display.set_mode(self.size, pygame.FULLSCREEN)
            if event.key == pygame.K_q:
                self.params['radius_cursor'] = self.params['radius_cursor']-1
            if event.key == pygame.K_w:
                self.params['radius_cursor'] = self.params['radius_cursor']+1
            #if event.key == pygame.K_c:
            #    if (self.params['use_eye_tracker']):
            #        EyeLink.tracker(self.params['width'], self.params['height'])

    def drawTgt(self):
        pygame.draw.circle(self._display_surf, self.colorTarget,
                           self.target_coords[self.tgti_to_show],
                           self.params['radius_target'], 0)

    def drawHome(self):
        pygame.draw.circle(self._display_surf, [0, 0, 0],
                           self.home_position,
                           int(self.params['radius_cursor']*2.0), 2)

    def drawCursorFeedback(self):
        pygame.draw.circle(self._display_surf, [255, 255, 255],
                           (self.feedbackX, self.feedbackY),
                           self.params['radius_cursor'], 0)

    def drawCursorOrig(self, debug=0):
        c = [255, 255, 255]
        if debug:
            c = [200, 100, 100]
        pygame.draw.circle(self._display_surf, c,
                           (self.cursorX, self.cursorY),
                           self.params['radius_cursor'], 0)


    def drawTraj(self, pert=0):
        for i,(x,y) in enumerate(zip(self.trajX, self.trajY ) ):
            c = [100, 50, 60]

            if pert:
                trial_info = self.trial_infos[self.trial_counter]
                ctxtpl = (trial_info['vis_feedback_type'], trial_info['tgti'] )
                vft, tgti = ctxtpl
                self.apply_visuomotor_pert(self, (x,y), vft)
            pygame.draw.circle(self._display_surf, c,
                (x,y), self.params['radius_cursor'] / 2.5 , 0 )

    def drawTextCenter(self, text, length_info):
        self._display_surf.blit(self.text,
            (int(round(((self.params['width'] - length_info[0]) / 2.0))),
             int(round(((self.params['height'] - length_info[1]) / 2.0)))))

    def on_render(self):
        if (self.start_task):
            # Remove components screen
            self._display_surf.fill([100, 100, 100])
            if self.current_phase == self.REST_PHASE:
                # if home, draw cursor
                if self.point_in_circle(self.home_position,
                                        (self.cursorX, self.cursorY),
                                        self.params['radius_cursor'] * 5):
                    self.drawCursorOrig()
                self.drawHome()

            if self.current_phase == self.TARGET_PHASE:
                self.drawTgt()
                self.drawHome()

            if (self.current_phase == self.FEEDBACK_PHASE):
                self.drawTgt()
                self.drawHome()
                self.drawCursorFeedback()

            pygame.draw.rect(self._display_surf, self.color_photodiode,
                             (0, 0, self.params['width'], 30), 0)

            # maybe show info on participant hitting performance
            if (self.params['show_text']):
                self._display_surf.blit(self.currentText, (0, 0))

            if (self.current_phase == self.BREAK_PHASE):
                self.drawTextCenter(self.break_text, self.length_text)
        # if not start_task
        else:
            self._display_surf.fill([100, 100, 100])
            self.drawTextCenter(self.instructions_text, self.length_text)
            #self._display_surf.blit(self.instructions_text,
            #(int(round(((self.params['width'] - self.length_text[0]) / 2.0))),
            # int(round(((self.params['height'] - self.length_text[1]) / 2.0)))))


        if self.debug:
            debugstr = f'on_render: X={self.cursorX},Y={self.cursorY},  Phase={self.current_phase}'
            debugstr += f' ctrinside={self.counter_inside}'
            debug_text = self.myfont.render(debugstr, True, (255, 255, 255))
            ldt = self.myfont.size(debugstr)
            self._display_surf.blit(debug_text, (5, self.params['height']-30))


            self.drawCursorOrig(debug=1)
            self.drawTraj()


        pygame.display.update()


    def radius_reached(self):
        '''
        called from vars_update
        '''
        centeredX = self.cursorX - self.home_position[0]
        centeredY = self.cursorY - self.home_position[1]
        return (centeredX**2+centeredY**2)**(1/2.) >= self.params['radius']


    def apply_visuomotor_pert(self, coordinates, perturbation):
        '''
        called in vars_update
        rotates coordinates
        '''
        assert perturbation in self.vis_feedback_types

        scale = 1.
        rotang_deg = 0.
        sign_x = 1.
        if perturbation.startswith('rot'):
            rotang_deg = int(perturbation[3:])
        elif perturbation.startswith('scale'):
            scale = scale_params[perturbation]
        elif perturbation.startswith('reverse'):
            sign_x = -1.

        rotang_deg = perturbation*(np.pi/180)
        #perturbAngle = perturbation*(np.pi/180)
        my_coords = [-1, -1]
        # subtract center
        my_coords[0] = coordinates[0] - self.params['width']/2.0
        my_coords[1] = coordinates[1] - self.params['height']/2.0
        # rotate
        cursorReachX = np.cos(np.arctan2(my_coords[1], my_coords[0]) +
                              rotang_deg) * self.params['radius']
        cursorReachY = np.sin(np.arctan2(my_coords[1], my_coords[0]) +
                              rotang_deg) * self.params['radius']
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


    def vars_update(self):
        # only if task is running (i.e. when task_start is True)
        if self.use_joystick:
            self.cursorX = self.my_joystick.get_axis(0)
            self.cursorY = self.my_joystick.get_axis(1)

            self.cursorX = int(round(((self.cursorX - -1) / (1 - -1)) *
                                (self.params['width'] - 0) + 0))
            self.cursorY = int(round(((self.cursorY - -1) / (1 - -1)) *
                                (self.params['height'] - 0) + 0))
        else:
            self.cursorX, self.cursorY = self.my_joystick.get_pos()


        #print('alall')

        trial_info = self.trial_infos[self.trial_counter]
        ctxtpl = (trial_info['vis_feedback_type'], trial_info['tgti'] )
        vft, tgti = ctxtpl

        if (self.current_phase == self.REST_PHASE):
            # if at home, increase counter
            if self.point_in_circle(self.home_position, (self.cursorX, self.cursorY),
                                    self.params['radius_cursor'], verbose=0):
                self.counter_inside = self.counter_inside+1
            else:
                # if we leave center, reset to zero
                self.counter_inside = 0

            # if we spent inside time at home than show target
            if (self.counter_inside == self.params['FPS']*self.params['time_inside']):
                self.current_phase = self.TARGET_PHASE

                self.trajX = []
                self.trajY = []
                #print(self.counter_inside, self.params['FPS']*self.params['time_inside'])
                print('Start target display')

                # self.trial_infos[self.trial_counter, 0]
                # depending on whether random or stable change target
                self.tgti_to_show = tgti
                self.send_trigger(self.CONTEXT_TRIGGER_DICT[ctxtpl] )
                ## TODO: change
                #if (self.trial_infos[self.trial_counter, 2] ==  \
                #        self.STABLE_CONDITION_ID):
                #    self.tgti_to_show = int(self.trial_infos[self.trial_counter, 0])
                #elif (self.trial_infos[self.trial_counter, 2] == \
                #        self.RANDOM_CONDITION_ID):
                #    self.tgti_to_show = self.block_stack[self.counter_random_block][2]

                ## depending on whether random or stable change target trigger
                ## TODO: change
                #if self.trial_infos[self.trial_counter, 2] == 0:
                #    # target trigger
                #    self.send_trigger(self.current_phase + self.tgti_to_show)
                #elif self.trial_infos[self.trial_counter, 2] == 1:
                #    self.send_trigger(self.current_phase + self.tgti_to_show + 5)
                self.color_photodiode = [255, 255, 255]

        elif (self.current_phase == self.TARGET_PHASE):
            # if we have reached radius, set to FEEDBACK
            if (self.radius_reached()):
                #if (self.trial_infos[self.trial_counter, 2] == self.STABLE_CONDITION_ID):
                #    self.feedbackX, self.feedbackY = self.apply_visuomotor_pert((self.cursorX, self.cursorY),
                #                self.trial_infos[self.trial_counter, 1])
                #    self.block_stack.append([self.feedbackX, self.feedbackY, self.tgti_to_show])

                #    if ((self.trial_counter+1) != len(self.trial_infos)):
                #        if (self.trial_infos[self.trial_counter+1, 2] == self.RANDOM_CONDITION_ID):
                #            self.block_stack = np.array(self.block_stack)
                #            self.block_stack = np.random.permutation(self.block_stack)
                #            self.counter_random_block = 0

                #elif (self.trial_infos[self.trial_counter, 2] == self.RANDOM_CONDITION_ID):
                #    self.feedbackX, self.feedbackY = self.block_stack[self.counter_random_block][0:2]
                #    self.counter_random_block = self.counter_random_block+1
                #    if ((self.trial_counter+1) != len(self.trial_infos)):
                #        if (self.trial_infos[self.trial_counter+1, 2] == self.STABLE_CONDITION_ID):
                #            self.block_stack = []

                #else:
                #    raise ValueError('Unknown condition')


                self.feedbackX, self.feedbackY = \
                        self.apply_visuomotor_pert((self.cursorX, self.cursorY), vft)

                # This variable saves the unperturbed, unprojected feedback
                self.unpert_feedbackX, self.unpert_feedbackY = self.cursorX, self.cursorY
                self.error_distance = np.sqrt((self.feedbackX - self.target_coords[self.tgti_to_show][0])**2 +
                                              (self.feedbackY - self.target_coords[self.tgti_to_show][1])**2)

                if self.point_in_circle(self.target_coords[self.tgti_to_show],
                                        (self.feedbackX, self.feedbackY),
                                        self.params['radius_target'] +
                                        self.params['radius_cursor']):
                    self.colorTarget = [255, 0, 0]
                    self.counter_hit_trials = self.counter_hit_trials+1
                self.current_phase = self.FEEDBACK_PHASE
                if self.trial_infos[self.trial_counter, 2] == 0:
                    self.send_trigger(self.current_phase)
                elif self.trial_infos[self.trial_counter, 2] == 1:
                    self.send_trigger(self.current_phase + 5)
                self.color_photodiode = [0, 0, 0]

        elif (self.current_phase == self.FEEDBACK_PHASE):
            self.counter_feedback = self.counter_feedback+1
            if (self.counter_feedback == \
                    self.params['FPS'] * self.params['time_feedback']):
                if ((self.trial_counter+1) != len(self.trial_infos)):
                    # if change between RANDOM and STABLE, give a break
                    if (((self.trial_infos[self.trial_counter+1, 2] == self.RANDOM_CONDITION_ID) &
                       (self.trial_infos[self.trial_counter, 2] == self.STABLE_CONDITION_ID)) |
                        ((self.trial_infos[self.trial_counter+1, 2] == self.STABLE_CONDITION_ID) &
                       (self.trial_infos[self.trial_counter, 2] == self.RANDOM_CONDITION_ID))):

                        self.current_phase = self.BREAK_PHASE
                        self.free_from_break = 0
                    else:
                        self.current_phase = self.ITI_PHASE
                else:
                    self.current_phase = self.ITI_PHASE



                self.ITI_jitter = self.params['time_ITI'] + self.params['jitter'] * np.random.random_sample()
                if self.trial_infos[self.trial_counter, 2] == 0:
                    self.send_trigger(self.current_phase)
                elif self.trial_infos[self.trial_counter, 2] == 1:
                    self.send_trigger(self.current_phase + 5)
                self.currentText = self.myfont.render('Hits: ' + str(self.counter_hit_trials) + '/' + str(self.trial_counter+1), True, (255,255,255))

        elif (self.current_phase == self.BREAK_PHASE):
            if (self.free_from_break):
                self.current_phase = self.ITI_PHASE

        elif (self.current_phase == self.PAUSE_1MIN_PHASE):
            self.counter_PAUSE = self.counter_PAUSE + 1
            if (self.counter_PAUSE == int(self.params['FPS'] * 60)):
                self.current_phase = self.REST_PHASE

                self.counter_feedback = 0
                self.counter_PAUSE = 0
                self.counter_inside = 0
                self.feedbackX = 0
                self.feedbackY = 0
                self.unpert_feedbackX = 0
                self.unpert_feedbackY = 0
                self.error_distance = 0
                self.colorTarget = [0, 255, 0]
                self.trial_counter = self.trial_counter + 1
                print ('Trial: ' + str(self.trial_counter))

        elif (self.current_phase == self.ITI_PHASE):
            self.counter_ITI = self.counter_ITI + 1
            if (self.counter_ITI == int(self.params['FPS'] * self.ITI_jitter)):
                self.current_phase = self.REST_PHASE
                #if self.trial_infos[self.trial_counter, 2] == 0:
                #    self.send_trigger(self.current_phase)
                #elif self.trial_infos[self.trial_counter, 2] == 1:
                #    self.send_trigger(self.current_phase + 5)
                self.counter_feedback = 0
                self.counter_ITI = 0
                self.counter_inside = 0
                self.feedbackX = 0
                self.feedbackY = 0
                self.unpert_feedbackX = 0
                self.unpert_feedbackY = 0
                self.error_distance = 0
                self.colorTarget = [0, 255, 0]
                self.trial_counter = self.trial_counter + 1
                print ('Trial: ' + str(self.trial_counter))
        else:
            print("Error")
            raise ValueError('wrong phase')
            self.on_cleanup(-1)


    def update_log(self):
        '''
        saves everything to the log
        called from on_execute
        '''
        self.current_log = []
        self.current_log.append(self.trial_counter)
        self.current_log.append(self.current_phase)
        #self.current_log.append(self.trial_infos[self.trial_counter, 0]) #target number
        self.current_log.append(self.tgti_to_show) #target number
        self.current_log.append(self.trial_infos[self.trial_counter, 1]) #perturbation
        self.current_log.append(self.cursorX)
        self.current_log.append(self.cursorY)
        self.current_log.append(self.feedbackX)
        self.current_log.append(self.feedbackY)
        self.current_log.append(self.unpert_feedbackX) # before was called 'org_feedback'
        self.current_log.append(self.unpert_feedbackY)
        self.current_log.append(self.error_distance)  # Euclidean distance
        self.current_log.append(self.trial_infos[self.trial_counter, 2]) #stable (0) or random(1) block
        self.current_log.append(self.current_time - self.initial_time)
        self.logfile.write(",".join(str(x) for x in self.current_log) + '\n')


    def on_cleanup(self, exit_type):
        '''
        at the end of on_execute
        '''
        self.send_trigger(0)
        if (exit_type == 0):
            print("Yay")
        elif (exit_type == -1):
            print("Ouch")
        self.logfile.close()
        pygame.quit()
        exit()


    def on_execute(self):
        if self.on_init() is False:
            self._running = False
        clock = pygame.time.Clock()
        self.initial_time = time.time()
        #if (self.params['use_eye_tracker']):
        #    EyeLink.tracker(self.params['width'], self.params['height'])
        self.initial_time = time.time()
        while(self._running):
            self.current_time = time.time()
            for event in pygame.event.get():
                self.on_event(event)
            if (self.start_task):
                self.vars_update()
                if (self.trigger_countdown > 0):
                    self.trigger_countdown = self.trigger_countdown-1
                elif (self.trigger_countdown == 0):
                    self.trigger_countdown = -1
                    self.send_trigger(0)
                if self.trial_counter == len(self.trial_infos):
                    print("Success ! Experiment Done ")
                    break
                self.update_log()

            #print(clock)


            self.on_render()
            # print(time.time()-self.current_time)
            msElapsed = clock.tick_busy_loop(self.params['FPS'])

        self.on_cleanup(0)


if __name__ == "__main__":

    #app = VisuoMotor()

    info = {}

    #from psychopy import gui, core
    #info['participant'] = ''
    #info['session'] = ''
    #dlg = gui.DlgFromDict(info)
    #if not dlg.OK:
    #    core.quit()
    info['participant'] = 'dima'
    info['session'] = 'session1'


    app = VisuoMotor(info, use_triggers=0, use_joystick=0)
    app.on_execute()

    # it starts a loop in which it check for events using pygame.event.get()
    # then it executes on_event for all dected events
    # after than if we are running a task it runs vars_update

    # keys:   ESC -- exit, f -- fullscreen,  q,w -- change cursor radius
    # pressing joystick button can change experiment phase if start_task == 0


    #start_task is more like "task is running now"

    # Q: what is trigger countdown? And how trigger sending works in general?
    # Q: what is rest_phase + 5
    # more generally, what does if event.type == pygame.JOYBUTTONDOWN in
    #   on_event do?

    # photodiode color is set to white when change from REST to TARGET
    # to black when change from TARGET to FEEDBACK

    # top left = (0,0), top right (max, right)
    # so first is X (goes right), second is Y (goes down)

    # REST is both during ITI and before task. In it we plot orig feedback at
    # home
    # free_from_break asks to transition from BREAK to ITI

    # after start task we go to REST

    # ITI phase after time_feedback
    # ITI then REST

    # during ITI show nothing, during REST show orig, wait a bit and then
    # switch to TARGET_PHASE

    # show feedback only in FEEDBACK_PHASE

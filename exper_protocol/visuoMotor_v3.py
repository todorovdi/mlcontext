from __future__ import print_function
import pygame
from psychopy import gui, core
from pygame.locals import *
import time
import random
import math
import logging
import os
from win32api import GetSystemMetrics
from ctypes import windll
from random import shuffle
import numpy as np
import math
import EyeLink
# Create dictionnary info for gui
info = {}
info['participant'] = ''
info['session'] = ''
dlg = gui.DlgFromDict(info)
if not dlg.OK:
    core.quit()




class VisuoMotor:

    self.debug = True

    def add_param(self, name, value):
        self.params.update({name: value})
        self.paramfile.write(name + ' = ' + str(value) + '\n')

    def add_param_comment(self, comment):
        self.paramfile.write(comment + '\n')

    def initialize_parameters(self):
        self.params = {}
        self.task_id = 'visuomotor'
        self.subject_id = info['participant']
        self.session_id = info['session']
        self.timestr = time.strftime("%Y%m%d_%H%M%S")
        self.filename = ('data/' + self.subject_id + '_' + self.task_id +
                         '_' + self.timestr)
        self.paramfile = open(self.filename + '.param', 'w')
        self.logfile = open(self.filename + '.log', 'w')
        self.add_param_comment('# Width of screen')
        self.add_param('width', 600)
        self.add_param_comment('# Height of screen')
        self.add_param('height', 600)
        self.add_param_comment('# Frames per second for plotting')
        self.add_param('FPS', 120)
        self.add_param_comment('# Radius of the cursor')
        self.add_param('radius_cursor', 8)
        self.add_param_comment('# Radius of the target')
        self.add_param('radius_target', 12)
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
        self.add_param('use_eye_tracker', 0)
        self.add_param_comment('# Use triggers?')
        self.add_param('use_triggers', 1)
        self.paramfile.close()

    def __init__(self, subject_name='', task_id='', session_id=''):

        self.initialize_parameters()
        self.use_triggers = self.params['use_triggers']
        if (self.use_triggers):
            print("Using triggers")
        else:
            print("NOT using triggers")
        self.size = self.params['width'], self.params['height']
        self.trigger_port = 0x378
        self.trigger_countdown = -1
        self.counter_trials = 0
        self.num_trials_task = 80
        self.counter_inside = 0
        self.counter_feedback = 0
        self.counter_ITI = 0
        self.counter_hit_trials = 0
        self.REST_PHASE = 10
        self.TARGET_PHASE = 20
        self.FEEDBACK_PHASE = 30
        self.ITI_PHASE = 40
        self.init_target_positions()

        pygame.init()
        self.myfont = pygame.font.SysFont('Calibri', 24)
        self.string_instructions = 'We will start soon. Please wait for \
                                    instructions !'
        self.length_text = self.myfont.size(self.string_instructions)
        self.currentText = self.myfont.render('Hits: 0/0', True,
                                              (255, 255, 255))
        self.instructions_text = self.myfont.render(self.string_instructions,
                                                    True, (255, 255, 255))
        self.start_task = 0
        self.color_photodiode = [0, 0, 0]
        self.home_position = (int(round(self.params['width']/2.0)),
                              int(round(self.params['height']/2.0)))
        self.my_joystick = pygame.joystick.Joystick(0)
        self.my_joystick.init()
        self.joyX = 0
        self.joyY = 0
        self.feedbackX = 0
        self.feedbackY = 0
        self.org_feedbackX = 0
        self.org_feedbackY = 0
        self.error_distance = 0
        self.colorTarget = [0, 255, 0]
        self.trialsArray = np.empty([1, 2])  # Initialize trialsArray

        # choose ranomdly from two sts of trials and perturb
        # note that trials_vector does not change
        a = np.random.sample()
        if a >= 0.5:
            trials_vector = [5, 15, 5, 15, 5, 15, 5, 15]
            perturb_vector = [0, 45, 0, -45, 0, 45, 0, -45]
        else:
            trials_vector = [5, 15, 5, 15, 5, 15, 5, 15]
            perturb_vector = [0, -45, 0, 45, 0, -45, 0, 45]

        for nbTrial, perturbation in zip(trials_vector, perturb_vector):
            x = np.zeros(nbTrial)
            x1 = np.concatenate((x, x + 1, x + 2, x + 3))
            x1 = np.random.permutation(x1)
            x2 = np.ones(nbTrial*4) * perturbation
            x1x2 = np.column_stack((x1, x2))
            self.trialsArray = np.vstack((self.trialsArray, x1x2))
        self.trialsArray = np.delete(self.trialsArray, (0), axis=0)

        self.target_to_show = 0
        if (self.use_triggers):
            self.trigger = windll.inpout32
            self.send_trigger(0)

    def init_target_positions(self):
        targetAngs = [22.5+180, 67.5+180, 112.5+180, 157.5+180]
        self.target_types = []
        for x in range(0, len(targetAngs)):
            current = targetAngs[x]*(np.pi/180)
            self.target_types.append((int(round(self.params['width']/2.0 +
                                          np.cos(current) * self.params['radius'])),
                                      int(round(self.params['height']/2.0 +
                                          np.sin(current) * self.params['radius']))))

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
        if event.type == pygame.JOYBUTTONDOWN:
            if self.start_task == 0:
                self.current_phase = self.REST_PHASE
                self.send_trigger(self.current_phase)
                self.start_task = 1
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
            if event.key == pygame.K_c:
                if (self.params['use_eye_tracker']):
                    EyeLink.tracker(self.params['width'], self.params['height'])

    def on_render(self):
        if (self.start_task):
            # Remove components screen
            self._display_surf.fill([100, 100, 100])

            if self.current_phase == self.REST_PHASE:
                pygame.draw.circle(self._display_surf, [255, 255, 255],
                                   (self.joyX, self.joyY),
                                   self.params['radius_cursor'], 0)
                pygame.draw.circle(self._display_surf, [0, 0, 0],
                                   self.home_position,
                                   int(self.params['radius_cursor']*2.0), 2)

            if (self.current_phase == self.TARGET_PHASE) or (self.current_phase == self.FEEDBACK_PHASE):
                pygame.draw.circle(self._display_surf, self.colorTarget,
                                   self.target_types[self.target_to_show],
                                   self.params['radius_target'], 0)
                pygame.draw.circle(self._display_surf, [0, 0, 0],
                                   self.home_position,
                                   int(self.params['radius_cursor']*2.0), 2)

            if (self.current_phase == self.FEEDBACK_PHASE):
                pygame.draw.circle(self._display_surf, [255, 255, 255],
                                   (self.feedbackX, self.feedbackY),
                                   self.params['radius_cursor'], 0)

            pygame.draw.rect(self._display_surf, self.color_photodiode,
                             (0, 0, self.params['width'], 30), 0)
            if (self.params['show_text']):
                self._display_surf.blit(self.currentText, (0, 0))

        else:
            self._display_surf.fill([100, 100, 100])
            self._display_surf.blit(self.instructions_text,
                                    (int(round(((self.params['width'] - self.length_text[0]) / 2.0))),
                                     int(round(((self.params['height'] - self.length_text[1]) / 2.0)))))

        pygame.display.update()

    def radius_reached(self):
        centeredX = self.joyX - self.home_position[0]
        centeredY = self.joyY - self.home_position[1]
        return (centeredX**2+centeredY**2)**(1/2.) >= self.params['radius']

    def project_coordinates(self, coordinates, perturbation):
        perturbAngle = perturbation*(np.pi/180)
        my_coords = [-1, -1]

        my_coords[0] = coordinates[0] - self.params['width']/2.0
        my_coords[1] = coordinates[1] - self.params['height']/2.0

        cursorReachX = np.cos(np.arctan2(my_coords[1], my_coords[0]) +
                              perturbAngle) * self.params['radius']
        cursorReachY = np.sin(np.arctan2(my_coords[1], my_coords[0]) +
                              perturbAngle) * self.params['radius']

        cursorReachX = cursorReachX + self.params['width']/2.0
        cursorReachY = cursorReachY + self.params['height']/2.0

        cursorReachX = int(round(cursorReachX))
        cursorReachY = int(round(cursorReachY))
        return cursorReachX, cursorReachY

    def point_in_circle(self, point, circle_center, circle_radius):
        d = math.sqrt(math.pow(point[0]-circle_center[0], 2) +
                      math.pow(point[1]-circle_center[1], 2))
        return d < circle_radius

    def on_loop(self):
        self.joyX = self.my_joystick.get_axis(0)
        self.joyY = self.my_joystick.get_axis(1)
        self.joyX = int(round(((self.joyX - -1) / (1 - -1)) *
                              (self.params['width'] - 0) + 0))
        self.joyY = int(round(((self.joyY - -1) / (1 - -1)) *
                              (self.params['height'] - 0) + 0))
        if (self.current_phase == self.REST_PHASE):
            if self.point_in_circle(self.home_position, (self.joyX, self.joyY),
                                    self.params['radius_cursor']):
                self.counter_inside = self.counter_inside+1
            else:
                self.counter_inside = 0
            if (self.counter_inside == self.params['FPS']*self.params['time_inside']):
                self.current_phase = self.TARGET_PHASE
                self.target_to_show = int(self.trialsArray[self.counter_trials, 0])
                self.send_trigger(self.current_phase + self.target_to_show)
                self.color_photodiode = [255, 255, 255]

        elif (self.current_phase == self.TARGET_PHASE):
            if (self.radius_reached()):
                self.feedbackX, self.feedbackY = self.project_coordinates((self.joyX, self.joyY), self.trialsArray[self.counter_trials, 1])
                # This variable saves the unperturbed, unprojected feedback
                self.org_feedbackX, self.org_feedbackY = self.joyX, self.joyY

                self.error_distance = np.sqrt((self.feedbackX - self.target_types[self.target_to_show][0])**2 +
                                              (self.feedbackY - self.target_types[self.target_to_show][1])**2)

                if self.point_in_circle(self.target_types[self.target_to_show],
                                        (self.feedbackX, self.feedbackY),
                                        self.params['radius_target'] +
                                        self.params['radius_cursor']):
                    self.colorTarget = [255, 0, 0]
                    self.counter_hit_trials = self.counter_hit_trials+1
                self.current_phase = self.FEEDBACK_PHASE
                self.send_trigger(self.current_phase)
                self.color_photodiode = [0, 0, 0]

        elif (self.current_phase == self.FEEDBACK_PHASE):
            self.counter_feedback = self.counter_feedback+1
            if (self.counter_feedback == self.params['FPS'] * self.params['time_feedback']):
                self.current_phase = self.ITI_PHASE
                self.ITI_jitter = self.params['time_ITI'] + self.params['jitter'] * np.random.random_sample()
                self.send_trigger(self.current_phase)
                self.currentText = self.myfont.render('Hits: ' + str(self.counter_hit_trials) + '/' + str(self.counter_trials+1), True, (255,255,255))

        elif (self.current_phase == self.ITI_PHASE):
            self.counter_ITI = self.counter_ITI+1
            if (self.counter_ITI == self.params['FPS'] * self.ITI_jitter):
                self.current_phase = self.REST_PHASE
                self.send_trigger(self.current_phase)
                self.counter_feedback = 0
                self.counter_ITI = 0
                self.counter_inside = 0
                self.feedbackX = 0
                self.feedbackY = 0
                self.org_feedbackX = 0
                self.org_feedbackY = 0
                self.error_distance = 0
                self.colorTarget = [0, 255, 0]
                self.counter_trials = self.counter_trials + 1

        else:
            print("Error")
            self.on_cleanup(-1)

    def update_log(self):
        self.current_log = []
        self.current_log.append(self.counter_trials)
        self.current_log.append(self.current_phase)
        self.current_log.append(self.trialsArray[self.counter_trials, 0]) #target number
        self.current_log.append(self.trialsArray[self.counter_trials, 1]) #perturbation
        self.current_log.append(self.joyX)
        self.current_log.append(self.joyY)
        self.current_log.append(self.feedbackX)
        self.current_log.append(self.feedbackY)
        self.current_log.append(self.org_feedbackX)
        self.current_log.append(self.org_feedbackY)
        self.current_log.append(self.error_distance)
        self.current_log.append(self.current_time - self.initial_time)
        self.logfile.write(",".join(str(x) for x in self.current_log) + '\n')

    def on_cleanup(self, exit_type):
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
        if (self.params['use_eye_tracker']):
            EyeLink.tracker(self.params['width'], self.params['height'])
        self.initial_time = time.time()
        while(self._running):
            self.current_time = time.time()
            for event in pygame.event.get():
                self.on_event(event)
            if (self.start_task):
                self.on_loop()
                if (self.trigger_countdown > 0):
                    self.trigger_countdown = self.trigger_countdown-1
                elif (self.trigger_countdown == 0):
                    self.trigger_countdown = -1
                    self.send_trigger(0)
                if self.counter_trials == len(self.trialsArray):
                    print("Success ! Experiment Done ")
                    break
                self.update_log()

            self.on_render()
            # print(time.time()-self.current_time)
            msElapsed = clock.tick_busy_loop(self.params['FPS'])

        self.on_cleanup(0)

if __name__ == "__main__":
    app = VisuoMotor()
    app.on_execute()

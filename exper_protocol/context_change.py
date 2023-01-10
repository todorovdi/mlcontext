from __future__ import print_function
# needed for joystick
import pygame
# from pygame.locals import *
import time
# import logging
# from win32api import GetSystemMetrics
import sys
if sys.platform.startswith('win32'):
    from ctypes import windll
import numpy as np
import math
# import pylink
# import EyeLink


class VisuoMotor:

    def add_param(self, name, value):
        self.params.update({name: value})
        self.paramfile.write(name + ' = ' + str(value) + '\n')

    def add_param_comment(self, comment):
        self.paramfile.write(comment + '\n')

    def initialize_parameters(self, info):
        # self.debug = False
        self.params = {}
        self.task_id = 'visuomotor'
        self.subject_id = info['participant']
        self.session_id = info['session']
        self.timestr = time.strftime("%Y%m%d_%H%M%S")
        self.filename = ('data/' + self.subject_id + '_' + self.task_id +
                         '_' + self.timestr)
        self.paramfile = open(self.filename + '.param', 'w')
        self.logfile = open(self.filename + '.log', 'w')
        # for debug mostly
        self.trigger_logfile = open(self.filename + '_trigger.log', 'w')

        self.add_param_comment('# Width of screen')      # updates self.param dictionary
        self.add_param('width', 800)
        self.add_param_comment('# Height of screen')
        self.add_param('height', 800)
        self.add_param_comment('# Frames per second for plotting')
        self.add_param('FPS', 120)
        self.add_param_comment('# Radius of the cursor')
        self.add_param('radius_cursor', 10)

        self.add_param('radius_home', self.params['radius_cursor'] * 2)

        self.add_param_comment('# Radius of the target')
        self.add_param('radius_target', 14)
        # distance from the start location to the target center
        self.add_param_comment('# Radius of the invisible boundary')
        self.add_param('radius', int(round(self.params['height']*0.5*0.8)))
        self.add_param_comment('# Time inside the home position before trial \
                        starts (seconds)')
        self.add_param('time_at_home', 0.5)
        self.add_param_comment('# Time for feedback (seconds)')
        #self.add_param('time_feedback', 0.25)
        self.add_param_comment('# if online feedback, duration of reach  (seconds)')
        if self.debug:
            self.add_param('time_feedback', 2.)
        else:
            self.add_param('time_feedback', 0.85)

        if self.debug:
            self.add_param('return_duration', 60)
        else:
            self.add_param('return_duration', 10)
        self.add_param_comment('# Time for intertrial interval (seconds)')
        self.add_param('ITI_duration', 1.5)
        self.add_param_comment('# Max jitter during ITI (seconds)')
        self.add_param('ITI_jitter', 0.1)
        self.add_param_comment('# Show text?')
        self.add_param('show_text', 0)
        self.add_param_comment('# Use eye tracker?')
        self.add_param('use_eye_tracker', 1)
        self.add_param_comment('# Use triggers?')
        self.add_param('use_triggers', 1)
        self.add_param_comment('# How long should we stay at one point before the movement is judged finished?')
        self.add_param('stopping_min_dur_s', 0.2)

        self.add_param('pause_duration', 60)
        if self.debug:
            self.add_param('pause_duration', 10)


        # in milliseconds
        self.add_param('trigger_duration', 50)

        # whether feedback is shown always on circle with fixed radius or
        # normally
        self.add_param('feedback_fixed_distance', 0)

        #self.pause_duration = 60  # in sec
        #if self.debug:
        #    self.pause_duration = 10  # in sec


        self.paramfile.close()


    def __init__(self, info, task_id='',
                 use_triggers = 1, use_joystick=1, feedback_type='online', debug=False):
        self.debug = debug # affects fullscreen or not and other things
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
        # inc in the end of 'PAUSE' and 'ITI'
        self.trial_index = 0  # index of current trials
        self.counter_hit_trials = 0
        #
        #self.counter_inside = 0 # it is actually counter_rest or counter_home
        #self.counter_feedback = 0
        #self.counter_ITI = 0
        #self.counter_PAUSE = 0
        #self.counter_RETURN = 0

        # currently not used
        self.frame_counters = {'at_home':0, 'feedback_shown':0, 'pause':0,
                               'return':0, 'ITI':0}

        #self.reach_end_event = 'target_reached'
        self.reach_end_event = 'distance_reached'
        self.reach_end_event = 'stopped'


        self.feedback_type = feedback_type
        self.last_reach_too_slow = 0
        self.last_reach_not_full_rad = 0
        #if self.feedback_type == 'offline':
        #self.REST_PHASE = 10
        #self.TARGET_PHASE = 20
        #self.FEEDBACK_PHASE = 30
        #elif self.feedback_type == 'online':
        #self.TARGET_AND_FEEDBACK_PHASE = 31
        #self.ITI_PHASE = 40
        #self.BREAK_PHASE = 50
        #self.PAUSE_PHASE = 60
        #self.ERROR_CLAMP_PHASE = 70  # NO, THIS IS NOT A PHASE! It is feedback type
        #self.STABLE_CONDITION_ID = 0
        #self.RANDOM_CONDITION_ID = 1
        self.current_phase_trigger = None
        self.current_phase = None
        self.CONTEXT_TRIGGER_DICT = {}

        self.first_phase_after_start = 'REST'
        # ITI, then RETURN then REST
        self.rest_after_return = True

        self.phase2trigger = {'REST':10, 'RETURN':15,
                              'TARGET':20, 'TARGET_AND_FEEDBACK':31,
                           'ITI':40, 'BREAK':50, 'PAUSE':60 }
        self.trigger2phase = dict((v, k) for k, v in self.phase2trigger.items())
        self.phase_start_times = dict( zip(self.phase2trigger.keys(), len(self.phase2trigger) * [-1.] ) )
        self.phase_start_times['at_home'] = 0.

        self.use_joystick = use_joystick
        if use_joystick:
            self.phase_shift_event_type = pygame.JOYBUTTONDOWN
        else:
            self.phase_shift_event_type = pygame.MOUSEBUTTONDOWN
        self.init_target_positions()
        pygame.init()
        pygame.mouse.set_visible(0)
        self.foruser_font_size = 24
        self.myfont = pygame.font.SysFont('Calibri', self.foruser_font_size)
        self.myfont_debug = pygame.font.SysFont('Calibri', 16)
        self.myfont_popup = pygame.font.SysFont('Calibri', 30)
        #self.string_instructions = 'We will start soon. Please wait for instructions'
        self.string_instructions = 'Click to start'
        self.length_text = self.myfont.size(self.string_instructions)


        #self.cursor_shown =

        self.break_start_str = 'BREAK'
        self.color_hit = [255, 0, 0]   # red
        self.color_feedback = [255, 255, 255]

        #self.currentText = self.myfont.render('Hits: 0/0', True,
        #                                      (255, 255, 255))
        self.instructions_text = self.myfont.render(self.string_instructions,
                                                    True, (255, 255, 255))


        self.break_text = self.myfont.render(self.break_start_str,
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
        self.colorTarget = [0, 255, 0]  # green

        self.block_stack = []
        #self.counter_random_block = 0

        self.free_from_break = 0
        # trajectory starting from last target phase
        self.trajX = []  # true traj
        self.trajY = []
        self.trajfbX = []
        self.trajfbY = []

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

        if self.debug:
            block_len_min = 3
            block_len_max = 6

        target_inds = np.arange(len(self.target_coords) )
        from itertools import product,repeat
        vfti_seq0 = list( product( self.vis_feedback_types, target_inds ) )
        n_contexts = len(vfti_seq0)
        vfti_seq_noperm = list(vfti_seq0) * n_context_appearences
        # TODO: manage seed here, make it participant or date depenent explicitly
        self.seed = 1
        np.random.seed(self.seed)
        ct_inds = np.random.permutation(np.arange(len(vfti_seq_noperm) ) )
        print('ct_inds',ct_inds)
        self.vfti_seq = [vfti_seq_noperm[i] for i in ct_inds] # I don't want to convert to numpy here
        n_blocks = len(self.vfti_seq)
        ns_context_repeat = np.random.randint( block_len_min, block_len_max,
                                              size=n_blocks )

        #n_blocks = n_contexts * n_context_appearences
        #seq0 = np.tile( np.arange(n_contexts), n_context_appearences)
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

        self.trial_infos = [] # this is the sequence of trial types
        d = {'vis_feedback_type':'veridical', 'tgti':0,
                'trial_type': 'veridical',
             'special_block_type': None }
        num_initial_veridical = 7
        if self.debug:
            num_initial_veridical = 0
        self.trial_infos += [d] * num_initial_veridical

        # I want list of tuples -- target id, visual feedback type, phase
        # (needed for trigger)
        trigger = self.phase2trigger['PAUSE'] + 30



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


            if self.debug:
                pause_trial_middle, error_clamp_pair_middle,\
                        error_clamp_sandwich_middle = 0,1,0

            #if bi > 2 or self.debug:
            #    if bi % 4 == 0:
            #        pause_trial_middle, error_clamp_pair_middle,\
            #                error_clamp_sandwich_middle = 1,0,0
            #    elif bi % 4 == 2:
            #        pause_trial_end, error_clamp_pair_end,\
            #                error_clamp_sandwich_end = 1,0,0
            #    elif bi % 4 == 3:
            #        pause_trial_end, error_clamp_pair_end,\
            #                error_clamp_sandwich_end = 0,1,0

            hd = num_context_repeats // 2
            rhd = num_context_repeats - hd
            self.trial_infos += [d] * hd

            # insert in the middle
            self.trial_infos += genSpecTrialBlock(pause_trial_middle,
                error_clamp_pair_middle, error_clamp_sandwich_middle,d)

            self.trial_infos += [d] * rhd

            self.trial_infos += genSpecTrialBlock(pause_trial_end,
                error_clamp_pair_end, error_clamp_sandwich_end,d)


            tpl = (vis_feedback_type, tgti)
            if tpl not in self.CONTEXT_TRIGGER_DICT:
                self.CONTEXT_TRIGGER_DICT[ tpl ] = trigger
                trigger += 1
            #if bi % 4 == 0:
            # INSERT clamp or pause


        if self.debug:
            for tc in range(10):
                print(tc, self.trial_infos[tc]['trial_type'] )

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
        #print(self.trial_infos)
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
        # We block it if it has sent something already
        if (self.trigger_countdown == -1):
            if (self.use_triggers):
                self.trigger.Out32(self.trigger_port, value)
            else:
                self.trigger_logfile.write(str(value) + '\n' )
        # print("Sent trigger " + str(value))
        # For how long the trigger is gonna be raised? (in ms)
        if (value != 0):
            td = self.params['trigger_duration']
            self.trigger_countdown = int(round(self.params['FPS']*(td/1000.0)))


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
                  self.trial_infos[self.trial_index] )

            # if task was not started and a button was pressed, start the task
            if self.start_task == 0:
                self.current_phase = self.first_phase_after_start
                # TODO: send env info?
                #if self.trial_infos[self.trial_index, 2] == 0:
                #    self.send_trigger(self.current_phase_trigger)
                #elif self.trial_infos[self.trial_index, 2] == 1:
                #    self.send_trigger(self.current_phase_trigger+5)
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

    def drawTextMultiline(self, lines, pos_label = 'lower_left'):
        if pos_label != 'lower_left':
            raise ValueError(f'not implemented {pos_label}')
        voffset = 0
        for line in lines[::-1]:
            text_render = self.myfont_debug.render(line, True, (255, 255, 255))
            ldt = self.myfont_debug.size(line)
            self._display_surf.blit(text_render, (5, self.params['height']-ldt[1] - voffset))
            voffset += ldt[1]

    def drawPopupText(self, text, pos = 'center', font_size = None,
                      length_info = None, text_render = None):
        #, font_size=30
        if font_size is not None:
            font = pygame.font.SysFont('Calibri', font_size)
        else:
            font = self.myfont_popup

        if text_render is None:
            text_render = font.render(text,
                    True, (255, 255, 255))

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
        thickness = 3

        # if the last was scale then we wante to scale return radius as well
        # we need to subtract 1 becasue RETURN goes after ITI, where it was
        # increased
        trial_info = self.trial_infos[self.trial_index - 1]
        ctxtpl = (trial_info['vis_feedback_type'], trial_info['tgti'] )
        vft, tgti = ctxtpl
        if vft.startswith('scale'):
            scale = self.scale_params[vft]
            dist *= scale
        pygame.draw.circle(self._display_surf, self.colorTarget,
                           self.home_position,
                           dist, thickness)

    def drawHome(self):
        pygame.draw.circle(self._display_surf, [0, 0, 0],
                           self.home_position,
                           int(self.params['radius_home']), 2)

    def drawCursorFeedback(self):
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
        pygame.draw.circle(self._display_surf, self.color_feedback,
                        (self.feedbackX, self.feedbackY),
                        self.params['radius_cursor'], 0)

    def drawCursorOrig(self, debug=0):
        r = self.params['radius_cursor']
        c = [255, 255, 255]  # white
        if debug:
            c = [200, 100, 100]  # reddish
            r /= 2
        pygame.draw.circle(self._display_surf, c,
                           (self.cursorX, self.cursorY),
                           r, 0)


    def drawTraj(self, pert=0):
        #print('drawTraj beg')
        if (not pert) and (len(self.trajX) < 2):
            return
        if pert and (len(self.trajX) < 2):
            return

        if pert:
            thickness = 6
            tpls = list(zip(self.trajfbX, self.trajfbY ))
            c = [200, 200, 200]  # whitish for feedback
        else:
            thickness = 2
            tpls = list(zip(self.trajX, self.trajY ))
            c = [100, 50, 60]  # redish for true movement
        #    trial_info = self.trial_infos[self.trial_index]
        #    if self.current_phase in ['REST', 'RETURN']:
        #        trial_info = self.trial_infos[self.trial_index - 1]
        #    ctxtpl = (trial_info['vis_feedback_type'], trial_info['tgti'] )
        #    vft, tgti = ctxtpl
        #    tpls2 = []
        #    for (x,y) in tpls:
        #        x2,y2 = self.apply_visuomotor_pert((x,y), vft)
        #        tpls2 += [ (x2,y2) ]

        #    tpls = tpls2
        #for i,(x,y) in enumerate(zip(self.trajX, self.trajY ) ):
        #    if pert:
        #        vft, tgti = ctxtpl
        #        self.apply_visuomotor_pert(self, (x,y), vft)
        #    pygame.draw.circle(self._display_surf, c,
        #        (x,y), self.params['radius_cursor'] / 2.5 , 0 )

        pygame.draw.lines(self._display_surf, c, False,
                          tpls, thickness )


    #def drawTextCenter(self, text, length_info):
    #    self.drawPopupText(text,

    #    self._display_surf.blit(text,
    #        (int(round(((self.params['width'] - length_info[0]) / 2.0))),
    #         int(round(((self.params['height'] - length_info[1]) / 2.0)))))

    def on_render(self):
        '''
        called from on_execute
        '''
        if (self.start_task):
            # Remove components screen
            self._display_surf.fill([100, 100, 100])
            #self.
            ldt = self.drawPopupText(
                f'Trial N={self.trial_index}/{len(self.trial_infos)}',
                               pos='upper_right', font_size = 30)
            self.drawPopupText( f'Nhits = {self.counter_hit_trials}',
                               pos=('upper_right',(0,ldt[1]) ),
                               font_size = 30)

            if self.current_phase == 'ITI':
                if self.last_reach_too_slow:
                    s = 'Reach was too slow'
                    self.drawPopupText(s)
                if self.last_reach_not_full_rad:
                    s = 'Reach did not stop at target distance in required time'
                    self.drawPopupText(s)

            if self.current_phase == 'REST':
                # if home, draw cursor
                if self.point_in_circle(self.home_position,
                                        (self.cursorX, self.cursorY),
                                        self.params['radius_cursor'] * 5):
                    self.drawCursorOrig()
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
                self.drawPopupText(pause_str)

            if self.feedback_type == 'offline':
                if self.current_phase == 'TARGET':
                    self.drawTgt()
                    self.drawHome()

                if (self.current_phase == 'FEEDBACK'):
                    self.drawTgt()
                    self.drawHome()
                    self.drawCursorFeedback()
            elif self.feedback_type == 'online':
                if self.current_phase == 'TARGET_AND_FEEDBACK':
                    self.drawTgt()
                    self.drawHome()
                    self.drawCursorFeedback()

            if self.current_phase == 'RETURN':
                self.drawReturnCircle()

            show_diode = 1
            diode_width = 20
            diode_height = 20
            if show_diode:
                pygame.draw.rect(self._display_surf, self.color_photodiode,
                                (0, 0, diode_width, diode_height), 0)

            # maybe show info on participant hitting performance
            #if (self.params['show_text']):
            #    self._display_surf.blit(self.currentText, (0, 0))

            if (self.current_phase == 'BREAK'):
                #self.drawTextCenter(self.break_text, self.length_text)
                self.drawPopupText(self.break_start_str, font_size = self.foruser_font_size)
        # if not start_task
        else:
            self._display_surf.fill([100, 100, 100])
            #self.drawTextCenter(self.instructions_text, self.length_text)
            self.drawPopupText(self.string_instructions,
                               font_size = self.foruser_font_size)
            #self._display_surf.blit(self.instructions_text,
            #(int(round(((self.params['width'] - self.length_text[0]) / 2.0))),
            # int(round(((self.params['height'] - self.length_text[1]) / 2.0)))))


        if self.debug:
            trial_info = self.trial_infos[self.trial_index]
            if self.current_phase is None:
                phase_str = 'None'
            else:
                phase_str = self.current_phase

            m = min(self.trial_index + 40, len(self.trial_infos) )
            next_spec_trial_ind = None
            next_spec_trial_type = None
            for tc in range(self.trial_index, m ):
                ti = self.trial_infos[tc]
                if ti['trial_type'] not in ['veridical', 'perturbation']:
                    next_spec_trial_ind = tc
                    next_spec_trial_type = ti['trial_type']
            trd = next_spec_trial_ind - self.trial_index

            debugstrs = []
            debugstr = (f'Phase={phase_str}; X,Y={self.feedbackX},{self.feedbackY} (Xt,Yt={self.cursorX},{self.cursorY}), '
                f'{list(trial_info.values())}')

            tdif = time.time() - self.phase_start_times["at_home"]
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
            for tc in range(self.trial_index + 1, m ):
                ti = self.trial_infos[tc]
                debugstr = f'trial_infos[{tc}]={ti}'
                debugstrs += [debugstr]
            #
            self.drawTextMultiline(debugstrs)


            self.drawCursorOrig(debug=1) # with diff color and size
            self.drawTraj()
            self.drawTraj(pert=1)

        pygame.display.update()

    def test_reach_finished(self, ret_ext = False):
        at_home = self.point_in_circle(self.home_position, (self.cursorX, self.cursorY),
                                self.params['radius_cursor'], verbose=0)
        stopped = self.test_stopped()
        radius_reached = self.test_radius_reached()
        r = at_home,stopped,radius_reached
        if self.reach_end_event == 'stopped':
            b = stopped and (not at_home)
        elif self.reach_end_event == 'distance_reached':
            b = radius_reached
        if self.debug and b:
            print('test_reach_finished: ', self.reach_end_event)
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
    #    return (centeredX**2+centeredY**2)**(1/2.) >= self.params['radius']

    def test_stopped(self, stop_rad_px = 3):
        '''
        called from vars_update
        '''
        ntimebins = int( self.params['FPS'] * self.params['stopping_min_dur_s'] )
        if (len(self.trajX) < ntimebins) or (self.get_dist_from_home() < self.params['radius_home'] ):
            return False
        #else:
        #    print('dist_from_home=' , self.get_dist_from_home() )

        #stdX = np.std( self.trajfbX[-ntimebins:] )
        #stdY = np.std( self.trajfbY[-ntimebins:] )
        # if I use trajfb then for error clamp it works bad
        stdX = np.std( self.trajX[-ntimebins:] )
        stdY = np.std( self.trajY[-ntimebins:] )

        return max(stdX , stdY) <= stop_rad_px

    def get_dist_from_home(self):
        centeredX = self.cursorX - self.home_position[0]
        centeredY = self.cursorY - self.home_position[1]
        return (centeredX**2+centeredY**2)**(1/2.)

    def test_radius_reached(self):
        '''
        called from vars_update
        says if the cursor have reached the tgt pos
        '''
        return self.get_dist_from_home() >= ( self.params['radius'] -\
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
        if perturbation.startswith('rot'):
            rotang_deg = int(perturbation[3:])
        elif perturbation.startswith('scale'):
            scale = self.scale_params[perturbation]
        elif perturbation.startswith('reverse'):
            sign_x = -1.

        rotang_rad = rotang_deg*(np.pi/180)
        #perturbAngle = perturbation*(np.pi/180)
        my_coords = [-1, -1]
        # subtract center
        # home position but float
        my_coords[0] = coordinates[0] - self.params['width']/2.0
        my_coords[1] = coordinates[1] - self.params['height']/2.0
        # rotate
        if self.params['feedback_fixed_distance']:
            mult = self.params['radius']
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

    def timer_check(self, phase, parname, thr = None, use_frame_counter = False):
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
        return r

    def vars_update(self):
        '''
        is called if the task is running
        it govers phase change
        '''
        prev_phase = self.current_phase
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

        trial_info = self.trial_infos[self.trial_index]
        ctxtpl = (trial_info['vis_feedback_type'], trial_info['tgti'] )
        vft, tgti = ctxtpl

        if (self.current_phase == 'REST'):
            # if at home, increase counter
            at_home = self.point_in_circle(self.home_position, (self.cursorX, self.cursorY),
                                    self.params['radius_cursor'], verbose=0)
            if at_home:
                self.frame_counters["at_home"] = self.frame_counters["at_home"]+1
            else:
                # if we leave center, reset to zero
                self.frame_counters["at_home"] = 0
                self.phase_start_times["at_home"] = time.time()

            # if we spent inside time at home than show target
            #at_home_enough = (self.frame_counters["at_home"] == self.params['FPS']*\
            #        self.params['time_at_home'])
            at_home_enough = self.timer_check("at_home", 'time_at_home')
            if at_home_enough:
                if trial_info['trial_type'] == 'pause':
                    self.current_phase = 'PAUSE'
                else:
                    if self.feedback_type == 'online':
                        self.current_phase = 'TARGET_AND_FEEDBACK'
                    else:
                        self.current_phase = 'TARGET'

                    self.trajX = []
                    self.trajY = []
                    self.trajfbX = []
                    self.trajfbY = []
                    #print(self.frame_counters["at_home"], self.params['FPS']*self.params['time_at_home'])
                    print(f'Start target and feedback display {trial_info}')

                    # self.trial_infos[self.trial_index, 0]
                    # depending on whether random or stable change target
                    self.tgti_to_show = tgti
                self.send_trigger(self.CONTEXT_TRIGGER_DICT[ctxtpl] )
                ## TODO: change
                #if (self.trial_infos[self.trial_index, 2] ==  \
                #        self.STABLE_CONDITION_ID):
                #    self.tgti_to_show = int(self.trial_infos[self.trial_index, 0])
                #elif (self.trial_infos[self.trial_index, 2] == \
                #        self.RANDOM_CONDITION_ID):
                #    self.tgti_to_show = self.block_stack[self.counter_random_block][2]

                ## depending on whether random or stable change target trigger
                ## TODO: change
                #if self.trial_infos[self.trial_index, 2] == 0:
                #    # target trigger
                #    self.send_trigger(self.current_phase + self.tgti_to_show)
                #elif self.trial_infos[self.trial_index, 2] == 1:
                #    self.send_trigger(self.current_phase + self.tgti_to_show + 5)
                self.color_photodiode = [255, 255, 255]

        elif (self.current_phase == 'TARGET_AND_FEEDBACK'):
            self.trajX += [ self.cursorX ]
            self.trajY += [ self.cursorY ]

            #reach_time_finished = (self.frame_counters["feedback_shown"] == \
            #        self.params['FPS'] * self.params['time_feedback'])

            reach_time_finished = self.timer_check(self.current_phase,
                                    'time_feedback')
            # if time is up we switch to ITI, else
            reach_finished, extinfo = self.test_reach_finished(ret_ext = 1)
            at_home,stopped,radius_reached = extinfo
            if reach_time_finished or reach_finished:
                if reach_time_finished:
                    self.last_reach_too_slow = 1
                    print('SLOOOOW')
                else:
                    self.last_reach_too_slow = 0

                print('at_home,stopped,radius_reached = ',at_home,stopped,radius_reached)
                if stopped and (not radius_reached):
                    self.last_reach_not_full_rad = 1
                else:
                    self.last_reach_not_full_rad = 0

                # for the coming ITI update its duration
                self.ITI_jittered = self.params['ITI_duration'] +\
                        self.params['ITI_jitter'] * np.random.random_sample()
                print(f'Trial {self.trial_index}: stopping condition met')

                if self.get_dist_from_home() < self.params['radius_target']:
                    print('Too long staying at the target')
                self.current_phase = 'ITI'
                self.color_photodiode = [0, 0, 0]

                if self.point_in_circle(self.target_coords[self.tgti_to_show],
                                        (self.feedbackX, self.feedbackY),
                                        self.params['radius_target'] +
                                        self.params['radius_cursor']):
                    self.counter_hit_trials = self.counter_hit_trials+1
            # else draw feedback and check hit
            else:
                self.frame_counters["feedback_shown"] += 1

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

                # check hit
                if self.point_in_circle(self.target_coords[self.tgti_to_show],
                                        (self.feedbackX, self.feedbackY),
                                        self.params['radius_target'] +
                                        self.params['radius_cursor']):
                    # hit color
                    self.colorTarget = self.color_hit

                # Q: why do we send trigger here?
                #self.send_trigger(self.current_phase)

            #self.currentText = self.myfont.render(
            #    'Hits: ' + str(self.counter_hit_trials) + '/' +\
            #    str(self.trial_index+1), True, (255,255,255))


        elif (self.current_phase == 'TARGET'):
            if self.feedback_type == 'online':
                raise ValueError('nooo!')
            self.trajX += [ self.cursorX ]
            self.trajY += [ self.cursorY ]
            print(f'len traj = {len(self.trajX) }')
            # if we have reached radius, set to FEEDBACK
            if (self.test_reach_finished()):
                #if (self.trial_infos[self.trial_index, 2] == self.STABLE_CONDITION_ID):
                #    self.feedbackX, self.feedbackY = self.apply_visuomotor_pert((self.cursorX, self.cursorY),
                #                self.trial_infos[self.trial_index, 1])
                #    self.block_stack.append([self.feedbackX, self.feedbackY, self.tgti_to_show])

                #    if ((self.trial_index+1) != len(self.trial_infos)):
                #        if (self.trial_infos[self.trial_index+1, 2] == self.RANDOM_CONDITION_ID):
                #            self.block_stack = np.array(self.block_stack)
                #            self.block_stack = np.random.permutation(self.block_stack)
                #            self.counter_random_block = 0

                #elif (self.trial_infos[self.trial_index, 2] == self.RANDOM_CONDITION_ID):
                #    self.feedbackX, self.feedbackY = self.block_stack[self.counter_random_block][0:2]
                #    self.counter_random_block = self.counter_random_block+1
                #    if ((self.trial_index+1) != len(self.trial_infos)):
                #        if (self.trial_infos[self.trial_index+1, 2] == self.STABLE_CONDITION_ID):
                #            self.block_stack = []

                #else:
                #    raise ValueError('Unknown condition')


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
                    self.colorTarget = self.color_hit
                    self.counter_hit_trials = self.counter_hit_trials+1
                self.current_phase = 'FEEDBACK'
                #self.current_phase_trigger = self.trigger2phase[self.current_phase]
                #self.send_trigger(self.current_phase_trigger)
                self.color_photodiode = [0, 0, 0]

        elif (self.current_phase == 'FEEDBACK'):
            if self.feedback_type == 'online':
                raise ValueError('nooo!')
            self.frame_counters["feedback_shown"] += 1
            if (self.frame_counters["feedback_shown"] == \
                    self.params['FPS'] * self.params['time_feedback']):
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
                #self.current_phase_trigger = self.trigger2phase[self.current_phase]
                #self.send_trigger(self.current_phase_trigger)
                #self.currentText = self.myfont.render('Hits: ' + \
                #        str(self.counter_hit_trials) + '/' + \
                #        str(self.trial_index+1), True, (255,255,255))


        elif (self.current_phase == 'BREAK'):
            if (self.free_from_break):
                self.current_phase = 'ITI'


        elif (self.current_phase == 'PAUSE'):
            self.frame_counters["pause"] = self.frame_counters["pause"] + 1
            #pause_finished = self.frame_counters["pause"] == int(self.params['FPS'] *\
            #                              self.params['pause_duration'] )
            pause_finished = self.timer_check(self.current_phase, "pause_duration")
            if pause_finished:
                self.current_phase = 'REST'

                for ctr in self.frame_counters:
                    self.frame_counters[ctr] = 0

                #self.frame_counters["feedback_shown"] = 0
                #self.frame_counters["pause"] = 0
                #self.frame_counters["at_home"] = 0
                self.feedbackX = 0
                self.feedbackY = 0
                self.unpert_feedbackX = 0
                self.unpert_feedbackY = 0
                self.error_distance = 0
                self.colorTarget = [0, 255, 0]  #
                self.trial_index += 1

                pygame.mouse.set_pos(self.home_position)
                #print ('Start Trial: ' + str(self.trial_index))

        elif (self.current_phase == 'ITI'):
            self.frame_counters["ITI"] += 1
            #ITI_finished =(self.frame_counters["ITI"] == \
            #        int(self.params['FPS'] * self.ITI_jittered))
            ITI_finished = self.timer_check(self.current_phase, parname=None,
                                            thr = self.ITI_jittered)
            if ITI_finished:
                if self.rest_after_return:
                    self.current_phase = 'RETURN'
                else:
                    self.current_phase = 'REST'
                #if self.trial_infos[self.trial_index, 2] == 0:
                #    self.send_trigger(self.current_phase_trigger)
                #elif self.trial_infos[self.trial_index, 2] == 1:
                #    self.send_trigger(self.current_phase_trigger + 5)

                for ctr in self.frame_counters:
                    self.frame_counters[ctr] = 0
                #self.counter_feedback = 0
                #self.counter_ITI = 0
                #self.frame_counters["pause"] = 0
                #self.frame_counters["at_home"] = 0
                self.feedbackX = 0
                self.feedbackY = 0
                self.unpert_feedbackX = 0
                self.unpert_feedbackY = 0
                self.error_distance = 0
                self.colorTarget = [0, 255, 0]  # green
                self.trial_index += 1
                print ('Start trial: ' + str(self.trial_index))

        elif (self.current_phase == 'RETURN'):
            at_home = self.point_in_circle(self.home_position, (self.cursorX, self.cursorY),
                                    self.params['radius_cursor'], verbose=0)
            #time_is_up = ( self.frame_counters["return"] == int(self.params['FPS'] *\
            #        self.return_max_duration) )
            time_is_up = self.timer_check(self.current_phase,'return_duration')

            if at_home or time_is_up:
                self.current_phase = 'REST'
                self.frame_counters["return"] = 0
                pygame.mouse.set_pos(self.home_position)
            else:
                self.frame_counters["return"] += 1
        else:
            print("Error")
            raise ValueError('wrong phase')
            self.on_cleanup(-1)

        self.current_phase_trigger = self.phase2trigger[self.current_phase]
        if self.current_phase != prev_phase:
            self.phase_start_times[self.current_phase] = time.time()
            self.send_trigger(self.current_phase_trigger)
            print(f'Phase change {prev_phase} -> {self.current_phase}')


    def update_log(self):
        '''
        saves everything to the log
        called from on_execute
        '''
        self.current_log = []
        self.current_log.append(self.trial_index)
        self.current_log.append(self.current_phase_trigger)
        #self.current_log.append(self.trial_infos[self.trial_index, 0]) #target number
        self.current_log.append(self.tgti_to_show) #target number
        self.current_log.append(self.trial_infos[self.trial_index]['vis_feedback_type']) #perturbation
        self.current_log.append(self.cursorX)
        self.current_log.append(self.cursorY)
        self.current_log.append(self.feedbackX)
        self.current_log.append(self.feedbackY)
        self.current_log.append(self.unpert_feedbackX) # before was called 'org_feedback'
        self.current_log.append(self.unpert_feedbackY)
        self.current_log.append(self.error_distance)  # Euclidean distance
        # NOTE: this is different!
        #self.current_log.append(self.trial_infos[self.trial_index, 2]) #stable (0) or random(1) block
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
        self.trigger_logfile.close()
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

        # MAIN LOOP
        while(self._running):
            self.current_time = time.time()
            # process events
            for event in pygame.event.get():
                self.on_event(event)

            # when task is running
            if (self.start_task):
                self.vars_update()

                # count time since last trigger (to send trigger reset signal
                # later)
                if (self.trigger_countdown > 0):
                    self.trigger_countdown -= 1
                elif (self.trigger_countdown == 0):
                    self.trigger_countdown = -1
                    # trigger reset
                    self.send_trigger(0)

                if self.trial_index == len(self.trial_infos):
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


    app = VisuoMotor(info, use_triggers=0, use_joystick=0, debug=True)
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

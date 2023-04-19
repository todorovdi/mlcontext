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
from itertools import product  # ,repeat
import argparse
import pylink
#import EyeLink
#from utils import get_target_angles

from context_change import (fnuniquify,trial_info2trgtpl_upd,gget,get_target_angles)
from context_change import VisuoMotor

# eyelink calib
from CalibrationGraphicsPygame import CalibrationGraphics

#/usr/share/EyeLink/SampleExperiments/Python/examples/Pygame_examples

from eyelink_helpers import *

# at CRNL we have EyeLink 1000 Plus

class VisuoMotorMEG(VisuoMotor):

    def initialize_parameters(self, info):
        VisuoMotor.initialize_parameters(self, info)

        self.add_param('dummy_mode', 1)

    def __init__(self, info, task_id='',
                 use_true_triggers = 1, debug=False, 
                 seed= None, start_fullscreen = 0):
        info['show_diode'] = 1
        VisuoMotor.__init__(self, info, task_id, use_true_triggers, debug,
                            seed, start_fullscreen,
                            save_tigger_and_trial_infos_paramfile = 0,
                            parafile_close = 0)

        self.params['delay_exit_break_after_keypress'] = 3.

        self.el_tracker = EL_init(self.params['dummy_mode'] )
        self.edf_file = open_edf_file(self.el_tracker, info)


        preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
        self.el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)
        self.el_tracker.setOfflineMode()

        EL_config(self.el_tracker, self.params['dummy_mode'])



        self.instuctions_eyelink_calib_str = (f"To calibrate eyetracker, press 'e'\n")
        self.phase_after_restart = 'REST'

        # redefine
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
                           'ITI':40, 'BREAK':50, 'PAUSE':60,
                              'BUTTON_PRESS':70,
                              'EYELINK_CALIBRATION_PRE':71,
                              'EYELINK_CALIBRATION':72,
                              'TASK_INSTRUCTIONS':73,
                              }


        self.trigger2phase = dict((v, k) for k, v in self.phase2trigger.items())
        #ct = time.time()
        self.phase_start_times = dict( zip(self.phase2trigger.keys(), len(self.phase2trigger) * [-1.] ) )
        self.phase_start_times['at_home'] = 0.
        self.phase_start_times['trigger'] = 0.
        self.phase_start_times['current_trial'] = time.time()

        # start phase
        self.current_phase = "EYELINK_CALIBRATION_PRE"

        self.dummy_eyelink_counter = self.params['FPS'] * 1

        # Q do I switch to REST or to ITI from break (currently to REST)?

        # Q: what do we do with eyetracker during the break? -- apparently nothing unless we feel it's really necessary

        # Q: how will EL_calibration work if called from on_render? Will is disrupt the rest of the loop?
        # Q Coum: when to call eyeAvailable()?  -- apres calibration
        # Q Coum: when to call doTrackerSetup?

        # Q Coum: how to use his conda env?

        # TODO: how to align eyelink with other data? -- Romain says its automatic because signal goes to MEG

        # Q: sendCommand vs sendMessage?
        # Q send COmmand 'calibration_area_proportion ? And validation_area as well
        # Q send command clear screen
        # Q send command draw filled box
        
        # Q: diff between msecDelay and pumpDelay?


        # todo?: need to write param file differently. -- not really, because I use only come phases to code triggers and they don't include calibration anyway


        # send a "TRIALID" message to mark the start of the first trial, so
        # the number of trials will be properly parsed by Data Viewer
        # el_tracker.sendMessage('TRIALID %d' % trial_index)

        # OPTIONAL--record_status_message : show some info on the Host PC
        # for illustration purposes, here we show an "example_MSG"
        # el_tracker.sendCommand("record_status_message 'Block %d'" % block)

        ############################################################

        maxMEGrec_dur = 20 * 60
        mult = int( np.ceil( self.durtot_all / maxMEGrec_dur ) )
        approx_pause_step = len(self.trial_infos)  // mult
        print(f"mult = {mult}; approx_pause_step = {approx_pause_step}")

        N = self.params['spec_trial_modN']
        def iscandpause(x): 
            # 1 because enumerate
            bi = x[1]['block_ind']
            r = x[1]['trial_type'] == 'pause'
        #     if r:
        #         print(bi)
            r = r and ((bi > 2) and (bi % N == 4))
            r = r and (x[0] > approx_pause_step - self.params[ 'block_len_max'])
            return r
            

        f = filter( iscandpause,  enumerate(self.trial_infos) )
        f = list(f)
        tinds = np.array (list(zip(*f))[0])

        #print( tinds % approx_pause_step )

        assert np.all(np.diff(tinds) > 0 )
        tistis = np.sort( np.argsort(tinds % approx_pause_step)[:mult - 1] )
        # print(tistis)
        tinds_pauses_turn_breaks = tinds[tistis]
        print('Tinds_pauses_turn_breaks =', tinds_pauses_turn_breaks )

        for tind in tinds_pauses_turn_breaks:
            self.trial_infos[tind]['trial_type'] = 'break'

        #raise ValueError('fd')


        phases_trigger_coded = [ 'TRAINING_START', 'TRAINING_END',
            'REST', 'GO_CUE_WAIT_AND_SHOW',
            'ITI', 'BREAK', 'PAUSE', 'BUTTON_PRESS' ]

        if self.params['rest_after_return']:
             phases_trigger_coded += ['RETURN']

        if self.params['feedback_type'] == 'offline':
             phases_trigger_coded += ['TARGET', 'FEEDBACK']
        else:
             phases_trigger_coded += [ 'TARGET_AND_FEEDBACK']


        self.spec_phases = [ 'BREAK', 'TRAINING_START', 'TRAINING_END', 'PAUSE' ]
        #spec_tt = ['break', 'pause' ]
        self.spec_tt = {'break':['REST','BREAK', 'BUTTON_PRESS' ] , 
                   'pause':['REST','PAUSE' ] }

        # 'TARGET':20, 'FEEDBACK': 35,

        # I want list of tuples -- target id, visual feedback type, phase
        # (needed for trigger)
        self.CONTEXT_TRIGGER_DICT = {}
        CTD_to_export = {}
        # 0,
        #trigger = self.phase2trigger['PAUSE'] + 40   # to get 100
        trigger = 1   # to get 100
        for tind, ti in enumerate(self.trial_infos):
            for phase in phases_trigger_coded:
                tpl = trial_info2trgtpl_upd(ti, phase, self.spec_phases, self.spec_tt)
                if (ti['trial_type'] in ['pause', 'break']) or (phase in ['BREAK', 'PAUSE'] ):
                    print(tind, phase, tpl)
                if tpl is None:
                    continue
                if tpl not in self.CONTEXT_TRIGGER_DICT:
                    self.CONTEXT_TRIGGER_DICT[ tpl ] = trigger
                    s = f'{tpl[0]},{tpl[1]},{tpl[2]},{tpl[3]}'
                    CTD_to_export[s  ] = trigger
                    trigger += 1

                    if ti['trial_type'] == 'break':
                        print('    ' ,trigger, s)

                    assert trigger < self.MEG_start_trigger


        import json
        s = json.dumps(CTD_to_export, indent=4)
        self.paramfile.write( '# CTD_to_export \n' )
        self.paramfile.write( '# trial param and phase 2 trigger values \n' )
        self.paramfile.write( '# trial_type, vis feedback, tgti, phase \n' )
        self.paramfile.write( s + '\n' )
        self.paramfile.write( '# phase2trigger \n' )
        s = json.dumps(self.phase2trigger, indent=4)
        self.paramfile.write( s + '\n' )

        self.paramfile.write( '# trial_infos = \n' )
        for tc in range( len(self.trial_infos) ):
            ti = self.trial_infos[tc]
            s = '{} = {}, {}, {}, {}'.format(tc, ti['trial_type'], ti['tgti'],
                  ti['vis_feedback_type'], ti['special_block_type'] )
            #print(s)
            self.paramfile.write( f'# {s}\n' )


        self.paramfile.close()

        ############################################################


        test_trial_ind = info.get('test_trial_ind',3)
        if test_trial_ind >= 0:
            if info['test_break'] and test_trial_ind >= 0:
                dspec = {'vis_feedback_type':'veridical', 'tgti':0,
                             'trial_type': 'break', 'special_block_type': None }
                self.trial_infos = self.trial_infos[:test_trial_ind] + [dspec] +\
                    self.trial_infos[test_trial_ind:]


        #f2 = filter( lambda x: x['trial_type'] == 'break', self.trial_infos )
        #f2 = list(f2)
        #len(f2), f2


    def send_trigger_cur_trial_info(self):
        ti = self.trial_infos[self.trial_index]
        tpl = trial_info2trgtpl_upd(ti, self.current_phase,
                                    self.spec_phases, self.spec_tt)
        if tpl is None:
            print(ti, self.trial_index, self.current_phase )
        self.send_trigger( self.CONTEXT_TRIGGER_DICT[tpl], tpl )


    def playSound(self):
        pygame.mixer.init()
        oggfile = 'beep-06.mp3'
        pygame.mixer.music.load(oggfile)
        pygame.mixer.music.play()

    def on_init(self):
        pygame.init()

        self.playSound()

        disp_sizes = pygame.display.get_desktop_sizes()
        print(  f'disp_sizes = {disp_sizes}' )
        if len(disp_sizes) > 1:
            # take widest
            srt = sorted( list( enumerate(disp_sizes) ), key = lambda x: x[1][0] )
            srt = list(srt)
            self.preferred_display = srt[-1][0]
            print(' self.preferred_display = ',self.preferred_display)
            display = self.preferred_display
        else:
            display = 0

        if self.debug:
            self._display_surf = pygame.display.set_mode(self.size, display = display)
        else:
            if self.start_fullscreen:
                self._display_surf = pygame.display.set_mode(self.size, pygame.FULLSCREEN,  display = display)
            else:
                self._display_surf = pygame.display.set_mode(self.size,  display = display)
        self._running = True

    def on_render(self):
        '''
        called from on_execute
        '''
        # end of the task
        if (self.task_started == 2):
            self._display_surf.fill(self.color_bg)

            ###################   FRENCH TEXT BEGIN #####################
            endstrs = [ 'La tache est maintenant finie. Vous avez été super, bravo !' ]
            delay = self.ctr_endmessage_show / self.params['FPS']
            endstrs += [f'La fenêtre va se fermer dans {delay:.0f} secondes']
            ###################   FRENCH TEXT END #####################

            #self.drawPopupText(pause_str)
            rnd = False
            if self.params['reward_rounding'] == 'end':
                rnd = True
            perfstrs = self.getPerfInfoStrings(reward_type = ['money'],
                                         inc_mvt_num = False,
                                         inc_last_reward = False,
                                               round = rnd)
            if self.ctr_endmessage_show == self.ctr_endmessage_show_def:
                print('Vos résultats ',perfstrs)
                #subdir = 'data'
                #fnf = pjoin(subdir,
                #    f'final_perf_subj={self.subject_id}_sess={self.session_id}.txt')
                #with open(fnf, 'w') as f:
                #    f.writelines(perfstrs )

                self.logfile.write('\n\n' )
                monetary_value_tot = self.reward_accrued * self.reward2EUR_coef # without rounding!
                perfstrs[-1] = perfstrs[-1] + f'; reward_accrued={self.reward_accrued}; monetary_value_tot={monetary_value_tot}'
                self.logfile.write(";".join(perfstrs) )

            self.drawTextMultiline( endstrs + [''] + perfstrs,
                                   font = self.myfont_popup,
                                    pos_label= 'center')


        elif (self.task_started == 1):
            # Clear screen
            self._display_surf.fill(self.color_bg)
            if self.debug:
                perfstrs = self.getPerfInfoStrings()
                self.drawTextMultiline( perfstrs, font = self.myfont_popup,
                                       pos_label = 'upper_right')

            if self.current_phase in ['TRAINING_START', 'TRAINING_END']:
                txt = self.phase2text[self.current_phase]
                self.drawTextMultiline(txt.split('\n'), pos_label = 'center', font = self.myfont_popup )
            if self.current_phase == 'ITI':
                if self.params['ITI_show_home'] and not \
                        (self.params['hit_notif'] == 'home_color_change' or \
                        self.params['miss_notif'] == 'home_color_change'):
                    self.drawHome() 
                    #print('ITI draw home 1')

                tdif = time.time() - \
                    self.phase_start_times[self.current_phase]

                # we could have gotten good reward also when we hit but don't
                # stay at target long enough
                if abs( self.reward - 1.) < 1e-6 and self.last_trial_full_success:
                    tdif = time.time() - self.phase_start_times[self.current_phase]
                    if self.params['hit_notif'] == 'explosion':
                        self.color_target = self.color_hit
                        self.drawTgt( 1 + 1.2 * \
                            np.sin((tdif/self.params['perf_feedback_duration']) * np.pi))
                    elif self.params['hit_notif'] == 'home_color_change':
                        #print('self.params[hit_notif] == home_color_change' )
                        tdifdif = tdif - self.params['perf_notif_start_delay']
                        if (tdifdif >= 0) and (tdifdif <= self.params['perf_feedback_duration']):
                            self.color_home = self.color_hit
                        else:
                            self.color_home = self.color_home_def
                        self.drawHome()
                        #print('ITI draw home 2', tdifdif)
                    elif self.params['hit_notif'] != 'no':
                        raise ValueError(f'wrong hit notif {self.params["hit_notif"]}')
                else:
                    if self.params['miss_notif'] == 'text':
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
                    elif self.params['miss_notif'] == 'cursor_explode':
                        self.color_feedback = self.color_miss
                        self.drawCursorFeedback( 1 + 1.2 * np.sin( (tdif/self.params['perf_feedback_duration']) * np.pi  ) )
                    elif self.params['miss_notif'] == 'target_explode':
                        self.color_target = self.color_miss
                        tdif = time.time() - \
                            self.phase_start_times[self.current_phase]
                        self.drawTgt( 1 + 1 * np.sin( (tdif/self.params['perf_feedback_duration']) * np.pi  ) )
                    # if 'no' then just do nothing
                    elif self.params['miss_notif'] == 'home_color_change':
                        tdifdif = tdif - self.params['perf_notif_start_delay']
                        if (tdifdif >= 0) and (tdifdif <= self.params['perf_feedback_duration']):
                            self.color_home = self.color_miss
                        else:
                            self.color_home = self.color_home_def
                        self.drawHome()
                    elif self.params['miss_notif'] != 'no':
                        raise ValueError(f'wrong miss notif {self.params["miss_notif"]}')


                #elif self.last_reach_stopped_away:
                #    s = 'Reach stopped in some strange place'
                #    self.drawPopupText(s)

            if self.current_phase == 'REST':
                # if not very far from home, draw cursor
                self.color_home = self.color_home_def

                at_home_ext = self.is_home('unpert_cursor', 'radius_home_strict_inside', 2)
                at_home =     self.is_home('unpert_cursor', 'radius_home_strict_inside', 1)
                if at_home_ext:
                    smooth = self.params['smooth_traj_home']
                    xy = self.drawCursorOrig(verbose=0, smooth = smooth)
                #if not at_home_ext:
                if not at_home:  # not at_home_ext
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

            if self.current_phase == 'BREAK':
                break_str = 'Pause longue commence'
                self.drawTextMultiline( [break_str] , 
                   font = self.myfont_popup,
                   pos_label= 'center')

            if self.current_phase == 'PAUSE':
                self.color_home = self.color_home_def
                self.drawHome()

                # time left
                #R = ( ( int(self.params['FPS'] * self.params['pause_duration']  )) -\
                #        self.frame_counters["pause"] ) / (self.params['FPS'] )
                #R = int(R)

                timedif = time.time() - self.phase_start_times[self.current_phase]
                rem = self.params['pause_duration'] - timedif
                R = int(rem)
                if (R % 5 == 0) and (rem - R) <= 1/self.params['FPS']:
                    print(f'ongoing pause, {R} sec left')
                if self.params['pause_text_info'] == 'countdown_and_reward':
                    pause_str = f'Pause, time left={R} seconds'
                    #self.drawPopupText(pause_str)
                    perfstrs = self.getPerfInfoStrings(reward_type = ['money'] )
                    self.drawTextMultiline( [pause_str] + [''] + perfstrs, font = self.myfont_popup,
                                           pos_label= 'center')
                elif self.params['pause_text_info'] == 'reward':
                    if timedif <= self.params['pause_text_show_duration']:
                        pause_str = 'Pause commence'
                        perfstrs = self.getPerfInfoStrings(reward_type = ['money'],
                            inc_last_reward = False)
                        self.drawTextMultiline( [pause_str] + [''] + perfstrs,
                            font = self.myfont_popup, pos_label= 'center',
                                               voffset_glob = -80)

                elif self.params['pause_text_info'] == 'progress_bar':
                    if timedif <= self.params['pause_text_show_duration']:
                        self.drawProgressBar()

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
                    at_home = self.is_home('unpert_cursor', 'radius_home')
                    if at_home:
                        smooth = self.params['smooth_traj_feedback_when_home'] 
                    else:
                        smooth = 0
                    self.drawCursorFeedback(smooth = smooth)

            if self.current_phase == 'GO_CUE_WAIT_AND_SHOW':
                self.color_target = self.color_wait_for_go_cue
                if self.params['motor_prep_show_target']:
                    self.drawTgt()
                self.drawHome()
                mpsc = self.params['motor_prep_show_cursor']
                smooth = self.params['smooth_traj_home']
                if mpsc == 'orig':
                    xy = self.drawCursorOrig(smooth = smooth)
                elif mpsc == 'feedback':
                    self.drawCursorFeedback(smooth = smooth)
                elif mpsc != 'no':
                    raise ValueError(f'Wrong param motor_prep_show_cursor == {mpsc}')
                #self.drawPopupText('WAIT', 'center')

            if self.current_phase == 'RETURN':
                self.drawReturnCircle()

                thickness = 2
                pygame.draw.circle(self._display_surf, self.color_target_def,
                                   self.home_position,
                                   self.params['radius_return'], thickness)

            show_diode = self.params['show_diode']
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

        # if not task_started
        else:
            if self.current_phase == 'EYELINK_CALIBRATION':
                phase_after_calib = 'TASK_INSTRUCTIONS'
                # TODO: I am not sure if fullscreen here should be this or 
                # I have to actually chekc whether I am fullscreen or not
                if self.params['dummy_mode']:
                    if self.dummy_eyelink_counter % 10 == 0:
                        print(f'Dummy eyelink calibration, dummy_eyelink_counter = {self.dummy_eyelink_counter}')
                    self.dummy_eyelink_counter -= 1
                    if self.dummy_eyelink_counter < 0:
                    #    self.restartTask(very_first = 1, phase = phase_after_calib)
                        self.current_phase = phase_after_calib 
                else:
                    EL_calibration(self.el_tracker, self.start_fullscreen)
                    #self.restartTask( very_first = 1, phase = phase_after_calib)
                    self.current_phase = phase_after_calib 
            elif self.current_phase in self.phase2dir:

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
            elif self.current_phase == "TASK_INSTRUCTIONS":
                self._display_surf.fill(self.color_bg)
 
                instr = self.instuctions_str.split('\n')
                #self.drawPopupText(instr,
                #                   font_size = self.foruser_font_size)
                self.drawTextMultiline(instr, font = self.myfont_popup,
                                       pos_label= 'center', voffset_glob = -300 )
            elif self.current_phase == "EYELINK_CALIBRATION_PRE":
                self._display_surf.fill(self.color_bg)
                instr = self.instuctions_eyelink_calib_str.split('\n')
                #self.drawPopupText(instr,
                #                   font_size = self.foruser_font_size)
                self.drawTextMultiline(instr, font = self.myfont_popup,
                                       pos_label= 'center' )

                #self.drawProgressBar(0.23)

                #perfstrs = self.getPerfInfoStrings(reward_type = ['money'],
                #    inc_last_reward = False, inc_mvt_num = 0)
                #pause_str = 'Pause commence, '
                #perfstrs = [ pause_str ] +  perfstrs
                #self.drawTextMultiline(perfstrs, font = self.myfont_popup,
                #                       pos_label = 'center',
                #    voffset_glob = self.params['progress_bar_height'] )
                #ax2 = self.HID_controller.get_axis(1)
                #max_ax1 = ax1
                #max_ax2 = ax2


        # debug print
        if self.debug and self.debug_type == 'render_extra_info':
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
        # does not check time

        #at_home = self.point_in_circle(self.home_position, (self.cursorX, self.cursorY),
        #                        self.params['radius_cursor'], verbose=0)
        at_home = self.is_home('unpert_cursor', 'radius_home_strict_inside')

        stop_rad_px = self.params['stop_rad_px']
        stopped = self.test_stopped(stop_rad_px = stop_rad_px)
        radius_reached = self.test_radius_reached()
        r = at_home,stopped,radius_reached
        if self.params['early_reach_end_event'] == 'stopped':
            b = stopped and (not at_home)
        elif self.params['early_reach_end_event'] == 'distance_reached':
            b = radius_reached
        if self.debug and b:
            print('test_reach_finished: ', self.params['early_reach_end_event'])
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

    def restartTask(self, very_first = False, phase = None):
        if phase is None:
            if very_first:
                self.current_phase = self.first_phase_after_start
            else:
                self.current_phase = self.phase_after_restart 

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

        # arguments: sample_to_file, events_to_file, sample_over_link,
        # event_over_link (1-yes, 0-no)
        dummy_mode = self.params['dummy_mode']
        try:
            EL_driftCorrect(self.el_tracker, dummy_mode)

            self.el_tracker.setOfflineMode()
            self.el_tracker.sendMessage('MEG start trigger sent')
            self.el_tracker.startRecording(1, 1, 1, 1)
            # Allocate some time for the tracker to cache some samples
            pylink.pumpDelay(100)
            self.el_tracker.sendMessage('Eyelink rec start after 100ms delay')

            # record a message to mark the start of scanning
        except RuntimeError as error:
            print("Eyelink ERROR:", error)
            EL_disconnect(self.el_tracker, dummy_mode)


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
            r = self.apply_visuomotor_pert(coordinates,
                                              perturbation)
        elif alteration_type == 'error_clamp':
            vec_ideal = np.array(self.target_coords[self.tgti_to_show]) - \
                np.array(self.home_position)
            vec_feedback = np.array( coordinates ) - \
                np.array(self.home_position)
            lvf = np.linalg.norm(vec_feedback)
            lvi = np.linalg.norm(vec_ideal)
            vec = (float(lvf) / float(lvi)) * vec_ideal
            vec = vec.astype(int) + np.array(self.home_position)
            #print('EC ',coordinates, vec)
            r = tuple(vec)
        else:
            raise ValueError(f'wrong {alteration_type}')

        return r

    def applyNoise(self, coordinates, noise = None  ):
        if noise is None:
            noise = self.params['noise_fb'] 
        r = coordinates
        
        if noise > 1e-10:
            r = np.array(r) + np.random.uniform(-1,1) * noise
        return r

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
        my_coords[0] = float( coordinates[0] - self.home_position[0] )
        my_coords[1] = float( coordinates[1] - self.home_position[1] )
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
        cursorReachX = cursorReachX + self.home_position[0]
        cursorReachY = cursorReachY + self.home_position[1]
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

    def is_home(self, coords_label = 'unpert_cursor',
                param_name = 'radius_return', mult=1., pos = None):
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
                cursorX = self.home_position[0] + ax1 * (self.params['width_for_cccomp'] / 2)
                cursorY = self.home_position[1] + ax2 * (self.params['height_for_cccomp'] / 2)
            elif self.params['joystick_angle2cursor_control_type'] == 'velocity':
                coefX = 0.05
                coefY = 0.05
                ml = 10
                #  self.current_phase == 'TARGET_AND_FEEDBACK'
                noisesc_start = 1.1
                noisesc_cont  = 0.5
                if abs(ax1orig) > noise1 * noisesc_start or ( len(self.trajX) > ml and abs(ax1orig) > noise1 * noisesc_cont ) :
                    cursorX = self.cursorX + ax1 * (self.params['width_for_cccomp'] / 2)   * coefX
                if abs(ax2orig) > noise2 * noisesc_start or ( len(self.trajY) > ml and abs(ax1orig) > noise1 * noisesc_cont ):
                    cursorY += self.cursorY + ax2 * (self.params['height_for_cccomp'] / 2)  * coefY
            else:
                raise ValueError(self.params['joystick_angle2cursor_control_type'])

            # old
            #self.cursorX = int(round(((ax1 - jaxlims[0]) / jaxrng) *
            #                    (self.params['width'] - 0) + 0))
            #self.cursorY = int(round(((ax2 - jaxlims[0]) / jaxrng) *
            #                    (self.params['height'] - 0) + 0))
        else:
            cursorX, cursorY = self.HID_controller.get_pos()
            #cursorX, cursorY = self.cursorX, self.cursorY

        self.cursorX = int(  cursorX  )
        self.cursorY = int(  cursorY  )

        #float coords
        return cursorX, cursorY

    def reset_traj(self):
        self.trajX     = [] # true traj
        self.trajY     = []
        self.trajfbX   = []
        self.trajfbY   = []
        self.traj_jax1 = []  # joystick angles raw
        self.traj_jax2 = []
        print('Reset traj')

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
        cursorX, cursorY = self.cursor_pos_update()
        self.cursorX, self.cursorY = self.applyNoise( (cursorX,cursorY), self.params['noise_fb'] )
        # only if task is running (i.e. when task_start is True)
        self.trajX += [ cursorX ]
        self.trajY += [ cursorY ]

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
                #print(f'REST: Not at home, reset: {self.cursorX,self.cursorY}, home pos = {self.home_position}')

            # if we spent inside time at home than show target
            #at_home_enough = (self.frame_counters["at_home"] == self.params['FPS']*\
            #        self.params['time_at_home'])
            at_home_enough = self.timer_check("at_home", 'time_at_home')
            if at_home_enough:
                if trial_info['trial_type'] == 'pause':
                    self.current_phase = 'PAUSE'
                if trial_info['trial_type'] == 'break':
                    self.current_phase = 'BREAK'
                    self.playSound()
                    self.free_from_break = 0
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

        # motor preparation
        elif (self.current_phase == 'GO_CUE_WAIT_AND_SHOW'):
            self.tgti_to_show = tgti
            wait_finished = self.timer_check(self.current_phase,
                                    'motor_prep_duration')

            at_home = self.is_home('unpert_cursor', 'radius_home_strict_inside', 1)
            #self.reset_traj()
            if not at_home:
                self.left_home_during_prep = 1
                # we return to rest (NOT to ITI, so don't update trial counter)
                self.current_phase = 'REST'
                self.reset_traj()    # reset traj if leave home
            elif wait_finished:
                self.left_home_during_prep = 0
                if self.params['feedback_type'] == 'online':
                    self.current_phase = 'TARGET_AND_FEEDBACK'
                else:
                    self.current_phase = 'TARGET'
                #print(self.frame_counters["at_home"], self.params['FPS']*self.params['time_at_home'])
                print(f'Start target and feedback display {trial_info}')

                # self.trial_infos[self.trial_index, 0]
                #self.tgti_to_show = tgti
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
            #self.trajX += [ self.cursorX ]
            #self.trajY += [ self.cursorY ]

            self.color_target = self.color_target_def

            #reach_time_finished = (self.frame_counters["feedback_shown"] == \
            #        self.params['FPS'] * self.params['time_feedback'])

            hit_cond = self.point_in_circle(self.target_coords[self.tgti_to_show],
                                    (self.feedbackX, self.feedbackY),
                                    self.params['radius_target'] +
                                    self.params['radius_cursor'] / 2.)

            reach_time_finished = self.timer_check(self.current_phase,
                                    'time_feedback')
            if reach_time_finished:
                self.timer_check(self.current_phase,
                                    'time_feedback', verbose=1)

            rtc = self.params['reach_termination_condition']

            reach_finished, extinfo = self.test_reach_finished(ret_ext = 1)
            at_home,stopped,radius_reached = extinfo
            if rtc == 'time':
                reach_finished = reach_time_finished

            # if time is up we switch to ITI, else
            if reach_time_finished or reach_finished:
                print(f'reach_time_finished={reach_time_finished}, '
                      f'at_home={at_home}, stopped={stopped}, '
                      f'radius_reached={radius_reached}, hit_cond={hit_cond}')

                if self.debug:
                    self.save_scr(trial_info, prefix='endreach')

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
                        print(f'Reach did not finish early, according to '
                            f'{self.params["early_reach_end_event"]}: tdif={tdif:.2f} sec')
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
                    #  self.error_distance was computed on previous call of this function
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
                # TODO: perhaps move this outside "else"
                self.feedbackX, self.feedbackY = \
                    self.alter_feedback(
                    (self.cursorX, self.cursorY), vft, trial_info['trial_type'])
                # no noise here because noise is already applied to self.cursorX
                #self.applyNoise( (self.feedbackX, self.feedbackY), self.params['noise_fb'] )

                self.trajfbX += [ self.feedbackX ]
                self.trajfbY += [ self.feedbackY ]

                # This variable saves the unperturbed, unprojected feedback
                self.unpert_feedbackX, self.unpert_feedbackY = self.cursorX, self.cursorY
                d = (self.feedbackX - self.target_coords[self.tgti_to_show][0])**2 + \
                    (self.feedbackY - self.target_coords[self.tgti_to_show][1])**2
                self.error_distance = np.sqrt(float(d))
                #####

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

        elif (self.current_phase == 'BREAK'):
            if (self.free_from_break): # set somewhere esle
                #self.current_phase = 'ITI' # would do trial index update, but requries allowing special trigger for it

                self.current_phase = 'REST'
                self.trial_index += 1  
                print ('Start Trial: ' + str(self.trial_index)) # ITI will do it 

                if self.params['return_home_after_pause']:
                    self.moveHome()
                    self.reset_traj()


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
                    self.reset_traj()
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


                    # to avoid jump of the target when we start running REST
                    # for the first time (because during ITI it is hidden and
                    # then we show first the wrong one until vars update is ran
                    # for the first time)
                    if self.trial_index < len(self.trial_infos):
                        trial_info2 = self.trial_infos[self.trial_index]
                        self.tgti_to_show = trial_info2['tgti']

                if self.params['return_home_after_ITI']:
                    self.moveHome()
                    self.reset_traj()
                print (f'Start trial # {self.trial_index} / {len(self.trial_infos)}')

                if self.trial_index < len(self.trial_infos):
                    trial_info2 = self.trial_infos[self.trial_index]
                    if trial_info['special_block_type'] == 'pretraining' and \
                        trial_info2['special_block_type'] != 'pretraining':
                        self.just_finished_pretraining = 1
                        print('Just finished pretraining')
                        self.current_phase = 'TRAINING_END'
                        self.reset_main_vars()

                # update joystick calibration at the end of ITI (normally one
                # would expect participant to return joystick to vertical at
                # this moment)
                if self.params['controller_type'] == 'joystick' and \
                        self.params['autmatic_joystick_center_calib_adjust'] == 'end_ITI':
                    # uses cursorX,cursorY values
                    at_home = self.is_home('unpert_cursor',
                                           'radius_home')
                    # only update calibration if at home otherwise we assume
                    # they moved too much
                    if at_home:
                        # normally we should be in the center if we are not, ther
                        # is a problem and we have to update calibration
                        #discrepancy_x = self.cursorX - self.home_position[0]
                        #discrepancy_y = self.cursorY - self.home_position[1]
                        ax1 = self.HID_controller.get_axis(0)
                        ax2 = self.HID_controller.get_axis(1)
                        ax1c = self.jaxcenter['ax1']
                        ax2c = self.jaxcenter['ax2']
                        jaxc_upd = (ax1 - ax1c) * self.discrepancy_red_lr,\
                            (ax2 - ax2c) * self.discrepancy_red_lr

                        sold = repr(self.jaxcenter)

                        self.jaxcenter['ax1'] += jaxc_upd[0]
                        self.jaxcenter['ax2'] += jaxc_upd[1]

                        snew = repr(self.jaxcenter)

                        print(f'AUTO RECALIB: Update jaxcenter, add {jaxc_upd}:  {sold} -> {snew}')

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
                self.reset_traj()
            #else:
            #    self.frame_counters["return"] += 1
        else:
            #print("Error")
            raise ValueError(f'wrong phase {self.current_phase}')
            self.on_cleanup(-1)

        if self.trial_index == len(self.trial_infos):
            self.task_started = 2

        self.current_phase_trigger = self.phase2trigger[self.current_phase]
        if self.current_phase != prev_phase:
            self.el_tracker.sendMessage('phase change to %s' % self.current_phase)

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

            self.el_tracker.sendMessage('TRIALID %d' % self.trial_index)

            ## OPTIONAL--record_status_message : show some info on the Host PC
            ## for illustration purposes, here we show an "example_MSG"
            #el_tracker.sendCommand("record_status_message 'Block %d'" % block)
            # TODO look
            ## For illustration purpose,
            ## send interest area messages to record in the EDF data file
            ## here we draw a rectangular IA, for illustration purposes
            ## format: !V IAREA RECTANGLE <id> <left> <top> <right> <bottom> [label]
            ## for all supported interest area commands, see the Data Viewer Manual,
            ## "Protocol for EyeLink Data to Viewer Integration"
            # left = int(scn_width/2.0) - 60
            # top = int(scn_height/2.0) - 60
            # right = int(scn_width/2.0) + 60
            # bottom = int(scn_height/2.0) + 60
            # ia_pars = (1, left, top, right, bottom, 'screen_center')
            # el_tracker.sendMessage('!V IAREA RECTANGLE %d %d %d %d %d %s' % ia_pars)
            # imgload_msg = '!V IMGLOAD CENTER %s %d %d %d %d' % imgload_pars
            # el_tracker.sendMessage('fix_onset')
            # el_tracker.sendMessage('!V CLEAR 128 128 128')
            # hor = (scn_width/2-20, scn_height/2, scn_width/2+20, scn_height/2)
            # ver = (scn_width/2, scn_height/2-20, scn_width/2, scn_height/2+20)
            # el_tracker.sendMessage('!V DRAWLINE 0 0 0 %d %d %d %d' % hor)
            # el_tracker.sendMessage('!V DRAWLINE 0 0 0 %d %d %d %d' % ver)
            ## record trial variables to the EDF data file, for details, see Data
            ## Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
            #el_tracker.sendMessage('!V TRIAL_VAR condition %s' % cond)
            #el_tracker.sendMessage('!V TRIAL_VAR image %s' % pic)
            ## send a 'TRIAL_RESULT' message to mark the end of trial, see Data
            ## Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
            #el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_OK)


            ## Set up the camera and calibrate the tracker, if not running in Dummy mode
            #if not dummy_mode:
            #    try:
            #        el_tracker.doTrackerSetup()
            #    except RuntimeError as err:
            #        print('ERROR:', err)
            #        el_tracker.exitCalibration()

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
        self.current_log.append(ti.get('block_ind',-100) )
        #self.current_log.append(self.cursorX)
        #self.current_log.append(self.cursorY)
        self.current_log.append(self.feedbackX)
        self.current_log.append(self.feedbackY)
        # this is the same as cursorX
        self.current_log.append(self.unpert_feedbackX) # before was called 'org_feedback'
        self.current_log.append(self.unpert_feedbackY)
        self.current_log.append(self.error_distance)  # Euclidean distance

        tx,ty = self.target_coords[self.tgti_to_show]
        self.current_log.append(tx)
        self.current_log.append(ty)



        # TODO: save area difference of traj
        # TODO: save angular coords?
        # TODO: trial within block?

        # maybe I will add it several times if I cross more than once
        # or stay around dist. But it is okay, I can select later first or last
        # crossing during data processing
        self.current_log.append(self.feedbackX_when_crossing)
        self.current_log.append(self.feedbackY_when_crossing)

        if self.params['controller_type'] == 'joystick':
            ax1 = self.HID_controller.get_axis(0)
            ax2 = self.HID_controller.get_axis(1)
            self.current_log.append(ax1)
            self.current_log.append(ax2)
        else:
            self.current_log.append(-1)
            self.current_log.append(-1)

        self.current_log.append(self.reward)

        #tgtcolor = ':'.join( map(str,self.color_target) )
        #self.current_log.append(tgtcolor)
        #homecolor = ':'.join( map(str,self.color_home) )
        #self.current_log.append(homecolor)

        self.current_log.append(self.current_time - self.initial_time)
        self.logfile.write(",".join(str(x) for x in self.current_log) + '\n')

    #
    #trial_index,   current_phase_trigger, tgti_to_show, vis_feedback_type, trial_type, special_block_type, cursorX, cursorY, feedbackX, feedbackY, unpert_feedbackX, unpert_feedbackY, error_distance, feedbackX_when_crossing, feedbackY_when_crossing, time



    def on_cleanup(self, exit_type):
        '''
        at the end of on_execute
        '''
        self.send_trigger(self.MEG_stop_trigger)
        time.sleep(0.05)
        self.MEG_rec_started = 0

        self.send_trigger(0)
        print(f"on_cleanup exit code = {exit_type}")
        self.logfile.close()
        self.trigger_logfile.close()

        EL_abort(self.el_tracker)
        EL_disconnect(self.params['dummy_mode'],
                      self.edf_file)
        pygame.quit()
        exit()

    def on_event(self, event):
        if event.type not in [pygame.MOUSEMOTION, pygame.JOYBALLMOTION,
                              pygame.JOYHATMOTION, pygame.JOYAXISMOTION,
                              pygame.WINDOWMOVED, pygame.AUDIODEVICEADDED,
                              pygame.WINDOWSHOWN, pygame.WINDOWTAKEFOCUS,
                              pygame.KEYUP]:
            if pygame.mouse.get_focused():
                print('on_event: ',event)

        if event.type in [ pygame.JOYBUTTONDOWN, pygame.MOUSEBUTTONDOWN  ]:
            self.send_trigger(self.phase2trigger['BUTTON_PRESS'] )

        if (event.type == pygame.MOUSEMOTION) and self.just_moved_home:
            self.just_moved_home = 0

        if event.type == pygame.QUIT:
            self._running = False
        if event.type == self.phase_shift_event_type:
            print(self.cursorX, self.cursorY, self.current_phase,
                  self.trial_infos[self.trial_index] )

            # if task was not started and a button was pressed, start the task
            if self.task_started == 0:
                self.restartTask(very_first = 1)
            # if task was started and a button was pressed, release break
            else:
                self.moveHome()
                self.free_from_break = 0

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.on_cleanup(-1)
            if event.key == pygame.K_F11 and pygame.mouse.get_focused():
                disp_sizes = pygame.display.get_desktop_sizes()
                print(  f'disp_sizes = {disp_sizes}' )
                if len(disp_sizes) > 1:
                    display = self.preferred_display
                else:
                    display = 0

                if (self._display_surf.get_flags() & pygame.FULLSCREEN):
                    self._display_surf = pygame.display.set_mode(self.size, display = display)
                else:
                    self._display_surf = pygame.display.set_mode(self.size,
                        pygame.FULLSCREEN, display = display)
                    #self._display_surf = pygame.display.set_mode(0,0,
            #if event.key == pygame.K_q:
            #    self.params['radius_cursor'] = self.params['radius_cursor']-1
            #if event.key == pygame.K_w:
            #    self.params['radius_cursor'] = self.params['radius_cursor']+1
            if event.key == pygame.K_r:
                self.moveHome()
                self.reset_traj()
            if event.key == pygame.K_s:
                self.save_scr(prefix='keypress')
            # forcefully start task, should not be used unless debug
            if event.key == pygame.K_y:
                self.free_from_break = 1
                print('Debug restart task by key')
                self.restartTask(very_first = 1, phase = self.first_phase_after_start)
            # release from break
            if (event.key == pygame.K_g) and (not self.free_from_break):
                self.free_from_break = 1

                self.playSound()
                time.sleep(self.params['delay_exit_break_after_keypress']) # in sec
                self.restartTask(very_first = 0, phase = self.phase_after_restart  )
            if (event.key == pygame.K_e) and (not self.task_started == 1):
                self.current_phase = 'EYELINK_CALIBRATION'
            if (event.key == pygame.K_j) and (not self.task_started == 1) and\
                    (self.params['controller_type'] == 'joystick' ):
                self.current_phase = self.calib_seq[0]
                self.reset_traj()
                pygame.mouse.set_visible(0)

                print('Start joystick calibration')
            #if event.key == pygame.K_c:
            #    if (self.params['use_eye_tracker']):
            #        EyeLink.tracker(self.params['width'], self.params['height'])

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
                                  f' in the center position. Now it is at ax1={ax1:.3f}, ax2={ax2:.3f}'))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default='no' )
    parser.add_argument('--fullscreen', default=0, type=int )
    parser.add_argument('--joystick',   default=1, type=int )
    parser.add_argument('--seed',   default=None )
    parser.add_argument('--participant', default='debug' )
    parser.add_argument('--session', default='debugsession' )
    parser.add_argument('--show_dialog', default=1, type=int )
    # can be  <int width>x<int hegiht>
    parser.add_argument('--screen_size', default='fixed', type=str )
    parser.add_argument('--test_err_clamp', default=0, type=int)
    parser.add_argument('--test_pause',    default=0,  type=int)
    parser.add_argument('--test_break',    default=0,  type=int)
    parser.add_argument('--test_end_task',    default=0,  type=int)
    parser.add_argument('--num_training',       type=int)
    parser.add_argument('--test_trial_ind', default =-1,   type=int)
    parser.add_argument('--noise_fb',    type=int)
    parser.add_argument('--smooth_traj_home',  type=int)
    parser.add_argument('--smooth_traj_feedback_when_home',  type=int)
    parser.add_argument('--time_feedback',  type=float)
    parser.add_argument('--motor_prep_duration',  type=float)
    parser.add_argument('--ITI_duration',  type=float)
    parser.add_argument('--time_at_home',  type=float)
    parser.add_argument('--pause_duration',  type=float)
    parser.add_argument('--training_text_show_duration',  type=float)
 


    args = parser.parse_args()
    par = vars(args)

    assert par['debug'] in ['no', 'render_extra_info', 'render_final']
    assert ( par['screen_size'] in ['fixed', 'auto']) | ( par['screen_size'].find('x') > 0 )


    info = {}

    info['participant'] = par['participant']
    info['session']     = par['session']

    show_dialog = par['show_dialog']
    if show_dialog:
        info['participant'] = ''
        info['session'] = ''
        dlg = gui.DlgFromDict(info)
        if not dlg.OK:
            core.quit()
    else:
        info['participant'] = 'Dmitrii_test'
        info['session'] = 'session1'

    for p,v in par.items():
        if p not in ['participant', 'session']:
            info[p] = v
    #info['joystick'] = par['joystick']
    #info['screen_size'] = par['screen_size']


    if par['seed'] is not None:
        seed = par['seed']
    else:
        seed = None
    print(f'seed = {seed}')
    start_fullscreen = 0
    #app = VisuoMotor(info, use_true_triggers=0, debug=1, seed=seed,
    #                 start_fullscreen = 0)
    app = VisuoMotorMEG(info, use_true_triggers=0, debug=par['debug'],
                     seed=seed, start_fullscreen=par['fullscreen'])
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


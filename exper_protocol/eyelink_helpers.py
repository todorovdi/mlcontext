import pylink # need to register on https://www.sr-research.com/support/index.php
from CalibrationGraphicsPygame import CalibrationGraphics
import pygame
from pygame.locals import *
import sys

def EL_init(dummy_mode):
    """Connect to the EyeLink Host PC.

    The Host IP address, by default, is "100.1.1.1".
    the "el_tracker" objected created here can be accessed through the Pylink
    Set the Host PC address to "None" (without quotes) to run the script in "Dummy Mode"""

    if dummy_mode:
        el_tracker = pylink.EyeLink(None)
    else:
        try:
            el_tracker = pylink.EyeLink("100.1.1.1")
        except RuntimeError as error:
            print('ERROR:', error)
            pygame.quit()
            sys.exit()

    return el_tracker

def open_edf_file(el_tracker, info):
    """Open an EDF data file on the Host PC"""

    subj = info['participant']
    s = ''
    if len(subj) and ( subj != 'Dmitrii_test' ):
        print(subj)
        nn = 4
        if len(subj) > nn:
            subjN = subj[:-nn]
        else:
            subjN = subj
        assert subj.isalnum()
    else:
        subjN = ''
    edf_file = f"EL_{subjN}_{s}.edf"
    #maxlen = 8
    #assert len(edf_file) < maxlen


    print(f'open_edf_file: using file {edf_file}')
    #edf_file = f"eyelink_dat_{subj}_{self.settings.current_session}.edf"
    #edf_file = "eL_file.edf"
    try:
        el_tracker.openDataFile(edf_file)
    except RuntimeError as err:
        print('EDF file ERROR:', err)
        # close the link if we have one open
        if el_tracker.isConnected():
            el_tracker.close()
        pygame.quit()
        sys.exit()
    
    preamble_text = 'SUBJECT ID: %s' % subj
    el_tracker.sendCommand(
        "add_file_preamble_text '%s'" % preamble_text)

def EL_config(el_tracker, dummy_mode):
    """ Configure the tracker"""
    print('EL_config')

    # Get the software version:  1-EyeLink I, 2-EyeLink II, 3/4-EyeLink 1000,
    # 5-EyeLink 1000 Plus, 6-Portable DUO
    eyelink_ver = 0  # set version to 0, in case running in Dummy mode
    if not dummy_mode:
        vstr = el_tracker.getTrackerVersionString()
        eyelink_ver = int(vstr.split()[-1].split('.')[0])
        # print out some version info in the shell
        print('Running experiment on %s, version %d' % (vstr, eyelink_ver))

    # File and Link data control
    # what eye events to save in the EDF file, include everything by default
    file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
    # what eye events to make available over the link, include everything by default
    link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'
    # what sample data to save in the EDF data file and to make available
    # over the link, include the 'HTARGET' flag to save head target sticker
    # data for supported eye trackers
    if eyelink_ver > 3:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
    else:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
    el_tracker.sendCommand(
        "file_event_filter = %s" % file_event_flags)
    el_tracker.sendCommand(
        "file_sample_data = %s" % file_sample_flags)
    el_tracker.sendCommand(
        "link_event_filter = %s" % link_event_flags)
    el_tracker.sendCommand(
        "link_sample_data = %s" % link_sample_flags)
    # Optional tracking parameters
    # Sample rate, 250, 500, 1000, or 2000, check your tracker specification
    # if eyelink_ver > 2:
    #     el_tracker.sendCommand("sample_rate 1000")
    # Choose a calibration type, H3, HV3, HV5, HV13 (HV = horizontal/vertical),
    el_tracker.sendCommand("calibration_type = HV9")
    # Set a gamepad button to accept calibration/drift check target
    # You need a supported gamepad/button box that is connected to the Host PC
    el_tracker.sendCommand(
        "button_function 5 'accept_target_fixation'")

    # Optional -- Shrink the spread of the calibration/validation targets
    # if the default outermost targets are not all visible in the bore.
    # The default <x, y display proportion> is 0.88, 0.83 (88% of the display
    # horizontally and 83% vertically)

    # orig code
    #el_tracker.sendCommand('calibration_area_proportion 0.88 0.83')
    #el_tracker.sendCommand('validation_area_proportion 0.88 0.83')

    # Coum vals. The second coord is lower in order to avoid overlapping with photodiode
    el_tracker.sendCommand('calibration_area_proportion 0.88 0.60')
    el_tracker.sendCommand('validation_area_proportion 0.88 0.60')

    # Optional: online drift correction.
    # See the EyeLink 1000 / EyeLink 1000 Plus User Manual
    #
    # Online drift correction to mouse-click position:
    # el_tracker.sendCommand('driftcorrect_cr_disable = OFF')
    # el_tracker.sendCommand('normal_click_dcorr = ON')

    # Online drift correction to a fixed location, e.g., screen center
    # el_tracker.sendCommand('driftcorrect_cr_disable = OFF')
    # el_tracker.sendCommand('online_dcorr_refposn %d,%d' % (int(scn_width/2.0),
    #                                                        int(scn_height/2.0)))
    # el_tracker.sendCommand('online_dcorr_button = ON')
    # el_tracker.sendCommand('normal_click_dcorr = OFF')

# beg of each trial
# Q: what does drift correct do? Does it take time?
def EL_driftCorrect(el_tracker, dummy_mode):
    print('EL_driftCorrect')
    if not dummy_mode:
        while True:  # exit when succeed
            if (not el_tracker.isConnected()) or el_tracker.breakPressed():
                terminate_task()

            # drift-check and re-do camera setup if ESCAPE is pressed
            try:
                error = el_tracker.doDriftCorrect(int(scn_width/2.0),
                                                  int(scn_height/2.0), 1, 1)
                # break following a success drift-check
                if error is not pylink.ESC_KEY:
                    break
            except:
                pass

def EL_calibration(el_tracker, full_screen):
    """ Set up a graphic environment for calibration/"""
    print('EL_calibration')

    # Step 4: set up a graphics environment for calibration
    #
    # open a Pygame window
    win=None
    if full_screen:
        win = pygame.display.set_mode((0, 0), FULLSCREEN | DOUBLEBUF)
    else:
        win = pygame.display.set_mode((0, 0), 0)
        
    scn_width, scn_height = win.get_size()
    pygame.mouse.set_visible(False)  # hide mouse cursor

    # Pass the display pixel coordinates (left, top, right, bottom) to the tracker
    # see the EyeLink Installation Guide, "Customizing Screen Settings"
    el_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_width - 1, scn_height - 1)
    el_tracker.sendCommand(el_coords)

    # Write a DISPLAY_COORDS message to the EDF file
    # Data Viewer needs this piece of info for proper visualization, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (scn_width - 1, scn_height - 1)
    el_tracker.sendMessage(dv_coords)

    # Configure a graphics environment (genv) for tracker calibration
    genv = CalibrationGraphics(el_tracker, win)

    # Set background and foreground colors
    foreground_color = (0, 0, 0)
    background_color = (128, 128, 128)
    genv.setCalibrationColors(foreground_color, background_color)

    # Set up the calibration target
    #
    # The target could be a "circle" (default) or a "picture",
    # To configure the type of calibration target, set
    # genv.setTargetType to "circle", "picture", e.g.,
    # genv.setTargetType('picture')
    #
    # Use gen.setPictureTarget() to set a "picture" target, e.g.,
    # genv.setPictureTarget(os.path.join('images', 'fixTarget.bmp'))

    # Use the default calibration target
    genv.setTargetType('circle')

    # Configure the size of the calibration target (in pixels)
    genv.setTargetSize(24)

    # Beeps to play during calibration, validation and drift correction
    # parameters: target, good, error
    #     target -- sound to play when target moves
    #     good -- sound to play on successful operation
    #     error -- sound to play on failure or interruption
    # Each parameter could be ''--default sound, 'off'--no sound, or a wav file
    # e.g., genv.setCalibrationSounds('type.wav', 'qbeep.wav', 'error.wav')
    genv.setCalibrationSounds('', '', '')

    # Request Pylink to use the Pygame window we opened above for calibration
    pylink.openGraphicsEx(genv)

    # I am not sure, maybe  pylink.openGraphicsEx(genv) already does it
    # no, actually, apparently it does not 
    try:
        # it launches eyelink software on host PC which waits for 'C' to be pressed
        el_tracker.doTrackerSetup()
    except RuntimeError as err:
        print('ERROR:', err)
        el_tracker.exitCalibration()

def EL_getEyeInfo(el_tracker):
    # determine which eye(s) is/are available
    # 0- left, 1-right, 2-binocular
    print('EL_getEyeInfo: Trying to get eye information')
    eye_used = el_tracker.eyeAvailable()
    if eye_used == 1:
        el_tracker.sendMessage("EYE_USED 1 RIGHT")
    elif eye_used == 0 or eye_used == 2:
        el_tracker.sendMessage("EYE_USED 0 LEFT")
        eye_used = 0
    else:
        print("Error in not getting the eye information!")
        return pylink.TRIAL_ERROR
    
    return eye_used

def EL_abort(el_tracker):
    print('EL_abort')
    """Ends recording """

    # Stop recording
    if el_tracker.isRecording():
        # add 100 ms to catch final trial events
        pylink.pumpDelay(100)
        el_tracker.stopRecording()

def EL_disconnect(el_tracker,  edf_file, dummy_mode):
    print('EL_disconnect')
    if dummy_mode:
        return

    #el_tracker = pylink.getEYELINK()
    if el_tracker.isConnected():

        # Put tracker in Offline mode
        el_tracker.setOfflineMode()

        # Clear the Host PC screen and wait for 500 ms
        el_tracker.sendCommand('clear_screen 0')
        pylink.msecDelay(500)

        # Close the edf data file on the Host
        el_tracker.closeDataFile()

        # Show a file transfer message on the screen
        msg = 'EDF data is transferring from EyeLink Host PC...'
        #show_message(msg, (0, 0, 0), (128, 128, 128))

        # Download the EDF data file from the Host PC to a local data folder
        # parameters: source_file_on_the_host, destination_file_on_local_drive
        subject_number = 0
        import os
        local_edf = os.path.join("results", "%s.edf" % subject_number)
        try:
            el_tracker.receiveDataFile(edf_file, local_edf)
        except RuntimeError as error:
            print('ERROR:', error)

        # Close the link to the tracker.
        el_tracker.close()


# # initialize EyeLink
# self.EL_init()
# self.open_edf_file()
# self.EL_config()
# 
# # calibration
# self.EL_calibration()
# 
# # start recording
# self.el_tracker = pylink.getEYELINK()
# self.el_tracker.setOfflineMode()
# 
# try:
#     self.el_tracker.startRecording(1, 1, 1, 1)
# except RuntimeError as error:
#     print("ERROR:", error)
#     return pylink.TRIAL_ERROR
# 
# # Allocate some time for the tracker to cache some samples
# pylink.pumpDelay(100)
# 
# 
# # ------- your task ---------
# 
# # stop Recording
# self.EL_disconnect()

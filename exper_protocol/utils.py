import numpy as np

def readLogFile(fnf):
    with open(fnf, 'r') as f:
        lines = f.readlines()

def readParamFile(fnf):
    with open(fnf, 'r') as f:
        lines = f.readlines()

    ##################   process param file
    triggerdict_start_line = '# trial param and phase 2 trigger values'
    stl = -1
    for linei,line in enumerate(lines):
        if line.startswith(triggerdict_start_line):
            stl = linei
            break

    ##########  params
    params = {}
    for line in lines[:stl]:
        if line.startswith('#'):
            continue
        lhs,rhs = line.replace(' ','').replace('\n','').split('=')
        params[lhs] = rhs

    early_reach_end_event = params.get('early_reach_end_event')
    if early_reach_end_event is None:
        early_reach_end_event = params.get('reach_end_event')
    print('early_reach_end_event = ', early_reach_end_event)


    stage2pars = {}

    #phase2trigs = {}
    #phase2trigs[phase_to_collect] = []
    #for line in lines[stl + 3:]:
    #    if line.startswith('}'):
    #        break
    #    k,v = line.split(':')
    #    v = int(v[:-2])
    #    k = k.replace('"','').replace(' ','')
    #    tt,vft,tgti,phase = k.split(',')
    #    if len(tgti) > 0:
    #        tgti = int(tgti)
    #    else:
    #        tgti = None
    #    stage2pars[ v ] = tt,vft,tgti,phase
    #    if phase == phase_to_collect:
    #        phase2trigs[phase_to_collect] += [v]
    #    #print(line)

    params['width'] = int(params['width'])
    params['height'] = int(params['height'])
    params['dist_tgt_from_home'] = int(params['dist_tgt_from_home'])

    #return params, phase2trigger, trigger2phase, stage2pars
    return params


def get_target_angles(num_targets, target_location_pattern, spread=15, defloc = 0):
    # defloc -- angle from vertical line
    if num_targets == 3:
        assert target_location_pattern == 'fan'
        targetAngs = np.array([defloc-spread,defloc,defloc + spread])
    elif num_targets == 2:
        targetAngs = np.array([defloc-spread,defloc + spread])
    elif num_targets == 1:
        targetAngs = np.array([defloc])
    elif num_targets == 4:
        if target_location_pattern == 'diamond':
            mult = 90 # in deg
            targetAngs = np.arange(num_targets) * mult
        elif target_location_pattern == 'fan':
            mult = spread # in deg
            m = num_targets * mult
            targetAngs = np.arange(num_targets) * mult -  m / 2
        else:
            raise ValueError(f'target_location_pattern = {target_location_pattern} not implemented')
    elif num_targets >= 4:
        mult = spread # in deg
        m = num_targets * mult
        targetAngs = np.arange(num_targets) * mult -  m / 2

    #targetAngs = targetAngs + (180 + 90)
    targetAngs = targetAngs + 90
    print('Angles counting CCW from right pointing Oy, mostly above home')
    return targetAngs

def calc_target_positions(targetAngs, home_pos, dist_tgt_from_home,
                          consistent_with_app = True):
    '''
    called in class constructor
    returns in screen coords
    '''
    #targetAngs = [22.5+180, 67.5+180, 112.5+180, 157.5+180]
    #targetAngs = get_target_angles(self.params['num_targets'])

    # list of 2-tuples
    target_coords = []
    for tgti,tgtAngDeg in enumerate(targetAngs):
        tgtAngRad = float(tgtAngDeg)*(np.pi/180)
        # this will be given to pygame.draw.circle as 3rd arg
        # half screen width + cos * radius
        # half screen hight + sin * radius
        X = (np.cos(tgtAngRad) * dist_tgt_from_home)
        Y = (np.sin(tgtAngRad) * dist_tgt_from_home)


        if consistent_with_app:
            X += home_pos[0]
            Y += home_pos[1]
        else:
            # this would more logical but inconsistent with the code in context
            # change
            X,Y = homec2screen(X,Y, home_pos)
        target_coords.append((X,Y) )

    return target_coords

def calc_err_eucl(feedbackXY, target_coords, tgti_to_show):
    '''
    tuple as input
    '''
    feedbackX, feedbackY = feedbackXY
    a = feedbackX -target_coords[tgti_to_show][0]
    b = feedbackY -target_coords[tgti_to_show][1]
    #print(a,b)
    d = a**2 + b**2
    error_distance = np.sqrt(float(d))
    return error_distance

def screen2homec(X,Y,home_position):
    from collections.abc import Iterable
    if isinstance(X,Iterable):
        X = np.array(X)
    if isinstance(Y,Iterable):
        Y = np.array(Y)
    # home_positino is in screen coords
    return X - home_position[0], - ( Y - home_position[1] )

def homec2screen(X,Y,home_position):
    from collections.abc import Iterable
    if isinstance(X,Iterable):
        X = np.array(X)
    if isinstance(Y,Iterable):
        Y = np.array(Y)
    # home_positino is in screen coords
    return X + home_position[0], ( home_position[1] - Y   )

def coords2anglesRad(X, Y, home_position, radius = None ):
    # X,Y screen coords
    assert home_position is not None
    #from collections.abc import Iterable
    #if isinstance(X,Iterable):
    #    X = np.array(X)
    #if isinstance(Y,Iterable):
    #    Y = np.array(Y)

    #if home_position is not None:
    #    if isinstance(X,np.ndarray):
    #        X = X.copy()
    #        Y = Y.copy()
    #    X -= home_position[0]
    #    Y -= home_position[1]

    X,Y = screen2homec(X,Y,home_position)

    if radius is None:
        radius = np.sqrt( X*X + Y*Y )

    angles = np.arctan2(Y/float(radius),
                        X/float(radius))
    # change the 0 angle (0 is now bottom vertical in the circle)
    ##angles = angles + np.pi/2.
    # make the angle between 0 and 2*np.pi

    #if isinstance(X,np.ndarray):
    #    for i in np.where(angles < 0):
    #        angles[i] = angles[i] + 2*np.pi
    #    for i in np.where(angles > 2*np.pi):
    #        angles[i] = angles[i] - 2*np.pi
    #else:
    #    if angles < 0:
    #        angles = angles + 2*np.pi
    #    if angles > 2*np.pi:
    #        angles = angles - 2*np.pi
    return angles

def getLastInfo(fn, nread):
    import pandas as pd
    with open(fn + '.log', 'r') as f:
        lines = f.readlines()

    r = ('trial_index, current_phase_trigger, tgti_to_show,'
       ' vis_feedback_type, trial_type, special_block_type, block_ind, '
        ' feedbackX, feedbackY, unpert_feedbackX, unpert_feedbackY,'
         ' error_distance, target_coordX, target_coordY, '
         'feedbackX_when_crossing, feedbackY_when_crossing, '
         'jax1, jax2, reward, time')
    r = r.replace(' ','')
    colnames = r.split(',')

    print('Last line = ')
    print( dict(zip(colnames, lines[-1].split(',') ) ) )
    n_rows = len(lines)
    del lines
    skip = int(n_rows - nread)
    if skip < nread:
        skip = 0

    #def skip_comment(row):
    #    # check if the row starts with #
    #    return row.startswith("##")


    df = pd.read_csv(fn + '.log', skiprows=skip, names = colnames, comment = '#')
    #df = pd.read_csv(fn + '.log', skiprows=skip_comment, names = colnames)

    grp_pertim = df.groupby('trial_index').min('time')
    grp_blockim = df.groupby('block_ind').min('time')
    trial_starts = grp_pertim['time']
    block_starts = grp_blockim['time']
    phase_starts = df.groupby('current_phase_trigger').min('time')['time']


    last_pause, last_block = None, None
    p = df.query('trial_type == "pause"')
    if len(p):
        p = p.groupby('trial_index')
        last_pause = p['time'].idxmax()
    p = df.query('trial_type == "block"')
    if len(p):
        p = p.groupby('trial_index')
        last_block = p['time'].idxmax()

    last_trial_ind = trial_starts.idxmax()
    last_phase = phase_starts.idxmax()
    last_trial_block_ind = block_starts.idxmax()
 
    last_block_first_trial_ind = df.query(f'block_ind == {last_trial_block_ind}')['trial_index'].min()

    reward_accrued_before_last_trial = grp_pertim.\
        query(f'trial_index < {last_trial_ind}')['reward'].sum()

    reward_accrued_before_last_block = grp_blockim.\
        query(f'block_ind < {last_trial_block_ind}')['reward'].sum()

    return (last_phase, last_trial_ind, last_trial_block_ind, last_block_first_trial_ind, 
            last_pause, last_block,
        reward_accrued_before_last_trial, reward_accrued_before_last_block)

def getLastFname(subdir):
    import os, glob
    # Get a list of all files in 'subdir' ending with '.xls'
    if not os.path.exists(subdir):
        print('Dir not exists')
        return None

    from os.path import join as pjoin
    
    sp = pjoin(os.getcwd(), subdir, '*.param' )
    files = glob.glob(sp )

    if len(files) == 0:
        print(f'No files in {sp}')
        return None
    # Find the file with the latest creation time
    latest_file = max(files, key=os.path.getctime)

    latest_fn_base, extension = os.path.splitext(latest_file)

    return latest_fn_base

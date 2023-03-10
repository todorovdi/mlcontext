import numpy as np

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

def calc_target_positions(targetAngs, home_pos, dist_tgt_from_home):
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
        X,Y = homec2screen(X,Y, home_pos)
        target_coords.append((X,Y) )

    return target_coords

def calc_err_eucl(feedbackXY, target_coords, tgti_to_show):
    feedbackX, feedbackY = feedbackXY
    d = (feedbackX -target_coords[tgti_to_show][0])**2 + \
        (feedbackY -target_coords[tgti_to_show][1])**2
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

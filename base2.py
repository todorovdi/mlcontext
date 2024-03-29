import numpy as np
import os.path as op
# from nose.tools import assert_true
import math
from sklearn.base import BaseEstimator
from sklearn.model_selection import ShuffleSplit
from scipy.linalg import pinv
# for NIH data
width = 800  # need to match the screen size during the task
height = 800
# this is radius on which NIH targets appear
radius = int(round(height*0.5*0.8))
radius_target = 12
radius_cursor = 8

#event_ids = [20, 21, 22, 23, 25, 26, 27, 28]  # _tgt


from config2 import target_angs
#target_coords = calc_target_coordinates_centered(target_angs)


def int_to_unicode(array):
    return ''.join([str(chr(int(ii))) for ii in array])


def bincount(a):
    """Count the number of each different values in a."""
    y = np.bincount(a)
    ii = np.nonzero(y)[0]
    return np.vstack((ii, y[ii])).T


def test_trigger_accuracy(events):
    """Test the correct sequence of the triggers."""
    for i, event in enumerate(events):
        if events[event, 2] == 10:
            assert ((events[i+1, 2] >= 20 and events[i+1, 2] <= 23))


class ScoringAUC():
    """Score AUC for multiclass problems.
    Average of one against all.
    """
    def __call__(self, clf, X, y, **kwargs):
        from sklearn.metrics import roc_auc_score

        # Generate predictions
        if hasattr(clf, 'decision_function'):
            y_pred = clf.decision_function(X)
        elif hasattr(clf, 'predict_proba'):
            y_pred = clf.predict_proba(X)
        else:
            y_pred = clf.predict(X)

        # score
        classes = set(y)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]

        _score = list()
        for ii, this_class in enumerate(classes):
            _score.append(roc_auc_score(y == this_class,
                                        y_pred[:, ii]))
            if (ii == 0) and (len(classes) == 2):
                _score[0] = 1. - _score[0]
                break
            return np.mean(_score, axis=0)

def decod_stats2(X, n_jobs=None, n_permutations=2**10):
    from mne.stats import permutation_cluster_test
    """Statistical test applied across subjects"""
    # check input
    X = np.array(X)

    # stats function report p_value for each cluster
    # null = np.repeat(chance, len(times))
    # Non-corrected t-test...
    # T_obs, p_values_ = ttest_1samp(X, null, axis=0)
    if n_jobs is None:
        from config2 import n_jobs
    T_obs_, clusters, p_values, _ = permutation_cluster_test(
        X, out_type='mask', n_permutations = n_permutations, n_jobs=n_jobs,
        verbose=False)

    #print(type(T_obs_), type(clusters), type(p_values) )
    #print(T_obs_.shape, len(clusters), p_values.shape)
    #print(clusters[0], p_values)

    # clusters is a list of array slices
    # each cluster is a slice
    # format p_values to get same dimensionality as X
    p_values_ = np.ones_like(X[0][0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster] = pval

    r = np.squeeze(p_values_)
    #print('rshpe',r.shape)
    return r

def decod_stats(X, n_jobs=None, n_permutations=2**10, tail=0):
    from mne.stats import permutation_cluster_1samp_test, ttest_1samp_no_p
    """Statistical test applied across subjects"""
    # check input
    X = np.array(X)

    # stats function report p_value for each cluster
    # null = np.repeat(chance, len(times))
    # Non-corrected t-test...
    # T_obs, p_values_ = ttest_1samp(X, null, axis=0)
    stat_fun = ttest_1samp_no_p
    if n_jobs is None:
        from config2 import n_jobs
    T_obs_, clusters, p_values, _ = permutation_cluster_1samp_test(
        X, out_type='mask', n_permutations = n_permutations, n_jobs=n_jobs,
        stat_fun = stat_fun, tail=tail, verbose=False)

    #print(type(T_obs_), type(clusters), type(p_values) )
    #print(T_obs_.shape, len(clusters), p_values.shape)
    ## clusters is a lit of array
    #print(clusters[0], p_values)

    # format p_values to get same dimensionality as X
    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster] = pval

    r = np.squeeze(p_values_)
    #print('rshpe',r.shape)
    return r


def gat_stats(X):
    from mne.stats import spatio_temporal_cluster_1samp_test
    """Statistical test applied across subjects"""
    # check input
    X = np.array(X)
    X = X[:, :, None] if X.ndim == 2 else X

    # stats function report p_value for each cluster
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
        X, out_type='mask',
        n_permutations=2**12, n_jobs=-1, verbose=False)

    # format p_values to get same dimensionality as X
    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval

    return np.squeeze(p_values_).T


def scorer_angle(y_true, y_pred, n_jobs=1):
    """Scoring function dedicated to AngularRegressor"""
    y_true, y_pred, shape = _check_y(y_true, y_pred)
    accuracy = _parallel_scorer(y_true, y_pred, _angle_accuracy, n_jobs)
    if (len(shape) > 1) and (np.sum(shape[1:]) > 1):
        accuracy = np.reshape(accuracy, shape[1:])
    else:
        accuracy = accuracy[0]
    return accuracy


def _check_y(y_true, y_pred):
    """Aux function to apply scorer across multiple dimensions."""
    # Reshape to get 2D
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert (len(y_pred) == len(y_pred))
    shape = y_pred.shape
    y_pred = np.reshape(y_pred, [shape[0], -1])
    y_true = np.squeeze(y_true)
    assert (y_true.ndim == 1)
    # remove nan values XXX non-adjacency need memory!
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        sel = np.where(~np.isnan(y_true[:, np.newaxis] + y_pred))[0]
        y_true = y_true[sel]
        y_pred = y_pred[sel, :]
    return y_true, y_pred, shape


def _parallel_scorer(y_true, y_pred, func, n_jobs=1):
    from mne.parallel import parallel_func, check_n_jobs
    # check dimensionality
    assert (y_true.ndim == 1)
    assert (y_pred.ndim == 2)
    # set jobs not > to n_chunk
    n_jobs = min(y_pred.shape[1], check_n_jobs(n_jobs))
    parallel, p_func, n_jobs = parallel_func(func, n_jobs)
    chunks = np.array_split(y_pred.transpose(), n_jobs)
    # run parallel
    out = parallel(p_func(chunk.T, y_true) for chunk in chunks)
    # gather data
    return np.concatenate(out, axis=0)


def _angle_accuracy(y_pred, y_true):
    from astropy.stats import circcorrcoef
    rhos = list()
    for ii in range(y_pred.shape[1]):
        rho = circcorrcoef(y_pred[:, ii], y_true)
        rhos.append(rho)
    rhos = np.array(rhos)
    return rhos


def create_bem_surf(subject, subjects_dir=None, overwrite=False):
    from mne.bem import make_watershed_bem
    # Set file name ----------------------------------------------------------
    bem_dir = op.join(subjects_dir, subject, 'bem')
    src_fname = op.join(bem_dir, subject + '-oct-6-src.fif')
    bem_fname = op.join(bem_dir, subject + '-5120-bem.fif')
    bem_sol_fname = op.join(bem_dir, subject + '-5120-bem-sol.fif')

    # Create watershed BEM surfaces
    if overwrite or not op.isfile(op.join(bem_dir, subject + '-head.fif')):
        make_watershed_bem(subject=subject, subjects_dir=subjects_dir,
                           overwrite=True, volume='T1', atlas=False,
                           gcaatlas=False, preflood=None, show=False)
    # Setup source space
    if overwrite or not op.isfile(src_fname):
        from mne import setup_source_space
        files = ['lh.white', 'rh.white', 'lh.sphere', 'rh.sphere']
        for fname in files:
            if not op.exists(op.join(subjects_dir, subject, 'surf', fname)):
                raise RuntimeError('missing: %s' % fname)

        src = setup_source_space(subject=subject, subjects_dir=subjects_dir,
                                 spacing='oct6', surface='white',
                                 add_dist=True, n_jobs=-1, verbose=None)
        src.save(src_fname, overwrite=True)
    # Prepare BEM model
    if overwrite or not op.exists(bem_sol_fname):
        from mne.bem import (make_bem_model, write_bem_surfaces,
                             make_bem_solution, write_bem_solution)
        # run with a single layer model (enough for MEG data)
        surfs = make_bem_model(subject, conductivity=[0.3],
                               subjects_dir=subjects_dir)
        write_bem_surfaces(fname=bem_fname, surfs=surfs)
        bem = make_bem_solution(surfs)
        write_bem_solution(fname=bem_sol_fname, bem=bem)


def read_hpi_mri(fname):
    landmark = dict()
    f = open(fname)
    text = [ln.strip('\n') for ln in f.readlines()]
    f.close()
    idx = 0
    while idx < len(text):
        line = text[idx]
        if line[:9] in ('# Created'):
            if line.split()[-1] == '2.3.10':
                version = '2.3.10'
            elif line.split()[-1] == '2.3.9':
                version = '2.3.9'
        idx += 1
    idx = 0
    while idx < len(text):
        line = text[idx]
        if line[:4] in ('NEC\t', 'LEC\t', 'REC\t'):
            if version == '2.3.9':
                code, _, _, _, x, y, z = line.split('\t')[:7]
            elif version == '2.3.10':
                code, _, _, x, y, z = line.split('\t')[:6]
            landmark[code] = [float(x), float(y), float(z)]
        if line[:5] in 'le\tSe':
            _, _, x, y, z = line.split('\t')[:5]
            landmark['lpa'] = [float(x), float(y), float(z)]
        elif line[:5] in 're\tSe':
            _, _, x, y, z = line.split('\t')[:5]
            landmark['rpa'] = [float(x), float(y), float(z)]
        elif line[:5] in 'rn\tSe':
            _, _, x, y, z = line.split('\t')[:5]
            landmark['nasion'] = [float(x), float(y), float(z)]
        idx += 1
    return landmark


def point_in_circle_single(target_ind, target_coords, feedbackX,
                    feedbackY, circle_radius):
    non_hit = list()
    d = math.sqrt(math.pow(target_coords[target_ind][0]-feedbackX, 2) +
                  math.pow(target_coords[target_ind][1]-feedbackY, 2))
    if d > circle_radius:
        non_hit = True
    else:
        non_hit = False
    return non_hit

# for each dim tell whether we hit target or not
def point_in_circle(targets, target_coords, feedbackX,
                    feedbackY, circle_radius, target_info_type = 'inds'):
    non_hit = list()
    for ii in range(len(targets)):
        if target_info_type == 'inds':
            tgtloc = target_coords[targets[ii]]
        elif target_info_type == 'locs':
            tgtloc = targets[ii]
        else:
            raise ValueError('Wrong target info type')
        d = math.sqrt(math.pow(tgtloc[0]-feedbackX[ii], 2) +
                      math.pow(tgtloc[1]-feedbackY[ii], 2))
        if d > circle_radius:
            non_hit.append(True)
        else:
            non_hit.append(False)
    return non_hit

#def ishit(targets, feeback, radius):
#    non_hit = list()
#    for ii in range(len(targets)):
#
#        if d > circle_radius:
#            non_hit.append(True)
#        else:
#            non_hit.append(False)
#    return non_hit


def init_target_positions():
    # height and width of the screen = 600
    # radius of the invisible boundary = 240
    targetAngs = [22.5+180, 67.5+180, 112.5+180, 157.5+180]
    target_types = []
    for x in range(0, len(targetAngs)):
        current = targetAngs[x]*(np.pi/180)
        target_types.append((int(round(600/2.0 +
                                       np.cos(current) * 240)),
                             int(round(600/2.0 +
                                       np.sin(current) * 240))))
    return target_types

def calc_target_coordinates_centered(target_angs):
    target_coords = list()
    for x in range(0, len(target_angs)):
        rad_ang = (target_angs[x]-(90*np.pi/180))
        target_coords.append([int(round(np.cos(rad_ang) * radius)),
                              int(round(np.sin(rad_ang) * radius))])
    return target_coords


def calc_rad_angle_from_coordinates(X, Y, radius_ = None):
    '''
    angle counting from bottom direction CCW (i.e. right)
    so 1,0 gives 90
    '''
    if radius_ is None:
        radius_cur = radius
    else:
        radius_cur = radius_

    angles = np.arctan2(Y/float(radius_cur),
                        X/float(radius_cur)) # [-pi,pi]
    # change the 0 angle (0 is now bottom vertical in the circle)
    angles = angles + np.pi/2. 
    # make the angle between 0 and np.pi

    c = angles < 0
    angles[c] = angles[c] + 2*np.pi
    c = angles > np.pi
    angles[c] = angles[c] - 2*np.pi

    #for i in np.where(angles < 0):
    #    angles[i] = angles[i] + 2*np.pi
    #for i in np.where(angles > np.pi):
    #    angles[i] = angles[i] - 2*np.pi
    return angles


class B2B(BaseEstimator):
    def __init__(self, G, H, n_splits=30, each_fit_is_parallel=True, n_jobs=None, random_state_split = None, ensemble = None):
        self.n_splits = n_splits
        self.G = G
        self.H = H
        if n_jobs is None:
            from config2 import n_jobs
        self.n_jobs = n_jobs
        self.each_fit_is_parallel = each_fit_is_parallel
        self.random_state_split = random_state_split
        self.ensemble = ensemble
        assert not ( ( self.random_state_split is not None ) and ( self.ensemble is not None )  )

    def fit(self, X, Y, verbose=True):
        from joblib import Parallel, delayed
        if self.ensemble is None:
            ensemble = ShuffleSplit(n_splits=self.n_splits, test_size=.5, random_state = self.random_state_split)
        else:
            ensemble = self.ensemble

        split = ensemble.split(X, Y)
        if self.each_fit_is_parallel:
            if verbose:
                print('B2B start NOT parallel (across splits)')
            r = []
            for G_set, H_set in split:
                r_ = _b2b_fit_one_split2( X,Y,G_set,H_set,self.G,self.H  )
                #Y_hat = self.G.fit(X[G_set], Y[G_set]).predict(X)
                #H_hat = self.H.fit(Y[H_set], Y_hat[H_set]).coef_
                #H_hats.append(H_hat)
                r.append(r_)
            rr = list( zip(*r) )
            H_hats = np.array(rr[0])
            self.E_ = np.mean(H_hats, 0)

            self.H_hats = H_hats
        else:
            if verbose:
                print(f'B2B start parallel (across splits) with {self.n_jobs} jobs')
            r = Parallel(n_jobs=self.n_jobs)(
                delayed(_b2b_fit_one_split2)( X,Y,G_set,H_set,self.G,self.H  ) \
                    for (G_set, H_set) in split)
            rr = zip(*r)
            H_hats = np.array(rr[0])
            self.E_ = np.mean(H_hats, 0)

            self.H_hats = H_hats

        return self

    def _fit_orig(self, X, Y):
        ensemble = ShuffleSplit(n_splits=self.n_splits, test_size=.5)
        H_hats = list()
        for G_set, H_set in ensemble.split(X, Y):
            Y_hat = self.G.fit(X[G_set], Y[G_set]).predict(X)
            H_hat = self.H.fit(Y[H_set], Y_hat[H_set]).coef_
            H_hats.append(H_hat)
        self.E_ = np.mean(H_hats, 0)
        return self

    def fit2(self, X, Y):
        # for decoding predictions
        self.coef_ = pinv(Y.T @ X) @ (Y.T @ Y)
        return self

    def predict(self, X):
        return X @ self.coef_

def _b2b_fit_one_split2(X,Y,G_set,H_set,G,H):
    import mne
    with mne.use_log_level('warning'):
        Y_hat = G.fit(X[G_set], Y[G_set]).predict(X)
        H_hat = H.fit(Y[H_set], Y_hat[H_set]).coef_
    return H_hat, G.coef_, G.intercept_, H.coef_, H_intercept_


# this one cycles over dims so within a dim it does not make sense to
# parallelize perhaps
def _b2b_fit_one_split(isplit, X,Y,G_set,H_set,G,H, verbose = 0):
    #print(f'_b2b_fit_one_split: isplit = {isplit};')
    #Y_hats = [0] * dim
    dim = Y.shape[1]
    Y_hats = np.zeros( Y.shape )
    # over all dims
    import mne
    
    with mne.use_log_level('warning'):
        # across dims, take corres Y dimension, fig G on X[G_set],y[G_set]
        if verbose:
            print(f'_b2b_fit_one_split: (isplit={isplit}) G_set = {G_set}\n')

        G_coefs = []
        G_intercepts = []

        # we are doing it separately per dim here because SPoC 
        # cannot take multidim y
        for dim_ind in range(dim):
            y = Y[:, dim_ind]
            Y_hat = G.fit(X[G_set], y[G_set]).predict(X)
                    
            from sklearn.pipeline import Pipeline
            if isinstance(G,Pipeline):
                ridge = G.steps[1][1]
            else:
                ridge = G
            G_coefs += [ridge.coef_]
            G_intercepts += [ridge.intercept_]

            Y_hats[:,dim_ind] = Y_hat
            if verbose:
                print(f'_b2b_fit_one_split: (isplit={isplit}, {dim_ind}) y[G_set] = {y[G_set]}\n')
        #Y_hats = np.array(Y_hats).T
        #Y_hats = Y_hats.T
        H_hat = H.fit(Y[H_set], Y_hats[H_set]).coef_
        if verbose:
            print(f'_b2b_fit_one_split: (isplit={isplit}) H_set = {H_set}\n')
            print(f'_b2b_fit_one_split: (isplit={isplit}) Y[H_set] = {Y[H_set]}\n')
            print(f'_b2b_fit_one_split: (isplit={isplit}) Yshape={Y.shape}, Yhatsshape ={Y_hats.shape} \n')
            print(f'_b2b_fit_one_split: (isplit={isplit}) H_hat={H_hat}\n')

    return H_hat, np.array(G_coefs), np.array(G_intercepts), H.coef_, H.intercept_


def _b2b_fit_one_split_back(isplit, Y, Yhat, H_set, H):
    return H.fit(Y[H_set], Yhat[H_set]).coef_

# this one cycles over dims so within a dim it does not make sense to
# parallelize perhaps
def _b2b_fit_one_split_one_dim(isplit,dimi,X,y,G_set,G):
    #Y_hats = [0] * dim
    #Y_hats = np.zeros( Y.shape )
    # over all dims
    #for dim_ind in range(dim):
    #y = Y[:, dim_ind]

    verbose = 0
    if verbose:
        print(f'_b2b_fit_one_split_one_dim: (isplit={isplit}, {dimi}) G_set = {G_set}\n')
        print(f'_b2b_fit_one_split_one_dim: (isplit={isplit}, {dimi}) y[G_set] = {y[G_set]}\n')

    verbose = 1

    import mne
    with mne.use_log_level('warning'):
        if verbose:
            print(f'Fit split {isplit}')
        y_hat = G.fit(X[G_set], y[G_set]).predict(X)
    #Y_hat = G.fit(X[G_set], y[G_set]).predict(X)
    #Y_hats[:,dim_ind] = Y_hat
    #Y_hats = np.array(Y_hats).T
    #Y_hats = Y_hats.T

    #H_hat = H.fit(Y[H_set], Y_hats[H_set]).coef_

    return isplit,dimi,y_hat

def _b2b_fit_one_split_one_dim_back(isplit, Y,Y_hats,H_set,H):
    import mne
    verbose = 0
    if verbose:
        print(f'_b2b_fit_one_split_one_dim_back: (isplit={isplit}) H_set = {H_set}\n')
        print(f'_b2b_fit_one_split_one_dim_back: (isplit={isplit}) Y[H_set] = {Y[H_set]}\n')
    with mne.use_log_level('warning'):
    #Y_hats = [0] * dim
    #Y_hats = np.zeros( Y.shape )
    # over all dims
    #for dim_ind in range(dim):
    #y = Y[:, dim_ind]
    #y_hat = G.fit(X[G_set], y[G_set]).predict(X)
    #Y_hat = G.fit(X[G_set], y[G_set]).predict(X)
    #Y_hats[:,dim_ind] = Y_hat
    #Y_hats = np.array(Y_hats).T
    #Y_hats = Y_hats.T

        H_hat = H.fit(Y[H_set], Y_hats[H_set]).coef_
        verbose = 0
        if verbose:
            print(f'_b2b_fit_one_split_one_dim_back: (isplit={isplit}) Yshape={Y.shape}, Yhatsshape ={Y_hats.shape} \n')
            print(f'_b2b_fit_one_split_one_dim_back: (isplit={isplit}) H_hat={H_hat}\n')

    return H_hat


class B2B_SPoC(BaseEstimator):
    def __init__(self, G, H, n_splits=30, parallel_type = 'across_splits_and_dims',
                 n_jobs=None, random_state_split = None, ensemble = None,
                 back_each_dim_sep = 0):
        # H is usually LinearRegression w/o intercept
        # G is usually  SPoC + Ridge
        self.n_splits = n_splits
        self.G = G
        self.H = H
        if n_jobs is None:
            from config2 import n_jobs
        self.n_jobs = n_jobs
        #self.each_fit_is_parallel = each_fit_is_parallel
        #self.parallel_type = 'across_splits'
        #self.
        assert parallel_type in ['no','across_splits','across_splits_and_dims']
        self.parallel_type = parallel_type

        self.random_state_split = random_state_split
        self.ensemble = ensemble
        self.parallel_backend = 'multiprocessing'
        self.back_each_dim_sep = back_each_dim_sep
        assert not ( ( self.random_state_split is not None ) and ( self.ensemble is not None )  )


    def fit(self, X, Y):
        assert X.shape[0] == Y.shape[0], (X.shape, Y.shape)
        import mne
        from joblib import Parallel, delayed
        if self.ensemble is None:
            ensemble = ShuffleSplit(n_splits=self.n_splits, test_size=.5, random_state = self.random_state_split)
        else:
            ensemble = self.ensemble

        dim = Y.shape[1]
        # 0. split data into two parts half-half: G_set and H_set
        # 1. using G (SPoC + Ridge) fit and
        #    predict G_set _on training data_, call it Y_hat
        #   1.1 do it separately for every variable Y[:,i]
        # 2. fit with H (linear reg w/o intercept) Y on Y_hat
        # 3. diag of this fit is said to be the outcome of the alg
        # NOTE!:  in the paper by J.R.King et al the X and Y are swapped
        # and they have Y being MEG signal (dep vars) and
        # X being behav vars (indep vars, putative causes)
        # "causal inﬂuences" matrix S of shape dimX x dimX
        # (square of putative causes)
        # Ghat -- lin reg coef from "Y to X" and calc Xhat = Y*Ghat
        # [in Romain's code called Y_hat]
        # S is unbiased, in the sense that it is centered around zero
        # when there is no effect, only if the second regression
        # H is not regularized.
        # NOT! in the paper S is binary (unless in case of additive noise)!

        #mne.use_log_level('warning')
        # cycle over splits
        #G is a regressor, H too
        split = ensemble.split(X, Y)
        if self.parallel_type == 'no':
            print('B2B_SPoC start NOT parllel (across splits)')
            H_hats = list()
            for isplit,(G_set, H_set) in enumerate(split):
                H_hat = _b2b_fit_one_split(isplit, X,Y,G_set,H_set,self.G,self.H)
                H_hats.append(H_hat)

            rr = list( zip(*r) )
            H_hats = rr[0]
            self.E_ = np.mean(H_hats, 0)

            self.H_hats = H_hats
        else:
            if self.parallel_type == 'across_splits':
                print(f'B2B_SPoC start parllel (across splits) with {self.n_jobs} jobs')
                r = Parallel(n_jobs=self.n_jobs, backend =self.parallel_backend)(
                    delayed(_b2b_fit_one_split)( isplit,X,Y,G_set,H_set,self.G,self.H  ) \
                        for isplit,(G_set, H_set) in enumerate(split) )

                rr = list( zip(*r) )
                H_hats = rr[0]
                H_hats = np.array(H_hats)
                self.E_ = np.mean(H_hats, 0)

                self.H_hats = H_hats
            elif self.parallel_type == 'across_splits_and_dims':
                import itertools
                #a=[1,2,3]
                #b=[4,5,6]
                split = list(split)
                splits_x_dims = list( itertools.product( enumerate(split), range(dim) ) )

                print(f'B2B_SPoC start parllel (across {len(splits_x_dims)} splits x dims) with {self.n_jobs} jobs')
                pl = Parallel(n_jobs=self.n_jobs, backend =self.parallel_backend)
                r = pl( delayed(_b2b_fit_one_split_one_dim)(isplit, dimi, X, Y[:,dimi],G_set,self.G  ) \
                        for ((isplit, (G_set, H_set_unused)), dimi) in splits_x_dims)
                #return r
                #rT = list( zip(*r) )
                #print('LEN (r) = ',len(r) )
                #y_hat_per_dim = len(r) * [ [] ]

                y_hat_per_dim = {}
                for isplit,dimi,y_hat in r:
                    #if dimi not in y_hat_per_dim:
                    y_hat_per_dim[ (isplit, dimi) ] = y_hat.T

                if self.back_each_dim_sep:

                   # for dimi,y_hats in enumerate(y_hat_per_dim):
                   #     # we want dim dim to be the last one
                   #     y_hat_per_dim[dimi] = np.array(y_hat_per_dim[dimi]).T

                    # now parallel across dims and other splits
                    H_hats = pl(
                        delayed(_b2b_fit_one_split_one_dim_back)\
                        (isplit, Y, y_hat_per_dim[isplit,dimi], H_set, self.H  ) \
                            for ( (isplit, (G_set_unused, H_set) ),dimi) in splits_x_dims)

                    H_hats = np.array(H_hats).reshape( ( len(split), dim, dim)  )
                else:
                    Y_hat_per_split = {}
                    for isplit in range(len(split)):
                        Y_hat_per_split[isplit] = []
                        for dimi in range(dim):
                            Y_hat_per_split[isplit] += [y_hat_per_dim[isplit,dimi]]
                            #print( Y_hat_per_split[isplit] ))
                        Y_hat_per_split[isplit] = np.vstack(Y_hat_per_split[isplit]).T
                        #print('Y_hat_per_split[isplit].shape', Y_hat_per_split[isplit].shape)
                        assert Y_hat_per_split[isplit].shape == Y.shape

                    H_hats = pl( delayed(_b2b_fit_one_split_back)\
                        (isplit, Y, Y_hat_per_split[isplit], H_set, self.H  ) \
                            for  (isplit, (G_set_unused, H_set) ) in enumerate(split) )

                self.E_ = np.mean(H_hats, 0)

                self.H_hats = H_hats

        #R_full = corr(Y_test * H, X_test * G)
        # Y_test cannot be H_set because H was fitted on it
        # it also cannot be G_set because G was fitted on it
        #R_full = corr(H.predict(Y_test), G.predict(X_test) )
        # it is pretty close to H_hat.coef_ perhaps 
        #   modulo different folds and if we assume that corr is well approximated by linear regression coefficient


        return self

    def fit2(self, X, Y):
        # for decoding predictions
        self.coef_ = pinv(Y.T @ X) @ (Y.T @ Y)
        return self

    def predict(self, X):
        return X @ self.coef_


def partial_reg(Y=None, X=None, n_ensemble=100, clf1=None, clf2=None):
    """JR Partial regression-Jean-Remi King.

    Finds E in:

        Y = F(EX+N1)+N2

    Where Y is the recordings (samples * sensors),
          F is an unknown forward operator (neurons * sensors),
          X is the factors (samples * sources),
          N1 and N2 are noise matrices (samples * sources/sensors)

    Proof for classification is not done yet.
    """
    ensemble = ShuffleSplit(n_splits=n_ensemble, test_size=.5)
    G = clf1
    H = clf2

    H_hats = list()
    for G_set, H_set in ensemble.split(Y, X):
        Y_hat = G.fit(X[G_set], Y[G_set]).predict(X)
        H_hat = H.fit(Y[H_set], Y_hat[H_set]).steps[-1][1].coef_
        H_hats.append(H_hat)
    # E_hat = np.diag(np.mean(H_hats, 0))
    E_hat = np.mean(H_hats, 0)
    return E_hat

def getGPUavail():
    try:
        import GPUtil
        GPUs_list = GPUtil.getAvailable()
    except:
        GPUs_list = []

    return GPUs_list

def getXGBparams(n_jobs=None, tree_method='gpu_hist'):
    if tree_method is None:
        from config2 import XGB_tree_method_def
    else:
        XGB_tree_method = tree_method
    assert XGB_tree_method in ['auto','approx','hist','exact','gpu_hist']
    # hist or gpu_hist are faster than 'approx' for large datasets.
    # approx is for intermediate size datasets
    # exact works better but slower so for small datasets
    if n_jobs is None:
        from config2 import n_jobs

    n_jobs_XGB=n_jobs

    method_params = {'tree_method': XGB_tree_method}

    if XGB_tree_method == 'gpu_hist':
        #XGB_tree_method = 'gpu_hist'
        GPUs_list = getGPUavail()
        if len(GPUs_list):
        #assert len(GPUs_list)
            method_params['gpu_id'] = GPUs_list[0]
        else:
            print('Empty GPU list avail, return to auto')
            method_params['tree_method'] = 'auto'



    XGB_min_child_weight = None
    XGB_max_depth = None
    XGB_eta = None    #0.3
    add_clf_creopts={ 'n_jobs':n_jobs_XGB,
         'importance_type': 'gain',
         'max_depth': XGB_max_depth,
         'min_child_weight': XGB_min_child_weight,
         'eta':XGB_eta, 'subsample': 1 }
    add_clf_creopts.update(method_params)
    return add_clf_creopts

def adjustTrajCoords(XY, tgtcur, home_position, params,
                     rot_origin='trajstart', auto_scale = 1,
                     shift_home = 1):
    '''
    tgtcur and XY are in screen coords
    '''
    from exper_protocol.utils import screen2homec
    fbXhc, fbYhc = screen2homec(XY[:,0], XY[:,1], home_position  )
    fbXYhc = np.array( [fbXhc, fbYhc], dtype=float )

    fb0pt = np.array( [fbXhc[0], fbYhc[0] ] , dtype=float)
    if rot_origin == 'trajstart':
        origin = fb0pt
    elif rot_origin == 'home':
        origin = np.zeros(2)
    else:
        raise ValueError('wrong')
    tgtcur_adj = tgtcur - origin

    d = np.linalg.norm( tgtcur - fb0pt  )
    d0 = float(params['dist_tgt_from_home'])

    ang_tgt = np.math.atan2(*tuple(tgtcur_adj) )

    fbXYhc_ta = rot( *fbXYhc, ang_tgt, origin )
    #fbXhc_ta, fbYhc_ta = fbXYhc_ta

    tgtcur_ta = rot( *tgtcur, ang_tgt, origin )

    rh = float( params['radius_home'])
    dirx = rh * np.sin(ang_tgt) + origin[0]
    diry = rh * np.cos(ang_tgt) + origin[1]
    dirtgt0 = np.array( [dirx,diry] )

    dirtgt_ta = rot(*dirtgt0, ang_tgt, origin) #- origin[0],0
    #print('dirtgt_ta = ',dirtgt_ta, dirtgt0, ang_tgt, origin)
    #return 0
    dirtgt_ta[1] = np.sqrt( rh**2 - dirtgt_ta[0]**2 )

    ds = np.sqrt( fbXhc**2 + fbYhc**2 )
    leave_home_coef = 1.
    inds_leavehome = np.where(ds > rh * leave_home_coef )[0]

    if len(inds_leavehome) > 1:
        ind_first_lh  = inds_leavehome[0]  # first index when fb traj is outside home
        curveX, curveY   = fbXYhc_ta

        if shift_home:
            curveX,  curveY   = fbXYhc_ta  - origin[:,None]
            dirtgt_ta -= origin
            tgtcur_ta -= origin

        if auto_scale:
            if shift_home:
                curveX *= d0 / d
                curveY *= d0 / d
            else:
                # I'd need to shfit before rescalign and then shift back
                raise ValueError('not implemented')

        fbXYlh_ta = fbXYhc_ta[:,ind_first_lh]
    else:
        curveX, curveY = None,None
        ind_first_lh = None

    return curveX, curveY, ind_first_lh, dirtgt_ta, tgtcur_ta


def areaBetween(xs,ys, xs2, ys2, start ,end):
    import shapely
    assert shapely.__version__ == "2.0.1"
    from shapely.algorithms.cga import signed_area
    from shapely.geometry import LineString
    if end is None:
        endx_ = []
        endy_ = []
    else:
        endx_ = list( end[:1] )
        endy_ = list( end[1:] )
    xs = np.hstack( [start[0] ] + list( xs ) + endx_ + list(xs2[::-1] )  + [ start[0] ] )
    ys = np.hstack( [start[1] ] + list( ys ) + endy_ + list(ys2[::-1] )  + [ start[1] ] )
    lr = LineString(np.c_[xs,ys])

    #display(lr)

    #mp = shapely.validation.make_valid(shapely.geometry.Polygon(np.c_[xs, ys]))
    #return mp.area
    return signed_area(lr)

def areaOne(xs,ys,start,end):
    a = areaBetween(xs,ys,[],[], start,end)
    return a

def calcNormCoefSectorArea(params):
    # 0.5 r**2 theta
    full_sector_area = 0.5 * float(params['target_location_spread']) *(np.pi / 180) *\
            float(params['dist_tgt_from_home'])**2
    norm_coef = 10. / full_sector_area
    return norm_coef

def areaBetweenTraj(dftraj1, dftraj2, home_position,
                    target_coords, params, invalid_val = np.nan,
                    endpoint = 'last_point',
                    ax = None):
    #fbXY = dfcurtr[['feedbackX', 'feedbackY']].to_numpy()
    '''
    assumes all entires in dftrajs are valied (i.e. index 0 was stripped already
    '''
    import matplotlib.pyplot as plt
    from exper_protocol.utils import screen2homec
    target_coords_homec = screen2homec( *tuple(zip(*target_coords)), home_position  )
    txc,tyc = target_coords_homec

    rs = []
    for dftraj in [dftraj1, dftraj2]:
        if len(dftraj) == 0:
            return np.nan
        tgti = dftraj['tgti_to_show'].to_numpy()[0]
        tgtcur = np.array(  [ txc[tgti], tyc[tgti] ] )

        ofbXY = dftraj[['unpert_feedbackX', 'unpert_feedbackY']].to_numpy()
        r = adjustTrajCoords(ofbXY, tgtcur, home_position, params,
                        rot_origin='trajstart', auto_scale = 1,
                        shift_home = 1)
        rs += [r]
        if r[0] is None:
            return invalid_val

    xs, ys, ind_first_lh, tgtline_homec_crossing, \
        tgt = rs[0]
    xs = xs[ind_first_lh:]
    ys = ys[ind_first_lh:]

    xs2, ys2, ind_first_lh2, tgtline_homec_crossing2, \
        tgt2 = rs[1]
    xs2 = xs2[ind_first_lh2:]
    ys2 = ys2[ind_first_lh2:]

    #print(ys)
    #print(ys2)

    start = tgtline_homec_crossing
    if endpoint == 'target':
        end = tgt
    elif endpoint == 'last_point':
        end = None
    else:
        raise ValueError(f'endpoint = {endpoint} not impl')

    ab = areaBetween(xs,ys,xs2,ys2, start,end)

    ##################################################################
    ##########################  plotting  #########################
    ##################################################################

    if ax is not None:
        rt = float(params['radius_target'] )
        rh = float( params['radius_home'])
        xlim = (-170,170)
        crc = plt.Circle(tgt, rt, color='blue', lw=2, fill=False,
                         alpha=0.6)
        ax.add_patch(crc)


        ##################
        # mark black first exit point
        #ax.scatter( [fbXYlh_ta[0]], [fbXYlh_ta[1]] , alpha=0.8, c='k', s= 10)

        # aligned ofb
        ax.scatter(xs , ys, alpha=0.4,
                   label='ofb ta (homec)',
                   marker='*', s = 30, c='brown')

        # aligned ofb
        ax.scatter(xs2 , ys2 , alpha=0.4,
                   label='ofb2 ta (homec)',
                   marker='*', s = 30, c='green')

        r = zip( tgtline_homec_crossing , tgt)
        ax.plot( *r, ls=':', label='tgtpath ta')

        #################

        #ax.scatter( *list(zip(dirtgt0)) , alpha=0.8, c='k', s= 10,
        #           marker = 'x', label='dirtgt0')

        #ax.scatter( *list(zip(dirtgt) ) , alpha=0.8, c='k', s= 24,
        #           marker = '+', label='dirtgt shiftscaled')

        #ax.scatter( *list(zip(dirtgt_ta)) , alpha=0.8, c='k', s= 10,
        #           marker = '*', label='dirtgt_ta')

        ############
        crc = plt.Circle((0, 0), rh, color='r', lw=2, fill=False,
                         alpha=0.6)
        ax.add_patch(crc)

        #################

        vft1  = dftraj1['vis_feedback_type'].to_numpy()[0]
        ti1   = dftraj1['trial_index'].to_numpy()[0]
        tgti1 = dftraj1['tgti_to_show'].to_numpy()[0]

        vft2  = dftraj2['vis_feedback_type'].to_numpy()[0]
        ti2   = dftraj2['trial_index'].to_numpy()[0]
        tgti2 = dftraj2['tgti_to_show'].to_numpy()[0]
        #td = time_lh - dfcurtr["time"].to_numpy()[0]

        #s = '\n'
        #if addinfo is not None:
        #    r = addinfo
        #    for cols_ in titlecols:
        #        for col in cols_:
        #            s += f'{col}='
        #            colv = r[col]
        #            if isinstance(colv,float):
        #                s += f'{colv:.2f}'
        #            else:
        #                s += f'{colv}'
        #            s+='; '
        #        s += '\n'
        #    #s = f'\nerror={r["error_endpoint_ang"]:.1f}; trialwb={r["trialwb"]}; tt={r["trial_type"]}'

        ttl =  f'ti={ti1}; vft={vft1}; tgti={tgti1}\n'
        ttl += f'ti={ti2}; vft={vft2}; tgti={tgti2}\n'
        ttl += f'area={ab:.2f}'
        ax.set_title(ttl)
        ax.legend(loc='lower left')
        ax.set_xlim(xlim)

    return ab

def rot(xs,ys, ang=20. * np.pi / 180., startpt =(0.,0.) ):
    # ang is in radians
    xs = np.array(xs, dtype = float) - startpt[0]
    ys = np.array(ys, dtype = float) - startpt[1]
    assert ang < np.pi + 1e-5, ang
    xs2 = xs * np.cos(ang) - ys * np.sin(ang)
    ys2 = xs * np.sin(ang) + ys * np.cos(ang)

    xs2 += startpt[0]
    ys2 += startpt[1]
    return np.array( [xs2, ys2])


############################## extracted from jr tools

def repeated_corr(X, y, dtype=float):
    """Computes pearson correlations between a vector and a matrix.

    Adapted from Jona-Sassenhagen's PR #L1772 on mne-python.

    Parameters
    ----------
        X : np.array, shape (n_samples, n_measures)
            Data matrix onto which the vector is correlated.
        y : np.array, shape (n_samples)
            Data vector.
        dtype : type, optional
            Data type used to compute correlation values to optimize memory.

    Returns
    -------
        rho : np.array, shape (n_measures)
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.ndim == 1:
        X = X[:, None]
    shape = X.shape
    X = np.reshape(X, [shape[0], -1])
    if X.ndim not in [1, 2] or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError('y must be a vector, and X a matrix with an equal'
                         'number of rows.')
    if X.ndim == 1:
        X = X[:, None]

    # subtract mean over samples 
    ym = np.array(y.mean(0), dtype=dtype)
    Xm = np.array(X.mean(0), dtype=dtype)
    # chane in place to save memory
    y -= ym
    X -= Xm

    y_sd = y.std(0, ddof=1)
    X_sd = X.std(0, ddof=1)[:, None if y.shape == X.shape else Ellipsis]
    # corr / (len * product of stds)
    R = (np.dot(y.T, X) / float(len(y) - 1)) / (y_sd * X_sd)
    R = np.reshape(R, shape[1:])

    # cleanup variable changed in place
    y += ym
    X += Xm
    return R

def repeated_spearman(X, y, dtype=None):
    """Computes spearman correlations between a vector and a matrix.

    Parameters
    ----------
        X : np.array, shape (n_samples, n_measures ...)
            Data matrix onto which the vector is correlated.
        y : np.array, shape (n_samples)
            Data vector.
        dtype : type, optional
            Data type used to compute correlation values to optimize memory.

    Returns
    -------
        rho : np.array, shape (n_measures)
    """
    from scipy.stats import rankdata
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.ndim == 1:
        X = X[:, None]
    shape = X.shape
    X = np.reshape(X, [shape[0], -1])
    if X.ndim not in [1, 2] or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError('y must be a vector, and X a matrix with an equal'
                         'number of rows.')

    # Rank (indices in sorted array, caring about multiplicity)
    X = np.apply_along_axis(rankdata, 0, X)
    y = np.apply_along_axis(rankdata, 0, y)
    # Double rank to ensure that normalization step of compute_corr
    # (X -= mean(X)) remains an integer.
    X *= 2
    y *= 2
    X = np.array(X, dtype=dtype)
    y = np.array(y, dtype=dtype)
    R = repeated_corr(X, y, dtype=type(y[0]))
    R = np.reshape(R, shape[1:])
    return R

def _parallel_scorer(y_true, y_pred, func, n_jobs=1):
    # split long y into part to evaluate in parallel

    #from nose.tools import assert_true
    from mne.parallel import parallel_func, _check_n_jobs
    # check dimensionality
    assert(y_true.ndim == 1)
    assert(y_pred.ndim == 2)
    # set jobs not > to n_chunk
    n_jobs = min(y_pred.shape[1], _check_n_jobs(n_jobs))
    parallel, p_func, n_jobs = parallel_func(func, n_jobs)
    chunks = np.array_split(y_pred.transpose(), n_jobs)
    # run parallel
    out = parallel(p_func(chunk.T, y_true) for chunk in chunks)
    # gather data
    return np.concatenate(out, axis=0)

def scorer_spearman(y_true, y_pred, n_jobs=1):
    #from jr.stats import repeated_spearman
    y_true, y_pred, shape = _check_y(y_true, y_pred)
    rho = _parallel_scorer(y_true, y_pred, repeated_spearman, n_jobs)
    if (len(shape) > 1) and (np.sum(shape[1:]) > 1):
        rho = np.reshape(rho, shape[1:])
    else:
        rho = rho[0]
    return rho

def rescaleIfNeeded(Xcur, Y, par, centering = False): 
    # X shape is trials x channels x time
    # Y shape is trials x variables
    # with centering=false it changes number of positive and negative
     
    onedim = False
    if Y.ndim == 1:
        onedim = True
        Y = Y[:,None]
    from sklearn.preprocessing import RobustScaler
    if par['scale_X_robust']:
        # X shape is trials x channels x time
        #Xcur_reshape = Xcur.transpose((0, 2, 1)).reshape(Xcur.shape[0], -1)

        # for RobustScaler we want samples x features and we want to rescale within channel

        # bring channels in the end and rescale trials x time
        Xcur_reshape0 = Xcur.transpose((0, 2, 1))  # to trials x time x channels  
        Xcur_reshape = Xcur_reshape0.reshape(-1, Xcur_reshape0.shape[2] )
        rscale = RobustScaler(with_centering=centering).fit(Xcur_reshape)
        Xcur_reshape = rscale.transform(Xcur_reshape)

        Xcur = Xcur_reshape.reshape(Xcur_reshape0.shape).transpose((0, 2, 1))

        #Xcur = Xcur_reshape.reshape(Xcur.shape).transpose((0, 2, 1))
        #Xcur = Xcur_reshape.reshape(Xcur.shape)

    if par['scale_Y_robust'] == 1:
        #if 'err_sens' in varnames:
        #    raise ValueError('need to be more careful when scaling err sens')
        rscale = RobustScaler(with_centering=centering).fit(Y)
        Y = rscale.transform(Y)
    elif par['scale_Y_robust'] == 2:
        from sklearn.preprocessing import scale
        # in Romain's orig code  centering = True (he does not specify it but this is the default value)
        Y = scale(Y, with_mean = centering)
    else:
        print('Not scaling Y at all')
    
    if onedim:
        Y = Y[:,0]

    return Xcur, Y

def pipeline2vars(ppl):
    from sklearn.pipeline import Pipeline
    if isinstance(ppl,list):
        r = []
        for ppl_ in ppl:
            r.append( pipeline2vars(ppl_) )
        return r
    else:
        r = []
        for pplel in ppl.steps:
            tpl = (pplel[0],vars(pplel[1]))
            r.append(tpl)
        return r

def subAngles(ang1, ang2):
    # angles should be in radians
    import pandas as pd
    if isinstance(ang1, pd.Series):
        ang1 = ang1.values
    if isinstance(ang2, pd.Series):
        ang2 = ang2.values
    r = np.exp(ang1 * 1j) * np.exp(-ang2 * 1j)
    return np.log(r).imag


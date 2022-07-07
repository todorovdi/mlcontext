import numpy as np
import os.path as op
# from nose.tools import assert_true
import math
from sklearn.base import BaseEstimator
from sklearn.model_selection import ShuffleSplit
from scipy.linalg import pinv
width = 800  # need to match the screen size during the task
height = 800
radius = int(round(height*0.5*0.8))
radius_target = 12
radius_cursor = 8

event_ids = [20, 21, 22, 23, 25, 26, 27, 28]

env2envcode = dict(stable=0, random=1)
env2subtr   = dict(stable=20, random=25)

target_angs = (np.array([157.5, 112.5, 67.5, 22.5]) + 90) * \
              (np.pi/180)


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


def decod_stats(X):
    from mne.stats import permutation_cluster_1samp_test
    """Statistical test applied across subjects"""
    # check input
    X = np.array(X)

    # stats function report p_value for each cluster
    # null = np.repeat(chance, len(times))
    # Non-corrected t-test...
    # T_obs, p_values_ = ttest_1samp(X, null, axis=0)
    T_obs_, clusters, p_values, _ = permutation_cluster_1samp_test(
        X, out_type='mask', n_permutations=2**10, n_jobs=6,
        verbose=False)

    # format p_values to get same dimensionality as X
    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster] = pval

    return np.squeeze(p_values_)


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
    text = [l.strip('\n') for l in f.readlines()]
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


# for each dim tell whether we hit target or not
def point_in_circle(targets, target_coords, feedbackX,
                    feedbackY, circle_radius):
    non_hit = list()
    for ii in range(len(targets)):
        d = math.sqrt(math.pow(target_coords[targets[ii]][0]-feedbackX[ii], 2) +
                      math.pow(target_coords[targets[ii]][1]-feedbackY[ii], 2))
        if d > circle_radius:
            non_hit.append(True)
        else:
            non_hit.append(False)
    return non_hit


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


def calc_rad_angle_from_coordinates(X, Y):
    angles = np.arctan2(Y/float(radius),
                        X/float(radius))
    # change the 0 angle (0 is now bottom vertical in the circle)
    angles = angles + np.pi/2.
    # make the angle between 0 and 2*np.pi
    for i in np.where(angles < 0):
        angles[i] = angles[i] + 2*np.pi
    for i in np.where(angles > 2*np.pi):
        angles[i] = angles[i] - 2*np.pi
    return angles


class B2B(BaseEstimator):
    def __init__(self, G, H, n_splits=30, each_fit_is_parallel=True, n_jobs=None):
        self.n_splits = n_splits
        self.G = G
        self.H = H
        if n_jobs is None:
            from config2 import n_jobs
        self.n_jobs = n_jobs
        self.each_fit_is_parallel = each_fit_is_parallel

    def fit(self, X, Y):
        from joblib import Parallel, delayed
        ensemble = ShuffleSplit(n_splits=self.n_splits, test_size=.5)

        split = ensemble.split(X, Y)
        if self.each_fit_is_parallel:
            print(f'B2B start NOT parllel (across splits)')
            H_hats = list()
            for G_set, H_set in split:
                H_hat = _b2b_fit_one_split2( X,Y,G_set,H_set,self.G,self.H  )
                #Y_hat = self.G.fit(X[G_set], Y[G_set]).predict(X)
                #H_hat = self.H.fit(Y[H_set], Y_hat[H_set]).coef_
                H_hats.append(H_hat)
            self.E_ = np.mean(H_hats, 0)
        else:
            print(f'B2B start parllel (across splits) with {self.n_jobs} jobs')
            H_hats = Parallel(n_jobs=self.n_jobs)(
                delayed(_b2b_fit_one_split2)( X,Y,G_set,H_set,self.G,self.H  ) \
                    for (G_set, H_set) in split)
            H_hats = np.array(H_hats)
            self.E_ = np.mean(H_hats, 0)

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
    return H_hat


# this one cycles over dims so within a dim it does not make sense to
# parallelize perhaps
def _b2b_fit_one_split(X,Y,G_set,H_set,G,H):
    #Y_hats = [0] * dim
    dim = Y.shape[1]
    Y_hats = np.zeros( Y.shape )
    # over all dims
    import mne
    with mne.use_log_level('warning'):
        for dim_ind in range(dim):
            y = Y[:, dim_ind]
            Y_hat = G.fit(X[G_set], y[G_set]).predict(X)
            Y_hats[:,dim_ind] = Y_hat
        #Y_hats = np.array(Y_hats).T
        #Y_hats = Y_hats.T
        H_hat = H.fit(Y[H_set], Y_hats[H_set]).coef_

    return H_hat


# this one cycles over dims so within a dim it does not make sense to
# parallelize perhaps
def _b2b_fit_one_split_one_dim(dimi,X,y,G_set,G):
    #Y_hats = [0] * dim
    #Y_hats = np.zeros( Y.shape )
    # over all dims
    #for dim_ind in range(dim):
    #y = Y[:, dim_ind]
    import mne
    with mne.use_log_level('warning'):
        y_hat = G.fit(X[G_set], y[G_set]).predict(X)
    #Y_hat = G.fit(X[G_set], y[G_set]).predict(X)
    #Y_hats[:,dim_ind] = Y_hat
    #Y_hats = np.array(Y_hats).T
    #Y_hats = Y_hats.T

    #H_hat = H.fit(Y[H_set], Y_hats[H_set]).coef_

    return dimi,y_hat

def _b2b_fit_one_split_one_dim_back(Y,Y_hats,H_set,H):
    import mne
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

    return H_hat


class B2B_SPoC(BaseEstimator):
    def __init__(self, G, H, n_splits=30, parallel_type = 'across_splits_and_dims',
                 n_jobs=None):
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


    def fit(self, X, Y):
        import mne
        from joblib import Parallel, delayed
        ensemble = ShuffleSplit(n_splits=self.n_splits, test_size=.5)
        dim = Y.shape[1]

        #mne.use_log_level('warning')
        # cycle over splits
        #G is a regressor, H too
        split = ensemble.split(X, Y)
        if self.parallel_type == 'no':
            print(f'B2B_SPoC start NOT parllel (across splits)')
            H_hats = list()
            for G_set, H_set in split:
                H_hat = _b2b_fit_one_split(X,Y,G_set,H_set,self.G,self.H)
                H_hats.append(H_hat)
            self.E_ = np.mean(H_hats, 0)
        else:
            if self.parallel_type == 'across_splits':
                print(f'B2B_SPoC start parllel (across splits) with {self.n_jobs} jobs')
                H_hats = Parallel(n_jobs=self.n_jobs)(
                    delayed(_b2b_fit_one_split)( X,Y,G_set,H_set,self.G,self.H  ) \
                        for (G_set, H_set) in split)
                H_hats = np.array(H_hats)
                self.E_ = np.mean(H_hats, 0)
            elif self.parallel_type == 'across_splits_and_dims':
                import itertools
                #a=[1,2,3]
                #b=[4,5,6]
                splits_x_dims = list( itertools.product( split, range(dim) ) )

                print(f'B2B_SPoC start parllel (across {len(splits_x_dims)} splits x dims) with {self.n_jobs} jobs')
                r = Parallel(n_jobs=self.n_jobs)(
                    delayed(_b2b_fit_one_split_one_dim)(dimi, X,Y[:,dimi],G_set,self.G  ) \
                        for ((G_set, H_set),dimi) in splits_x_dims)
                #return r
                #rT = list( zip(*r) )
                y_hat_per_dim = dim * [ [] ]
                for dimi,y_hat in r:
                    #if dimi not in y_hat_per_dim:
                    y_hat_per_dim[dimi] += [ y_hat ]

                for dimi,y_hats in enumerate(y_hat_per_dim):
                    # we want dim dim to be the last one
                    y_hat_per_dim[dimi] = np.array(y_hat_per_dim[dimi]).T

                # now parallel across dims and other splits
                H_hats = Parallel(n_jobs=self.n_jobs)(
                    delayed(_b2b_fit_one_split_one_dim_back)\
                    (Y,y_hat_per_dim[dimi],H_set,self.H  ) \
                        for ((G_set, H_set),dimi) in splits_x_dims)

                H_hats = np.array(H_hats)
                self.E_ = np.mean(H_hats, 0)

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
        import GPUtil
        GPUs_list = GPUtil.getAvailable()
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

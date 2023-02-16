import numpy as np

class Shadmehr_model:
    def __init__(self,ntrials,min_err = -10, max_err = 10, basis_len = 10,
                beta = 0.05, sigma = None, decay=1., err_sens_def=0.1,
                weights_def = None):
        #weights_def = 0.05
        # here everything is relative to the target. 
        # So we assume that it does not depend on the target        
        self.basis_err = np.linspace(min_err,max_err,basis_len)
        self.basis_len = len(self.basis_err)
        self.ntrials = ntrials
        
        self.err_sens = np.zeros(ntrials)
        #self.feedback = np.zeros((ntrials,2))
        #self.errors   = np.zeros((ntrials,2))
        self.pert_est = np.zeros(ntrials)
        self.feedback = np.zeros(ntrials)
        self.errors   = np.zeros(ntrials)
        self.weights  = np.zeros((ntrials,self.basis_len))
        
        # pert_est = pert_est env
        self.feedback_orig_prev = 0  # was participant _expects_ without perturbation
                                     # 0 because we did not move right before
        self.feedback_prev      = np.nan  # known to particpant
        self.pert_est_prev      = 0  # pert_est about pert, initially expect precise movement
        self.err_prev           = 0  # konwn to participant
        self.err_sens_prev      = err_sens_def  
        
        # params from Herzfield supp material v2 p7 (for fig 3B)
        # width or bell curve of error around basis error
        if sigma is None:
            self.sigma = (self.basis_err[1]-self.basis_err[0]) / basis_len 
        else:
            self.sigma = sigma
        # another possible value from Herzfeld is 0.001 for Fig. 3C
        self.beta  = beta # speed of update of the weights to compute err sens        
        self.decay = decay
        self.prev_trial_ind = -1  # last actualized


        weights_init_adjust_step = 0.001
        if weights_def is None:
            w = np.zeros(self.basis_len);
            while self.err2err_sens(w,0) < err_sens_def:
                w += weights_init_adjust_step; #% get sensitivities as close to initial sensitivity as possible
            self.weights_prev =  w
        else:
            self.weights_prev  = np.ones(self.basis_len) * weights_def

        print(f'weights_prev = {self.weights_prev}')


    # bold g (Herzfeld eq 3)
    def transfer_fun_vec(self,err):
        r = np.zeros(self.basis_len)
        for i in range(self.basis_len):
            r[i] = self.basis_fun(i,err)
        return r

    def basis_fun(self,i,err):
        exparg = -(err - self.basis_err[i])**2 
        exparg /= (2 * (self.sigma**2) )
        r  = np.exp( exparg )
        return r

    def err2err_sens(self,weights,err):
        assert len(weights) == self.basis_len
        r = 0
        for i in range(self.basis_len):
            r += weights[i] * self.basis_fun(i,err)
        return r
    
    def update(self, feedback_new):    
        # here error is more like discrepancy, not the reaching error
        # self.feedback_orig_prev is unknown, to be fitted
        if self.prev_trial_ind == -1:
            #feedback_predict_prev = feedback_new
            feedback_predict_prev = 0
        else:
            feedback_predict_prev = self.feedback_orig_prev + self.pert_est_prev
        # error = difference between truth and prediction
        err_new = feedback_new - feedback_predict_prev        
        # read_behave2: 'belief' in Romain == orig_feedback (unk to particpant) - target
        # Herzfeld sm p4 bottom pert_est = \hat{y} = u + \hat{x}
        
        # update weights (Herzfeld eq 3)
        for i in range(self.basis_len):
            bold_g = self.transfer_fun_vec(self.err_prev)  # G 
            factor = bold_g / np.linalg.norm(bold_g,2)
            assert factor.shape == self.basis_err.shape
            weights_cur = self.weights_prev + \
                self.beta * np.sign(err_new * self.err_prev) * factor

        # (Herzfeld eq 2)
        err_sens_cur = self.err2err_sens(weights_cur, err_new)
        
        # (Herzfeld eq 1)
        pert_est_cur = self.decay * self.pert_est_prev + \
            self.err_sens_prev * err_new        
        
        # what I'll see on next trial
        self.prev_trial_ind += 1
        
        assert self.prev_trial_ind >= 0
        self.feedback[ self.prev_trial_ind ] = feedback_new
        self.errors[ self.prev_trial_ind ]   = err_new
        self.weights[ self.prev_trial_ind ]  = weights_cur
        self.err_sens[ self.prev_trial_ind ] = err_sens_cur
        self.pert_est[ self.prev_trial_ind ] = pert_est_cur
        
        self.feedback_prev = feedback_new
        self.weights_prev = weights_cur
        self.err_prev = err_new
        self.pert_est_prev = pert_est_cur
        self.err_sens_prev = err_sens_cur
        
        #self.err_prev_prev = self.err_prev
        #self.feedback_orig_prev = feedback - predict
        #self.feedback_predict_prev = feedback - self.pert_est_cur
        
        # next move (feedback_orig) should be like feedback_predict_prev - pert_est_prev
        
        #err_cur = feedback_cur - feedback_predict_cur
    def print_basic(self):
        me,mx = np.mean(self.weights_prev),np.max(self.weights_prev)
        print((f'prev: fb={self.feedback_prev:.3f}, err={self.err_prev:.4f},'
              f' pert_est={self.pert_est_prev:.4f}, err_sens={self.err_sens_prev:.8f}'
              f' wmax={mx:.4f}') )

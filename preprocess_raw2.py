import os
import os.path as op
import mne
from mne.io import read_raw_ctf
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
import matplotlib.pyplot as plt
from scipy.signal import detrend
import numpy as np
from config2 import path_data
import sys

#subject = sys.argv[1]

#hpass = '0.05'  # '0.1', '0.05, ''detrend', no_hpass, no_filter, no_hpass_tf
#### Dmitrii's change
#hpass = '0.1'  
####
with_ICA = True
task = 'VisuoMotor'
# images_folder = op.join(op.join(path_data, subject, 'preproc_images'))
# if not op.exists(images_folder):
#     os.mkdir(images_folder)

mne.cuda.init_cuda()

sub_files = os.listdir(op.join(path_data, subject))
fname = [file for file in sub_files if (task in file and '.ds' in file)][0]

raw = read_raw_ctf(op.join(path_data, subject, fname), preload=True)
# remove misc channels
ch_to_drop = np.array(raw.ch_names)[mne.pick_types(raw.info,
                                                   meg=False,
                                                   ref_meg=False,
                                                   misc=True)]
# keep the ADC and stim
ch_to_drop = np.delete(ch_to_drop,
                       np.where((ch_to_drop == 'UPPT001') |
                                (ch_to_drop == 'UADC003-2104') |
                                (ch_to_drop == 'UADC004-2104') |
                                (ch_to_drop == 'UADC009-2104') |
                                (ch_to_drop == 'UADC010-2104') |
                                (ch_to_drop == 'UADC013-2104') |
                                (ch_to_drop == 'UADC016-2104')))
raw.drop_channels(list(ch_to_drop))
# Remove bad channel 'MZF03-1609' for subject 15
if subject == 'sub15_TAMMXQQS':
    raw.drop_channels(['MZF03-1609'])

if with_ICA:
    filt_raw = raw.copy()
    raw_orig = raw.copy()
    filt_raw.notch_filter(np.arange(60, 241, 60), n_jobs=n_jobs)
    filt_raw.filter(l_freq=1, h_freq=None, n_jobs=n_jobs)

    # ICA
    ica = ICA(n_components=30)
    reject = dict(mag=5e-12, grad=4000e-13)  # from MNE website
    picks_meg = mne.pick_types(raw.info, meg='mag', misc=False,
                               stim=False, exclude='bads')
    ica.fit(filt_raw, reject=reject, decim=3)

    # Identify EOG artifact components
    eog_inds = []
    # eye tracker
    Xeye = filt_raw[:][0][mne.pick_channels(raw.ch_names, include=['UADC009-2104'])][0]
    Yeye = filt_raw[:][0][mne.pick_channels(raw.ch_names, include=['UADC010-2104'])][0]

    xcor = ica.score_sources(filt_raw, target=Xeye)
    ycor = ica.score_sources(filt_raw, target=Yeye)
    eog_inds.extend(i for i, cor in enumerate(xcor) if abs(cor) > 0.2)
    eog_inds.extend(i for i, cor in enumerate(ycor) if abs(cor) > 0.2)
    # Identify ECG artifact components
    ecg_inds, ecg_scores = ica.find_bads_ecg(filt_raw, method='correlation')
    # create new raw data without bad components and save it
    exclude = list(set(eog_inds + ecg_inds))
    raw = ica.apply(raw, exclude=exclude)
# Filter the clean raw data
raw.notch_filter(np.arange(60, 241, 60), n_jobs=n_jobs)
if hpass == 'no_hpass':
    raw.filter(l_freq=None, h_freq=30, n_jobs=n_jobs)
elif hpass == 'no_hpass_tf':
    raw.filter(l_freq=None, h_freq=150, n_jobs=n_jobs)
elif hpass == 'no_filter':
    pass
else:
    # 30 -- just to smooth
    raw.filter(l_freq=float(hpass), h_freq=30, n_jobs=n_jobs)
if with_ICA:
    fname = op.join(path_data, subject, f'raw_{task}_{hpass}_with_ICA.fif')
else:
    fname = op.join(path_data, subject, f'raw_{task}_{hpass}.fif' )
raw.save(fname, overwrite=True)

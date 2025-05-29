#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infer state spectra.

@author: sebastiancoleman
"""

import os
import os.path as op
from osl_dynamics.data import Data
from osl_dynamics.models import load
from osl_dynamics.analysis import spectral
from osl_dynamics.inference import modes
import pickle
import numpy as np
from glob import glob

#%% set up paths

data_path = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/data'
deriv_path = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/derivatives'
subjects_dir = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/subjects_dir'
output_dir = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/dynemo/6_modes'

#%% get list of subjects

dir_list = sorted(glob(op.join(deriv_path, '*')))
dir_list = [entry for entry in dir_list if op.isdir(entry)]
subjects = sorted([op.basename(d) for d in dir_list])
subjects = [sub for sub in subjects if not sub=='bad']

#%% load trained model

model = load(op.join(output_dir, "model"))

#%% load data

files = []
for s, subject in enumerate(subjects):
    print(subject)
    fname = op.join(deriv_path, subject, subject + '_prepped-raw.fif')
    files.append(fname)
    
# load data
data = Data(files, sampling_frequency=250)
print(data)

# trim data
trimmed_data = data.trim_time_series(n_embeddings=15, sequence_length=100)

# get alpha
alpha = pickle.load(open(op.join(output_dir, "inf_params", "alp_rw.pkl"), "rb"))

# Calculate regression spectra for each mode and subject (will take a few minutes)
f, psd, coh, w = spectral.regression_spectra(
    data=trimmed_data,
    alpha=alpha,
    sampling_frequency=250,
    frequency_range=[1, 45],
    window_length=1000,
    step_size=20,
    n_sub_windows=8,
    return_coef_int=True,
    return_weights=True,
    n_jobs=8,
)

# re-scale psd
psd_rs = spectral.rescale_regression_coefs(psd, alpha, window_length=1000, 
                                           step_size=20, n_sub_windows=8)

os.makedirs(op.join(output_dir, "spectra"), exist_ok=True)
np.save(op.join(output_dir, "spectra", "f.npy"), f)
np.save(op.join(output_dir, "spectra", "psd.npy"), psd)
np.save(op.join(output_dir, "spectra", "psd_rs.npy"), psd_rs)
np.save(op.join(output_dir, "spectra", "coh.npy"), coh)
np.save(op.join(output_dir, "spectra", "w.npy"), w)
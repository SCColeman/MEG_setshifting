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

# paths
root = r"/d/gmi/1/sebastiancoleman/MEG_setshifting"
data_dir = op.join(root, 'dynemo', 'data')
output_dir = op.join(root, 'dynemo', '6_modes')

# load trained model
model = load(op.join(output_dir, "model"))

# load data
data = Data(data_dir)
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
    frequency_range=[3, 45],
    window_length=1000,
    step_size=20,
    n_sub_windows=8,
    return_coef_int=True,
    return_weights=True,
    n_jobs=16,
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
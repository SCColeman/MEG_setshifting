#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sign flipping and save into a folder.

@author: sebastiancoleman
"""

import mne
import numpy as np
from matplotlib import pyplot as plt
import os.path as op
import glob
import pandas as pd
from glob import glob
from scipy.stats import zscore
from osl_dynamics.data import Data

#%% set up paths

data_path = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/data'
deriv_path = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/derivatives'
subjects_dir = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/subjects_dir'
output_dir = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/dynemo'

#%% get list of subjects

dir_list = sorted(glob(op.join(deriv_path, '*')))
dir_list = [entry for entry in dir_list if op.isdir(entry)]
subjects = [op.basename(d) for d in dir_list]
subjects = [sub for sub in subjects if not sub=='bad']

#%% get all data

all_data = []
for s, subject in enumerate(subjects):
    print(subject)
    fname = op.join(deriv_path, subject, subject + '_source_orth-raw.fif')
    raw = mne.io.Raw(fname, preload=True, verbose=False)
    
    # store out data
    data = raw.get_data(reject_by_annotation='omit', verbose=False)
    all_data.append(data.T)
    
#%% sign flip

data = Data(all_data, sampling_frequency=250, n_jobs=16)
print(data)

# prepare
methods = {"standardize": {},
           "align_channel_signs": {}}
data.prepare(methods)
print(data)

#%% save

data_prepped = data.time_series()
for s, subject in enumerate(subjects):
    fname = op.join(deriv_path, subject, subject + '_source_orth-raw.fif')
    raw = mne.io.Raw(fname, preload=True, verbose=False)
    raw = mne.io.RawArray(data_prepped[s].T, raw.info)
    raw.save(op.join(deriv_path, subject, subject + '_prepped-raw.fif'))




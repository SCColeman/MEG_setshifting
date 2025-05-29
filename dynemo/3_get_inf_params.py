#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use trained model to predict the mode timecourses.

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

#%% prepare

methods = {"tde_pca": {"n_embeddings": 15, "n_pca_components": 90},
           "standardize": {}}
data.prepare(methods)
print(data)

#%% get parameters

alpha = model.get_alpha(data)
os.makedirs(op.join(output_dir, "inf_params"), exist_ok=True)
pickle.dump(alpha, open(op.join(output_dir, "inf_params", "alp.pkl"), "wb"))

# get covs and fix alpha
means, covs = model.get_means_covariances()
alpha_rw = modes.reweight_alphas(alpha, covs)
np.save(op.join(output_dir, "inf_params", "covs.npy"), covs)
pickle.dump(alpha_rw, open(op.join(output_dir, "inf_params", "alp_rw.pkl"), "wb"))

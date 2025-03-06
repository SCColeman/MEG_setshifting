#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train DyNeMo on prepared data.

@author: sebastiancoleman
"""

import os.path as op
from osl_dynamics.data import Data
from osl_dynamics.models.dynemo import Config
from osl_dynamics.models.dynemo import Model
from glob import glob

# paths
root = r"/d/gmi/1/sebastiancoleman/MEG_setshifting"
data_dir = op.join(root, 'dynemo', 'data')
output_dir = op.join(root, 'dynemo', '6_modes')
files = sorted(glob(op.join(data_dir, '*')))

# load data
data = Data(files, sampling_frequency=250, n_jobs=16)
print(data)

# prepare
methods = {"tde_pca": {"n_embeddings": 15, "n_pca_components": 90},
           "standardize": {}}
data.prepare(methods)
print(data)

# set training parameters
config = Config(
    n_modes=6,
    n_channels=data.n_channels,
    sequence_length=100,
    inference_n_units=64,
    inference_normalization="layer",
    model_n_units=64,
    model_normalization="layer",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=False,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=5,
    n_kl_annealing_epochs=15,
    batch_size=32,
    learning_rate=0.001,
    n_epochs=30,
)

model = Model(config)
model.summary()

# initialisation
init_history = model.random_subset_initialization(data, n_epochs=1, n_init=3, take=0.2)

# full training
history = model.fit(data)

# save model
model.save(op.join(output_dir, "model"))
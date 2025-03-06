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
import scipy
from scipy.stats import pearsonr, zscore, kurtosis
import glob
from nilearn import image, datasets, plotting
from nibabel import nifti1, Nifti1Image
import pandas as pd
from glob import glob

#%% functions

def isoutlier(data, thresh=5):
    from scipy.stats import iqr
    data = np.asarray(data)
    q75, q25 = np.percentile(data, [75 ,25])
    iqr_value = iqr(data)
    lower_bound = q25 - thresh * iqr_value
    upper_bound = q75 + thresh * iqr_value
    outliers = data > upper_bound
    return outliers

#%% set up paths

data_path = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/data'
deriv_path = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/derivatives'
subjects_dir = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/subjects_dir'
output_dir = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/dynemo'

#%% load atlas 

atlas_file = '/d/gmi/1/sebastiancoleman/atlases/MEG_atlas_38_regions_3D.nii.gz'
atlas = image.load_img(atlas_file)
coords = np.load('/d/gmi/1/sebastiancoleman/atlases/MEG_atlas_38_regions_coords.npy')
names = np.squeeze(pd.read_csv('/d/gmi/1/sebastiancoleman/atlases/MEG_atlas_38_regions_names.csv', header=None).to_numpy())

#%% get list of subjects

dir_list = sorted(glob(op.join(deriv_path, '*')))
dir_list = [entry for entry in dir_list if op.isdir(entry)]
subjects = [op.basename(d) for d in dir_list]
subjects = [sub for sub in subjects if not sub=='bad']

#%% calculate covariance of all data

i, j = np.triu_indices(len(names), 1)
all_covs = []
for s, subject in enumerate(subjects):
    print(subject)
    fname = op.join(deriv_path, subject, subject + '_source_orth-raw.fif')
    raw = mne.io.Raw(fname, preload=True, verbose=False)
    
    # annotate bad segments (treat as resting state)
    data = raw.get_data()
    segment_len = int(1 * raw.info['sfreq'])
    variances = np.array([np.mean(np.var(data[:, i:i+segment_len], axis=1)) for i in np.arange(0, data.shape[1]-segment_len+1, segment_len)])
    outliers = isoutlier(variances, thresh=3)
    annot = raw.annotations
    for ii, ind in enumerate(np.arange(0, data.shape[1]-segment_len, segment_len)):
        if outliers[ii]:
            onset = raw.times[ind]
            duration = segment_len * (1/raw.info['sfreq'])
            description = 'BAD_var'
            annot += mne.Annotations(onset, duration, description, orig_time=annot.orig_time)
    raw.set_annotations(annot, verbose=False)
    
    data = raw.get_data(reject_by_annotation='omit', verbose=False)
    cov = np.cov(data)
    all_covs.append(cov[i,j])
    
#%% find median cov

all_covs_cat = np.array(all_covs)
all_covs_corr = np.corrcoef(all_covs_cat)
corr_strength = np.mean(all_covs_corr, 0)
med_cov = np.argmin(np.abs(corr_strength - np.median(corr_strength)))
template = all_covs[med_cov]

#%% calculate flips

all_flips = []
for c, cov in enumerate(all_covs):
    flips = []
    for e in range(len(template)):
        unflipped = np.abs(template[e] - cov[e])
        flipped = np.abs(template[e] + cov[e])
        if flipped < unflipped:
            flips.append(-1)
        else:
            flips.append(1)
    flips = np.array(flips)
    flips_mat = np.zeros((len(names),len(names)))
    flips_mat[i,j] = flips
    flips_mat[j,i] = flips
    flips_mat[np.diag_indices(len(names))] = np.nan
    flips_vec = np.nanmean(flips_mat, 1)
    flips_vec[flips_vec < 0] = -1
    flips_vec[flips_vec > 0] = 1
    all_flips.append(flips_vec)
    
#%% apply flips and save out data

i, j = np.triu_indices(len(names), 1)
for s, subject in enumerate(subjects):
    print(subject)
    fname = op.join(deriv_path, subject, subject + '_source_orth-raw.fif')
    raw = mne.io.Raw(fname, preload=True, verbose=False)
    
    # annotate bad segments (treat as resting state)
    data = raw.get_data()
    segment_len = int(1 * raw.info['sfreq'])
    variances = np.array([np.mean(np.var(data[:, i:i+segment_len], axis=1)) for i in np.arange(0, data.shape[1]-segment_len+1, segment_len)])
    outliers = isoutlier(variances, thresh=3)
    annot = raw.annotations
    for ii, ind in enumerate(np.arange(0, data.shape[1]-segment_len, segment_len)):
        if outliers[ii]:
            onset = raw.times[ind]
            duration = segment_len * (1/raw.info['sfreq'])
            description = 'BAD_segment'
            annot += mne.Annotations(onset, duration, description, orig_time=annot.orig_time)
    raw.set_annotations(annot, verbose=False)
    
    # save bad mask
    bads = raw.get_data(reject_by_annotation='NaN')[0,:]
    bads = np.isnan(bads)
    np.save(op.join(output_dir, 'bads', subject + '_bad_mask.npy'), bads)
    
    data = raw.get_data(reject_by_annotation='omit', verbose=False)
    data = zscore(data,1)
    data_flipped = data.T * all_flips[s]
    raw = mne.io.RawArray(data_flipped.T, raw.info)
    raw.save(op.join(output_dir, 'data', subject + '_prepped-raw.fif'))

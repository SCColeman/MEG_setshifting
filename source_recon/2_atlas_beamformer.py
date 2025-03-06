#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast and simple MEG source reconstruction. Convert MEG data from 
sensor space to atlas space.

Note this script treats the data as if it were resting state, i.e., no epoching
prior to source reconstruction. This is necessary for HMM/DyNeMo.

Script assumes you already have a subjects_dir with the required FreeSurfer
outputs in subjects_dir, along with a saved coreg file (-trans.fif).

@author: Sebastian C. Coleman, sebastian.coleman@sickkids.ca
"""

import mne
from mne_connectivity import symmetric_orth
import os.path as op
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import zscore
from nilearn import plotting, datasets, image
import nibabel as nib
from scipy.ndimage import uniform_filter1d, label
import pandas as pd

#%% functions

def make_4d_atlas_nifti(atlas_img, values):

    # load fsaverage and atlas   
    mni = datasets.load_mni152_template()
    atlas_data = atlas_img.get_fdata()

    # place values in each parcel region
    regs = []
    for reg in range(atlas_data.shape[-1]):
        atlas_reg = atlas_data[:,:,:,reg]
        atlas_reg[atlas_reg>0] = 1
        regs.append(atlas_reg * values[reg])
    atlas_new = np.sum(regs, 0)

    # make image from new atlas data
    new_img = nib.Nifti1Image(atlas_new, atlas_img.affine, atlas_img.header)
    
    # interpolate image
    img_interp = image.resample_img(new_img, mni.affine)
    
    return img_interp

def surface_brain_plot(img, subjects_dir, surf='inflated', cmap='cold_hot', symmetric=True, 
                       threshold=0, fade=True, cbar_label=None, figsize=(10,7)):
    
    # some package imports inside function - ignore my bad practice
    from nilearn import surface
    import matplotlib as mpl
    
    # make MNE stc out of nifti
    lh_surf = op.join(subjects_dir, 'fsaverage', 'surf', 'lh.pial')
    lh = surface.vol_to_surf(img, lh_surf)
    rh_surf = op.join(subjects_dir, 'fsaverage', 'surf', 'rh.pial')
    rh = surface.vol_to_surf(img, rh_surf)
    data = np.hstack([lh, rh])
    vertices = [np.arange(len(lh)), np.arange(len(rh))]
    stc = mne.SourceEstimate(data, vertices, tmin=0, tstep=1)

    # set up axes
    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_axes([0, 0.60, 0.35, 0.35])  # top-left
    ax2 = fig.add_axes([0.65, 0.60, 0.35, 0.35])  # top-right
    ax3 = fig.add_axes([0.0, 0.15, 0.35, 0.35])  # bottom-left
    ax4 = fig.add_axes([0.65, 0.15, 0.35, 0.35])  # bottom-right
    ax5 = fig.add_axes([0.32, 0.3, 0.36, 0.5])  # center 
    cax = fig.add_axes([0.25, 0.1, 0.5, 0.03]) # colorbar ax
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_facecolor('none')
        ax.axis(False)
        
    # set up threshold
    if symmetric:
        vmax = np.max(np.abs(data))
        vmin = -vmax
        mid = threshold + ((vmax-threshold)/2)
        if fade:
            clim = {'kind': 'value', 'pos_lims':(threshold, mid, vmax)}
        else:
            clim = {'kind': 'value', 'pos_lims':(threshold, threshold, vmax)}
    else:
        vmax = np.max(data)
        vmin = np.min(data)
        mid = threshold + ((vmax-threshold)/3)
        if fade:
            clim = {'kind': 'value', 'lims':(threshold, mid, vmax)}
        else:
            clim = {'kind': 'value', 'lims':(threshold, threshold, vmax)}
        
    if surf=='inflated':
        cortex='low_contrast'
    elif surf=='pial':
        cortex=(0.6, 0.6, 0.6)
    else:
        cortex=(0.6, 0.6, 0.6)
    plot_kwargs = dict(subject='fsaverage',
                       subjects_dir=subjects_dir,
                       surface=surf,
                       cortex=cortex,
                       background='white',
                       colorbar=False,
                       time_label=None,
                       time_viewer=False,
                       transparent=True,
                       clim=clim,
                       colormap=cmap,
                       )
    
    def remove_white_space(imdata):
        nonwhite_pix = (imdata != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        imdata_cropped = imdata[nonwhite_row][:, nonwhite_col]
        return imdata_cropped

    # top left
    views = ['lat']
    hemi = 'lh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax1.imshow(screenshot)

    # top right
    views = ['lat']
    hemi = 'rh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax2.imshow(screenshot)

    # bottom left
    views = ['med']
    hemi = 'lh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax3.imshow(screenshot)

    # bottom right
    views = ['med']
    hemi = 'rh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax4.imshow(screenshot)

    # middle
    views = ['dorsal']
    hemi = 'both'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    background = np.sum(screenshot, -1) == 3*255
    alpha = np.ones(screenshot.shape[:2])  
    alpha[background] = 0
    ax5.imshow(screenshot, alpha=alpha)

    # colorbar
    cmap = plt.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='horizontal', label=cbar_label)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cbar_label, fontsize=16, labelpad=0)
    
    return fig

def finish_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(alpha=0.3)
    
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

#%% load atlas 

atlas_file = '/d/gmi/1/sebastiancoleman/atlases/MEG_atlas_38_regions_4D.nii.gz'
atlas = image.load_img(atlas_file)
coords = np.load('/d/gmi/1/sebastiancoleman/atlases/MEG_atlas_38_regions_coords.npy')
names = np.squeeze(pd.read_csv('/d/gmi/1/sebastiancoleman/atlases/MEG_atlas_38_regions_names.csv', header=None).to_numpy())

#%% get list of subjects

dir_list = sorted(glob(op.join(deriv_path, '*')))
dir_list = [entry for entry in dir_list if op.isdir(entry)]
subjects = [op.basename(d) for d in dir_list]

#%% quick beamformer

for s, subject in enumerate(subjects):
    
    ### PREPROCESSING ###
    
    # load raw unprocessed data
    fname = glob(op.join(data_path, subject, '*-raw.fif'))[0]
    raw = mne.io.Raw(fname, preload=True)
    
    # get head position coil data
    chpi_locs = mne.chpi.extract_chpi_locs_ctf(raw)
    head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=False)
    head_xyz = head_pos[:,1:4]*100 - head_pos[0,1:4]*100  
    times = head_pos[:,0]
    
    # plot head position and rotation
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(times, head_xyz[:,0], label='x')
    ax.plot(times, head_xyz[:,1], label='y')
    ax.plot(times, head_xyz[:,2], label='z')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Position (mm)', fontsize=12)
    ax.set_title('Head Position', fontsize=14)
    ax.legend()
    finish_plot(ax)
    plt.tight_layout()
    fig.savefig(op.join(deriv_path, subject, subject + '_head_position.png'))
    plt.close()
    
    # basic preprocessing
    raw.apply_gradient_compensation(3)
    raw.pick('mag')
    raw.resample(250)
    raw.filter(3,45)  
    
    # annotate segments with high movement
    threshold = 8 # mm
    bad_mask = np.any(np.abs(head_xyz) > threshold, 1) 
    bad_labels, _ = label(bad_mask)
    bad_times = []
    for cluster in np.unique(bad_labels[bad_labels>0]):
        cluster_times = times[bad_labels==cluster]
        time0, time1 = cluster_times[0], cluster_times[-1]
        bad_times.append([time0, time1])

    # join together bad segments within threshold
    if len(bad_times) > 1:
        keep_threshold = 20 # seconds
        bad_times_joined = []
        bad_times_joined.append(bad_times[0])
        for i in np.arange(1, len(bad_times)):
            gap = bad_times[i][0] - bad_times[i-1][1]
            if gap < keep_threshold:
                bad_times_joined[-1][1] = bad_times[i][1].copy()
            else:
                bad_times_joined.append(bad_times[i])
        bad_times = bad_times_joined.copy()
        del bad_times_joined
        
    # annotate
    annot = raw.annotations
    for bad_time in bad_times:
        onset = bad_time[0]
        duration = bad_time[1] - bad_time[0]
        description = 'BAD_pos'
        annot += mne.Annotations(onset, duration, description, orig_time=annot.orig_time)
    raw.set_annotations(annot, verbose=False)
        
    if (raw.times[-1]-raw.times[0]) > 120: # 2 minutes 
    
        # remove bad channels
        chan_var = np.var(raw.get_data(), axis=1)
        outliers = isoutlier(chan_var)
        bad_chan = [raw.ch_names[ch] for ch in range(len(raw.ch_names)) if outliers[ch]]
        raw.info['bads'] = bad_chan
        fig = raw.plot_sensors()
        fig.savefig(op.join(deriv_path, subject, subject + '_bad_sensors.png'))
        plt.close()
        raw.drop_channels(bad_chan)
        
        # annotate bad segments (treat as resting state)
        data = raw.get_data()
        segment_len = int(1 * raw.info['sfreq'])
        variances = np.array([np.mean(np.var(data[:, i:i+segment_len], axis=1)) for i in np.arange(0, data.shape[1]-segment_len+1, segment_len)])
        outliers = isoutlier(variances, thresh=3)
        annot = raw.annotations
        for i, ind in enumerate(np.arange(0, data.shape[1]-segment_len, segment_len)):
            if outliers[i]:
                onset = raw.times[ind]
                duration = segment_len * (1/raw.info['sfreq'])
                description = 'BAD_var'
                annot += mne.Annotations(onset, duration, description, orig_time=annot.orig_time)
        raw.set_annotations(annot, verbose=False)
        
        # plot preprocessed sensor-level PSD
        psd = raw.compute_psd(fmin=3, fmax=45, method='welch', n_fft=500, reject_by_annotation=True)
        psd_data, freqs = psd.data, psd.freqs
        psd_data = 10 * np.log10(psd_data)
        fig, ax = plt.subplots(figsize=(5,3.5))
        ax.plot(freqs, psd_data.T, color='gray', alpha=0.2)
        ax.plot(freqs, np.mean(psd_data,0), color='black', linewidth=2.5)
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title('Sensor-Level PSD', fontsize=14)
        finish_plot(ax)
        plt.tight_layout()
        fig.savefig(op.join(deriv_path, subject, subject + '_sensor_psd.png'))
        plt.close()
        
        # plot preprocessed sensor-level power maps
        fig, ax = plt.subplots(1,3, figsize=(10,4))
        bands = {'Theta (4-7 Hz)': (4, 7), 'Alpha (8-12 Hz)': (8, 12), 'Beta (13-30 Hz)': (13, 30)}
        psd.plot_topomap(bands, axes=ax, normalize=True)
        plt.tight_layout()
        fig.savefig(op.join(deriv_path, subject, subject + '_sensor_power.png'))
        plt.close()
        
        # save preprocessed data
        raw.save(op.join(deriv_path, subject, subject + '_preproc-raw.fif'), overwrite=True)
        
        ### FORWARD MODEL ###
        
        # single-shell conduction model
        conductivity = (0.3,)
        model = mne.make_bem_model(
                subject=subject, ico=4,
                conductivity=conductivity,
                subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model)
        
        # get mri-->MNI transform and apply inverse to atlas
        mri_mni_t = mne.read_talxfm(subject, subjects_dir=subjects_dir)['trans']
        mni_mri_t = np.linalg.inv(mri_mni_t)
        centroids_mri = mne.transforms.apply_trans(mni_mri_t, coords / 1000) # in m
        
        # create atlas source space
        rr = centroids_mri # positions
        nn = np.zeros((rr.shape[0], 3)) # normals
        nn[:,-1] = 1.
        src = mne.setup_volume_source_space(
            subject,
            pos={'rr': rr, 'nn': nn},
            subjects_dir=subjects_dir,
            verbose=True,
        )
        
        # forward solution
        trans_fname = op.join(deriv_path, subject, subject + '-trans.fif')
        fwd = mne.make_forward_solution(
            raw.info,
            trans=trans_fname,
            src=src,
            bem=bem,
            meg=True,
            eeg=False
            )
        mne.write_forward_solution(op.join(deriv_path, subject, subject + '-fwd.fif'), fwd, overwrite=True)
        
        ### SOURCE RECON ###
        
        # calculate covariance of all data and plot
        cov = mne.compute_raw_covariance(raw, reject_by_annotation=True)
        cov_data = cov.data
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(cov_data)
        ax.set_xlabel('Sensor')
        ax.set_ylabel('Sensor')
        ax.set_title('Data Covariance')
        plt.tight_layout()
        fig.savefig(op.join(deriv_path, subject, subject + '_data_cov.png'))
        plt.close()
    
        # construct beamformer
        filters = mne.beamformer.make_lcmv(
                raw.info,
                fwd,
                cov,
                reg=0.05,
                noise_cov=None,
                pick_ori='max-power',
                weight_norm='unit-noise-gain-invariant',
                rank=None,
                reduce_rank=True,
                verbose=False,
                )
    
        # apply beamformer
        stc =  mne.beamformer.apply_lcmv_raw(raw, filters, verbose=False)
        source_data = stc.data 
        
        if source_data.shape[0]==len(names): # make sure all parcels included
    
            # make source raw
            info = mne.create_info(list(names), raw.info['sfreq'], 'misc', verbose=False)
            source_raw = mne.io.RawArray(source_data, info, verbose=False)
            source_raw.set_meas_date(raw.info['meas_date'])
            source_raw.set_annotations(raw.annotations, verbose=False)
            source_raw.save(op.join(deriv_path, subject, subject + '_source-raw.fif'), overwrite=True)
        
            # orthogonalise
            source_data_orth = zscore(symmetric_orth(source_data), 1)
            source_raw_orth = mne.io.RawArray(source_data_orth, source_raw.info)
            source_raw_orth.set_meas_date(raw.info['meas_date'])
            source_raw_orth.set_annotations(raw.annotations, verbose=False)
            source_raw_orth.save(op.join(deriv_path, subject, subject + '_source_orth-raw.fif'), overwrite=True)
            
            # compute source level PSD
            psd = source_raw_orth.compute_psd(method='welch', n_fft=500, fmin=3, fmax=45, picks='all')
            freqs = psd.freqs
            psd_data = psd.data
            
            # plot PSD
            psd_data = 10 * np.log10(psd_data)
            fig, ax = plt.subplots(figsize=(5,3.5))
            ax.plot(freqs, psd_data.T, color='gray', alpha=0.2)
            ax.plot(freqs, np.mean(psd_data,0), color='black', linewidth=2.5)
            ax.set_xlabel('Frequency (Hz)', fontsize=12)
            ax.set_ylabel('Amplitude', fontsize=12)
            ax.set_title('Source-Level PSD', fontsize=14)
            finish_plot(ax)
            plt.tight_layout()
            fig.savefig(op.join(deriv_path, subject, subject + '_source_psd.png'))
            plt.close()
            
            # plot source-level power maps
            fmin = (4, 8, 13)
            fmax = (7, 12, 30)
            band_names = ['Theta', 'Alpha', 'Beta']
            psd_data = psd.data
            for freq in range(len(fmin)):
                freq_range = (freqs >= fmin[freq]) * (freqs <= fmax[freq])
                power_map = np.mean(psd_data[:,freq_range], -1)
                power_map = zscore(power_map)
                power_img = make_4d_atlas_nifti(atlas, power_map)
                fig = surface_brain_plot(power_img, subjects_dir, 'pial', 'RdBu_r', threshold=0, fade=True, symmetric=True, cbar_label='Power (z-score)')
                fig.savefig(op.join(deriv_path, subject, subject + '_source_' + band_names[freq] + '.png'))
                plt.close()
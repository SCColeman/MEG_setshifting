#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mega post-hoc analysis of DyNeMo modes.

@author: sebastiancoleman
"""

import mne
import os
import os.path as op
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from nilearn import image, datasets, plotting
import nibabel as nib
from scipy.stats import zscore, pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import pandas as pd
from scipy.ndimage import label
import seaborn as sns
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

#%% functions

def make_atlas_nifti(atlas_img, values):
    
    from nilearn import image, datasets
    import nibabel as nib
    
    mni = datasets.load_mni152_template()
    atlas_data = atlas_img.get_fdata()
    atlas_new = np.zeros(np.shape(atlas_data))
    indices = np.unique(atlas_data[atlas_data>0])
    for reg in range(len(values)):
        reg_mask = atlas_data == indices[reg]
        atlas_new[reg_mask] = values[reg]
    new_img = nib.Nifti1Image(atlas_new, atlas_img.affine, atlas_img.header)
    img = image.resample_img(new_img, mni.affine, mni.shape)
    img = image.new_img_like(img, (mni.get_fdata()>0)*img.get_fdata())
    
    return img

def make_4d_atlas_nifti(atlas_img, values):

    # load fsaverage and atlas   
    mni = datasets.load_mni152_template()
    atlas_data = atlas_img.get_fdata()

    # place values in each parcel region
    regs = []
    for reg in range(atlas_data.shape[-1]):
        atlas_reg = atlas_data[:,:,:,reg]
        atlas_reg /= np.max(atlas_reg)
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
    fig = plt.figure(figsize=(5.2, 4))
    ax1 = fig.add_axes([0.05, 0.58, 0.45, 0.40])  # top-left
    ax2 = fig.add_axes([0.5, 0.58, 0.45, 0.40])  # top-right
    ax3 = fig.add_axes([0.05, 0.17, 0.45, 0.38])  # bottom-left
    ax4 = fig.add_axes([0.5, 0.17, 0.45, 0.38])  # bottom-right
    cax = fig.add_axes([0.25, 0.12, 0.5, 0.03]) # colorbar ax
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('none')
        ax.axis(False)
        
    # set up threshold
    if symmetric:
        vmax = np.max(np.abs(data))
        vmin = -vmax
        mid = threshold + ((vmax-threshold)/3)
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
        cortex=(0.7, 0.7, 0.7)
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
                       alpha=0.8,
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

    # colorbar
    mne.viz.plot_brain_colorbar(cax, clim, cmap, orientation='horizontal', label=cbar_label)
    
    return fig

def glass_brain_plot(adjacency, atlas_coords, threshold, cbar_label):
    
    import pyvista as pv
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from nilearn import surface
    
    def add_spheres(plotter, points, radius=2.0, color="black"):
        for point in points:
            sphere = pv.Sphere(radius=radius, center=point)
            plotter.add_mesh(sphere, color=color)
    
    def remove_white_space(imdata, decim=None):
        nonwhite_pix = (imdata != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        if decim:
            nonwhite_row[::decim] = True
        nonwhite_col = nonwhite_pix.any(0)
        if decim:
            nonwhite_col[::decim] = True
        imdata_cropped = imdata[nonwhite_row][:, nonwhite_col]
        return imdata_cropped
    
    # get upper triangle only
    triu = np.triu_indices(adjacency.shape[0],1)
    mask = np.zeros((adjacency.shape[0],adjacency.shape[0]))
    mask[triu] = 1
    adjacency *= mask
    
    # get indices
    threshold = threshold
    indices = np.argwhere(np.abs(adjacency) > threshold)
    values = adjacency[np.abs(adjacency) > threshold]
    
    # get coordinates of indices
    lines = []
    for ind in range(indices.shape[0]):
        line = np.array([atlas_coords[indices[ind,0],:], atlas_coords[indices[ind,1],:]])
        lines.append(line)
    lines = np.concatenate(lines, 0)
    
    # get colors
    vmin, vmax = -np.max(np.abs(adjacency[triu])), np.max(np.abs(adjacency[triu]))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.RdBu_r
    rgb_values = cmap(norm(values))
    
    # get fsaverage meshes
    fsaverage = datasets.fetch_surf_fsaverage()
    lh, rh = surface.load_surf_mesh(fsaverage.pial_left), surface.load_surf_mesh(fsaverage.pial_right)
    
    # lh mesh
    coords, faces = lh.coordinates, lh.faces
    faces_vtk = np.column_stack((np.full(faces.shape[0], 3), faces)).ravel()
    lh_mesh = pv.PolyData(coords, faces_vtk)

    # rh mesh
    coords, faces = rh.coordinates, rh.faces
    faces_vtk = np.column_stack((np.full(faces.shape[0], 3), faces)).ravel()
    rh_mesh = pv.PolyData(coords, faces_vtk)
    
    # plot brain
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(lh_mesh, color="gray", opacity=0.06)
    plotter.add_mesh(rh_mesh, color="gray", opacity=0.06)
    
    # add line
    for l, ll in enumerate(np.arange(0, lines.shape[0], 2)):
        color = rgb_values[l,:]
        color[-1] = 0.6
        plotter.add_lines(lines[ll:ll+2,:], color=rgb_values[l,:], width=7)
    
    # add coordinates
    add_spheres(plotter, atlas_coords, radius=2.0, color="gray")
    
    # get up view
    plotter.view_xy() # up view
    img_up = plotter.screenshot()
    img_up = remove_white_space(img_up)
    
    # get side view
    plotter.view_yz() # side view
    img_side = plotter.screenshot()
    img_side = remove_white_space(img_side)
    
    # get back view
    plotter.view_xz() # back view
    img_back = plotter.screenshot()
    img_back = remove_white_space(img_back)
    
    # plot 
    fig = plt.figure(figsize=(9,5.5))
    ax1 = fig.add_axes([0.15, 0.2, 0.4, 0.7])  # top-left
    ax2 = fig.add_axes([0.55, 0.55, 0.3, 0.35])  # top-right
    ax3 = fig.add_axes([0.55, 0.2, 0.3, 0.35])  # bottom-left
    cax = fig.add_axes([0.32, 0.12, 0.4, 0.03]) # colorbar ax
    
    # insert images
    ax1.imshow(img_up)
    ax1.axis(False)
    ax2.imshow(img_side)
    ax2.axis(False)
    ax3.imshow(img_back)
    ax3.axis(False)
    
    # add cbar
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='horizontal', label=cbar_label)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cbar_label, fontsize=16, labelpad=0)
    
    return fig

def regression_plot(x, y, ax, color='purple', label=None):
    
    # regression
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    results = model.fit()
    slope = results.params[1]
    intercept = results.params[0]
    y_pred = intercept + slope*x
    predictions = results.get_prediction(X)
    ci = predictions.conf_int()
    sort_i = np.argsort(x)
    stat, p = pearsonr(x,y)
    
    # plot
    ax.scatter(x,y, color=color, alpha=0.5)
    ax.fill_between(x[sort_i], ci[sort_i, 0], ci[sort_i, 1], color=color, alpha=0.2)
    ax.plot(x[sort_i],y_pred[sort_i], color=color, label=label, linewidth=2.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    #ax.grid('on', alpha=0.3)
    
    return stat, p, slope

def line_with_error(ax, data, times, color, label):
    values = np.mean(data, 0)
    err = np.std(data, 0) / np.sqrt(len(data))
    ax.plot(times, values, color=color, label=label)
    ax.fill_between(times, values-err, values+err, alpha=0.2, color=color)

#%% set up paths

data_path = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/data'
deriv_path = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/derivatives'
subjects_dir = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/subjects_dir'

#%% load dynemo outputs

n_modes = 6
run = 3
dynemo_dir = sorted(glob('/d/gmi/1/sebastiancoleman/MEG_setshifting/dynemo/' + str(n_modes) + '_modes*'))[run]
alpha = np.load(op.join(dynemo_dir, 'inf_params', 'alp_rw.pkl'),allow_pickle=True)
covs = np.load(op.join(dynemo_dir, 'inf_params', 'covs.npy'))
psd = np.load(op.join(dynemo_dir, 'spectra', 'psd_rs.npy'))
psd_base = psd[:,1]
psd_coef = psd[:,0]
coh = np.load(op.join(dynemo_dir, 'spectra', 'coh.npy'))
freqs = np.load(op.join(dynemo_dir, 'spectra', 'f.npy'))

#%% load pheno csv

df = pd.read_table(op.join(data_path, 'pheno_updated.csv'), delimiter=',')
subj_id = df['subj_id'].to_numpy()
ages = df['age'].to_numpy()
is_adhd = df['is_ADHD'].to_numpy()
sexes = df['sex'].to_numpy()
meds = df['Medication (24 hrs)'].to_numpy()

#%% load atlas 

#atlas_file = '/d/gmi/1/sebastiancoleman/atlases/MEG_atlas_38_regions_3D.nii.gz'
atlas_file = '/d/gmi/1/sebastiancoleman/atlases/MEG_atlas_38_regions_4D.nii.gz'
atlas = image.load_img(atlas_file)
coords = np.load('/d/gmi/1/sebastiancoleman/atlases/MEG_atlas_38_regions_coords.npy')
names = np.squeeze(pd.read_csv('/d/gmi/1/sebastiancoleman/atlases/MEG_atlas_38_regions_names.csv', header=None).to_numpy())

#%% get list of subjects

dir_list = sorted(glob(op.join(deriv_path, '*')))
dir_list = [entry for entry in dir_list if op.isdir(entry)]
subjects = [op.basename(d) for d in dir_list]
subjects = [sub for sub in subjects if not sub=='bad']

#%% plot mode psd

colors = plt.cm.tab10.colors[:8]

mode = 2
mode_power = np.mean(psd_coef[:,mode,:,:], (0,-1))
max_reg = np.argmax(np.mean(psd_coef[:,mode,:,:], (0,-1)))
base_mean = np.mean(psd_base[:,mode,:,:], 0)
coef_mean = np.mean(psd_coef[:,mode,:,:], 0)
mode_psd = (base_mean.T + (coef_mean.T)).T

fig, ax = plt.subplots(figsize=(3.2,2))
ax.plot(freqs[3:], np.average(mode_psd[:,3:], 0, weights=mode_power), color=colors[6], linewidth=2.5, alpha=1, label='Mode PSD')
ax.plot(freqs[3:], np.mean(base_mean,0)[3:], color='black', linestyle='--', linewidth=2, label='Static PSD')
ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.set_xlim([4, 40])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.legend()
plt.tight_layout()

#%% plot mode map

for mode in range(n_modes):
    coef_mean = np.mean(psd_coef[:,mode,:,:], 0)
    freq_range = (freqs>=4) * (freqs<=15)
    power = np.mean(coef_mean, -1)
    power *= 100
    #img = make_atlas_nifti(atlas, power)
    img = make_4d_atlas_nifti(atlas, power)
    fig = surface_brain_plot(img, subjects_dir, 'pial', 'RdBu_r', threshold=0, cbar_label='Coefficient')
    fig.savefig(dynemo_dir + '/figures/mode' + str(mode+1) + '_power.png')
    plt.close()

#%% plot mode coh

for mode in range(n_modes):
    mode_coh = np.mean(coh[:,mode,:,:,:], 0)
    freq_range = (freqs>=4) * (freqs<=15)
    conn = np.nanmean(mode_coh[:,:,freq_range], -1)
    triu = np.triu_indices(len(names), k=1)
    np.fill_diagonal(conn, np.nan)
    threshold = np.nanpercentile(np.abs(conn[triu]), 97)
    fig = glass_brain_plot(conn, coords, threshold, 'Mode Coherence')
    fig.savefig(dynemo_dir + '/figures/mode' + str(mode+1) + '_coh.png')
    plt.close()

#%% get mode timecourses and behaviour

all_adhd = []
all_age = []
all_sex = []
all_epochs = []
all_conditions = []
all_RT = []
all_correct = []
all_subjects = []

for s, subject in enumerate(subjects):
    
    # get clinical variables
    sub_ind = subj_id==subject
    age = ages[sub_ind][0]
    adhd = is_adhd[sub_ind][0]
    sex = sexes[sub_ind][0]
    
    # load data
    fname = op.join(deriv_path, subject, subject + '_source_orth-raw.fif')
    raw = mne.io.Raw(fname, preload=True)
    
    # load subject alpha and bad samples
    sub_alpha = alpha[s].T
    bads = np.isnan(raw.get_data(reject_by_annotation='nan')[0,:])
    
    # pad alpha
    keep_index = np.ones(len(raw))
    keep_index[bads] = 0
    n_embeddings = 15
    embed_loss = (n_embeddings-1) // 2
    keep_index[:embed_loss] = 0
    keep_index[-embed_loss:] = 0
    keep_pos = np.squeeze(np.argwhere(keep_index==1))
    alpha_pad = np.zeros((sub_alpha.shape[0], len(raw)))
    alpha_pad[:,keep_pos[:sub_alpha.shape[1]]] = sub_alpha.copy() 
    
    # make raw out of alpha
    alpha_info = mne.create_info(['mode' + str(mode+1) for mode in range(n_modes)], 250)
    alpha_raw = mne.io.RawArray(alpha_pad, alpha_info)
    alpha_raw.set_meas_date(raw.info['meas_date'])
    alpha_raw.set_annotations(raw.annotations)
    
    # get all events
    events, ids = mne.events_from_annotations(alpha_raw)
    onsets = events[:,0]
    
    # create new events with only relevant triggers
    new_events = []
    for e in range(events.shape[0]):
        event = events[e,:]
        event_onset = event[0]
        if event[2]==ids['LeftTarget'] or event[2]==ids['RightTarget']:
            new_event = event.copy()
            other_events = events[onsets==event_onset,:]
            other_vals = other_events[:,2]
            if np.any(other_vals==ids['ExtraDim']):
                new_event[2] = 3
            elif np.any(other_vals==ids['IntraDim']):
                new_event[2] = 2
            else:
                new_event[2] = 1
            new_events.append(new_event)
        elif event[2]==ids['Correct']:
            new_event = event.copy()
            new_event[2] = 4
            new_events.append(new_event)
        elif event[2]==ids['Incorrect']:
            new_event = event.copy()
            new_event[2] = 5
            new_events.append(new_event)
    new_events = np.concatenate([new_events], 0)
    
    # calculate all RTs and correct status           
    stim_codes = [1, 2, 3]
    resp_codes = [4, 5]
    max_rt_s = 4
    sfreq = raw.info['sfreq']
    
    stim_events = new_events[np.isin(new_events[:, 2], stim_codes)]
    resp_events = new_events[np.isin(new_events[:, 2], resp_codes)]
    
    RTs = []
    corrects = []
    for i, stim in enumerate(stim_events):
        stim_time = stim[0]
        next_stim_time = stim_events[i + 1][0] if i + 1 < len(stim_events) else np.inf
    
        # Find first response after stim, before next stim and within max_rt_s
        valid_resps = resp_events[
            (resp_events[:, 0] > stim_time) &
            (resp_events[:, 0] < next_stim_time) &
            (resp_events[:, 0] - stim_time < max_rt_s * sfreq)
        ]
    
        if len(valid_resps) > 0:
            resp = valid_resps[0]
            rt = (resp[0] - stim_time) / sfreq
            correct = 1 if resp[2] == 4 else 0
        else:
            rt = np.nan
            correct = np.nan
    
        RTs.append(rt)
        corrects.append(correct)
    RTs = np.array(RTs)
    corrects = np.array(corrects)
    
    # epoch around all conditions
    epochs = mne.Epochs(alpha_raw, stim_events, tmin=-0.8, tmax=2, preload=True)
    
    # line up conditions, RTs and correct status with retained epochs
    RTs = RTs[epochs.selection]
    corrects = corrects[epochs.selection]
    conds = epochs.events[:,2]
    
    # only keep valid epochs
    valid = (np.isnan(RTs)==False)*(np.isnan(corrects)==False)
    
    # store out
    epoch_data = epochs.get_data()
    all_epochs.append(epoch_data[valid,:,:])
    all_conditions.append(conds[valid])
    all_RT.append(RTs[valid])
    all_correct.append(corrects[valid])
    all_adhd.append(np.repeat(adhd, np.sum(valid)))
    all_age.append(np.repeat(age, np.sum(valid)))
    all_sex.append(np.repeat(sex, np.sum(valid)))
    all_subjects.append(np.repeat(subject, np.sum(valid)))
    
#%% behaviour 

# RT stats
df = pd.DataFrame({
    'RT': np.concatenate(all_RT),
    'Correct': np.concatenate(all_correct),
    'Condition': np.concatenate(all_conditions),  
    'Age': np.concatenate(all_age),                      
    'ADHD': np.concatenate(all_adhd),                    
    'Subject': np.concatenate(all_subjects), 
})
adhd_labels = {1: 'ADHD', 0: 'Control'}
df['Diagnosis'] = df['ADHD'].map(adhd_labels)
model = mixedlm("RT ~ Condition + Age + Diagnosis + Correct", df, groups=df["Subject"])
result = model.fit()
print(result.summary())

# RT plot
df_rt = df.groupby(['Subject', 'Condition', 'Diagnosis'])['RT'].mean().reset_index()
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.boxplot(data=df, x='Condition', y='RT', hue='Diagnosis', palette='Set2', ax=ax)
ax.set_ylabel('Reaction Time (s)', fontsize=16)
ax.set_xticks([0,1,2])
ax.set_xticklabels(['Non-Shift', 'Implicit', 'Explicit'], fontsize=14)
plt.yticks(fontsize=12)
ax.set_xlabel(None)
plt.legend(title=None, fontsize=12, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()

# Accuracy stats
df = pd.DataFrame({
    'RT': np.concatenate(all_RT),
    'Correct': np.concatenate(all_correct),
    'Condition': np.concatenate(all_conditions),  
    'Age': np.concatenate(all_age),                      
    'ADHD': np.concatenate(all_adhd),                    
    'Subject': np.concatenate(all_subjects), 
})
adhd_labels = {1: 'ADHD', 0: 'Control'}
df['Diagnosis'] = df['ADHD'].map(adhd_labels)
model = mixedlm("Correct ~ Condition + Age + Diagnosis + RT", df, groups=df["Subject"])
result = model.fit()
print(result.summary())

# Accuracy plot
df_acc = df.groupby(['Subject', 'Condition', 'Diagnosis'])['Correct'].mean().reset_index()
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.boxplot(data=df_acc, x='Condition', y='Correct', hue='Diagnosis', palette='Set2', ax=ax)
ax.set_ylabel('Accuracy', fontsize=16)
ax.set_xticks([0,1,2])
ax.set_xticklabels(['Non-Shift', 'Implicit', 'Explicit'], fontsize=14)
plt.yticks(fontsize=12)
ax.set_xlabel(None)
plt.legend(title=None, fontsize=12, loc='lower left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()

# age on RT
df_rt = df.groupby(['Subject', 'Age'])['RT'].mean().reset_index()
fig, ax = plt.subplots(figsize=(4, 3.5))
sns.regplot(data=df_rt, x='Age', y='RT', ax=ax, color='gray')
ax.set_ylabel('Reaction Time (s)', fontsize=14)
ax.set_xlabel('Age (yrs)', fontsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()

# age on accuracy
df_acc = df.groupby(['Subject', 'Age'])['Correct'].mean().reset_index()
fig, ax = plt.subplots(figsize=(4, 3.5))
sns.regplot(data=df_acc, x='Age', y='Correct', ax=ax, color='gray')
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_xlabel('Age (yrs)', fontsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()

#%%  plot all mode timecourses, trial onset

times = epochs.times
baseline = (times>-0.1) * (times < 0)
mean_timecourses = np.array([np.mean(inst,0) for inst in all_epochs])
fig, ax = plt.subplots(figsize=(7,3))
colors = plt.cm.tab10.colors[:8]
color_order = [2, 4, 6, 1, 3, 0]
mode_names = ['Sensorimotor', 'Temporal', 'Background', 'Visual', 'Higher Visual', 'Frontotemporal']

for mode in range(n_modes):
    mode_timecourse = mean_timecourses[:,mode,:]
    values = np.mean(mode_timecourse, 0)
    values -= np.mean(values[baseline])
    err = np.std(mode_timecourse, 0) / np.sqrt(mode_timecourse.shape[0])
    ax.plot(times, values, label=mode_names[mode], color=colors[color_order[mode]])
    ax.fill_between(times, values-err, values+err, alpha=0.2, color=colors[color_order[mode]])
ax.axvline(0, linestyle='--', color=[0.2,0.2,0.2, 0.5], label=None)
ax.set_xlim([-0.25, 2.5])
ax.legend()
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Mode Activation', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()

#%% mega single mode analysis

mode = 4
mode_data = np.concatenate([inst[:,mode,:] for inst in all_epochs], 0)

# behaviour dataframe
df = pd.DataFrame({
    'RT': np.concatenate(all_RT),
    'Correct': np.concatenate(all_correct),
    'Condition': np.concatenate(all_conditions),  
    'Age': np.concatenate(all_age),                      
    'ADHD': np.concatenate(all_adhd),                    
    'Subject': np.concatenate(all_subjects), 
})

# mega stats (on decimated data for speed)
p_arr = []
t_arr = []
decim = 10
times = epochs.times
times_decim = times[np.arange(0, len(times)-decim, decim)]
for timepoint in np.arange(0, len(times)-decim, decim):
    
    # get data from timepoint
    window = np.arange(timepoint, timepoint+decim)
    data_i = np.mean(mode_data[:,window], -1)
    df_i = df.copy()
    df_i['Data'] = 1 * data_i
    
    # LME
    model = mixedlm("Data ~ Condition + Age + ADHD + RT + Correct", df_i, groups=df["Subject"])
    result = model.fit()
    p = result.pvalues[1:-1].to_numpy()
    t = result.tvalues[1:-1].to_numpy()
    p_arr.append(p)
    t_arr.append(t)
p_arr = np.array(p_arr)
t_arr = np.array(t_arr)

# separate effects
t_cond, t_age, t_adhd, t_rt, t_acc = t_arr[:,0], t_arr[:,1], t_arr[:,2], t_arr[:,3], t_arr[:,4]
p_cond, p_age, p_adhd, p_rt, p_acc = p_arr[:,0], p_arr[:,1], p_arr[:,2], p_arr[:,3], p_arr[:,4]

##### EFFECT OF CONDITION ######

# melt df
n_timepoints = mode_data.shape[1]
time_cols = [f't{i}' for i in range(n_timepoints)]
df_data = pd.DataFrame(mode_data, columns=time_cols)
df_full = pd.concat([df.reset_index(drop=True), df_data], axis=1)
mode_timecourses = df_full.groupby(['Subject', 'Condition'])[time_cols].mean().reset_index()

# plot conditions
colors = sns.color_palette("Set2", 3)
times = epochs.times
fig, ax = plt.subplots(figsize=(6,4))
nonshift = mode_timecourses[mode_timecourses['Condition']==1][time_cols].to_numpy()
line_with_error(ax, nonshift, times, colors[0], label='Non-Shift')
implicit = mode_timecourses[mode_timecourses['Condition']==2][time_cols].to_numpy()
line_with_error(ax, implicit, times, colors[1], label='Implicit')
explicit = mode_timecourses[mode_timecourses['Condition']==3][time_cols].to_numpy()
line_with_error(ax, explicit, times, colors[2], label='Explicit')

# shade significant effects
reject, _ = mne.stats.fdr_correction(p_cond)
reject_labelled, _ = label(reject)
for cluster in np.unique(reject_labelled)[1:]:
    tmin = np.min(times_decim[reject_labelled==cluster])
    tmax = np.max(times_decim[reject_labelled==cluster])
    ax.axvspan(tmin, tmax, color='gray', alpha=0.1)     

# finish plot
ax.axvline(0, color='black', linestyle='--', alpha=0.2)
ax.set_xlim([-0.5, 1.5])
ax.set_xlabel('Time (s)', fontsize=16)
ax.set_ylabel('Mode Activity', fontsize=16)
ax.set_title('Effect of Condition', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=12)
plt.tight_layout()

##### EFFECT OF RT #############

def label_rt_speed(group):
    q1 = group['RT'].quantile(0.25)
    q3 = group['RT'].quantile(0.75)
    return pd.Series(np.where(group['RT'] <= q1, 'Fast',
                       np.where(group['RT'] >= q3, 'Slow', 'Other')),
                     index=group.index)

# melt df
n_timepoints = mode_data.shape[1]
time_cols = [f't{i}' for i in range(n_timepoints)]
df_data = pd.DataFrame(mode_data, columns=time_cols)
df_full = pd.concat([df.reset_index(drop=True), df_data], axis=1)
df_full['RT_Speed'] = (df_full.groupby('Subject', group_keys=False)[['RT']].apply(label_rt_speed).reset_index(drop=True))
mode_timecourses = df_full.groupby(['Subject', 'RT_Speed'])[time_cols].mean().reset_index()

# plot conditions
colors = sns.color_palette("Set2", 3)
times = epochs.times
fig, ax = plt.subplots(figsize=(6,4))
fast = mode_timecourses[mode_timecourses['RT_Speed']=='Fast'][time_cols].to_numpy()
line_with_error(ax, fast, times, colors[0], label='Fast')
slow = mode_timecourses[mode_timecourses['RT_Speed']=='Slow'][time_cols].to_numpy()
line_with_error(ax, slow, times, colors[1], label='Slow')

# shade significant effects
reject, _ = mne.stats.fdr_correction(p_rt)
reject_labelled, _ = label(reject)
for cluster in np.unique(reject_labelled)[1:]:
    tmin = np.min(times_decim[reject_labelled==cluster])
    tmax = np.max(times_decim[reject_labelled==cluster])
    ax.axvspan(tmin, tmax, color='gray', alpha=0.1)     

# finish plot
ax.axvline(0, color='black', linestyle='--', alpha=0.2)
ax.set_xlim([-0.5, 1.5])
ax.set_xlabel('Time (s)', fontsize=16)
ax.set_ylabel('Mode Activity', fontsize=16)
ax.set_title('Effect of RT', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=12)
plt.tight_layout()

##### EFFECT OF INCORRECT #############

# melt df
n_timepoints = mode_data.shape[1]
time_cols = [f't{i}' for i in range(n_timepoints)]
df_data = pd.DataFrame(mode_data, columns=time_cols)
df_full = pd.concat([df.reset_index(drop=True), df_data], axis=1)
mode_timecourses = df_full.groupby(['Subject', 'Correct'])[time_cols].mean().reset_index()

# plot conditions
colors = sns.color_palette("Set2", 3)
times = epochs.times
fig, ax = plt.subplots(figsize=(6,4))
correct = mode_timecourses[mode_timecourses['Correct']==1][time_cols].to_numpy()
line_with_error(ax, correct, times, colors[0], label='Correct')
incorrect = mode_timecourses[mode_timecourses['Correct']==0][time_cols].to_numpy()
line_with_error(ax, slow, times, colors[1], label='Incorrect')

# shade significant effects
reject, _ = mne.stats.fdr_correction(p_acc)
reject_labelled, _ = label(reject)
for cluster in np.unique(reject_labelled)[1:]:
    tmin = np.min(times_decim[reject_labelled==cluster])
    tmax = np.max(times_decim[reject_labelled==cluster])
    ax.axvspan(tmin, tmax, color='gray', alpha=0.1)     

# finish plot
ax.axvline(0, color='black', linestyle='--', alpha=0.2)
ax.set_xlim([-0.5, 1.5])
ax.set_xlabel('Time (s)', fontsize=16)
ax.set_ylabel('Mode Activity', fontsize=16)
ax.set_title('Effect of Accuracy', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=12)
plt.tight_layout()

##### EFFECT OF AGE ######

# melt df
n_timepoints = mode_data.shape[1]
time_cols = [f't{i}' for i in range(n_timepoints)]
df_data = pd.DataFrame(mode_data, columns=time_cols)
df_full = pd.concat([df.reset_index(drop=True), df_data], axis=1)
subject_ages = df_full.groupby('Subject')['Age'].first()
median_age = subject_ages.median()
subject_group = (subject_ages <= median_age).map({True: 'Younger', False: 'Older'})
df_full['Young_Old'] = df_full['Subject'].map(subject_group)
mode_timecourses = df_full.groupby(['Subject', 'Young_Old'])[time_cols].mean().reset_index()

# plot conditions
colors = sns.color_palette("Set2", 3)
times = epochs.times
fig, ax = plt.subplots(figsize=(6,4))
young = mode_timecourses[mode_timecourses['Young_Old']=='Younger'][time_cols].to_numpy()
line_with_error(ax, young, times, colors[1], label='Younger')
old = mode_timecourses[mode_timecourses['Young_Old']=='Older'][time_cols].to_numpy()
line_with_error(ax, old, times, colors[0], label='Older')

# shade significant effects
reject, _ = mne.stats.fdr_correction(p_age)
reject_labelled, _ = label(reject)
for cluster in np.unique(reject_labelled)[1:]:
    tmin = np.min(times_decim[reject_labelled==cluster])
    tmax = np.max(times_decim[reject_labelled==cluster])
    ax.axvspan(tmin, tmax, color='gray', alpha=0.1)     

# finish plot
ax.axvline(0, color='black', linestyle='--', alpha=0.2)
ax.set_xlim([-0.5, 1.5])
ax.set_xlabel('Time (s)', fontsize=16)
ax.set_ylabel('Mode Activity', fontsize=16)
ax.set_title('Effect of Age', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=12)
plt.tight_layout()

##### EFFECT OF ADHD ######

# melt df
n_timepoints = mode_data.shape[1]
time_cols = [f't{i}' for i in range(n_timepoints)]
df_data = pd.DataFrame(mode_data, columns=time_cols)
df_full = pd.concat([df.reset_index(drop=True), df_data], axis=1)
mode_timecourses = df_full.groupby(['Subject', 'ADHD'])[time_cols].mean().reset_index()

# plot conditions
colors = sns.color_palette("Set2", 3)
times = epochs.times
fig, ax = plt.subplots(figsize=(6,4))
control = mode_timecourses[mode_timecourses['ADHD']==0][time_cols].to_numpy()
line_with_error(ax, control, times, colors[0], label='Control')
adhd = mode_timecourses[mode_timecourses['ADHD']==1][time_cols].to_numpy()
line_with_error(ax, adhd, times, colors[1], label='ADHD')

# shade significant effects
#reject, _ = mne.stats.fdr_correction(p_adhd)
reject = p_adhd < 0.05
reject_labelled, _ = label(reject)
for cluster in np.unique(reject_labelled)[1:]:
    tmin = np.min(times_decim[reject_labelled==cluster])
    tmax = np.max(times_decim[reject_labelled==cluster])
    ax.axvspan(tmin, tmax, color='pink', alpha=0.25)     

# finish plot
ax.axvline(0, color='black', linestyle='--', alpha=0.2)
ax.set_xlim([-0.5, 1.5])
ax.set_xlabel('Time (s)', fontsize=16)
ax.set_ylabel('Mode Activity', fontsize=16)
ax.set_title('Effect of ADHD', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=12)
plt.tight_layout()

##### EFFECT OF AGE and ADHD ON WHOLE TRIAL ######

# stats
window = (times>-0.5) * (times < 1.5)
mode_FO = np.mean(mode_data[:,window], -1)
df_i = df.copy()
df_i['Data'] = 1 * mode_FO
model = mixedlm("Data ~ Condition + Age + ADHD + RT + Correct", df_i, groups=df["Subject"])
result = model.fit()

# plot
age_FO = df_i.groupby(['Subject', 'Condition', 'Age', 'ADHD'])['Data'].mean().reset_index()
condition_labels = {1: 'Non-Shift', 2: 'Implicit', 3: 'Explicit'}
age_FO['ConditionLabel'] = age_FO['Condition'].map(condition_labels)
adhd_labels = {1: 'ADHD', 0: 'Control'}
age_FO['ADHDLabel'] = age_FO['ADHD'].map(adhd_labels)
g = sns.lmplot(
    data=age_FO,
    x='Age',
    y='Data',
    hue='ConditionLabel',
    palette='Set2',
    col='ADHD',
    aspect=1.1,
    height=3,
    facet_kws={'sharex':True, 'sharey':True},
    scatter_kws={'s': 30, 'alpha': 0.6},
    line_kws={'linewidth': 2},
)
g.despine(left=True, bottom=True)
g.set_xlabels('Age (yrs)', fontsize=12)
g.set_ylabels('Mode Activity', fontsize=12)

# plot ADHD vs controls
fig, ax = plt.subplots(figsize=(6,3))
g = sns.boxplot(
    data=age_FO,
    x='ConditionLabel',  # Conditions on the x-axis
    y='Data',            # 'Data' on the y-axis
    hue='ADHDLabel',          # Group by ADHD vs control
    order=['Non-Shift', 'Implicit', 'Explicit'],  # Ensure specific condition order
    palette='Set2',       # Optional: color palette for ADHD and control
    ax=ax,
    legend=False,
)
ax.set_xlabel(None)
ax.set_ylabel('Mode Activity', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()

#%% slightly less mega single mode analysis

mode = 0
mode_data = np.concatenate([inst[:,mode,:] for inst in all_epochs], 0)

# behaviour dataframe
df = pd.DataFrame({
    'RT': np.concatenate(all_RT),
    'Correct': np.concatenate(all_correct),
    'Condition': np.concatenate(all_conditions),  
    'Age': np.concatenate(all_age),                      
    'ADHD': np.concatenate(all_adhd),                    
    'Subject': np.concatenate(all_subjects), 
})

# mega stats (on decimated data for speed)
p_arr = []
t_arr = []
decim = 10
times = epochs.times
times_decim = times[np.arange(0, len(times)-decim, decim)]
for timepoint in np.arange(0, len(times)-decim, decim):
    
    # get data from timepoint
    window = np.arange(timepoint, timepoint+decim)
    data_i = np.mean(mode_data[:,window], -1)
    df_i = df.copy()
    df_i['Data'] = 1 * data_i
    
    # LME
    model = mixedlm("Data ~ Condition + Age + ADHD + RT + Correct", df_i, groups=df["Subject"])
    result = model.fit()
    p = result.pvalues[1:-1].to_numpy()
    t = result.tvalues[1:-1].to_numpy()
    p_arr.append(p)
    t_arr.append(t)
p_arr = np.array(p_arr)
t_arr = np.array(t_arr)

# separate effects
t_cond, t_age, t_adhd, t_rt, t_acc = t_arr[:,0], t_arr[:,1], t_arr[:,2], t_arr[:,3], t_arr[:,4]
p_cond, p_age, p_adhd, p_rt, p_acc = p_arr[:,0], p_arr[:,1], p_arr[:,2], p_arr[:,3], p_arr[:,4]

##### EFFECT OF CONDITION ######

# melt df
n_timepoints = mode_data.shape[1]
time_cols = [f't{i}' for i in range(n_timepoints)]
df_data = pd.DataFrame(mode_data, columns=time_cols)
df_full = pd.concat([df.reset_index(drop=True), df_data], axis=1)
mode_timecourses = df_full.groupby(['Subject', 'Condition'])[time_cols].mean().reset_index()

# plot conditions
colors = sns.color_palette("Set2", 3)
times = epochs.times
fig, ax = plt.subplots(figsize=(6,4))
nonshift = mode_timecourses[mode_timecourses['Condition']==1][time_cols].to_numpy()
line_with_error(ax, nonshift, times, colors[0], label='Non-Shift')
implicit = mode_timecourses[mode_timecourses['Condition']==2][time_cols].to_numpy()
line_with_error(ax, implicit, times, colors[1], label='Implicit')
explicit = mode_timecourses[mode_timecourses['Condition']==3][time_cols].to_numpy()
line_with_error(ax, explicit, times, colors[2], label='Explicit')

# shade significant effects
reject, _ = mne.stats.fdr_correction(p_cond)
reject_labelled, _ = label(reject)
for cluster in np.unique(reject_labelled)[1:]:
    tmin = np.min(times_decim[reject_labelled==cluster])
    tmax = np.max(times_decim[reject_labelled==cluster])
    ax.axvspan(tmin, tmax, color='gray', alpha=0.1)     

# finish plot
ax.axvline(0, color='black', linestyle='--', alpha=0.2)
ax.set_xlim([-0.5, 1.5])
ax.set_xlabel('Time (s)', fontsize=16)
ax.set_ylabel('Mode Activity', fontsize=16)
ax.set_title('Effect of Condition', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=12)
plt.tight_layout()

##### EFFECT OF RT #############

def label_rt_speed(group):
    q1 = group['RT'].quantile(0.25)
    q3 = group['RT'].quantile(0.75)
    return pd.Series(np.where(group['RT'] <= q1, 'Fast',
                       np.where(group['RT'] >= q3, 'Slow', 'Other')),
                     index=group.index)

# melt df
n_timepoints = mode_data.shape[1]
time_cols = [f't{i}' for i in range(n_timepoints)]
df_data = pd.DataFrame(mode_data, columns=time_cols)
df_full = pd.concat([df.reset_index(drop=True), df_data], axis=1)
df_full['RT_Speed'] = (df_full.groupby('Subject', group_keys=False)[['RT']].apply(label_rt_speed).reset_index(drop=True))
mode_timecourses = df_full.groupby(['Subject', 'RT_Speed'])[time_cols].mean().reset_index()

# plot conditions
colors = sns.color_palette("Set2", 3)
times = epochs.times
fig, ax = plt.subplots(figsize=(3.5,2.5))
fast = mode_timecourses[mode_timecourses['RT_Speed']=='Fast'][time_cols].to_numpy()
line_with_error(ax, fast, times, colors[0], label='Fast')
slow = mode_timecourses[mode_timecourses['RT_Speed']=='Slow'][time_cols].to_numpy()
line_with_error(ax, slow, times, colors[1], label='Slow')

# shade significant effects
reject, _ = mne.stats.fdr_correction(p_rt)
reject_labelled, _ = label(reject)
for cluster in np.unique(reject_labelled)[1:]:
    tmin = np.min(times_decim[reject_labelled==cluster])
    tmax = np.max(times_decim[reject_labelled==cluster])
    ax.axvspan(tmin, tmax, color='gray', alpha=0.1)     

# finish plot
ax.axvline(0, color='black', linestyle='--', alpha=0.4)
ax.set_xlim([-0.5, 1.5])
ax.set_xlabel('Time (s)', fontsize=16)
ax.set_ylabel('Mode Activity', fontsize=16)
ax.set_title('Effect of RT', fontsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=14)
plt.axis(False)
plt.tight_layout()

##### EFFECT OF INCORRECT #############

# melt df
n_timepoints = mode_data.shape[1]
time_cols = [f't{i}' for i in range(n_timepoints)]
df_data = pd.DataFrame(mode_data, columns=time_cols)
df_full = pd.concat([df.reset_index(drop=True), df_data], axis=1)
mode_timecourses = df_full.groupby(['Subject', 'Correct'])[time_cols].mean().reset_index()

# plot conditions
colors = sns.color_palette("Set2", 3)
times = epochs.times
fig, ax = plt.subplots(figsize=(3.5,2.5))
correct = mode_timecourses[mode_timecourses['Correct']==1][time_cols].to_numpy()
line_with_error(ax, correct, times, colors[0], label='Correct')
incorrect = mode_timecourses[mode_timecourses['Correct']==0][time_cols].to_numpy()
line_with_error(ax, slow, times, colors[1], label='Incorrect')

# shade significant effects
reject, _ = mne.stats.fdr_correction(p_acc)
reject_labelled, _ = label(reject)
for cluster in np.unique(reject_labelled)[1:]:
    tmin = np.min(times_decim[reject_labelled==cluster])
    tmax = np.max(times_decim[reject_labelled==cluster])
    ax.axvspan(tmin, tmax, color='gray', alpha=0.1)     

# finish plot
ax.axvline(0, color='black', linestyle='--', alpha=0.4)
ax.set_xlim([-0.5, 1.5])
ax.set_xlabel('Time (s)', fontsize=16)
ax.set_ylabel('Mode Activity', fontsize=16)
ax.set_title('Effect of Accuracy', fontsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=14)
plt.axis(False)
plt.tight_layout()

##### EFFECT OF AGE ######

# melt df
n_timepoints = mode_data.shape[1]
time_cols = [f't{i}' for i in range(n_timepoints)]
df_data = pd.DataFrame(mode_data, columns=time_cols)
df_full = pd.concat([df.reset_index(drop=True), df_data], axis=1)
subject_ages = df_full.groupby('Subject')['Age'].first()
median_age = subject_ages.median()
subject_group = (subject_ages <= median_age).map({True: 'Younger', False: 'Older'})
df_full['Young_Old'] = df_full['Subject'].map(subject_group)
mode_timecourses = df_full.groupby(['Subject', 'Young_Old'])[time_cols].mean().reset_index()

# plot conditions
colors = sns.color_palette("Set2", 3)
times = epochs.times
fig, ax = plt.subplots(figsize=(3.5,2.5))
old = mode_timecourses[mode_timecourses['Young_Old']=='Older'][time_cols].to_numpy()
line_with_error(ax, old, times, colors[0], label='Older')
young = mode_timecourses[mode_timecourses['Young_Old']=='Younger'][time_cols].to_numpy()
line_with_error(ax, young, times, colors[1], label='Younger')

# shade significant effects
reject, _ = mne.stats.fdr_correction(p_age)
reject_labelled, _ = label(reject)
for cluster in np.unique(reject_labelled)[1:]:
    tmin = np.min(times_decim[reject_labelled==cluster])
    tmax = np.max(times_decim[reject_labelled==cluster])
    ax.axvspan(tmin, tmax, color='gray', alpha=0.1)     

# finish plot
ax.axvline(0, color='black', linestyle='--', alpha=0.4)
ax.set_xlim([-0.5, 1.5])
ax.set_xlabel('Time (s)', fontsize=16)
ax.set_ylabel('Mode Activity', fontsize=16)
ax.set_title('Effect of Age', fontsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=14)
plt.axis(False)
plt.tight_layout()

##### EFFECT OF ADHD ######

# melt df
n_timepoints = mode_data.shape[1]
time_cols = [f't{i}' for i in range(n_timepoints)]
df_data = pd.DataFrame(mode_data, columns=time_cols)
df_full = pd.concat([df.reset_index(drop=True), df_data], axis=1)
mode_timecourses = df_full.groupby(['Subject', 'ADHD'])[time_cols].mean().reset_index()

# plot conditions
colors = sns.color_palette("Set2", 3)
times = epochs.times
fig, ax = plt.subplots(figsize=(3.5,2.5))
control = mode_timecourses[mode_timecourses['ADHD']==0][time_cols].to_numpy()
line_with_error(ax, control, times, colors[0], label='Control')
adhd = mode_timecourses[mode_timecourses['ADHD']==1][time_cols].to_numpy()
line_with_error(ax, adhd, times, colors[1], label='ADHD')

# shade significant effects
#reject, _ = mne.stats.fdr_correction(p_adhd)
reject = p_adhd < 0.05
reject_labelled, _ = label(reject)
for cluster in np.unique(reject_labelled)[1:]:
    tmin = np.min(times_decim[reject_labelled==cluster])
    tmax = np.max(times_decim[reject_labelled==cluster])
    ax.axvspan(tmin, tmax, color='pink', alpha=0.25)     

# finish plot
ax.axvline(0, color='black', linestyle='--', alpha=0.4)
ax.set_xlim([-0.5, 1.5])
ax.set_xlabel('Time (s)', fontsize=16)
ax.set_ylabel('Mode Activity', fontsize=16)
ax.set_title('Effect of ADHD', fontsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=12)
plt.axis(False)
plt.tight_layout()

##### EFFECT OF AGE and ADHD ON WHOLE TRIAL ######

# stats
window = (times>-0.5) * (times < 1.5)
mode_FO = np.mean(mode_data[:,window], -1)
df_i = df.copy()
df_i['Data'] = 1 * mode_FO
model = mixedlm("Data ~ Condition + Age + ADHD + RT + Correct", df_i, groups=df["Subject"])
result = model.fit()

# plot
age_FO = df_i.groupby(['Subject', 'Condition', 'Age', 'ADHD'])['Data'].mean().reset_index()
condition_labels = {1: 'Non-Shift', 2: 'Implicit', 3: 'Explicit'}
age_FO['ConditionLabel'] = age_FO['Condition'].map(condition_labels)
adhd_labels = {1: 'ADHD', 0: 'Control'}
age_FO['ADHDLabel'] = age_FO['ADHD'].map(adhd_labels)
g = sns.lmplot(
    data=age_FO,
    x='Age',
    y='Data',
    hue='ConditionLabel',
    palette='Set2',
    col='ADHD',
    aspect=1.1,
    height=3,
    facet_kws={'sharex':True, 'sharey':True},
    scatter_kws={'s': 30, 'alpha': 0.6},
    line_kws={'linewidth': 2},
)
g.despine(left=True, bottom=True)
g.set_xlabels('Age (yrs)', fontsize=12)
g.set_ylabels('Mode Activity', fontsize=12)

# plot ADHD vs controls
fig, ax = plt.subplots(figsize=(6,3))
g = sns.boxplot(
    data=age_FO,
    x='ConditionLabel',  # Conditions on the x-axis
    y='Data',            # 'Data' on the y-axis
    hue='ADHDLabel',          # Group by ADHD vs control
    order=['Non-Shift', 'Implicit', 'Explicit'],  # Ensure specific condition order
    palette='Set2',       # Optional: color palette for ADHD and control
    ax=ax,
    legend=False,
)
ax.set_xlabel(None)
ax.set_ylabel('Mode Activity', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()





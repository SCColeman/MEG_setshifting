#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse DyNeMo modes.

@author: sebastiancoleman
"""

import mne
import os
import os.path as op
from mne_connectivity import spectral_connectivity_epochs
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from nilearn import image, datasets, plotting
import nibabel as nib
from scipy.stats import pearsonr, ttest_ind, ttest_1samp, zscore, ttest_rel
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import pandas as pd
from scipy.ndimage import uniform_filter1d, label
import seaborn as sns

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
    img = image.new_img_like(img, mni.get_fdata()*img.get_fdata())
    
    return img

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
    ax.plot(x[sort_i],y_pred[sort_i], color=color, label=label)
    ax.fill_between(x[sort_i], ci[sort_i, 0], ci[sort_i, 1], color=color, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    #ax.grid('on', alpha=0.3)
    
    return stat, p, slope

def cluster_perm_ttest(a, b, n_permutations=1000, alpha=0.05):
   
    # Compute t-test
    t_values, p_values = ttest_ind(a, b, axis=0)
    
    # Identify clusters in observed data
    observed_clusters, n_clusters = label(p_values < alpha)
    observed_cluster_stats = [np.sum(observed_clusters == i + 1) for i in range(n_clusters)]
    
    # Permutation test
    combined = np.vstack((a, b))
    labels = np.array([0] * len(a) + [1] * len(b))
    cluster_stats_perm = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        np.random.shuffle(labels)
        a_perm = combined[labels == 0]
        b_perm = combined[labels == 1]
        t_perm, p_perm = ttest_ind(a_perm, b_perm, axis=0)
        perm_clusters, n_perm_clusters = label(p_perm < alpha)
        if n_perm_clusters > 0:
            cluster_stats_perm[i] = np.max([np.sum(perm_clusters == j + 1) for j in range(n_perm_clusters)], initial=0)
    
    # Compute p-values for observed clusters
    cluster_p_values = [np.mean(cluster_stats_perm > obs_stat) for obs_stat in observed_cluster_stats]
    
    # Extract significant clusters
    significant_clusters = [(np.where(observed_clusters == i + 1)[0][0], 
                             np.where(observed_clusters == i + 1)[0][-1], 
                             cluster_p_values[i])
                            for i in range(n_clusters) if cluster_p_values[i] < alpha]
    
    return significant_clusters

#%% set up paths

data_path = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/data'
deriv_path = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/derivatives'
subjects_dir = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/subjects_dir'
hmm_dir = r'/d/gmi/1/sebastiancoleman/MEG_setshifting/dynemo/6_modes'
alpha = np.load(op.join(hmm_dir, 'inf_params', 'alp_rw.pkl'),allow_pickle=True)
covs = np.load(op.join(hmm_dir, 'inf_params', 'covs.npy'))
psd = np.load(op.join(hmm_dir, 'spectra', 'psd_rs.npy'))
psd_base = psd[:,1]
psd_coef = psd[:,0]
coh = np.load(op.join(hmm_dir, 'spectra', 'coh.npy'))
freqs = np.load(op.join(hmm_dir, 'spectra', 'f.npy'))

#%% load pheno csv

df = pd.read_table(op.join(data_path, 'pheno.csv'), delimiter=',')
subj_id = df['subj_id'].to_numpy()
ages = df['age'].to_numpy()
is_adhd = df['is_ADHD'].to_numpy()

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

#%% plot mode psd at peak region

colors = plt.cm.tab10.colors[:8]

mode = 0
max_reg = np.argmax(np.mean(psd_coef[:,mode,:,:], (0,-1)))
base_mean = np.mean(psd_base[:,mode,:,:], 0)
coef_mean = np.mean(psd_coef[:,mode,:,:], 0)
mode_psd = (base_mean.T + (coef_mean.T)).T

fig, ax = plt.subplots(figsize=(3.2,2))
ax.plot(freqs, mode_psd[max_reg,:], color=colors[mode], linewidth=2.5, alpha=1, label='Mode PSD')
ax.plot(freqs, np.mean(base_mean,0), color='black', linestyle='--', linewidth=2, label='Static PSD')
ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.set_xlim([3, 40])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.legend()
plt.tight_layout()

#%% plot mode map

for mode in range(6):
    coef_mean = np.mean(psd_coef[:,mode,:,:], 0)
    freq_range = (freqs>=6) * (freqs<=15)
    power = np.mean(coef_mean[:,freq_range], -1)
    power *= 100
    img = make_atlas_nifti(atlas, power)
    fig = surface_brain_plot(img, subjects_dir, 'pial', 'RdBu_r', threshold=0, cbar_label='Coefficient')
    fig.savefig('/d/gmi/1/sebastiancoleman/MEG_setshifting/dynemo/6_modes/figures/mode' + str(mode+1) + '_power.png')
    plt.close()

#%% plot mode coh

for mode in range(6):
    mode_coh = np.mean(coh[:,mode,:,:,:], 0)
    freq_range = (freqs>=6) * (freqs<=15)
    conn = np.nanmean(mode_coh[:,:,freq_range], -1)
    triu = np.triu_indices(len(names), k=1)
    np.fill_diagonal(conn, np.nan)
    threshold = np.nanpercentile(np.abs(conn[triu]), 97)
    fig = glass_brain_plot(conn, coords, threshold, 'Mode Coherence')
    fig.savefig('/d/gmi/1/sebastiancoleman/MEG_setshifting/dynemo/6_modes/figures/mode' + str(mode+1) + '_coh.png')
    plt.close()

#%% get mode timecourses and behaviour

all_adhd = []
all_age = []
all_shift = []
all_nonshift = []
all_implicit = []
all_explicit = []
all_trials = []
all_RT = []
all_FO = []
all_fast = []
all_slow = []
all_resp_fast = []
all_resp_slow = []
all_resp = []
all_score = []
all_ntrials = []

for s, subject in enumerate(subjects):
    
    # get clinical variables
    sub_ind = subj_id==subject
    age = ages[sub_ind][0]
    adhd = is_adhd[sub_ind][0]
    
    # load data
    fname = op.join(deriv_path, subject, subject + '_source_orth-raw.fif')
    raw = mne.io.Raw(fname, preload=True)
    
    # annotate bad segments (treat as resting mode)
    # THIS NEEDS TO BE SAME AS PREPPED DATA
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
    
    # load subject alpha and bad samples
    sub_alpha = alpha[s].T
    FO = np.mean(sub_alpha, 1)
    bads = np.load(op.join('/d/gmi/1/sebastiancoleman/MEG_setshifting/dynemo/bads', subject + '_bad_mask.npy'))
    
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
    alpha_info = mne.create_info(['mode' + str(mode+1) for mode in range(6)], 250)
    alpha_raw = mne.io.RawArray(alpha_pad, alpha_info)
    alpha_raw.set_meas_date(raw.info['meas_date'])
    alpha_raw.set_annotations(raw.annotations)
    
    # epoch
    events, ids = mne.events_from_annotations(alpha_raw)
    onsets = events[:,0]
    
    # epoch explicit and implciit
    implicit_epochs = mne.Epochs(alpha_raw, events, ids['IntraDim'], tmin=-0.8, tmax=1.5, preload=True)
    explicit_epochs = mne.Epochs(alpha_raw, events, ids['ExtraDim'], tmin=-0.8, tmax=1.5, preload=True)
    implicit_evoked = implicit_epochs.average('all').get_data()
    explicit_evoked = explicit_epochs.average('all').get_data()
    
    # make new events related to set-shifting and correct responses
    new_events = []
    for e in range(events.shape[0]):
        event = events[e,:]
        event_onset = event[0]
        if event[2]==ids['LeftTarget'] or event[2]==ids['RightTarget']:
            new_event = event.copy()
            other_events = events[onsets==event_onset,:]
            other_vals = other_events[:,2]
            if np.any((other_vals==ids['ExtraDim']) + (other_vals==ids['IntraDim'])):
                new_event[2] = 2
            else:
                new_event[2] = 1
            new_events.append(new_event)
        elif event[2]==ids['Correct']:
            new_event = event.copy()
            new_event[2] = 3
            new_events.append(new_event)
    new_events = np.concatenate([new_events], 0)
    
    # epoch around set-shift trials
    nonshift_epochs = mne.Epochs(alpha_raw, new_events, 1, tmin=-0.8, tmax=1.5, preload=True)
    shift_epochs = mne.Epochs(alpha_raw, new_events, 2, tmin=-0.8, tmax=1.5, preload=True)
    epochs = mne.Epochs(alpha_raw, new_events, [1,2], tmin=-0.8, tmax=1.5, preload=True)
    nonshift_evoked = nonshift_epochs.average('all').get_data()
    shift_evoked = shift_epochs.average('all').get_data()
    all_evoked = epochs.average('all').get_data()
    nshift = len(shift_epochs)
    nnon = len(nonshift_epochs)
    
    # get RT and score from all trials
    trial_onsets = epochs.events[:,0]
    resp_onsets = new_events[new_events[:,2]==3,0]
    RTs = np.zeros(len(trial_onsets))
    for o, onset in enumerate(trial_onsets):
        lags = (resp_onsets - onset) / raw.info['sfreq']
        lags[lags<0] = 0
        if np.any(lags>0):
            RT = np.min(lags[lags>0])
            if RT < 4:
                RTs[o] = RT
    correct = RTs > 0
    RTs[RTs==0] = np.nan
    med = np.nanmedian(RTs)
    upper = np.nanpercentile(RTs, 70)
    lower = np.nanpercentile(RTs, 30)
    fast = RTs < lower
    slow = RTs > upper
    fast_evoked = epochs[fast].average('all').get_data()
    slow_evoked = epochs[slow].average('all').get_data()
    
    # epoch around resp
    resp_epochs = mne.Epochs(alpha_raw, new_events, 3, tmin=-1.5, tmax=0.8, preload=True)
    resp_evoked = resp_epochs.average('all').get_data()
    
    # get fast and slow resp
    resp_onsets = resp_epochs.events[:,0]
    RTs = np.zeros(len(resp_onsets))
    for o, onset in enumerate(resp_onsets):
        lags = (onset - trial_onsets) / raw.info['sfreq']
        lags[lags<0] = 0
        if np.any(lags>0):
            RT = np.min(lags[lags>0])
            if RT < 4:
                RTs[o] = RT
    correct = RTs > 0
    RTs[RTs==0] = np.nan
    upper = np.nanpercentile(RTs, 70)
    lower = np.nanpercentile(RTs, 30)
    fast = RTs < lower
    slow = RTs > upper
    fast_resp_evoked = resp_epochs[fast].average('all').get_data()
    slow_resp_evoked = resp_epochs[slow].average('all').get_data()
    
    # store out
    if not np.isnan(adhd) and not np.isnan(age):
        all_shift.append(shift_evoked)
        all_nonshift.append(nonshift_evoked)
        all_adhd.append(adhd)
        all_age.append(age)
        all_RT.append(RTs[np.isnan(RTs)==False])
        all_FO.append(FO)
        all_trials.append(all_evoked)
        all_fast.append(fast_evoked)
        all_slow.append(slow_evoked)
        all_explicit.append(explicit_evoked)
        all_implicit.append(implicit_evoked)
        all_resp.append(resp_evoked)
        all_resp_fast.append(fast_resp_evoked)
        all_resp_slow.append(slow_resp_evoked)
        all_score.append(np.mean(correct)*100)
        all_ntrials.append([nshift, nnon])
    
all_shift = np.array(all_shift)
all_nonshift = np.array(all_nonshift)
all_adhd = np.array(all_adhd)
all_age = np.array(all_age)
all_FO = np.array(all_FO)
all_trials = np.array(all_trials)
all_fast = np.array(all_fast)
all_slow = np.array(all_slow)
all_explicit = np.array(all_explicit)
all_implicit = np.array(all_implicit)
all_resp = np.array(all_resp)
all_score = np.array(all_score)
all_resp_fast = np.array(all_resp_fast)
all_resp_slow = np.array(all_resp_slow)
all_ntrials = np.array(all_ntrials)

#%%  plot all mode timecourses, trial onset

times = shift_epochs.times
baseline = (times>-0.1) * (times < 0)
mean_timecourses = np.mean([all_shift, all_nonshift], 0)
fig, ax = plt.subplots(figsize=(7,3))

for mode in range(6):
    mode_timecourse = mean_timecourses[:,mode,:]
    values = np.mean(mode_timecourse, 0)
    values -= np.mean(values[baseline])
    err = np.std(mode_timecourse, 0) / np.sqrt(mode_timecourse.shape[0])
    ax.plot(times, values, label='Mode ' + str(mode+1))
    ax.fill_between(times, values-err, values+err, alpha=0.2)
ax.axvline(0, linestyle='--', color=[0.2,0.2,0.2, 0.5], label=None)
ax.set_xlim([-0.25, 2])
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Mode Activation', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.legend()
plt.tight_layout()

#%%  plot all mode timecourses, resp 

times = resp_epochs.times
baseline = (times>-0.1) * (times < 0.1)
mean_timecourses = all_resp.copy()
fig, ax = plt.subplots(figsize=(7,3))

for mode in range(6):
    mode_timecourse = mean_timecourses[:,mode,:]
    values = np.mean(mode_timecourse, 0)
    values -= np.mean(values[baseline])
    err = np.std(mode_timecourse, 0) / np.sqrt(mode_timecourse.shape[0])
    ax.plot(times, values, label='Mode ' + str(mode+1))
    ax.fill_between(times, values-err, values+err, alpha=0.2)
ax.axvline(0, linestyle='--', color=[0.2,0.2,0.2, 0.5], label=None)
ax.set_xlim([-0.5, 1.2])
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Mode Activation', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.legend()
plt.tight_layout()

#%% plot FO

values = np.mean(all_FO, 0)
err = np.std(all_FO, 0) 

colors = plt.cm.tab10.colors[:8]
fig, ax = plt.subplots(figsize=(4,3.5))
ax.bar(np.arange(1,6+1), values, yerr=err, color=colors)
ax.set_ylim([0,None])
ax.set_xticks(np.arange(1,6+1))
ax.tick_params(labelsize=12)
ax.set_xlabel('Mode', fontsize=14)
ax.set_ylabel('Fractional Occupancy', fontsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()

#%% compare shift and non-shift trials

colors = sns.color_palette("Set2", 3)

a = all_explicit.copy()
b = all_implicit.copy()
c = all_nonshift.copy()
times = shift_epochs.times

# stats and plots
fig, ax = plt.subplots(2,3, figsize=(12,6))
ax = ax.flatten()
for mode in range(6):
    a_mode = a[:,mode,:]
    b_mode = b[:,mode,:]
    c_mode = c[:,mode,:]
    t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(
                                a_mode - c_mode, n_jobs=1)
    
    # plot
    values = np.mean(c_mode, 0)
    err = np.std(c_mode, 0) / np.sqrt(c_mode.shape[0])
    ax[mode].plot(times, values, color=colors[2], label='Non-Shift')
    ax[mode].fill_between(times, values-err, values+err, alpha=0.2, color=colors[2])
    values = np.mean(a_mode, 0)
    err = np.std(a_mode, 0) / np.sqrt(a_mode.shape[0])
    ax[mode].plot(times, values, color=colors[1], label='Extra-Dim')
    ax[mode].fill_between(times, values-err, values+err, alpha=0.2, color=colors[1]) 
    ymin, ymax = ax[mode].get_ylim()
    yrange = ymax - ymin
    for cluster in range(len(clusters)):
        if cluster_pv[cluster] < 0.05:
            onset = clusters[cluster][0][0]
            end = clusters[cluster][0][-1]
            ax[mode].axvspan(times[onset], times[end], color='gray', alpha=0.15)
    ax[mode].set_ylim([ymin-(0.05*yrange), ymax+(0.05*yrange)])
    ax[mode].axvline(0, color='black', linestyle='--', alpha=0.2)
    ax[mode].set_xlim([-0.25, 1.5])
    ax[mode].set_xlabel('Time (s)', fontsize=12)
    ax[mode].set_ylabel('Probability', fontsize=12)
    ax[mode].set_title('Mode ' + str(mode+1), fontsize=14)
    ax[mode].spines['top'].set_visible(False)
    ax[mode].spines['right'].set_visible(False)
    ax[mode].spines['bottom'].set_visible(False)
    ax[mode].spines['left'].set_visible(False)
    ax[mode].legend()
    
plt.tight_layout()

#%% compare slow and fast trial onsets

colors = sns.color_palette("Set2", 3)

a = all_slow.copy()
b = all_fast.copy()
times = shift_epochs.times

# stats and plots
fig, ax = plt.subplots(2,3, figsize=(12,6))
ax = ax.flatten()
for mode in range(6):
    a_mode = a[:,mode,:]
    b_mode = b[:,mode,:]
    t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(
                                a_mode - b_mode, n_jobs=1)
    
    # plot
    values = np.mean(b_mode, 0)
    err = np.std(b_mode, 0) / np.sqrt(b_mode.shape[0])
    ax[mode].plot(times, values, color=colors[2], label='Fast')
    ax[mode].fill_between(times, values-err, values+err, alpha=0.2, color=colors[2])
    values = np.mean(a_mode, 0)
    err = np.std(a_mode, 0) / np.sqrt(a_mode.shape[0])
    ax[mode].plot(times, values, color=colors[1], label='Slow')
    ax[mode].fill_between(times, values-err, values+err, alpha=0.2, color=colors[1]) 
    ymin, ymax = ax[mode].get_ylim()
    yrange = ymax - ymin
    for cluster in range(len(clusters)):
        if cluster_pv[cluster] < 0.05:
            onset = clusters[cluster][0][0]
            end = clusters[cluster][0][-1]
            #ax[mode].plot([times[onset], times[end]], [ymin, ymin], color='black', linewidth=2.5, alpha=0.7)
            ax[mode].axvspan(times[onset], times[end], color='gray', alpha=0.15)
    ax[mode].set_ylim([ymin-(0.05*yrange), ymax+(0.05*yrange)])
    ax[mode].axvline(0, color='black', linestyle='--', alpha=0.2)
    ax[mode].set_xlim([-0.25, 1.5])
    ax[mode].set_xlabel('Time (s)', fontsize=12)
    ax[mode].set_ylabel('Probability', fontsize=12)
    ax[mode].set_title('Mode ' + str(mode+1), fontsize=14)
    ax[mode].spines['top'].set_visible(False)
    ax[mode].spines['right'].set_visible(False)
    ax[mode].spines['bottom'].set_visible(False)
    ax[mode].spines['left'].set_visible(False)
    ax[mode].legend()
    
plt.tight_layout()

#%% compare slow and fast responses

colors = sns.color_palette("Set2", 3)

a = all_resp_slow.copy()
b = all_resp_fast.copy()
times = resp_epochs.times

# stats and plots
fig, ax = plt.subplots(2,3, figsize=(12,6))
ax = ax.flatten()
for mode in range(6):
    a_mode = a[:,mode,:]
    b_mode = b[:,mode,:]
    t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(
                                a_mode - b_mode, n_jobs=1)
    
    # plot
    values = np.mean(b_mode, 0)
    err = np.std(b_mode, 0) / np.sqrt(b_mode.shape[0])
    ax[mode].plot(times, values, color=colors[2], label='Fast')
    ax[mode].fill_between(times, values-err, values+err, alpha=0.2, color=colors[2])
    values = np.mean(a_mode, 0)
    err = np.std(a_mode, 0) / np.sqrt(a_mode.shape[0])
    ax[mode].plot(times, values, color=colors[1], label='Slow')
    ax[mode].fill_between(times, values-err, values+err, alpha=0.2, color=colors[1]) 
    ymin, ymax = ax[mode].get_ylim()
    yrange = ymax - ymin
    for cluster in range(len(clusters)):
        if cluster_pv[cluster] < 0.05:
            onset = clusters[cluster][0][0]
            end = clusters[cluster][0][-1]
            ax[mode].axvspan(times[onset], times[end], color='gray', alpha=0.15)
    ax[mode].set_ylim([ymin-(0.05*yrange), ymax+(0.05*yrange)])
    ax[mode].axvline(0, color='black', linestyle='--', alpha=0.2)
    ax[mode].set_xlim([-1, 0.8])
    ax[mode].set_xlabel('Time (s)', fontsize=12)
    ax[mode].set_ylabel('Probability', fontsize=12)
    ax[mode].set_title('Mode ' + str(mode+1), fontsize=14)
    ax[mode].spines['top'].set_visible(False)
    ax[mode].spines['right'].set_visible(False)
    ax[mode].spines['bottom'].set_visible(False)
    ax[mode].spines['left'].set_visible(False)
    ax[mode].legend()
    
plt.tight_layout()

#%% compare all trials, adhd vs controls

colors = ['#6a7edb','#f58b6a']

# trials
a = uniform_filter1d(all_trials[(all_adhd==1)], 1)
b = uniform_filter1d(all_trials[(all_adhd==0)], 1)
times = shift_epochs.times

# stats and plots
fig, ax = plt.subplots(2,3, figsize=(12,6))
ax = ax.flatten()
for mode in range(6):
    a_mode = a[:,mode,:]
    b_mode = b[:,mode,:]
    significant_clusters = cluster_perm_ttest(a_mode, b_mode)
    
    # plot
    values = np.mean(b_mode, 0)
    err = np.std(b_mode, 0) / np.sqrt(b_mode.shape[0])
    ax[mode].plot(times, values, color=colors[0], label='Control')
    ax[mode].fill_between(times, values-err, values+err, alpha=0.2, color=colors[0])
    values = np.mean(a_mode, 0)
    err = np.std(a_mode, 0) / np.sqrt(a_mode.shape[0])
    ax[mode].plot(times, values, color=colors[1], label='ADHD')
    ax[mode].fill_between(times, values-err, values+err, alpha=0.2, color=colors[1]) 
    ymin, ymax = ax[mode].get_ylim()
    yrange = ymax - ymin
    for cluster in significant_clusters:
        onset = cluster[0]
        end = cluster[1]
        ax[mode].axvspan(times[onset], times[end], color='gray', alpha=0.15)
    ax[mode].set_ylim([ymin-(0.05*yrange), ymax+(0.05*yrange)])
    ax[mode].axvline(0, color='black', linestyle='--', alpha=0.2)
    ax[mode].set_xlim([-0.2, 1.5])
    ax[mode].set_xlabel('Time (s)', fontsize=12)
    ax[mode].set_ylabel('Probability', fontsize=12)
    ax[mode].set_title('Mode ' + str(mode+1), fontsize=14)
    ax[mode].spines['top'].set_visible(False)
    ax[mode].spines['right'].set_visible(False)
    ax[mode].spines['bottom'].set_visible(False)
    ax[mode].spines['left'].set_visible(False)
    ax[mode].legend()
    
plt.tight_layout()

#%% compare all trials, slow vs fast RT

colors = ['#6a7edb','#f58b6a']

# trials
mean_RT = np.array([np.var(inst) for inst in all_RT])
a = uniform_filter1d(all_trials[mean_RT>np.median(mean_RT)], 1)
b = uniform_filter1d(all_trials[mean_RT<np.median(mean_RT)], 1)
times = shift_epochs.times

# stats and plots
fig, ax = plt.subplots(2,3, figsize=(12,6))
ax = ax.flatten()
for mode in range(6):
    a_mode = a[:,mode,:]
    b_mode = b[:,mode,:]
    significant_clusters = cluster_perm_ttest(a_mode, b_mode)
    
    # plot
    values = np.mean(b_mode, 0)
    err = np.std(b_mode, 0) / np.sqrt(b_mode.shape[0])
    ax[mode].plot(times, values, color=colors[0], label='Low Var')
    ax[mode].fill_between(times, values-err, values+err, alpha=0.2, color=colors[0])
    values = np.mean(a_mode, 0)
    err = np.std(a_mode, 0) / np.sqrt(a_mode.shape[0])
    ax[mode].plot(times, values, color=colors[1], label='High Var')
    ax[mode].fill_between(times, values-err, values+err, alpha=0.2, color=colors[1]) 
    ymin, ymax = ax[mode].get_ylim()
    yrange = ymax - ymin
    for cluster in significant_clusters:
        onset = cluster[0]
        end = cluster[1]
        ax[mode].axvspan(times[onset], times[end], color='gray', alpha=0.15)
    ax[mode].set_ylim([ymin-(0.05*yrange), ymax+(0.05*yrange)])
    ax[mode].axvline(0, color='black', linestyle='--', alpha=0.2)
    
    ax[mode].set_xlabel('Time (s)', fontsize=12)
    ax[mode].set_ylabel('Probability', fontsize=12)
    ax[mode].set_title('Mode ' + str(mode+1), fontsize=14)
    ax[mode].spines['top'].set_visible(False)
    ax[mode].spines['right'].set_visible(False)
    ax[mode].spines['bottom'].set_visible(False)
    ax[mode].spines['left'].set_visible(False)
    ax[mode].legend()
    
plt.tight_layout()

#%% compare all trials, low vs high score

colors = ['#6a7edb','#f58b6a']

# trials
mean_RT = np.array([np.median(inst) for inst in all_RT])
a = uniform_filter1d(all_trials[all_score<np.mean(all_score)], 1)
b = uniform_filter1d(all_trials[all_score>np.mean(all_score)], 1)
times = shift_epochs.times

# stats and plots
fig, ax = plt.subplots(2,3, figsize=(12,6))
ax = ax.flatten()
for mode in range(6):
    a_mode = a[:,mode,:]
    b_mode = b[:,mode,:]
    significant_clusters = cluster_perm_ttest(a_mode, b_mode)
    
    # plot
    values = np.mean(b_mode, 0)
    err = np.std(b_mode, 0) / np.sqrt(b_mode.shape[0])
    ax[mode].plot(times, values, color=colors[0], label='high score')
    ax[mode].fill_between(times, values-err, values+err, alpha=0.2, color=colors[0])
    values = np.mean(a_mode, 0)
    err = np.std(a_mode, 0) / np.sqrt(a_mode.shape[0])
    ax[mode].plot(times, values, color=colors[1], label='low score')
    ax[mode].fill_between(times, values-err, values+err, alpha=0.2, color=colors[1]) 
    ymin, ymax = ax[mode].get_ylim()
    yrange = ymax - ymin
    for cluster in significant_clusters:
        onset = cluster[0]
        end = cluster[1]
        ax[mode].axvspan(times[onset], times[end], color='gray', alpha=0.15)
    ax[mode].set_ylim([ymin-(0.05*yrange), ymax+(0.05*yrange)])
    ax[mode].axvline(0, color='black', linestyle='--', alpha=0.2)
    
    ax[mode].set_xlabel('Time (s)', fontsize=12)
    ax[mode].set_ylabel('Probability', fontsize=12)
    ax[mode].set_title('Mode ' + str(mode+1), fontsize=14)
    ax[mode].spines['top'].set_visible(False)
    ax[mode].spines['right'].set_visible(False)
    ax[mode].spines['bottom'].set_visible(False)
    ax[mode].spines['left'].set_visible(False)
    ax[mode].legend()
    
plt.tight_layout()


#%% compare all trials, old vs young

colors = ['#6a7edb','#f58b6a']

# trials
a = uniform_filter1d(all_trials[all_age<12], 1)
b = uniform_filter1d(all_trials[all_age>=12], 1)
times = shift_epochs.times

# stats and plots
fig, ax = plt.subplots(2,3, figsize=(12,6))
ax = ax.flatten()
for mode in range(6):
    a_mode = a[:,mode,:]
    b_mode = b[:,mode,:]
    significant_clusters = cluster_perm_ttest(a_mode, b_mode)
    
    # plot
    values = np.mean(b_mode, 0)
    err = np.std(b_mode, 0) / np.sqrt(b_mode.shape[0])
    ax[mode].plot(times, values, color=colors[0], label='12yrs+')
    ax[mode].fill_between(times, values-err, values+err, alpha=0.2, color=colors[0])
    values = np.mean(a_mode, 0)
    err = np.std(a_mode, 0) / np.sqrt(a_mode.shape[0])
    ax[mode].plot(times, values, color=colors[1], label='<12yrs')
    ax[mode].fill_between(times, values-err, values+err, alpha=0.2, color=colors[1]) 
    ymin, ymax = ax[mode].get_ylim()
    yrange = ymax - ymin
    for cluster in significant_clusters:
        onset = cluster[0]
        end = cluster[1]
        ax[mode].axvspan(times[onset], times[end], color='gray', alpha=0.15)
    ax[mode].set_ylim([ymin-(0.05*yrange), ymax+(0.05*yrange)])
    ax[mode].axvline(0, color='black', linestyle='--', alpha=0.2)
    
    ax[mode].set_xlabel('Time (s)', fontsize=12)
    ax[mode].set_ylabel('Probability', fontsize=12)
    ax[mode].set_title('Mode ' + str(mode+1), fontsize=14)
    ax[mode].spines['top'].set_visible(False)
    ax[mode].spines['right'].set_visible(False)
    ax[mode].spines['bottom'].set_visible(False)
    ax[mode].spines['left'].set_visible(False)
    ax[mode].legend()
    
plt.tight_layout()

#%% FO with age

colors = ['#003366','#cc4c00']
fig, ax = plt.subplots(2,3, figsize=(11,6))
ax = ax.flatten()
for mode in range(6):
    
    a = all_age[all_adhd<3]
    b = all_FO[all_adhd<3,mode]
    stat, p, slope = regression_plot(a, b, ax[mode], color=colors[0])
    print('mode ' + str(mode+1) + ': p = ' + str(p))
    print('mode ' + str(mode+1) + ': t = ' + str(stat))
    print('mode ' + str(mode+1) + ': slope = ' + str(slope))
    
    ax[mode].set_xlabel('Age (yrs)', fontsize=12)
    ax[mode].set_ylabel('Fractional Occupancy', fontsize=12)
    ax[mode].set_title('Mode ' + str(mode+1), fontsize=14)
    
plt.tight_layout()

#%% FO with RT

colors = ['#003366','#cc4c00']
mean_RT = np.array([np.median(inst) for inst in all_RT])
fig, ax = plt.subplots(2,3, figsize=(11,6))
ax = ax.flatten()
for mode in range(6):
    
    a = mean_RT[all_adhd<3]
    b = all_FO[all_adhd<3,mode]
    stat, p, slope = regression_plot(a, b, ax[mode], color=colors[0])
    print('mode ' + str(mode+1) + ': p = ' + str(p))
    print('mode ' + str(mode+1) + ': t = ' + str(stat))
    print('mode ' + str(mode+1) + ': slope = ' + str(slope))
    
    ax[mode].set_xlabel('Median RT (s)', fontsize=12)
    ax[mode].set_ylabel('Fractional Occupancy', fontsize=12)
    ax[mode].set_title('Mode ' + str(mode+1), fontsize=14)
    
plt.tight_layout()
  
#%% ADHD vs controls

fig, ax = plt.subplots(2,3, figsize=(11,6))
ax = ax.flatten()
fo_shift = np.mean(all_shift[:,:,:],-1)
for mode in range(6):
    
    cond1 = fo_shift[all_adhd==0,mode]
    cond2 = fo_shift[all_adhd==1,mode]
    
    # stats
    stat, p = ttest_ind(cond1, cond2)
    print('mode ' + str(mode+1) + ': p = ' + str(p))
    print('mode ' + str(mode+1) + ': t = ' + str(stat))
    
    # plot
    box = ax[mode].boxplot([cond1, cond2], labels=['Control', 'ADHD'], patch_artist=True)
    
    # Customize the colors and styles
    colors = ['#66B2FF', '#FF9999']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Customize the whiskers, caps, and medians
    for whisker in box['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.5)
    for cap in box['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.5)
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    ax[mode].set_ylabel('Probability', fontsize=12)
    ax[mode].set_title('mode ' + str(mode+1), fontsize=14)
    ax[mode].spines['top'].set_visible(False)
    ax[mode].spines['right'].set_visible(False)
    ax[mode].spines['bottom'].set_visible(False)
    ax[mode].spines['left'].set_visible(False)
plt.tight_layout()



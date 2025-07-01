#%%
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.stats import spatio_temporal_cluster_test
from mne.channels import find_ch_adjacency
import matplotlib.pyplot as plt
from Mypackage.Processed_toolbox import *
import pickle
#%%
alpha_v = 0.05
save_file_path = f'.\\Fig\\data'

# Standard:
# Load the arrays from the .npz file
data = np.load(os.path.join(save_file_path,'Standard_data_plot.npz'))
Std_350 = data['Std_350']
Std_350_rest = data['Std_350_rest']
# Local:
data = np.load(os.path.join(save_file_path, 'Local_effect.npz'))
# frequency deviants
Local_Dev_1 = data['Dev_1']
Local_STD_1 = data['STD_1']
# numerical deviants
Local_Dev_2 = data['Dev_2']
Local_STD_2 = data['STD_2']
# Global:
data = np.load(os.path.join(save_file_path, 'Global_effect.npz'))
# Global effect I:  num-1000
Global_Dev_1 = data['Dev_1']
Global_STD_1 = data['STD_1']
# Global effect II:  num-1500
Global_Dev_2 = data['Dev_2']
Global_STD_2 = data['STD_2']

# times:
with open(os.path.join(save_file_path,'times_350.pkl'), 'rb') as f:
    times_350 = pickle.load(f)
times_350 = times_350
with open(os.path.join(save_file_path,'times_800.pkl'), 'rb') as f:
    times_800 = pickle.load(f)
with open('desired_chan_order.pkl', 'rb') as f:
    desired_chan_order = pickle.load(f)
times_800 = times_800
#%%
Pathname = r'E:\MyWork\Graduation\ASAP_ICA_EEG_ArtifactRejection_v1\Re_data\healthy'
Listing = os.listdir(Pathname)
for isSub in [0]:     
    Sub_Path = os.path.join(Pathname, Listing[isSub]) 
    Cond_list = os.listdir(Sub_Path)
    for istype in [0]:
        Data_Path = os.path.join(Sub_Path, Cond_list[istype])
        raw = mne.io.read_raw_eeglab(input_fname=os.path.join(Data_Path, 'Spatial_filtered_ICA_Format.set'), preload=True)  
        raw = raw.reorder_channels(desired_chan_order)
        raw.load_data()
        # M1M2 reference
        reference = ['M1', 'M2']
        raw.set_eeg_reference(reference)
        events, event_id = mne.events_from_annotations(raw)
        tmin = -0.1
        tmax = 0.35
        baseline = (None, 0)
        epochs = mne.Epochs(raw, events=events, event_id=event_id, event_repeated='merge',
                            tmin=tmin, tmax=tmax, baseline=baseline, preload=True)
        adjacency, _ = find_ch_adjacency(epochs.info, "eeg")

#%%
x = times_350  
# Figure 2 Significant areas between Active and Passive
test_data = [Std_350-Std_350_rest]
significant_points = np.empty((len(test_data), 64, 114))
for i in range(len(test_data)):
    data1 = test_data[i]    
    active_data = np.squeeze(data1[:,0,:,:])
    passive_data = np.squeeze(data1[:,1,:,:])
    X = [
        active_data.transpose(0, 2, 1),
        passive_data.transpose(0, 2, 1),
    ]
    tfce = dict(start=0, step=0.1)  # ideally start and step would be smaller

    # Calculate statistical thresholds
    t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_test(
        X, tfce, adjacency=adjacency, n_permutations=1000, tail=0, n_jobs=-1
    )  # a more standard number would be 1000+
    significant_points[i,:,:] = cluster_pv.reshape(t_obs.shape).T <= alpha_v

Sign_STD_ACT_PAS = significant_points
#%%
chan_names = 'Fz'
chan_indice = 5  # Fz:5;Cz:15;Pz:25;
window_size = 5
y1 = -6e-6
y2 = 6e-6

Active_data1 = np.mean(Std_350[:,0,:,:], axis=0)-np.mean(Std_350_rest[:,0,:,:], axis=0)
Passive_data1 = np.mean(Std_350[:,1,:,:], axis=0)-np.mean(Std_350_rest[:,1,:,:], axis=0)
mean_std_350 = Active_data1
mean_std_350_rest = Passive_data1
mean_STD = mean_std_350-mean_std_350_rest

# Standard error
std_std_350 = np.std(Std_350[:,0,:,:]-Std_350_rest[:,0,:,:], axis=0)
std_std_350_rest = np.std(Std_350[:,1,:,:]-Std_350_rest[:,1,:,:], axis=0)

rgb_1 = (244, 140, 155)
rgb_2 = (88, 159, 243)
color_1 = tuple(value / 255 for value in rgb_1)
color_2 = tuple(value / 255 for value in rgb_2)

fig, ax = plt.subplots(figsize=(6, 6), dpi=600)
ax.plot(x, mean_std_350[chan_indice, :], color=color_1, label='active', linewidth=4)
ax.plot(x, mean_std_350_rest[chan_indice, :], color=color_2, label='passive', linewidth=4)
ax.fill_between(x, 
                mean_std_350[chan_indice, :] - std_std_350[chan_indice, :], 
                mean_std_350[chan_indice, :] + std_std_350[chan_indice, :], 
                color=color_1, alpha=0.3)
ax.fill_between(x, 
                mean_std_350_rest[chan_indice, :] - std_std_350_rest[chan_indice, :], 
                mean_std_350_rest[chan_indice, :] + std_std_350_rest[chan_indice, :], 
                color=color_2, alpha=0.3)


sig_point = np.squeeze(Sign_STD_ACT_PAS[0,:,:])
true_indices = np.where(sig_point[chan_indice,:])[0]
start1 = -6e-6
start2 = -5.2e-6
start_index = None
end_index = None
positive_mask = mean_STD[chan_indice,:] > 0
negative_mask = mean_STD[chan_indice,:] < 0   

# Traverse significant time points and draw corresponding significant regions
for index in true_indices:
    if start_index is None:
        start_index = index
        end_index = index
    elif index == end_index + 1:
        end_index = index
    else:
        x_start = x[start_index]
        x_end = x[end_index]
        x_range_mask = (x >= x_start) & (x <= x_end)    # & time_window
        ax.fill_between(x, start1, start2, where=contin_significant_window(positive_mask & x_range_mask,window_size), color='red', alpha=0.3)
        ax.fill_between(x, start1, start2, where=contin_significant_window(negative_mask & x_range_mask,window_size), color='blue', alpha=0.3)
        start_index = index
        end_index = index
# Draw the salient region of the last continuous area
if start_index is not None and end_index is not None:
    x_start = x[start_index]
    x_end = x[end_index]
    x_range_mask = (x >= x_start) & (x <= x_end)
    ax.fill_between(x, start1, start2, where=contin_significant_window(positive_mask & x_range_mask,window_size), color='red', alpha=0.3)
    ax.fill_between(x, start1, start2, where=contin_significant_window(negative_mask & x_range_mask,window_size), color='blue', alpha=0.3)

ax.plot(x, np.full(len(x), start2), color='black', linewidth=0.5)
ax.set_title(f"Standard Contrast", fontsize=24, fontweight='bold')
ax.set_xlabel("Time(s)", fontsize=20)
ax.set_ylabel("Amplitude(μV)", fontsize=20)
ax.set_xlim(-0.1, 0.35)
ax.set_ylim(y1, y2)
ax.set_xticks([0, 0.1, 0.2, 0.3])
ax.set_xticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                        if tick != 0 else "0" for tick in [0, 0.1, 0.2, 0.3]],fontsize=16, fontweight='bold')
ax.set_yticks([-6e-6, -4e-6, -2e-6, 0e-6, 2e-6, 4e-6, 6e-6])
ax.set_yticklabels([f"{tick * 1e6:.1f}"[:-2] for tick in ax.get_yticks()], fontsize=16, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
legend = ax.legend(loc='upper left', fontsize=20, frameon=False)      
plt.tight_layout()
# plt.show()
save_path = (
f"./Fig/waveform/Act_Pas/"
f"STD_{chan_names}.png"
)      
plt.savefig(save_path, bbox_inches='tight')
#%%
x = times_800 
# Figure 2 Significant areas between Active and Passive
test_data = [Global_Dev_1-Global_STD_1,Global_Dev_2-Global_STD_2]
significant_points = np.empty((len(test_data), 64, 226))
T_value_Global = np.empty((len(test_data), 64, 226))
for i in range(len(test_data)):
    data1 = test_data[i]    
    active_data = np.squeeze(data1[:,0,:,:])
    passive_data = np.squeeze(data1[:,1,:,:])
    X = [
        active_data.transpose(0, 2, 1),
        passive_data.transpose(0, 2, 1),
    ]
    tfce = dict(start=0, step=0.1)  # ideally start and step would be smaller

    # Calculate statistical thresholds
    t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_test(
        X, tfce, adjacency=adjacency, n_permutations=1000, tail=0, n_jobs=-1
    )  # a more standard number would be 1000+
    significant_points[i,:,:] = cluster_pv.reshape(t_obs.shape).T <= alpha_v
    T_value_Global[i,:,:] = t_obs.T

Sign_Global_ACT_PAS = significant_points
#%%
chan_names = 'Pz'
chan_indice = 25  # Fz:5;Cz:15;Pz:25;
window_size = 5
y1 = -9e-6
y2 = 9e-6
rgb_1 = (244, 140, 155)
rgb_2 = (88, 159, 243)
color_1 = tuple(value / 255 for value in rgb_1)
color_2 = tuple(value / 255 for value in rgb_2)

Active_data1 = np.mean(Global_Dev_1[:,0,:,:], axis=0)-np.mean(Global_STD_1[:,0,:,:], axis=0)
Passive_data1 = np.mean(Global_Dev_1[:,1,:,:], axis=0)-np.mean(Global_STD_1[:,1,:,:], axis=0)
Active_data2 = np.mean(Global_Dev_2[:,0,:,:], axis=0)-np.mean(Global_STD_2[:,0,:,:], axis=0)
Passive_data2 = np.mean(Global_Dev_2[:,1,:,:], axis=0)-np.mean(Global_STD_2[:,1,:,:], axis=0)

mean_active_data = Active_data1
mean_passive_data = Passive_data1
mean_dw = mean_active_data-mean_passive_data
std_active_data = np.std(Global_Dev_1[:,0,:,:]-Global_STD_1[:,0,:,:], axis=0)
std_passive_data = np.std(Global_Dev_1[:,1,:,:]-Global_STD_1[:,1,:,:], axis=0)

fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
ax.plot(x, mean_active_data[chan_indice, :], color=color_1, label='active', linewidth=4)
ax.plot(x, mean_passive_data[chan_indice, :], color=color_2, label='passive', linewidth=4)
ax.fill_between(x, 
                mean_active_data[chan_indice, :] - std_active_data[chan_indice, :], 
                mean_active_data[chan_indice, :] + std_active_data[chan_indice, :], 
                color=color_1, alpha=0.3)
ax.fill_between(x, 
                mean_passive_data[chan_indice, :] - std_passive_data[chan_indice, :], 
                mean_passive_data[chan_indice, :] + std_passive_data[chan_indice, :], 
                color=color_2, alpha=0.3)

sig_point = np.squeeze(Sign_Global_ACT_PAS[0,:,:])
TT = np.squeeze(T_value_Global[0,chan_indice,:])
true_indices = np.where(sig_point[chan_indice,:])[0]
start1 = -9e-6
start2 = -8e-6
start_index = None
end_index = None
positive_mask = mean_dw[chan_indice,:] > 0
negative_mask = mean_dw[chan_indice,:] < 0   

# Traverse significant time points and draw corresponding significant regions
for index in true_indices:
    if start_index is None:
        start_index = index
        end_index = index
    elif index == end_index + 1:
        end_index = index
    else:
        x_start = x[start_index]
        x_end = x[end_index]
        x_range_mask = (x >= x_start) & (x <= x_end) 
        ax.fill_between(x, start1, start2, where=contin_significant_window(positive_mask & x_range_mask,window_size), color='red', alpha=0.3)
        ax.fill_between(x, start1, start2, where=contin_significant_window(negative_mask & x_range_mask,window_size), color='blue', alpha=0.3)
        start_index = index
        end_index = index

# Draw the salient region of the last continuous area
if start_index is not None and end_index is not None:
    x_start = x[start_index]
    x_end = x[end_index]
    x_range_mask = (x >= x_start) & (x <= x_end)
    ax.fill_between(x, start1, start2, where=contin_significant_window(positive_mask & x_range_mask,window_size), color='red', alpha=0.3)
    ax.fill_between(x, start1, start2, where=contin_significant_window(negative_mask & x_range_mask,window_size), color='blue', alpha=0.3)
    # Find the duration window of the component
    positive_windows = contin_significant_window(positive_mask & x_range_mask, window_size)
    if np.any(positive_windows):
        positive_indices = np.where(positive_windows)[0]
        positive_start = x[positive_indices[0]]
        positive_end = x[positive_indices[-1]]
        print(f"Positive component duration: {positive_start} to {positive_end}")
        T_value_window = TT[positive_indices]
        max_T_value = np.max(T_value_window)
        max_T_index = np.argmax(T_value_window)
        max_T_time = x[positive_indices[max_T_index]]
        print(f"Max T_value in this window {max_T_time}: {max_T_value}")
ax.plot(x, np.full(len(x), start2), color='black', linewidth=0.5)
ax.set_title(f"Global I Contrast", fontsize=24, fontweight='bold')
ax.set_xlabel("Time(s)", fontsize=20)
ax.set_ylabel("Amplitude(μV)", fontsize=20)
ax.set_xlim(-0.1, 0.8)
ax.set_ylim(y1, y2)
ax.set_xticks([0, 0.2,0.4,0.6,0.8])
ax.set_xticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                        if tick != 0 else "0" for tick in [0, 0.2,0.4,0.6,0.8]],fontsize=16, fontweight='bold')
ax.set_yticks([-8e-6, -6e-6, -4e-6, -2e-6, 0e-6, 2e-6, 4e-6, 6e-6, 8e-6])
ax.set_yticklabels([f"{tick * 1e6:.1f}"[:-2] for tick in ax.get_yticks()], fontsize=16, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
legend = ax.legend(loc='upper left', fontsize=20, frameon=False)      
plt.tight_layout()
# plt.show()
save_path = (
f"./Fig/waveform/Act_Pas/"
f"GlobalI_{chan_names}.png"
)      
plt.savefig(save_path, bbox_inches='tight')
#%%
############# global II  ##########
mean_active_data = Active_data2
mean_passive_data = Passive_data2
mean_dw = mean_active_data-mean_passive_data
std_active_data = np.std(Global_Dev_2[:,0,:,:]-Global_STD_2[:,0,:,:], axis=0)
std_passive_data = np.std(Global_Dev_2[:,1,:,:]-Global_STD_2[:,1,:,:], axis=0)

rgb_1 = (244, 140, 155)
rgb_2 = (88, 159, 243)
color_1 = tuple(value / 255 for value in rgb_1)
color_2 = tuple(value / 255 for value in rgb_2)

fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
ax.plot(x, mean_active_data[chan_indice, :], color=color_1, label='active', linewidth=4)
ax.plot(x, mean_passive_data[chan_indice, :], color=color_2, label='passive', linewidth=4)
ax.fill_between(x, 
                mean_active_data[chan_indice, :] - std_active_data[chan_indice, :], 
                mean_active_data[chan_indice, :] + std_active_data[chan_indice, :], 
                color=color_1, alpha=0.3)
ax.fill_between(x, 
                mean_passive_data[chan_indice, :] - std_passive_data[chan_indice, :], 
                mean_passive_data[chan_indice, :] + std_passive_data[chan_indice, :], 
                color=color_2, alpha=0.3)

sig_point = np.squeeze(Sign_Global_ACT_PAS[1,:,:])
TT = np.squeeze(T_value_Global[1,chan_indice,:])
true_indices = np.where(sig_point[chan_indice,:])[0]
start1 = -9e-6
start2 = -8e-6
start_index = None
end_index = None
positive_mask = mean_dw[chan_indice,:] > 0
negative_mask = mean_dw[chan_indice,:] < 0   

# Traverse significant time points and draw corresponding significant regions
for index in true_indices:
    if start_index is None:
        start_index = index
        end_index = index
    elif index == end_index + 1:
        end_index = index
    else:
        x_start = x[start_index]
        x_end = x[end_index]
        x_range_mask = (x >= x_start) & (x <= x_end)    # & time_window
        ax.fill_between(x, start1, start2, where=contin_significant_window(positive_mask & x_range_mask,window_size), color='red', alpha=0.3)
        ax.fill_between(x, start1, start2, where=contin_significant_window(negative_mask & x_range_mask,window_size), color='blue', alpha=0.3)
        start_index = index
        end_index = index
# Draw the salient region of the last continuous area
if start_index is not None and end_index is not None:
    x_start = x[start_index]
    x_end = x[end_index]
    x_range_mask = (x >= x_start) & (x <= x_end)
    ax.fill_between(x, start1, start2, where=contin_significant_window(positive_mask & x_range_mask,window_size), color='red', alpha=0.3)
    ax.fill_between(x, start1, start2, where=contin_significant_window(negative_mask & x_range_mask,window_size), color='blue', alpha=0.3)
    # Find the duration window of the component
    positive_windows = contin_significant_window(positive_mask & x_range_mask, window_size)
    if np.any(positive_windows):
        positive_indices = np.where(positive_windows)[0]
        positive_start = x[positive_indices[0]]
        positive_end = x[positive_indices[-1]]
        print(f"Positive component duration: {positive_start} to {positive_end}")
        T_value_window = TT[positive_indices]
        max_T_value = np.max(T_value_window)
        max_T_index = np.argmax(T_value_window)
        max_T_time = x[positive_indices[max_T_index]]
        print(f"Max T_value in this window {max_T_time}: {max_T_value}")
ax.plot(x, np.full(len(x), start2), color='black', linewidth=0.5)
ax.set_title(f"Global II Contrast", fontsize=24, fontweight='bold')
ax.set_xlabel("Time(s)", fontsize=20)
ax.set_ylabel("Amplitude(μV)", fontsize=20)
ax.set_xlim(-0.1, 0.8)
ax.set_ylim(y1, y2)
ax.set_xticks([0, 0.2,0.4,0.6,0.8])
ax.set_xticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                        if tick != 0 else "0" for tick in [0, 0.2,0.4,0.6,0.8]],fontsize=16, fontweight='bold')
ax.set_yticks([-8e-6, -6e-6, -4e-6, -2e-6, 0e-6, 2e-6, 4e-6, 6e-6, 8e-6])
ax.set_yticklabels([f"{tick * 1e6:.1f}"[:-2] for tick in ax.get_yticks()], fontsize=16, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
legend = ax.legend(loc='upper left', fontsize=20, frameon=False)      
plt.tight_layout()
# plt.show()
save_path = (
f"./Fig/waveform/Act_Pas/"
f"GlobalII_{chan_names}.png"
)      
plt.savefig(save_path, bbox_inches='tight')

#%%
x = times_350  
# Figure 2 Significant areas between Active and Passive
test_data = [Local_Dev_1-Local_STD_1,Local_Dev_2-Local_STD_2]
significant_points = np.empty((len(test_data), 64, 114))
T_value_Local = np.empty((len(test_data), 64, 114))
for i in range(len(test_data)):
    data1 = test_data[i]    
    active_data = np.squeeze(data1[:,0,:,:])
    passive_data = np.squeeze(data1[:,1,:,:])
    X = [
        active_data.transpose(0, 2, 1),
        passive_data.transpose(0, 2, 1),
    ]
    tfce = dict(start=0, step=0.1)  # ideally start and step would be smaller

    # Calculate statistical thresholds
    t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_test(
        X, tfce, adjacency=adjacency, n_permutations=1000, tail=0, n_jobs=-1
    )  # a more standard number would be 1000+
    significant_points[i,:,:] = cluster_pv.reshape(t_obs.shape).T <= alpha_v
    T_value_Local[i,:,:] = t_obs.T

Sign_Local_ACT_PAS = significant_points
#%%
chan_names = 'Fz'
chan_indice = 5  # Fz:5;Cz:15;Pz:25;
window_size = 5
y1 = -8e-6
y2 = 8e-6
Active_data1 = np.mean(Local_Dev_1[:,0,:,:], axis=0)-np.mean(Local_STD_1[:,0,:,:], axis=0)
Passive_data1 = np.mean(Local_Dev_1[:,1,:,:], axis=0)-np.mean(Local_STD_1[:,1,:,:], axis=0)
Active_data2 = np.mean(Local_Dev_2[:,0,:,:], axis=0)-np.mean(Local_STD_2[:,0,:,:], axis=0)
Passive_data2 = np.mean(Local_Dev_2[:,1,:,:], axis=0)-np.mean(Local_STD_2[:,1,:,:], axis=0)
mean_active_data = Active_data1
mean_passive_data = Passive_data1
mean_dw = mean_active_data-mean_passive_data
std_active_data = np.std(Local_Dev_1[:,0,:,:]-Local_STD_1[:,0,:,:], axis=0)
std_passive_data = np.std(Local_Dev_1[:,1,:,:]-Local_STD_1[:,1,:,:], axis=0)

rgb_1 = (244, 140, 155)
rgb_2 = (88, 159, 243)
color_1 = tuple(value / 255 for value in rgb_1)
color_2 = tuple(value / 255 for value in rgb_2)

fig, ax = plt.subplots(figsize=(6, 6), dpi=600)
ax.plot(x, mean_active_data[chan_indice, :], color=color_1, label='active', linewidth=4)
ax.plot(x, mean_passive_data[chan_indice, :], color=color_2, label='passive', linewidth=4)
ax.fill_between(x, 
                mean_active_data[chan_indice, :] - std_active_data[chan_indice, :], 
                mean_active_data[chan_indice, :] + std_active_data[chan_indice, :], 
                color=color_1, alpha=0.3)
ax.fill_between(x, 
                mean_passive_data[chan_indice, :] - std_passive_data[chan_indice, :], 
                mean_passive_data[chan_indice, :] + std_passive_data[chan_indice, :], 
                color=color_2, alpha=0.3)

sig_point = np.squeeze(Sign_Local_ACT_PAS[0,:,:])
true_indices = np.where(sig_point[chan_indice,:])[0]
start1 = -8e-6
start2 = -7.2e-6
start_index = None
end_index = None
positive_mask = mean_dw[chan_indice,:] > 0
negative_mask = mean_dw[chan_indice,:] < 0   

# Traverse significant time points and draw corresponding significant regions
for index in true_indices:
    if start_index is None:
        start_index = index
        end_index = index
    elif index == end_index + 1:
        end_index = index
    else:
        x_start = x[start_index]
        x_end = x[end_index]
        x_range_mask = (x >= x_start) & (x <= x_end)
        ax.fill_between(x, start1, start2, where=contin_significant_window(positive_mask & x_range_mask,window_size), color='red', alpha=0.3)
        ax.fill_between(x, start1, start2, where=contin_significant_window(negative_mask & x_range_mask,window_size), color='blue', alpha=0.3)
        start_index = index
        end_index = index
if start_index is not None and end_index is not None:
    x_start = x[start_index]
    x_end = x[end_index]
    x_range_mask = (x >= x_start) & (x <= x_end)
    ax.fill_between(x, start1, start2, where=contin_significant_window(positive_mask & x_range_mask,window_size), color='red', alpha=0.3)
    ax.fill_between(x, start1, start2, where=contin_significant_window(negative_mask & x_range_mask,window_size), color='blue', alpha=0.3)
ax.plot(x, np.full(len(x), start2), color='black', linewidth=0.5)
ax.set_title(f"Local I Contrast", fontsize=24, fontweight='bold')
ax.set_xlabel("Time(s)", fontsize=20)
ax.set_ylabel("Amplitude(μV)", fontsize=20)
ax.set_xlim(-0.1, 0.35)
ax.set_ylim(y1, y2)
ax.set_xticks([0, 0.1, 0.2, 0.3])
ax.set_xticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                        if tick != 0 else "0" for tick in [0, 0.1, 0.2, 0.3]],fontsize=16, fontweight='bold')
ax.set_yticks([-8e-6, -6e-6, -4e-6, -2e-6, 0e-6, 2e-6, 4e-6, 6e-6])
ax.set_yticklabels([f"{tick * 1e6:.1f}"[:-2] for tick in ax.get_yticks()], fontsize=16, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
legend = ax.legend(loc='upper left', fontsize=20, frameon=False)      
plt.tight_layout()
# plt.show()
save_path = (
f"./Fig/waveform/Act_Pas/"
f"LocalI_{chan_names}.png"
)      
plt.savefig(save_path, bbox_inches='tight')
#%%
############# local II  ##########
mean_active_data = Active_data2
mean_passive_data = Passive_data2
mean_dw = mean_active_data-mean_passive_data
y1 = -12e-6
y2 = 18e-6
std_active_data = np.std(Local_Dev_2[:,0,:,:]-Local_STD_2[:,0,:,:], axis=0)
std_passive_data = np.std(Local_Dev_2[:,1,:,:]-Local_STD_2[:,1,:,:], axis=0)

rgb_1 = (244, 140, 155)
rgb_2 = (88, 159, 243)
color_1 = tuple(value / 255 for value in rgb_1)
color_2 = tuple(value / 255 for value in rgb_2)

fig, ax = plt.subplots(figsize=(6, 6), dpi=600)
ax.plot(x, mean_active_data[chan_indice, :], color=color_1, label='active', linewidth=4)
ax.plot(x, mean_passive_data[chan_indice, :], color=color_2, label='passive', linewidth=4)
ax.fill_between(x, 
                mean_active_data[chan_indice, :] - std_active_data[chan_indice, :], 
                mean_active_data[chan_indice, :] + std_active_data[chan_indice, :], 
                color=color_1, alpha=0.3)
ax.fill_between(x, 
                mean_passive_data[chan_indice, :] - std_passive_data[chan_indice, :], 
                mean_passive_data[chan_indice, :] + std_passive_data[chan_indice, :], 
                color=color_2, alpha=0.3)
sig_point = np.squeeze(Sign_Local_ACT_PAS[1,:,:])
TT = np.squeeze(T_value_Local[1,chan_indice,:])
true_indices = np.where(sig_point[chan_indice,:])[0]
start1 = -12e-6
start2 = -10.5e-6
start_index = None
end_index = None
positive_mask = mean_dw[chan_indice,:] > 0
negative_mask = mean_dw[chan_indice,:] < 0   

# Traverse significant time points and draw corresponding significant regions
for index in true_indices:
    if start_index is None:
        start_index = index
        end_index = index
    elif index == end_index + 1:
        end_index = index
    else:
        x_start = x[start_index]
        x_end = x[end_index]
        x_range_mask = (x >= x_start) & (x <= x_end) 
        ax.fill_between(x, start1, start2, where=contin_significant_window(positive_mask & x_range_mask,window_size), color='red', alpha=0.3)
        ax.fill_between(x, start1, start2, where=contin_significant_window(negative_mask & x_range_mask,window_size), color='blue', alpha=0.3)
        start_index = index
        end_index = index
        # Find the duration window of the component
        positive_windows = contin_significant_window(negative_mask & x_range_mask, window_size)
        if np.any(positive_windows):
            positive_indices = np.where(positive_windows)[0]
            positive_start = x[positive_indices[0]]
            positive_end = x[positive_indices[-1]]
            print(f"Positive component duration: {positive_start} to {positive_end}")
            T_value_window = TT[positive_indices]
            max_T_value = np.max(T_value_window)
            max_T_index = np.argmax(T_value_window)
            max_T_time = x[positive_indices[max_T_index]]
            print(f"Max T_value in this window {max_T_time}: {max_T_value}")        
if start_index is not None and end_index is not None:
    x_start = x[start_index]
    x_end = x[end_index]
    x_range_mask = (x >= x_start) & (x <= x_end)
    ax.fill_between(x, start1, start2, where=contin_significant_window(positive_mask & x_range_mask,window_size), color='red', alpha=0.3)
    ax.fill_between(x, start1, start2, where=contin_significant_window(negative_mask & x_range_mask,window_size), color='blue', alpha=0.3)
    # Find the duration window of the component
    positive_windows = contin_significant_window(positive_mask & x_range_mask, window_size)
    if np.any(positive_windows):
        positive_indices = np.where(positive_windows)[0]
        positive_start = x[positive_indices[0]]
        positive_end = x[positive_indices[-1]]
        print(f"Positive component duration: {positive_start} to {positive_end}")
        T_value_window = TT[positive_indices]
        max_T_value = np.max(T_value_window)
        max_T_index = np.argmax(T_value_window)
        max_T_time = x[positive_indices[max_T_index]]
        print(f"Max T_value in this window {max_T_time}: {max_T_value}")
ax.plot(x, np.full(len(x), start2), color='black', linewidth=0.5)
ax.set_title(f"Local II Contrast", fontsize=24, fontweight='bold')
ax.set_xlabel("Time(s)", fontsize=20)
ax.set_ylabel("Amplitude(μV)", fontsize=20)
ax.set_xlim(-0.1, 0.35)
ax.set_ylim(y1, y2)
ax.set_xticks([0, 0.1, 0.2, 0.3])
ax.set_xticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                        if tick != 0 else "0" for tick in [0, 0.1, 0.2, 0.3]],fontsize=16, fontweight='bold')
ax.set_yticks([-12e-6, -8e-6, -4e-6, 0e-6,  4e-6,  8e-6, 12e-6, 16e-6])
ax.set_yticklabels([f"{tick * 1e6:.1f}"[:-2] for tick in ax.get_yticks()], fontsize=16, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
legend = ax.legend(loc='upper left', fontsize=20, frameon=False)      
plt.tight_layout()
# plt.show()
save_path = (
f"./Fig/waveform/Act_Pas/"
f"LocalII_{chan_names}.png"
)      
plt.savefig(save_path, bbox_inches='tight')
#%%
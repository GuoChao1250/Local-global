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
from scipy.sparse import save_npz
#%%
# parameters
threshold = 100e-6
Std_350 = np.empty((30, 2, 64, 114))
Std_350_rest = np.empty((30, 2, 64, 114))
num_trials = np.empty((30, 2))
# path
Pathname = r'E:\MyWork\Graduation\ASAP_ICA_EEG_ArtifactRejection_v1\Re_data\healthy'
Listing = os.listdir(Pathname)
with open('desired_chan_order.pkl', 'rb') as f:
    desired_chan_order = pickle.load(f)
#%% Standard
for isSub in [0]:#range(len(Listing)):  
    Sub_Path = os.path.join(Pathname, Listing[isSub])
    Cond_list = os.listdir(Sub_Path)
    for istype in range(len(Cond_list)):
        Data_Path = os.path.join(Sub_Path, Cond_list[istype])
        raw = mne.io.read_raw_eeglab(input_fname=os.path.join(Data_Path, 'Spatial_filtered_ICA_Format.set'), preload=True)  
        # define fixed channels
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
        
        # Calculate adjacency matrix between sensors from their locations
        adjacency, _ = find_ch_adjacency(epochs.info, "eeg")         
        #  Removing paired epochs using threshold
        Full_epochs = epochs.copy()   
        # Calculate adjacency matrix between sensors from their locations
        adjacency, _ = find_ch_adjacency(epochs.info, "eeg") 
        #  Extracting Epochs
        local_std_350  = combine_standard(Full_epochs,'s1','s11',threshold=threshold)
        num_trials[isSub, istype] = local_std_350.shape[0]
        Std_350[isSub, istype, :, :] = np.mean(local_std_350,axis=0)    
        del raw

times_350 = epochs.times
#%% Resting
for isSub in range(len(Listing)):      
    Sub_Path = os.path.join(Pathname, Listing[isSub])
    Cond_list = os.listdir(Sub_Path)
    for istype in range(len(Cond_list)):
        Data_Path = os.path.join(Sub_Path, Cond_list[istype])
        raw = mne.io.read_raw_eeglab(input_fname=os.path.join(Data_Path, 'Spatial_filtered_ICA_Format.set'), preload=True)  
        raw = raw.reorder_channels(desired_chan_order)
        raw.load_data()
        # M1M2 reference
        reference = ['M1', 'M2']
        raw.set_eeg_reference(reference)
        events, event_id = mne.events_from_annotations(raw)
        tmin = -0.45
        tmax = 0.45
        baseline = (None, 0)
        epochs = mne.Epochs(raw, events=events, event_id=event_id, event_repeated='merge',
                            tmin=tmin, tmax=tmax, baseline=baseline, preload=True)
        
        #  Removing paired epochs using threshold
        Full_epochs = epochs.copy()    
        local_std_350  = combine_standard(Full_epochs,'s1','s11',threshold=threshold)

        time_450 = epochs.times
        interestTime_start = np.array(-0.45)
        interestTime_end = np.array(0)
        # Find the index of the element closest to interesting time range
        interest_point1 = np.abs(time_450 - interestTime_start).argmin()
        interest_point2 = np.abs(time_450 - interestTime_end).argmin()
        data_resting = local_std_350[:,:,interest_point1:interest_point2+2]
        Std_350_rest[isSub, istype, :, :] = np.mean(data_resting,axis=0)
        # clean
        del raw

#%%
save_file_path = f'.\\Fig\\data'
# Load the arrays from the .npz file
data = np.load(os.path.join(save_file_path,'Standard_data_plot.npz'))

# Access the arrays using the keys
Std_350 = data['Std_350']
Std_350_rest = data['Std_350_rest']
significant_points = data['significant_points']
T_value = data['T_value']
# with open(os.path.join(save_file_path,'times_350.pkl'), 'rb') as f:
#     times_350 = pickle.load(f)
# times = times_350       
#%% baseline correct
Std_350_rest_base = np.zeros_like(Std_350_rest)
for isSub in range(len(Listing)):     
    Sub_Path = os.path.join(Pathname, Listing[isSub]) 
    Cond_list = os.listdir(Sub_Path)
    for istype in range(len(Cond_list)):
        data = Std_350_rest[isSub,istype,:,:]
        baseline_mean = np.mean(data[:, 0:25], axis=1, keepdims=True)  
        # baseline correct
        data_baseline_corrected = data - baseline_mean
        Std_350_rest_base[isSub,istype,:,:] = data_baseline_corrected
#%% clustered permutation   
test_data1 = [Std_350]
test_data2 = [Std_350_rest_base]
significant_points = np.empty((len(Cond_list), len(test_data1), 64, 114))
T_value = np.empty((len(Cond_list), len(test_data1), 64, 114))
for i in range(len(test_data1)):
    data1 = test_data1[i]
    data2 = test_data2[i]
    for istype in range(len(Cond_list)):
        data11 = np.squeeze(data1[:,istype,:,:])
        data22 = np.squeeze(data2[:,istype,:,:])
        X = [
            data11.transpose(0, 2, 1),
            data22.transpose(0, 2, 1),
        ]
        tfce = dict(start=0, step=0.1)  # ideally start and step would be smaller

        # Calculate statistical thresholds
        t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_test(
            X, tfce, adjacency=adjacency, n_permutations=1000, tail=0, n_jobs=-1
        )  # a more standard number would be 1000+
        significant_points[istype, i,:,:] = cluster_pv.reshape(t_obs.shape).T <= 0.05
        T_value[istype, i,:,:] = t_obs.T


#%%
condition_type = ['Active', 'Passive']
window_size = 5
x = times_350
chan_indice = 5  # Fz:5;Cz:15;Pz:25;
chan_names = 'Fz'
y1 = -5e-6
y2 = 5e-6
# Define the desired range
interestTime_start = np.array(0)
interestTime_end = np.array(0.35)
# Find the index of the element closest to interesting time range
interest_point1 = np.abs(x - interestTime_start).argmin()
interest_point2 = np.abs(x - interestTime_end).argmin()
time_window_std = (x >= x[interest_point1]) & (x <= x[interest_point2])
rgb_1 = (244, 140, 155)
rgb_2 = (88, 159, 243)
color_1 = tuple(value / 255 for value in rgb_1)
color_2 = tuple(value / 255 for value in rgb_2)

for istype in range(len(condition_type)): 
    mean_std_350 = np.mean(Std_350[:, istype, :, :], axis=0)
    mean_std_350_rest = np.mean(Std_350_rest[:, istype, :, :], axis=0)
    mean_STD = mean_std_350-mean_std_350_rest
    # calculate std
    std_std_350 = np.std(Std_350[:, istype, :, :], axis=0)
    std_std_350_rest = np.std(Std_350_rest[:, istype, :, :], axis=0)

    fig, ax = plt.subplots(figsize=(6, 8), dpi=600)
    ax.plot(x, mean_std_350[chan_indice, :], color=color_2, label='Standard', linewidth=4)
    ax.plot(x, mean_std_350_rest[chan_indice, :], color=color_1, label='Resting', linewidth=4)
    # Standard deviation area
    ax.fill_between(x, 
                    mean_std_350[chan_indice, :] - std_std_350[chan_indice, :], 
                    mean_std_350[chan_indice, :] + std_std_350[chan_indice, :], 
                    color=color_2, alpha=0.3)
    ax.fill_between(x, 
                    mean_std_350_rest[chan_indice, :] - std_std_350_rest[chan_indice, :], 
                    mean_std_350_rest[chan_indice, :] + std_std_350_rest[chan_indice, :], 
                    color=color_1, alpha=0.3)

    sig_point = np.squeeze(significant_points[istype,0,:,:])
    true_indices = np.where(sig_point[chan_indice,:])[0]
    TT = np.squeeze(T_value[istype,0,chan_indice,:])
    start1 = -5e-6
    start2 = -4.5e-6
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
            x_range_mask = (x >= x_start) & (x <= x_end) 
            ax.fill_between(x, start1, start2, where=contin_significant_window(positive_mask & x_range_mask & time_window_std,window_size), color='red', alpha=0.3)
            ax.fill_between(x, start1, start2, where=contin_significant_window(negative_mask & x_range_mask & time_window_std,window_size), color='blue', alpha=0.3)
            start_index = index
            end_index = index
        # # Find the duration window of the component
        #     positive_windows = contin_significant_window(positive_mask & x_range_mask & time_window_std, window_size)
        #     if np.any(positive_windows):
        #         positive_indices = np.where(positive_windows)[0]
        #         positive_start = x[positive_indices[0]]
        #         positive_end = x[positive_indices[-1]]
        #         print(f"Positive component duration: {positive_start} to {positive_end}")
                
        #         T_value_window = TT[positive_indices]
        #         max_T_value = np.max(T_value_window)  
        #         max_T_index = np.argmax(T_value_window)  
        #         max_T_time = x[positive_indices[max_T_index]] 
        #         print(f"Max T_value in this window {max_T_time}: {max_T_value}")
    # Draw the salient region of the last continuous area
    if start_index is not None and end_index is not None:
        x_start = x[start_index]
        x_end = x[end_index]
        x_range_mask = (x >= x_start) & (x <= x_end)
        ax.fill_between(x, start1, start2, where=contin_significant_window(positive_mask & x_range_mask & time_window_std,window_size), color='red', alpha=0.3)
        ax.fill_between(x, start1, start2, where=contin_significant_window(negative_mask & x_range_mask & time_window_std,window_size), color='blue', alpha=0.3)
        # positive_windows = contin_significant_window(positive_mask & x_range_mask & time_window_std, window_size)
        # if np.any(positive_windows):
        #     positive_indices = np.where(positive_windows)[0]
        #     positive_start = x[positive_indices[0]]
        #     positive_end = x[positive_indices[-1]]
        #     print(f"Positive component duration: {positive_start} to {positive_end}")
            
        #     T_value_window = TT[positive_indices]
        #     max_T_value = np.max(T_value_window)  
        #     max_T_index = np.argmax(T_value_window)  
        #     max_T_time = x[positive_indices[max_T_index]] 
        #     print(f"Max T_value in this window {max_T_time}: {max_T_value}")
    # Add horizontal line
    ax.plot(x, np.full(len(x), start2), color='black', linewidth=0.5)
    # title and x/y label
    ax.set_title(f"{condition_type[istype]}: Standard", fontsize=24, fontweight='bold')
    ax.set_xlabel("Time(s)", fontsize=20)
    ax.set_ylabel("Amplitude(μV)", fontsize=20)
    ax.set_xlim(-0.1, 0.35)
    ax.set_ylim(y1, y2)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_xticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                            if tick != 0 else "0" for tick in [0, 0.1, 0.2, 0.3]],fontsize=16, fontweight='bold')    
    ax.set_yticks([-4e-6, -2e-6, 0e-6, 2e-6, 4e-6])
    ax.set_yticklabels([f"{tick * 1e6:.1f}"[:-2] for tick in ax.get_yticks()], fontsize=16, fontweight='bold')
    # Remove the top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # legend
    legend = ax.legend(loc='upper left', fontsize=20, frameon=False)    
    plt.tight_layout()
    # plt.show()
    save_path = (
    f"./Fig/waveform/STD/"
    f"{condition_type[istype]}_{chan_names}.png"
    )      
    plt.savefig(save_path, bbox_inches='tight')
# %%
save_file_path = f'.\\Fig\\data'
# Save the arrays to a .npz file
np.savez(os.path.join(save_file_path, 'Standard_data_plot.npz'), Std_350=Std_350, 
         Std_350_rest=Std_350_rest,significant_points=significant_points,T_value=T_value)

# %%

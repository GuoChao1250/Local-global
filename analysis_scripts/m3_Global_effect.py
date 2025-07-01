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
threshold=100e-6
Dev_1 = np.empty((30, 2, 64, 226))
STD_1 = np.empty((30, 2, 64, 226))
Dev_2 = np.empty((30, 2, 64, 226))
STD_2 = np.empty((30, 2, 64, 226))

I_num_trials = np.empty((30, 2))
II_num_trials = np.empty((30, 2))
# path
Pathname = r'E:\MyWork\Graduation\ASAP_ICA_EEG_ArtifactRejection_v1\Re_data\healthy'
Listing = os.listdir(Pathname)
with open('desired_chan_order.pkl', 'rb') as f:
    desired_chan_order = pickle.load(f)
#%%
for isSub in [0]:#range(len(Listing)):      
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
        tmin = -0.1
        tmax = 0.8
        baseline = (None, 0)
        epochs = mne.Epochs(raw, events=events, event_id=event_id, event_repeated='merge',
                            tmin=tmin, tmax=tmax, baseline=baseline, preload=True)    
        #  Removing paired epochs using threshold
        Full_epochs = epochs.copy()     

        # Calculate adjacency matrix between sensors from their locations
        adjacency, _ = find_ch_adjacency(epochs.info, "eeg")
        #  Extracting Epochs
        global_std_1, global_dev_1  = extract_special_pairwise_mark(Full_epochs,std_mark='s4',dev_mark='s2',mark_step=2,event_id=event_id,threshold=threshold)
        global_std_2, global_dev_2  = extract_pairwise_mark(Full_epochs,std_mark='s22',dev_mark='s2',mark_step=2,event_id=event_id,threshold=threshold)

        I_num_trials[isSub, istype] = global_std_1.shape[0]
        II_num_trials[isSub, istype] = global_std_2.shape[0]
        
        Dev_1[isSub, istype, :, :] = np.mean(global_dev_1,axis=0)
        STD_1[isSub, istype, :, :] = np.mean(global_std_1,axis=0)
        Dev_2[isSub, istype, :, :] = np.mean(global_dev_2,axis=0)
        STD_2[isSub, istype, :, :] = np.mean(global_std_2,axis=0)

        # 清理内存
        del raw
times = epochs.times

# %%
test_data = [Dev_1-STD_1,Dev_2-STD_2]
significant_points = np.empty((len(Cond_list), len(test_data), 64, 226))
T_value = np.empty((len(Cond_list), len(test_data), 64, 226))
for i in range(len(test_data)):
    data1 = test_data[i]
    for istype in range(len(Cond_list)):
        data11 = np.squeeze(data1[:,istype,:,:])
        zero_matrix = np.zeros(data11.shape)
        X = [
            data11.transpose(0, 2, 1),
            zero_matrix.transpose(0, 2, 1),
        ]
        tfce = dict(start=0, step=0.1)  # ideally start and step would be smaller

        # Calculate statistical thresholds
        t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_test(
            X, tfce, adjacency=adjacency, n_permutations=1000, tail=0, n_jobs=-1
        )  # a more standard number would be 1000+
        significant_points[istype, i,:,:] = cluster_pv.reshape(t_obs.shape).T <= 0.001
        T_value[istype, i,:,:] = t_obs.T

#%%
save_file_path = f'.\\Fig\\data'
# Load the arrays from the .npz file
data = np.load(os.path.join(save_file_path,'Global_effect.npz'))
# Access the arrays using the keys
Dev_1 = data['Dev_1']
STD_1 = data['STD_1']
Dev_2 = data['Dev_2']
STD_2 = data['STD_2']
significant_points = data['significant_points']
T_value = data['T_value']

#%% Global SS-SN
condition_type = ['Active', 'Passive']
window_size = 5
x = times 
chan_indice = 25  # Fz:5;Cz:15;Pz:25;
chan_names = 'Pz'
y1 = -8e-6
y2 = 8e-6
# Define the desired range
interestTime_start = np.array(0)
interestTime_end = np.array(0.6)
# Find the index of the element closest to interesting time range
interest_point1 = np.abs(x - interestTime_start).argmin()
interest_point2 = np.abs(x - interestTime_end).argmin()
time_window_std = (x >= x[interest_point1]) & (x <= x[interest_point2])
rgb_1 = (244, 140, 155)
rgb_2 = (88, 159, 243)
color_1 = tuple(value / 255 for value in rgb_1)
color_2 = tuple(value / 255 for value in rgb_2)

for istype in range(len(condition_type)):
    mean_dev = np.mean(Dev_1[:, istype, :, :], axis=0)
    mean_std = np.mean(STD_1[:, istype, :, :], axis=0)
    mean_dw = mean_dev-mean_std
    # calculate std
    std_dev = np.std(Dev_1[:, istype, :, :], axis=0)
    std_std = np.std(STD_1[:, istype, :, :], axis=0)
    
    fig, ax = plt.subplots(figsize=(6, 8), dpi=600)
    ax.plot(x, mean_dev[chan_indice, :], color=color_1, label='Deviant', linewidth=4)
    ax.plot(x, mean_std[chan_indice, :], color=color_2, label='Standard', linewidth=4)
    # Standard deviation area
    ax.fill_between(x, 
                    mean_dev[chan_indice, :] - std_dev[chan_indice, :], 
                    mean_dev[chan_indice, :] + std_dev[chan_indice, :], 
                    color=color_1, alpha=0.3)
    ax.fill_between(x, 
                    mean_std[chan_indice, :] - std_std[chan_indice, :], 
                    mean_std[chan_indice, :] + std_std[chan_indice, :], 
                    color=color_2, alpha=0.3)

    sig_point = np.squeeze(significant_points[istype,0,:,:])
    TT = np.squeeze(T_value[istype,0,chan_indice,:])
    true_indices = np.where(sig_point[chan_indice,:])[0]
    start1 = -8e-6
    start2 = -7.4e-6
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
            ax.fill_between(x, start1, start2, where=contin_significant_window(positive_mask & x_range_mask & time_window_std,window_size), color='red', alpha=0.3)
            ax.fill_between(x, start1, start2, where=contin_significant_window(negative_mask & x_range_mask & time_window_std,window_size), color='blue', alpha=0.3)
            start_index = index
            end_index = index
            # Find the duration window of the component
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
    # Draw the salient region of the last continuous area
    if start_index is not None and end_index is not None:
        x_start = x[start_index]
        x_end = x[end_index]
        x_range_mask = (x >= x_start) & (x <= x_end)
        ax.fill_between(x, start1, start2, where=contin_significant_window(positive_mask & x_range_mask & time_window_std,window_size), color='red', alpha=0.3)
        ax.fill_between(x, start1, start2, where=contin_significant_window(negative_mask & x_range_mask & time_window_std,window_size), color='blue', alpha=0.3)
    ax.plot(x, np.full(len(x), start2), color='black', linewidth=0.5)
    ax.set_title(f"{condition_type[istype]}: Global I", fontsize=24, fontweight='bold')
    ax.set_xlabel("Time(s)", fontsize=20)
    ax.set_ylabel("Amplitude(μV)", fontsize=20)
    ax.set_xlim(-0.1, 0.8)
    ax.set_ylim(y1, y2)
    ax.set_xticks([0, 0.2,0.4,0.6,0.8])
    ax.set_xticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                            if tick != 0 else "0" for tick in [0, 0.2,0.4,0.6,0.8]],fontsize=16, fontweight='bold')
    ax.set_yticks([-6e-6, -4e-6, -2e-6, 0e-6, 2e-6, 4e-6, 6e-6])
    ax.set_yticklabels([f"{tick * 1e6:.1f}"[:-2] for tick in ax.get_yticks()], fontsize=16, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend = ax.legend(loc='upper right', fontsize=18, frameon=False)      

    plt.tight_layout()
    # plt.show()
    save_path = (
    f"./Fig/waveform/Global_effect/GlobalSS-SN/"
    f"{condition_type[istype]}_{chan_names}.png"
    )      
    plt.savefig(save_path, bbox_inches='tight')       
#%% Global SF-SN
condition_type = ['Active', 'Passive']
window_size = 5
x = times
chan_indice = 25  # Fz:5;Cz:15;Pz:25;
chan_names = 'Pz'
y1 = -8e-6
y2 = 8e-6
# Define the desired range
interestTime_start = np.array(0)
interestTime_end = np.array(0.6)
# Find the index of the element closest to interesting time range
interest_point1 = np.abs(x - interestTime_start).argmin()
interest_point2 = np.abs(x - interestTime_end).argmin()
time_window_std = (x >= x[interest_point1]) & (x <= x[interest_point2])
rgb_1 = (244, 140, 155)
rgb_2 = (88, 159, 243)
color_1 = tuple(value / 255 for value in rgb_1)
color_2 = tuple(value / 255 for value in rgb_2)
for istype in range(len(condition_type)):
    mean_dev = np.mean(Dev_2[:, istype, :, :], axis=0)
    mean_std = np.mean(STD_2[:, istype, :, :], axis=0)
    mean_dw = mean_dev-mean_std
    # calculate std
    std_dev = np.std(Dev_2[:, istype, :, :], axis=0)
    std_std = np.std(STD_2[:, istype, :, :], axis=0)
    
    fig, ax = plt.subplots(figsize=(6, 8), dpi=600)
    ax.plot(x, mean_dev[chan_indice, :], color=color_1, label='Deviant', linewidth=4)
    ax.plot(x, mean_std[chan_indice, :], color=color_2, label='Standard', linewidth=4)
    ax.fill_between(x, 
                    mean_dev[chan_indice, :] - std_dev[chan_indice, :], 
                    mean_dev[chan_indice, :] + std_dev[chan_indice, :], 
                    color=color_1, alpha=0.3)
    ax.fill_between(x, 
                    mean_std[chan_indice, :] - std_std[chan_indice, :], 
                    mean_std[chan_indice, :] + std_std[chan_indice, :], 
                    color=color_2, alpha=0.3)

    sig_point = np.squeeze(significant_points[istype,1,:,:])
    TT = np.squeeze(T_value[istype,1,chan_indice,:])
    true_indices = np.where(sig_point[chan_indice,:])[0]
    start1 = -8e-6
    start2 = -7.4e-6
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
            ax.fill_between(x, start1, start2, where=contin_significant_window(positive_mask & x_range_mask & time_window_std,window_size), color='red', alpha=0.3)
            ax.fill_between(x, start1, start2, where=contin_significant_window(negative_mask & x_range_mask & time_window_std,window_size), color='blue', alpha=0.3)
            start_index = index
            end_index = index
            # Find the duration window of the component
            positive_windows = contin_significant_window(positive_mask & x_range_mask & time_window_std, window_size)
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
    # Draw the salient region of the last continuous area
    if start_index is not None and end_index is not None:
        x_start = x[start_index]
        x_end = x[end_index]
        x_range_mask = (x >= x_start) & (x <= x_end)
        ax.fill_between(x, start1, start2, where=contin_significant_window(positive_mask & x_range_mask & time_window_std,window_size), color='red', alpha=0.3)
        ax.fill_between(x, start1, start2, where=contin_significant_window(negative_mask & x_range_mask & time_window_std,window_size), color='blue', alpha=0.3)
    ax.plot(x, np.full(len(x), start2), color='black', linewidth=0.5)
    ax.set_title(f"{condition_type[istype]}: Global II", fontsize=24, fontweight='bold')
    ax.set_xlabel("Time(s)", fontsize=20)
    ax.set_ylabel("Amplitude(μV)", fontsize=20)
    ax.set_xlim(-0.1, 0.8)
    ax.set_ylim(y1, y2)
    ax.set_xticks([0, 0.2,0.4,0.6,0.8])
    ax.set_xticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                            if tick != 0 else "0" for tick in [0, 0.2,0.4,0.6,0.8]],fontsize=16, fontweight='bold')
    ax.set_yticks([-6e-6, -4e-6, -2e-6, 0e-6, 2e-6, 4e-6, 6e-6])
    ax.set_yticklabels([f"{tick * 1e6:.1f}"[:-2] for tick in ax.get_yticks()], fontsize=16, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend = ax.legend(loc='upper right', fontsize=18, frameon=False)    
    plt.tight_layout()
    # plt.show()
    save_path = (
    f"./Fig/waveform/Global_effect/GlobalSF-SN/"
    f"{condition_type[istype]}_{chan_names}.png"
    )      
    plt.savefig(save_path, bbox_inches='tight')   

# %%
save_file_path = f'.\\Fig\\data'
# Save the arrays to a .npz file
np.savez(os.path.join(save_file_path, 'Global_effect.npz'), Dev_1=Dev_1, STD_1=STD_1, Dev_2=Dev_2
         ,STD_2=STD_2, significant_points=significant_points,T_value=T_value)

# %%

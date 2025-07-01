#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from mne.channels import find_ch_adjacency
from Mypackage.Processed_toolbox import *
import pickle
import mne
from mne.viz import plot_topomap
#%%
save_file_path = f'.\\Fig\\data'
Pathname = r'E:\MyWork\Graduation\ASAP_ICA_EEG_ArtifactRejection_v1\Re_data\healthy'
Listing = os.listdir(Pathname)
threshold=100e-6
with open('desired_chan_order.pkl', 'rb') as f:
    desired_chan_order = pickle.load(f)
n_perm = 500
standard_8ch = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1','O2']
standard_16ch = [
    'Fp1', 'Fp2', 
    'F7', 'F3', 'Fz', 'F4', 'F8', 
    'T7', 'C3', 'Cz', 'C4', 'T8', 
    'P7', 'P3', 'Pz', 'P4', 'P8', 
    'O1', 'O2'
]
standard_32ch = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
    'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8',
    'POz', 'O1', 'O2', 'AF7', 'AF3', 'AF4', 'AF8'
]
# Retrieve indexes for 8, 16, and 32 channels
chan_8 = [desired_chan_order.index(ch) for ch in standard_8ch if ch in desired_chan_order]
chan_16 = [desired_chan_order.index(ch) for ch in standard_16ch if ch in desired_chan_order]
chan_32 = [desired_chan_order.index(ch) for ch in standard_32ch if ch in desired_chan_order]
chan_64 = list(range(len(desired_chan_order)))
chan_mix = [chan_8,chan_16,chan_32,chan_64]
chan_num = [8,16,32,64]
trial_percentages = [0.25, 0.50, 0.75, 1.0]
accuracy_true = np.empty((30, 2, len(trial_percentages), len(chan_mix), n_perm))
accuracy_false = np.empty((30, 2, len(trial_percentages), len(chan_mix), n_perm))

#%%
for isSub in range(len(Listing)):   
    Sub_Path = os.path.join(Pathname, Listing[isSub])   
    Cond_list = os.listdir(Sub_Path)
    # Active
    Data_Path = os.path.join(Sub_Path, Cond_list[0])
    raw = mne.io.read_raw_eeglab(input_fname=os.path.join(Data_Path, 'Spatial_filtered_ICA_Format.set'), preload=True)  
    raw = raw.reorder_channels(desired_chan_order)
    raw.load_data()
    reference = ['M1', 'M2']
    raw.set_eeg_reference(reference)
    events, event_id = mne.events_from_annotations(raw)
    tmin = -0.1
    tmax = 0.8
    baseline = (None, 0)
    epochs = mne.Epochs(raw, events=events, event_id=event_id, event_repeated='merge',
                        tmin=tmin, tmax=tmax, baseline=baseline, preload=True)
    times_800 = epochs.times
    interestTime_start = np.array(0)
    interestTime_end = np.array(0.4)
    # Find the index of the element closest to interesting time range
    interest_point1 = np.abs(times_800 - interestTime_start).argmin()
    interest_point2 = np.abs(times_800 - interestTime_end).argmin()   
    Full_epochs = epochs.copy()     
    #  Extracting Epochs
    global_std_1, global_dev_1  = extract_special_pairwise_mark(Full_epochs,std_mark='s4',dev_mark='s2',mark_step=2,event_id=event_id,threshold=threshold)
    global_std_2, global_dev_2  = extract_pairwise_mark(Full_epochs,std_mark='s22',dev_mark='s2',mark_step=2,event_id=event_id,threshold=threshold)
    act_diff_global1 = global_dev_1[:,:,interest_point1:interest_point2]-global_std_1[:,:,interest_point1:interest_point2]
    act_diff_global2 = global_dev_2[:,:,interest_point1:interest_point2]-global_std_2[:,:,interest_point1:interest_point2]

    # Passive
    Data_Path = os.path.join(Sub_Path, Cond_list[1])
    raw = mne.io.read_raw_eeglab(input_fname=os.path.join(Data_Path, 'Spatial_filtered_ICA_Format.set'), preload=True)  
    raw = raw.reorder_channels(desired_chan_order)
    raw.load_data()
    reference = ['M1', 'M2']
    raw.set_eeg_reference(reference)
    events, event_id = mne.events_from_annotations(raw)
    tmin = -0.1
    tmax = 0.8
    baseline = (None, 0)
    epochs = mne.Epochs(raw, events=events, event_id=event_id, event_repeated='merge',
                        tmin=tmin, tmax=tmax, baseline=baseline, preload=True)
    
    Full_epochs = epochs.copy()     
    #  Extracting Epochs
    global_std_1, global_dev_1  = extract_special_pairwise_mark(Full_epochs,std_mark='s4',dev_mark='s2',mark_step=2,event_id=event_id,threshold=threshold)
    global_std_2, global_dev_2  = extract_pairwise_mark(Full_epochs,std_mark='s22',dev_mark='s2',mark_step=2,event_id=event_id,threshold=threshold)
    pas_diff_global1 = global_dev_1[:,:,interest_point1:interest_point2]-global_std_1[:,:,interest_point1:interest_point2]
    pas_diff_global2 = global_dev_2[:,:,interest_point1:interest_point2]-global_std_2[:,:,interest_point1:interest_point2]
   
    for ip, percentage in enumerate(trial_percentages):
        for ic, chan_index in enumerate(chan_mix):
            # Process active data
            n_trials_act1 = int(act_diff_global1.shape[0] * percentage)
            n_trials_act2 = int(act_diff_global2.shape[0] * percentage)
            # Randomly select trials without replacement
            selected_trials_act1 = np.random.choice(act_diff_global1.shape[0], n_trials_act1, replace=False)
            selected_trials_act2 = np.random.choice(act_diff_global2.shape[0], n_trials_act2, replace=False)

            # Process passive data
            n_trials_pas1 = int(pas_diff_global1.shape[0] * percentage)
            n_trials_pas2 = int(pas_diff_global2.shape[0] * percentage)  
            selected_trials_pas1 = np.random.choice(pas_diff_global1.shape[0], n_trials_pas1, replace=False)
            selected_trials_pas2 = np.random.choice(pas_diff_global2.shape[0], n_trials_pas2, replace=False)

            chan_index = np.array(chan_index)
            data1 = [
                act_diff_global1[selected_trials_act1, :, :][:, chan_index, :],
                act_diff_global2[selected_trials_act2, :, :][:, chan_index, :]
            ]
            data2 = [
                pas_diff_global1[selected_trials_pas1, :, :][:, chan_index, :],
                pas_diff_global2[selected_trials_pas2, :, :][:, chan_index, :]
            ]
            for isdata in range(len(data1)):

                X_standards1 = data1[isdata]
                y_standards1 = np.zeros(len(X_standards1))
                X_deviants1 = data2[isdata]
                y_deviants1 = np.ones(len(X_deviants1))
            
                X1 = np.concatenate((X_standards1, X_deviants1), axis=0)
                Y1 = np.concatenate((y_standards1, y_deviants1), axis=0)  # label
                
                acc_perm_truelabel = SVM_classify(X1,Y1,n_perm,shuffle_type=0)
                acc_perm_falselabel = SVM_classify(X1,Y1,n_perm,shuffle_type=1)
                accuracy_true[isSub,isdata,ip,ic,:] = acc_perm_truelabel
                accuracy_false[isSub,isdata,ip,ic,:] = acc_perm_falselabel
 #%%
save_file_path = f'.\\SVM\\accuracy'
# Save the arrays to a .npz file
np.savez(os.path.join(save_file_path, 'acc_perm_Global_400ms_SVM_compare_diff_chanAndepoch.npz'), accuracy_true=accuracy_true,accuracy_false=accuracy_false)
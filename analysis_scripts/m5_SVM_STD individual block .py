#%%
import os
import numpy as np
from Mypackage.Processed_toolbox import *
import pickle
#%%
save_file_path = f'.\\Fig\\data'
Pathname = r'E:\MyWork\Graduation\ASAP_ICA_EEG_ArtifactRejection_v1\Re_data\healthy'
Listing = os.listdir(Pathname)
threshold=100e-6
with open('desired_chan_order.pkl', 'rb') as f:
    desired_chan_order = pickle.load(f)
n_perm = 1000
accuracy_true = np.empty((30, 1,n_perm))
accuracy_false = np.empty((30, 1,n_perm))
#%%
for isSub in range(len(Listing)):     
    Sub_Path = os.path.join(Pathname, Listing[isSub]) 
    Cond_list = os.listdir(Sub_Path)
    # Active
    Data_Path = os.path.join(Sub_Path, Cond_list[0])
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
    times_350 = epochs.times
    Full_epochs = epochs.copy()     
    interestTime_start = np.array(0)
    interestTime_end = np.array(0.3)
    # Find the index of the element closest to interesting time range
    interest_point1 = np.abs(times_350 - interestTime_start).argmin()
    interest_point2 = np.abs(times_350 - interestTime_end).argmin()
    local_std_350  = combine_standard(Full_epochs,'s1','s11',threshold=threshold)
    act_data_350 = local_std_350[:,:,interest_point1:interest_point2]

    # resting
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
    baseline_mean = np.mean(data_resting[:, :, 0:25], axis=2, keepdims=True)
    data_baseline_corrected = data_resting - baseline_mean
    interestTime_start = np.array(0)
    interestTime_end = np.array(0.3)
    # Find the index of the element closest to interesting time range
    interest_point1 = np.abs(times_350 - interestTime_start).argmin()
    interest_point2 = np.abs(times_350 - interestTime_end).argmin()
    act_data_resting = data_baseline_corrected[:,:,interest_point1:interest_point2]
        
    # Passive
    Data_Path = os.path.join(Sub_Path, Cond_list[1])
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
    times_350 = epochs.times
    Full_epochs = epochs.copy()     
    interestTime_start = np.array(0)
    interestTime_end = np.array(0.3)
    # Find the index of the element closest to interesting time range
    interest_point1 = np.abs(times_350 - interestTime_start).argmin()
    interest_point2 = np.abs(times_350 - interestTime_end).argmin()
    local_std_350  = combine_standard(Full_epochs,'s1','s11',threshold=threshold)
    pas_data_350 = local_std_350[:,:,interest_point1:interest_point2]

    # resting
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
    baseline_mean = np.mean(data_resting[:, :, 0:25], axis=2, keepdims=True)
    data_baseline_corrected = data_resting - baseline_mean
    interestTime_start = np.array(0)
    interestTime_end = np.array(0.3)
    # Find the index of the element closest to interesting time range
    interest_point1 = np.abs(times_350 - interestTime_start).argmin()
    interest_point2 = np.abs(times_350 - interestTime_end).argmin()
    pas_data_resting = data_baseline_corrected[:,:,interest_point1:interest_point2]

    n = 50  # split into 50 blocks
    #  np.array_split 
    chunks1 = np.array_split(act_data_resting, n, axis=0)
    chunks2 = np.array_split(act_data_350, n, axis=0)
    chunks3 = np.array_split(pas_data_resting, n, axis=0)
    chunks4 = np.array_split(pas_data_350, n, axis=0)

    rest_data1 = np.array([chunk.mean(axis=0) for chunk in chunks1])
    std_data1 = np.array([chunk.mean(axis=0) for chunk in chunks2])
    rest_data2 = np.array([chunk.mean(axis=0) for chunk in chunks3])
    std_data2 = np.array([chunk.mean(axis=0) for chunk in chunks4])
    data1 = [rest_data1]
    data2 = [std_data1]

    for isdata in range(len(data1)):
        X_standards1 = data1[isdata]
        y_standards1 = np.zeros(len(X_standards1))
        X_deviants1 = data2[isdata]
        y_deviants1 = np.ones(len(X_deviants1))
    
        # Merge two datasets together
        X1 = np.concatenate((X_standards1, X_deviants1), axis=0)
        Y1 = np.concatenate((y_standards1, y_deviants1), axis=0)  # label
        
        acc_perm_truelabel = SVM_classify(X1,Y1,n_perm,shuffle_type=0)
        acc_perm_falselabel = SVM_classify(X1,Y1,n_perm,shuffle_type=1)
        accuracy_true[isSub,isdata,:] = acc_perm_truelabel
        accuracy_false[isSub,isdata,:]  = acc_perm_falselabel
#%%
save_file_path = f'.\\SVM\\accuracy'
# Save the arrays to a .npz file
np.savez(os.path.join(save_file_path, 'acc_perm_STD_SVM_block_act_50.npz'), accuracy_true=accuracy_true,accuracy_false=accuracy_false)
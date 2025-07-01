#%%
import os
import numpy as np
from Mypackage.Processed_toolbox import *
import pickle
num_cores = os.cpu_count()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
#%%
save_file_path = f'.\\Fig\\data'
Pathname = r'E:\MyWork\Graduation\ASAP_ICA_EEG_ArtifactRejection_v1\Re_data\healthy'
Listing = os.listdir(Pathname)
threshold=100e-6
with open('desired_chan_order.pkl', 'rb') as f:
    desired_chan_order = pickle.load(f)
n_perm = 1000
accuracy_true = np.empty((30, 2,n_perm))
accuracy_false = np.empty((30, 2,n_perm))
#%%
for isSub in range(len(Listing)):#[0]:  # 处理所有的子目录        
    Sub_Path = os.path.join(Pathname, Listing[isSub]) # 'Raw_data.set'
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
    interestTime_start = np.array(0)
    interestTime_end = np.array(0.35)
    # Find the index of the element closest to interesting time range
    interest_point1 = np.abs(times_350 - interestTime_start).argmin()
    interest_point2 = np.abs(times_350 - interestTime_end).argmin()   
    Full_epochs = epochs.copy()     
    #  Extracting Epochs
    global_std_1, global_dev_1  = extract_pairwise_mark(Full_epochs,std_mark='s11',dev_mark='s22',mark_step=1,event_id=event_id,threshold=threshold)
    global_std_2, global_dev_2  = extract_pairwise_mark(Full_epochs,std_mark='s1',dev_mark='s2',mark_step=1,event_id=event_id,threshold=threshold)
    timespoint = global_std_1.shape[2]
    act_diff_global1 = global_dev_1[:,:,interest_point1:interest_point2]-global_std_1[:,:,interest_point1:interest_point2]
    act_diff_global2 = global_dev_2[:,:,interest_point1:interest_point2]-global_std_2[:,:,interest_point1:interest_point2]

    # split into 50 blocks
    # n = 50 
    # chunks1 = np.array_split(global_dev_1[:,:,interest_point1:interest_point2], n, axis=0)
    # chunks2 = np.array_split(global_std_1[:,:,interest_point1:interest_point2], n, axis=0)
    # chunks3 = np.array_split(global_dev_2[:,:,interest_point1:interest_point2], n, axis=0)
    # chunks4 = np.array_split(global_std_2[:,:,interest_point1:interest_point2], n, axis=0)
    # dev_data1 = np.array([chunk.mean(axis=0) for chunk in chunks1])
    # std_data1 = np.array([chunk.mean(axis=0) for chunk in chunks2])
    # dev_data2 = np.array([chunk.mean(axis=0) for chunk in chunks3])
    # std_data2 = np.array([chunk.mean(axis=0) for chunk in chunks4])
    # data1 = [dev_data1,dev_data2]
    # data2 = [std_data1,std_data2]
    
    # Passive
    Data_Path = os.path.join(Sub_Path, Cond_list[1])
    raw = mne.io.read_raw_eeglab(input_fname=os.path.join(Data_Path, 'Spatial_filtered_ICA_Format.set'), preload=True)  
    raw = raw.reorder_channels(desired_chan_order)
    raw.load_data()
    # M1M2 reference
    reference = ['M1', 'M2']
    raw.set_eeg_reference(reference)
    events, event_id = mne.events_from_annotations(raw)
    baseline = (None, 0)
    epochs = mne.Epochs(raw, events=events, event_id=event_id, event_repeated='merge',
                        tmin=tmin, tmax=tmax, baseline=baseline, preload=True)
    
    Full_epochs = epochs.copy()     
    #  Extracting Epochs
    global_std_1, global_dev_1  = extract_pairwise_mark(Full_epochs,std_mark='s11',dev_mark='s22',mark_step=1,event_id=event_id,threshold=threshold)
    global_std_2, global_dev_2  = extract_pairwise_mark(Full_epochs,std_mark='s1',dev_mark='s2',mark_step=1,event_id=event_id,threshold=threshold)
    pas_diff_global1 = global_dev_1[:,:,interest_point1:interest_point2]-global_std_1[:,:,interest_point1:interest_point2]
    pas_diff_global2 = global_dev_2[:,:,interest_point1:interest_point2]-global_std_2[:,:,interest_point1:interest_point2]

    # split into 50 blocks
    # n = 50 
    # chunks1 = np.array_split(global_dev_1[:,:,interest_point1:interest_point2], n, axis=0)
    # chunks2 = np.array_split(global_std_1[:,:,interest_point1:interest_point2], n, axis=0)
    # chunks3 = np.array_split(global_dev_2[:,:,interest_point1:interest_point2], n, axis=0)
    # chunks4 = np.array_split(global_std_2[:,:,interest_point1:interest_point2], n, axis=0)
    # dev_data1 = np.array([chunk.mean(axis=0) for chunk in chunks1])
    # std_data1 = np.array([chunk.mean(axis=0) for chunk in chunks2])
    # dev_data2 = np.array([chunk.mean(axis=0) for chunk in chunks3])
    # std_data2 = np.array([chunk.mean(axis=0) for chunk in chunks4])

    # data1 = [dev_data1,dev_data2]
    # data2 = [std_data1,std_data2]
    
    # split into 50 blocks
    n = 50  
    chunks1 = np.array_split(act_diff_global1, n, axis=0)
    chunks2 = np.array_split(act_diff_global2, n, axis=0)
    chunks3 = np.array_split(pas_diff_global1, n, axis=0)
    chunks4 = np.array_split(pas_diff_global2, n, axis=0)
    act_data1 = np.array([chunk.mean(axis=0) for chunk in chunks1])
    act_data2 = np.array([chunk.mean(axis=0) for chunk in chunks2])
    pas_data1 = np.array([chunk.mean(axis=0) for chunk in chunks3])
    pas_data2 = np.array([chunk.mean(axis=0) for chunk in chunks4])
    data1 = [act_data1,act_data2]
    data2 = [pas_data1,pas_data2]

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
np.savez(os.path.join(save_file_path, 'acc_perm_Local_SVM_block_compare_50.npz'), accuracy_true=accuracy_true,accuracy_false=accuracy_false)
import numpy as np
import mne
import os
from EEGModels import EEGNet
import tensorflow as tf
from tensorflow import keras
from keras import utils as np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def contin_significant_window(mixed_condition, window_size):
    """
    Detect continuous significant windows.

    Parameters:
    mixed_condition : ndarray
        Boolean array indicating the condition (True or False).
    window_size : int
        The size of the continuous window to check.

    Returns:
    result : ndarray
        An array of the same shape as the input, indicating the positions of significant windows (1 indicates significant, 0 indicates not significant).
    """
    # Convert False to 0 and True to 1
    condition_as_int = mixed_condition.astype(int)
    
    # Initialize the result array
    result = np.zeros_like(mixed_condition, dtype=int)

    # Use a sliding window to detect segments of continuous window_size
    for i in range(len(mixed_condition) - window_size + 1):
        # Check if the entire window is 1
        if np.all(condition_as_int[i:i + window_size] == 1):
            result[i:i + window_size] = 1

    return result

def extract_single_mark(epochs,mark_name,event_id,threshold):
    """
    Extract epochs corresponding to a specific event marker and remove epochs exceeding a threshold.

    Parameters:
    - epochs: mne.Epochs object containing the epochs data.
    - mark_name: str, the name of the event marker to extract.
    - event_id: dict, mapping of event names to their corresponding numerical IDs.
    - threshold: float, the absolute amplitude threshold in microvolts for removing epochs.

    Returns:
    - EEG_data: mne.Epochs object containing the filtered epochs.
    """
    EEG_Idx = []
    mark_num = event_id[mark_name]
    for i in range(2, len(epochs.events)):
        # 判断是否符合条件
        if epochs.events[i, 2] == mark_num:   
            # 符合条件的索引加入列表
            EEG_Idx.append(i)

    EEG_Idx = np.array(EEG_Idx)
    EEG_data = epochs[EEG_Idx]

    EEG_data, dropped_count = absolute_threshold(EEG_data,threshold)

    return EEG_data._data, dropped_count

def extract_pairwise_mark(epochs,std_mark,dev_mark,mark_step,event_id,threshold):
    # local: mark_step=1  global: mark_step=2
    EEG_Idx = []
    mark_dev = event_id[dev_mark]
    mark_std = event_id[std_mark]
    for i in range(2, len(epochs.events)):
        # 判断是否符合条件
        if epochs.events[i, 2] == mark_dev and epochs.events[i - mark_step, 2] == mark_std:   
            # 符合条件的索引加入列表
            EEG_Idx.append(i)

    EEG_Idx = np.array(EEG_Idx)
    DEV_data = epochs[EEG_Idx]
    STD_data = epochs[EEG_Idx - mark_step]

    DEV_data, STD_data = absolute_threshold_pairwise(DEV_data,STD_data,threshold)

    return STD_data._data, DEV_data._data

def extract_special_pairwise_mark(epochs,std_mark,dev_mark,mark_step,event_id,threshold):
    # local: mark_step=1  global: mark_step=2
    EEG_Idx = []
    mark_dev = event_id[dev_mark]
    mark_std = event_id[std_mark]
    specical_mark = event_id['s3']
    for i in range(2, len(epochs.events)):
        # 判断是否符合条件
        if epochs.events[i, 2] == mark_dev and epochs.events[i - mark_step, 2] == mark_std:   
            # 符合条件的索引加入列表
            EEG_Idx.append(i)
        elif epochs.events[i, 2] == mark_dev and epochs.events[i - mark_step, 2] == specical_mark:  
            EEG_Idx.append(i)

    EEG_Idx = np.array(EEG_Idx)
    DEV_data = epochs[EEG_Idx]
    STD_data = epochs[EEG_Idx - mark_step]

    DEV_data, STD_data = absolute_threshold_pairwise(DEV_data,STD_data,threshold)

    return STD_data._data, DEV_data._data

def single_mark_array(extracted_data,mark_name1):
    # 计算平均波形
    mean_mark1 = extracted_data[mark_name1].average()

    return mean_mark1.data

def combine_standard(epochs, *mark_names, threshold):
    """
    合并指定标记的 epochs，并根据阈值丢弃超出阈值的 epochs。

    参数：
    epochs : mne.Epochs
        包含 EEG 数据的 epochs 对象。
    mark_names : str
        要提取的标记名称，可以传入任意数量的标记。
    threshold : float
        按绝对值计算的幅度阈值。

    返回：
    mean_EEG_data : ndarray
        剩余 epochs 的均值。
    dropped_count : int
        由于阈值丢弃的 epochs 数量。
    """
    # 验证标记名称
    for mark_name in mark_names:
        if mark_name not in epochs.event_id:
            raise ValueError(f"标记 {mark_name} 没有在 epochs 中找到。")

    # 提取并合并 epochs
    epochs_list = [epochs[mark_name] for mark_name in mark_names]
    combined_epochs = mne.concatenate_epochs(epochs_list)

    combined_epochs, dropped_count = absolute_threshold(combined_epochs,threshold)


    return combined_epochs.get_data()

def absolute_threshold(epochs,threshold):
    # 绝对阈值法
    abs_amp_EEG = np.abs(epochs.get_data())
    abs_amp_EEG = np.reshape(abs_amp_EEG,(abs_amp_EEG.shape[0], -1))

    # Identify epochs exceeding the threshold
    exceed_thre_EEG = np.max(abs_amp_EEG, axis=1) > threshold

    # Count the number of epochs to be dropped
    dropped_count = np.sum(exceed_thre_EEG)

    # Drop epochs exceeding the threshold
    if dropped_count > 0:
        epochs = epochs.drop(np.where(exceed_thre_EEG)[0])

    return epochs, dropped_count

def absolute_threshold_pairwise(epoch_1,epoch_2,threshold):
    abs_amp_DEV = np.abs(epoch_1.get_data())
    abs_amp_STD = np.abs(epoch_2.get_data())
    abs_amp_DEV = np.reshape(abs_amp_DEV,(abs_amp_DEV.shape[0], -1))
    abs_amp_STD = np.reshape(abs_amp_STD,(abs_amp_STD.shape[0], -1))
    #
    exceed_thre_DEV = np.where(np.max(abs_amp_DEV, axis=1) > threshold)[0]
    exceed_thre_STD = np.where(np.max(abs_amp_STD, axis=1) > threshold)[0]
    exceed_thre_indices = np.concatenate((exceed_thre_DEV, exceed_thre_STD))

    # 对应删除超过100μV的试次
    if exceed_thre_indices.size > 0:
        epoch_1 = epoch_1.drop(exceed_thre_indices)
        epoch_2 = epoch_2.drop(exceed_thre_indices)

    return epoch_1, epoch_2

def compare_three_Dev(epochs,event_id,threshold,mark_STD,mark_DEV):
    # Type1: within stimuli pairs
    EEG_Idx = []
    mark_dev = event_id[mark_DEV]
    mark_std = event_id[mark_STD]
    for i in range(2, len(epochs.events)):
        # 判断是否符合条件
        if epochs.events[i, 2] == mark_dev and epochs.events[i - 1, 2] == mark_std:   
            # 符合条件的索引加入列表
            EEG_Idx.append(i)

    EEG_Idx = np.array(EEG_Idx)
    DEV_data_1 = epochs[EEG_Idx]
    STD_data_1 = epochs[EEG_Idx - 1]

    # 绝对阈值法
    DEV_data_1, STD_data_1 = absolute_threshold_pairwise(DEV_data_1,STD_data_1,threshold)


    # Type2: Dev STD_350ms
    DEV_data_2 = epochs[mark_DEV]
    epochs_list = [epochs['s11'], epochs['s1']]
    STD_data_2 = mne.concatenate_epochs(epochs_list)

    # 绝对阈值法
    DEV_data_2, dropped_count = absolute_threshold(DEV_data_2,threshold)  
    STD_data_2, dropped_count = absolute_threshold(STD_data_2,threshold)   

    # Type3: Difference wave of DEV-STD minus Difference wave of STD-STD

    difference_data = DEV_data_1.get_data()-STD_data_1.get_data()
    difference_epochs_1 = DEV_data_1.copy()
    difference_epochs_1._data = difference_data

    EEG_Idx = []
    mark_std_1000 = event_id['s4']
    mark_std_350 = event_id['s1']
    for i in range(2, len(epochs.events)):
        # 判断是否符合条件
        if epochs.events[i, 2] == mark_std_1000 and epochs.events[i - 1, 2] == mark_std_350:   
            # 符合条件的索引加入列表
            EEG_Idx.append(i)

    EEG_Idx = np.array(EEG_Idx)
    STD_data_1000 = epochs[EEG_Idx]
    STD_data_350 = epochs[EEG_Idx - 1]

    # 绝对阈值法
    STD_data_1000, STD_data_350 = absolute_threshold_pairwise(STD_data_1000,STD_data_350,threshold)

    difference_data = STD_data_1000.get_data()-STD_data_350.get_data()
    difference_epochs_2 = STD_data_1000.copy()
    difference_epochs_2._data = difference_data

    DEV_1 = DEV_data_1.average()
    STD_1 = STD_data_1.average()
    DEV_2 = DEV_data_2.average()
    STD_2 = STD_data_2.average()
    DW_1 = difference_epochs_1.average()
    DW_2 = difference_epochs_2.average()


    return DEV_1._data, STD_1._data, DEV_2._data, STD_2._data, DW_1._data, DW_2._data   

def extract_global_mark(epochs,global_std,global_dev,event_id,threshold):

    EEG_Idx = []
    mark_dev = event_id[global_dev]
    mark_std = event_id[global_std]
    for i in range(2, len(epochs.events)):
        # 判断是否符合条件
        if epochs.events[i, 2] == mark_dev and epochs.events[i - 2, 2] == mark_std:   
            # 符合条件的索引加入列表
            EEG_Idx.append(i)

    EEG_Idx = np.array(EEG_Idx)
    DEV_data = epochs[EEG_Idx]
    STD_data = epochs[EEG_Idx - 2]

    DEV_data, STD_data = absolute_threshold_pairwise(DEV_data,STD_data,threshold)

    return DEV_data._data, STD_data._data

def extract_pairwise1(epochs,std_mark,dev_mark,event_id,threshold):
    # local: mark_step=1  global: 
    EEG_Idx = []
    mark_dev = event_id[dev_mark]
    mark_std = event_id[std_mark]
    for i in range(2, len(epochs.events)):
        # 判断是否符合条件
        if epochs.events[i, 2] == mark_dev and epochs.events[i - 2, 2] == mark_std:   
            # 符合条件的索引加入列表
            EEG_Idx.append(i)

    EEG_Idx = np.array(EEG_Idx)
    DEV_data = epochs[EEG_Idx]
    STD_data = epochs[EEG_Idx - 2]

    DEV_data, STD_data = absolute_threshold_pairwise(DEV_data,STD_data,threshold)
    
    EEG_Idx = []
    mark_dev = event_id['s4']
    mark_std = event_id['s1']
    for i in range(2, len(epochs.events)):
        # 判断是否符合条件
        if epochs.events[i, 2] == mark_dev and epochs.events[i - 1, 2] == mark_std:   
            # 符合条件的索引加入列表
            EEG_Idx.append(i)

    EEG_Idx = np.array(EEG_Idx)
    DEV_data_1 = epochs[EEG_Idx]
    STD_data_1 = epochs[EEG_Idx - 1]

    DEV_data_1, STD_data_1 = absolute_threshold_pairwise(DEV_data_1,STD_data_1,threshold)

    return STD_data._data, DEV_data._data, STD_data_1._data, DEV_data_1._data

def extract_pairwise2(epochs,std_mark,dev_mark,event_id,threshold):
    EEG_Idx = []
    mark_dev = event_id[dev_mark]
    mark_std = event_id[std_mark]
    for i in range(2, len(epochs.events)):
        # 判断是否符合条件
        if epochs.events[i, 2] == mark_dev and epochs.events[i - 2, 2] == mark_std:   
            # 符合条件的索引加入列表
            EEG_Idx.append(i)

    EEG_Idx = np.array(EEG_Idx)
    DEV_data = epochs[EEG_Idx]
    STD_data = epochs[EEG_Idx - 2]

    DEV_data, STD_data = absolute_threshold_pairwise(DEV_data,STD_data,threshold)
    
    EEG_Idx = []
    mark_dev = event_id['s22']
    mark_std = event_id['s11']
    for i in range(2, len(epochs.events)):
        # 判断是否符合条件
        if epochs.events[i, 2] == mark_dev and epochs.events[i - 1, 2] == mark_std:   
            # 符合条件的索引加入列表
            EEG_Idx.append(i)

    EEG_Idx = np.array(EEG_Idx)
    DEV_data_1 = epochs[EEG_Idx]
    STD_data_1 = epochs[EEG_Idx - 1]

    DEV_data_1, STD_data_1 = absolute_threshold_pairwise(DEV_data_1,STD_data_1,threshold)

    return STD_data._data, DEV_data._data, STD_data_1._data, DEV_data_1._data


def EEGNET_classify(X,Y,save_subpath,n_permutation,shuffle_type):
    accuracy = np.empty((n_permutation))
    n_trials, n_channels, n_times = X.shape  

    for isperm in range(n_permutation):
        if shuffle_type==0:
            indices = np.random.permutation(n_trials)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
        elif shuffle_type==1:
            indices = np.random.permutation(n_trials)
            X_shuffled = X
            Y_shuffled = Y[indices]

        # train/test :  80/20%
        # Split X into train and test sets
        X1_train, X1_test, Y1_train, Y1_test = train_test_split(X_shuffled, Y_shuffled, test_size=0.2, random_state=42)  
        # Define EEGNET
        kernels, chans, samples = 1, n_channels, n_times  

        # Initialize performance metrics
        accuracy_scores = []

        Y_test = np_utils.to_categorical(Y1_test)
        X_test = X1_test.reshape(X1_test.shape[0], chans, samples, kernels)
        # Define 5-fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # scikit-learn库中的StratifiedKFold类，该类能够确保在划分数据集时，每个折中的类别比例都与原始数据集中的比例相同。
        # n_splits参数指定了要将数据集分成的折数，shuffle参数指定是否要在分折之前打乱数据集，random_state参数指定了随机种子，以确保每次运行时都会得到相同的结果。

        # 用 skf.split 函数将数据集分成训练集和测试集。其中 train_idx 和 test_idx 是训练集和测试集的索引数组。
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X1_train, Y1_train)):
            print('Fold:', fold_idx + 1)
            # Get current fold data
            X_train_fold, y_train_fold = X1_train[train_idx], Y1_train[train_idx]
            X_val_fold, y_val_fold = X1_train[test_idx], Y1_train[test_idx]
            # Convert labels to one-hot encodings
            Y_train_fold = np_utils.to_categorical(y_train_fold)
            Y_val_fold = np_utils.to_categorical(y_val_fold)        

            # Convert data to NHWC format
            X_train_fold = X_train_fold.reshape(X_train_fold.shape[0], chans, samples, kernels)
            X_val_fold = X_val_fold.reshape(X_val_fold.shape[0], chans, samples, kernels)        

            # Define the model and the optimizer
            model = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
                            dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16,
                            dropoutType='Dropout')  # try 8 2 16  428
            # nb_classes：分类任务中的类数。
            # Chans：脑电图通道数。
            # Samples：每个 EEG 通道中的时间样本数。
            # dropoutRate：训练期间使用的辍学率。
            # kernLength：时间卷积核的长度。，通常不超过采样数的一半
            # F1：第一个卷积块中的过滤器数量。
            # D：深度卷积块的深度乘数。
            # F2：第二个卷积块中的过滤器数量。
            # dropoutType：要使用的丢失类型，“Dropout”或“SpatialDropout2D”。
            # Create the CombinedOptimizerCallback    

            # combined_optimizer_callback = CombinedOptimizerCallback()
            # Set a valid path for model checkpoints
            if os.path.exists(save_subpath):
                pass
            else:
                os.makedirs(save_subpath)
            file_subpath = os.path.join(save_subpath,f'checkpoint_fold{fold_idx + 1}.h5')
            checkpointer = ModelCheckpoint(filepath=file_subpath, verbose=1, save_best_only=True)
            # Compile the model with the SGD optimizer
            # model.compile(loss='binary_crossentropy', optimizer=SGD(lr=lr_schedule(0)), metrics=['accuracy'])
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
            
            # count number of parameters in the model
            numParams = model.count_params()    

            # Fit the model with the combined optimizer callback
            fittedMode = model.fit(X_train_fold, Y_train_fold, batch_size=16, epochs=100, verbose=2,
                                validation_data=(X_val_fold, Y_val_fold),
                                callbacks=[checkpointer],workers=32,use_multiprocessing=True)

            # load optimal weights
            model.load_weights(file_subpath)

            y_pred = model.predict(X_test)
            # Convert one-hot encodings to labels
            y_pred_labels = np.argmax(y_pred, axis=1)
            acc = np.mean(y_pred_labels == Y_test.argmax(axis=-1))
            accuracy_scores.append(acc)

        accuracy[isperm] = np.mean(accuracy_scores,axis=0)

    return accuracy

# def SVM_classify(X, Y, n_permutation, shuffle_type):
#     accuracy = np.empty((n_permutation))
#     n_trials= X.shape[0]

#     for isperm in range(n_permutation):
#         # Shuffle data
#         if shuffle_type == 0:
#             indices = np.random.permutation(n_trials)
#             X_shuffled = X[indices]
#             Y_shuffled = Y[indices]
#         elif shuffle_type == 1:
#             indices = np.random.permutation(n_trials)
#             X_shuffled = X
#             Y_shuffled = Y[indices]

#         # Train/test split
#         X_train, X_test, Y_train, Y_test = train_test_split(X_shuffled, Y_shuffled, test_size=0.2, random_state=42)  
        
#         # Initialize performance metrics
#         accuracy_scores = []

#         # # Define 5-fold cross-validation
#         # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#         # for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_shuffled, Y_shuffled)):
#         #     print('Fold:', fold_idx + 1)

#         #     # Get current fold data
#         #     X_train_fold, y_train_fold = X_shuffled[train_idx], Y_shuffled[train_idx]
#         #     X_val_fold, y_val_fold = X_shuffled[test_idx], Y_shuffled[test_idx]

#         #     # Standardize data
#         #     scaler = StandardScaler()
#         #     X_train_fold = scaler.fit_transform(X_train_fold.reshape(X_train_fold.shape[0], -1))
#         #     X_val_fold = scaler.transform(X_val_fold.reshape(X_val_fold.shape[0], -1))

#         #     # Initialize SVM classifier
#         #     model = SVC(kernel='linear', probability=True)

#         #     # Fit the model
#         #     model.fit(X_train_fold, y_train_fold)

#         #     # Predict and calculate accuracy
#         #     y_pred = model.predict(X_val_fold)
#         #     acc = accuracy_score(y_val_fold, y_pred)
#         #     accuracy_scores.append(acc)

#         # # Calculate mean accuracy for the permutation
#         # accuracy[isperm] = np.mean(accuracy_scores)

#         # Standardize data
#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
#         X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1))        
#         # Define 5-fold cross-validation
#         skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#         # Initialize parameter grid for GridSearchCV
#         param_grid = {
#             'C': [0.1, 1, 10, 100],
#             'kernel': ['linear', 'rbf'],
#         }

#         # Initialize SVM classifier
#         model = SVC(probability=True)

#         # Initialize GridSearchCV with your StratifiedKFold instance
#         grid_search = GridSearchCV(model, param_grid, cv=skf, n_jobs=-1, verbose=1)

#         # Fit the model with grid search
#         grid_search.fit(X_train, Y_train)

#         # Predict using the best model found by GridSearchCV
#         y_pred = grid_search.predict(X_test_scaled)
#         acc = accuracy_score(Y_test, y_pred)

#         # Store the accuracy for the permutation
#         accuracy[isperm] = acc

#     return accuracy


def SVM_classify(X, Y, n_permutation, shuffle_type):
    accuracy = np.empty((n_permutation))
    n_trials = X.shape[0]  

    for isperm in range(n_permutation):
        # Shuffle data
        if shuffle_type == 0:
            indices = np.random.permutation(n_trials)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
        elif shuffle_type == 1:
            indices = np.random.permutation(n_trials)
            X_shuffled = X
            Y_shuffled = Y[indices]

        # Initialize performance metrics
        accuracy_scores = []

        # Define 5-fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_shuffled, Y_shuffled)):
            # print('Fold:', fold_idx + 1)

            # Get current fold data
            X_train_fold, y_train_fold = X_shuffled[train_idx], Y_shuffled[train_idx]
            X_val_fold, y_val_fold = X_shuffled[test_idx], Y_shuffled[test_idx]

            # Standardize data
            scaler = StandardScaler()
            X_train_fold = scaler.fit_transform(X_train_fold.reshape(X_train_fold.shape[0], -1))
            X_val_fold = scaler.transform(X_val_fold.reshape(X_val_fold.shape[0], -1))

            # Initialize SVM classifier
            model = SVC(kernel='linear', probability=True)

            # Fit the model
            model.fit(X_train_fold, y_train_fold)

            # Predict and calculate accuracy
            y_pred = model.predict(X_val_fold)
            acc = accuracy_score(y_val_fold, y_pred)
            accuracy_scores.append(acc)

        # Calculate mean accuracy for the permutation
        accuracy[isperm] = np.mean(accuracy_scores)

    return accuracy
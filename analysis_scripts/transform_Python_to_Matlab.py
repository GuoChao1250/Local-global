#%%
import numpy as np
import scipy.io
import os

save_file_path = f'.\\Fig\\data'
npzfile = np.load(os.path.join(save_file_path, 'NOV_MMN.npz'))
scipy.io.savemat('converted_NOV_MMN.mat', {key: npzfile[key] for key in npzfile})
# with open(os.path.join(save_file_path, 'times_800.pkl'), 'rb') as f:
#     data = pickle.load(f)

# # Save data as a. mat file
# scipy.io.savemat('times_800.mat', {'times_800': data})

# with open(os.path.join(save_file_path, 'times_350.pkl'), 'rb') as f:
#     data = pickle.load(f)

# # Save data as a. mat file
# scipy.io.savemat('times_350.mat', {'times_350': data})
# %%

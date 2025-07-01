#%%
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
#%% STD
alpha_p = 0.001/30
###### combine active and passive  ##########
# path
save_file_path = './SVM/accuracy'

# import passive data
data_passive = np.load(os.path.join(save_file_path, 'acc_perm_STD_SVM_block_pas_50.npz'))
accuracy_true_passive = data_passive['accuracy_true']
accuracy_false_passive = data_passive['accuracy_false']
# import active data
data_active = np.load(os.path.join(save_file_path, 'acc_perm_STD_SVM_block_act_50.npz'))
accuracy_true_active = data_active['accuracy_true']
accuracy_false_active = data_active['accuracy_false']
n_perm_passive = accuracy_true_passive.shape[2]
n_perm_active = accuracy_true_active.shape[2]

fig, axs = plt.subplots(figsize=(6, 10), tight_layout=True)
# passive
for j in range(accuracy_true_passive.shape[0]):
    axs.scatter([j + 1] * n_perm_passive, -accuracy_true_passive[j, 0, :], color=(1, 0, 0, 0.3), marker='s', s=20)  
    axs.scatter([j + 1] * n_perm_passive, -accuracy_false_passive[j, 0, :], color=(0, 0, 1, 0.3), marker='s', s=20)

# Calculate and plot the 95% locus curve for passive
percentile_95_passive = np.percentile(accuracy_false_passive, 95, axis=2).flatten()
axs.plot(range(1, accuracy_false_passive.shape[0] + 1), -percentile_95_passive, color='black', linewidth=4, label='Passive 95% Percentile')

# Initialize result array
result_pas = np.zeros(30, dtype=int)
# t test
for subject in range(30):
    t_stat, p_value = ttest_rel(
        accuracy_true_passive[subject, 0, :],
        np.full_like(accuracy_true_passive[subject, 0, :], percentile_95_passive[subject,]),
        alternative='greater' 
    )
    if p_value < alpha_p:
        result_pas[subject] = 1

# active
for j in range(accuracy_true_active.shape[0]):
    axs.scatter([j + 1] * n_perm_active, accuracy_true_active[j, 0, :], color=(1, 0, 0, 0.3), marker='s', s=20)  
    axs.scatter([j + 1] * n_perm_active, accuracy_false_active[j, 0, :], color=(0, 0, 1, 0.3), marker='s', s=20)
# Calculate and plot the 95% locus curve
percentile_95_active = np.percentile(accuracy_false_active, 95, axis=2).flatten()
axs.plot(range(1, accuracy_false_active.shape[0] + 1), percentile_95_active, color='black', linewidth=4, label='Active 95% Percentile')

result_act = np.zeros(30, dtype=int) 
# t test
for subject in range(30):
    t_stat, p_value = ttest_rel(
        accuracy_true_active[subject, 0, :],
        np.full_like(accuracy_true_active[subject, 0, :], percentile_95_active[subject,]),
        alternative='greater' 
    )
    if p_value < alpha_p: 
        result_act[subject] = 1

# axs.set_title('Accuracy by Participants', fontsize=24, fontweight='bold')
axs.set_ylabel('Accuracy', fontsize=20)  # 修改为ylabel
axs.set_xlabel('Participants', fontsize=20)  # 修改为xlabel
axs.set_ylim(-1, 1)  # 交换xlim和ylim
axs.set_xlim(0, 30)  # 交换xlim和ylim
axs.set_yticks([-1, -0.5, 0, 0.5, 1])  # 修改yticks
axs.set_yticklabels([f"{tick:.1f}" for tick in [1, 0.5, 0, 0.5, 1]], fontsize=16, fontweight='bold')  # 修改yticklabels
axs.set_xticks([10, 20, 30])  # 修改xticks
axs.set_xticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                    if tick != 0 else "0" for tick in [10, 20, 30]], fontsize=16, fontweight='bold')  # 修改xticklabels
axs.axhline(y=0, color='gray', linestyle='--', linewidth=2)  # 将axvline改为axhline
# Move y-axis to the right
axs.yaxis.tick_right()  # 将刻度移到右侧
axs.yaxis.set_label_position("right")  # 将标签移到右侧
# plt.show()
save_path = (
f"./Fig/individual/"
f"STD_combined.png"
)      
plt.savefig(save_path, bbox_inches='tight')
# %% local I 
alpha_p = 0.001/30
###### combine active and passive  ##########
# path
save_file_path = './SVM/accuracy'
# import passive data
data_passive = np.load(os.path.join(save_file_path, 'acc_perm_Local_SVM_block_pas_50.npz'))
accuracy_true_passive = data_passive['accuracy_true']
accuracy_false_passive = data_passive['accuracy_false']
# import active data
data_active = np.load(os.path.join(save_file_path, 'acc_perm_Local_SVM_block_act_50.npz'))
accuracy_true_active = data_active['accuracy_true']
accuracy_false_active = data_active['accuracy_false']
n_perm_passive = accuracy_true_passive.shape[2]
n_perm_active = accuracy_true_active.shape[2]

fig, axs = plt.subplots(figsize=(6, 10), tight_layout=True)
# passive
for j in range(accuracy_true_passive.shape[0]):
    axs.scatter([j + 1] * n_perm_passive, -accuracy_true_passive[j, 0, :], color=(1, 0, 0, 0.3), marker='s', s=20)  
    axs.scatter([j + 1] * n_perm_passive, -accuracy_false_passive[j, 0, :], color=(0, 0, 1, 0.3), marker='s', s=20)

# Calculate and plot the 95% locus curve for passive
percentile_95_passive = np.percentile(accuracy_false_passive[:, 0:1, :], 95, axis=2).flatten()
axs.plot(range(1, accuracy_false_passive.shape[0] + 1), -percentile_95_passive, color='black', linewidth=4, label='Passive 95% Percentile')
# Initialize result array
result_pas = np.zeros(30, dtype=int)
# t test
for subject in range(30):
    t_stat, p_value = ttest_rel(
        accuracy_true_passive[subject, 0, :],
        np.full_like(accuracy_true_passive[subject, 0, :], percentile_95_passive[subject,]),
        alternative='greater' 
    )
    if p_value < alpha_p:
        result_pas[subject] = 1

# active
for j in range(accuracy_true_active.shape[0]):
    axs.scatter([j + 1] * n_perm_active, accuracy_true_active[j, 0, :], color=(1, 0, 0, 0.3), marker='s', s=20)  
    axs.scatter([j + 1] * n_perm_active, accuracy_false_active[j, 0, :], color=(0, 0, 1, 0.3), marker='s', s=20)
# Calculate and plot the 95% locus curve
percentile_95_active = np.percentile(accuracy_false_active[:, 0:1, :], 95, axis=2).flatten()
axs.plot(range(1, accuracy_false_active.shape[0] + 1), percentile_95_active, color='black', linewidth=4, label='Active 95% Percentile')
result_act = np.zeros(30, dtype=int) 
# t test
for subject in range(30):
    t_stat, p_value = ttest_rel(
        accuracy_true_active[subject, 0, :],
        np.full_like(accuracy_true_active[subject, 0, :], percentile_95_active[subject,]),
        alternative='greater' 
    )
    if p_value < alpha_p: 
        result_act[subject] = 1


axs.set_ylabel('Accuracy', fontsize=20)  
axs.set_xlabel('Participants', fontsize=20) 
axs.set_ylim(-1, 1)  
axs.set_xlim(0, 30)  
axs.set_yticks([-1, -0.5, 0, 0.5, 1])  
axs.set_yticklabels([f"{tick:.1f}" for tick in [1, 0.5, 0, 0.5, 1]], fontsize=16, fontweight='bold')  
axs.set_xticks([10, 20, 30]) 
axs.set_xticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                    if tick != 0 else "0" for tick in [10, 20, 30]], fontsize=16, fontweight='bold')  
axs.axhline(y=0, color='gray', linestyle='--', linewidth=2) 
# Move y-axis to the right
axs.yaxis.tick_right()  
axs.yaxis.set_label_position("right")  
# plt.show()
save_path = (
f"./Fig/individual/"
f"LocalI_combined.png"
)      
plt.savefig(save_path, bbox_inches='tight')
#%%
##### Local II
fig, axs = plt.subplots(figsize=(6, 10), tight_layout=True)
# passive
for j in range(accuracy_true_passive.shape[0]):
    axs.scatter([j + 1] * n_perm_passive, -accuracy_true_passive[j, 1, :], color=(1, 0, 0, 0.3), marker='s', s=20)  
    axs.scatter([j + 1] * n_perm_passive, -accuracy_false_passive[j, 1, :], color=(0, 0, 1, 0.3), marker='s', s=20)
# Calculate and plot the 95% locus curve
percentile_95_passive = np.percentile(accuracy_false_passive[:, 1:2, :], 95, axis=2).flatten()
axs.plot(range(1, accuracy_false_passive.shape[0] + 1), -percentile_95_passive, color='black', linewidth=4, label='Passive 95% Percentile')

# Initialize result array
result_pas = np.zeros(30, dtype=int)  

# t test
for subject in range(30):
    t_stat, p_value = ttest_rel(
        accuracy_true_passive[subject, 1, :],
        np.full_like(accuracy_true_passive[subject, 1, :], percentile_95_passive[subject,]),
        alternative='greater' 
    )
    if p_value < alpha_p: 
        result_pas[subject] = 1
# active 
for j in range(accuracy_true_active.shape[0]):
    axs.scatter([j + 1 ] * n_perm_active, accuracy_true_active[j, 1, :], color=(1, 0, 0, 0.3), marker='s', s=20) 
    axs.scatter([j + 1 ] * n_perm_active, accuracy_false_active[j, 1, :], color=(0, 0, 1, 0.3), marker='s', s=20)
# Calculate and plot the 95% locus curve
percentile_95_active = np.percentile(accuracy_false_active[:, 1:2, :], 95, axis=2).flatten()
axs.plot(range(1, accuracy_false_active.shape[0] + 1), percentile_95_active, color='black', linewidth=4, label='Active 95% Percentile')

result_act = np.zeros(30, dtype=int) 
# t test
for subject in range(30):
    t_stat, p_value = ttest_rel(
        accuracy_true_active[subject, 1, :],
        np.full_like(accuracy_true_active[subject, 1, :], percentile_95_active[subject,]),
        alternative='greater' 
    )
    if p_value < alpha_p:
        result_act[subject] = 1

axs.set_ylabel('Accuracy', fontsize=20)  
axs.set_xlabel('Participants', fontsize=20) 
axs.set_ylim(-1, 1)  
axs.set_xlim(0, 30)  
axs.set_yticks([-1, -0.5, 0, 0.5, 1])  
axs.set_yticklabels([f"{tick:.1f}" for tick in [1, 0.5, 0, 0.5, 1]], fontsize=16, fontweight='bold')  
axs.set_xticks([10, 20, 30]) 
axs.set_xticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                    if tick != 0 else "0" for tick in [10, 20, 30]], fontsize=16, fontweight='bold')  
axs.axhline(y=0, color='gray', linestyle='--', linewidth=2) 
# Move y-axis to the right
axs.yaxis.tick_right()  
axs.yaxis.set_label_position("right")  
# plt.show()
save_path = (
f"./Fig/individual/"
f"LocalII_combined.png"
)      
plt.savefig(save_path, bbox_inches='tight')
# %% Global I 
###### combine active and passive  ##########
alpha_p = 0.001/30
# path
save_file_path = './SVM/accuracy'
# import passive data
data_passive = np.load(os.path.join(save_file_path, 'acc_perm_Global_400ms_SVM_block_pas_50.npz'))
accuracy_true_passive = data_passive['accuracy_true']
accuracy_false_passive = data_passive['accuracy_false']
# import active data
data_active = np.load(os.path.join(save_file_path, 'acc_perm_Global_400ms_SVM_block_act_50.npz'))
accuracy_true_active = data_active['accuracy_true']
accuracy_false_active = data_active['accuracy_false']
n_perm_passive = accuracy_true_passive.shape[2]
n_perm_active = accuracy_true_active.shape[2]

fig, axs = plt.subplots(figsize=(6, 10), tight_layout=True)
# passive
for j in range(accuracy_true_passive.shape[0]):
    axs.scatter([j + 1] * n_perm_passive, -accuracy_true_passive[j, 0, :], color=(1, 0, 0, 0.3), marker='s', s=20)  
    axs.scatter([j + 1] * n_perm_passive, -accuracy_false_passive[j, 0, :], color=(0, 0, 1, 0.3), marker='s', s=20)

# Calculate and plot the 95% locus curve for passive
percentile_95_passive = np.percentile(accuracy_false_passive[:, 0:1, :], 95, axis=2).flatten()
axs.plot(range(1, accuracy_false_passive.shape[0] + 1), -percentile_95_passive, color='black', linewidth=4, label='Passive 95% Percentile')
# Initialize result array
result_pas = np.zeros(30, dtype=int)
# t test
for subject in range(30):
    t_stat, p_value = ttest_rel(
        accuracy_true_passive[subject, 0, :],
        np.full_like(accuracy_true_passive[subject, 0, :], percentile_95_passive[subject,]),
        alternative='greater' 
    )
    if p_value < alpha_p:
        result_pas[subject] = 1

# active
for j in range(accuracy_true_active.shape[0]):
    axs.scatter([j + 1] * n_perm_active, accuracy_true_active[j, 0, :], color=(1, 0, 0, 0.3), marker='s', s=20)  
    axs.scatter([j + 1] * n_perm_active, accuracy_false_active[j, 0, :], color=(0, 0, 1, 0.3), marker='s', s=20)
# Calculate and plot the 95% locus curve
percentile_95_active = np.percentile(accuracy_false_active[:, 0:1, :], 95, axis=2).flatten()
axs.plot(range(1, accuracy_false_active.shape[0] + 1), percentile_95_active, color='black', linewidth=4, label='Active 95% Percentile')
result_act = np.zeros(30, dtype=int) 
# t test
for subject in range(30):
    t_stat, p_value = ttest_rel(
        accuracy_true_active[subject, 0, :],
        np.full_like(accuracy_true_active[subject, 0, :], percentile_95_active[subject,]),
        alternative='greater' 
    )
    if p_value < alpha_p: 
        result_act[subject] = 1


axs.set_ylabel('Accuracy', fontsize=20)  
axs.set_xlabel('Participants', fontsize=20) 
axs.set_ylim(-1, 1)  
axs.set_xlim(0, 30)  
axs.set_yticks([-1, -0.5, 0, 0.5, 1])  
axs.set_yticklabels([f"{tick:.1f}" for tick in [1, 0.5, 0, 0.5, 1]], fontsize=16, fontweight='bold')  
axs.set_xticks([10, 20, 30]) 
axs.set_xticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                    if tick != 0 else "0" for tick in [10, 20, 30]], fontsize=16, fontweight='bold')  
axs.axhline(y=0, color='gray', linestyle='--', linewidth=2) 
# Move y-axis to the right
axs.yaxis.tick_right()  
axs.yaxis.set_label_position("right")  
# plt.show()
save_path = (
f"./Fig/individual/"
f"GlobalI_combined.png"
)      
plt.savefig(save_path, bbox_inches='tight')
#%% Global II
fig, axs = plt.subplots(figsize=(6, 10), tight_layout=True)
# passive
for j in range(accuracy_true_passive.shape[0]):
    axs.scatter([j + 1] * n_perm_passive, -accuracy_true_passive[j, 1, :], color=(1, 0, 0, 0.3), marker='s', s=20)  
    axs.scatter([j + 1] * n_perm_passive, -accuracy_false_passive[j, 1, :], color=(0, 0, 1, 0.3), marker='s', s=20)
# Calculate and plot the 95% locus curve
percentile_95_passive = np.percentile(accuracy_false_passive[:, 1:2, :], 95, axis=2).flatten()
axs.plot(range(1, accuracy_false_passive.shape[0] + 1), -percentile_95_passive, color='black', linewidth=4, label='Passive 95% Percentile')

# Initialize result array
result_pas = np.zeros(30, dtype=int)  

# t test
for subject in range(30):
    t_stat, p_value = ttest_rel(
        accuracy_true_passive[subject, 1, :],
        np.full_like(accuracy_true_passive[subject, 1, :], percentile_95_passive[subject,]),
        alternative='greater' 
    )
    if p_value < alpha_p: 
        result_pas[subject] = 1
# active 
for j in range(accuracy_true_active.shape[0]):
    axs.scatter([j + 1 ] * n_perm_active, accuracy_true_active[j, 1, :], color=(1, 0, 0, 0.3), marker='s', s=20) 
    axs.scatter([j + 1 ] * n_perm_active, accuracy_false_active[j, 1, :], color=(0, 0, 1, 0.3), marker='s', s=20)
# Calculate and plot the 95% locus curve
percentile_95_active = np.percentile(accuracy_false_active[:, 1:2, :], 95, axis=2).flatten()
axs.plot(range(1, accuracy_false_active.shape[0] + 1), percentile_95_active, color='black', linewidth=4, label='Active 95% Percentile')

result_act = np.zeros(30, dtype=int) 
# t test
for subject in range(30):
    t_stat, p_value = ttest_rel(
        accuracy_true_active[subject, 1, :],
        np.full_like(accuracy_true_active[subject, 1, :], percentile_95_active[subject,]),
        alternative='greater' 
    )
    if p_value < alpha_p:
        result_act[subject] = 1

axs.set_ylabel('Accuracy', fontsize=20)  
axs.set_xlabel('Participants', fontsize=20) 
axs.set_ylim(-1, 1)  
axs.set_xlim(0, 30)  
axs.set_yticks([-1, -0.5, 0, 0.5, 1])  
axs.set_yticklabels([f"{tick:.1f}" for tick in [1, 0.5, 0, 0.5, 1]], fontsize=16, fontweight='bold')  
axs.set_xticks([10, 20, 30]) 
axs.set_xticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                    if tick != 0 else "0" for tick in [10, 20, 30]], fontsize=16, fontweight='bold')  
axs.axhline(y=0, color='gray', linestyle='--', linewidth=2) 
# Move y-axis to the right
axs.yaxis.tick_right()  
axs.yaxis.set_label_position("right")  
# plt.show()
save_path = (
f"./Fig/individual/"
f"GlobalII_combined.png"
)      
plt.savefig(save_path, bbox_inches='tight')

# %% compare between active and passive accuary
####### STD  #######
save_file_path = f'.\\SVM\\accuracy'
data = np.load(os.path.join(save_file_path, 'acc_perm_STD_SVM_block_compare_50.npz'))
# Access the arrays using the keys
accuracy_true = data['accuracy_true']
accuracy_false = data['accuracy_false']

n_perm = accuracy_true.shape[2]
data_types = ['STD']
fig, axs = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
for j in range(accuracy_true.shape[0]):
    axs.scatter([j + 1] * n_perm, accuracy_true[j, 0, :], color=(1, 0, 0, 0.3), marker='s', s=20)  
    axs.scatter([j + 1] * n_perm, accuracy_false[j, 0, :], color=(0, 0, 1, 0.3), marker='s', s=20)

# Calculate and plot the 95% locus curve
percentile_95 = np.percentile(accuracy_false, 95, axis=2).flatten()  
# Initialize result array
result = np.zeros(30, dtype=int)  
# t test
for subject in range(30):
    t_stat, p_value = ttest_rel(
        accuracy_true[subject, 0, :],
        np.full_like(accuracy_true[subject, 0, :], percentile_95[subject,]),
        alternative='greater' 
    )
    if p_value < alpha_p: 
        result_pas[subject] = 1
 
axs.plot(range(1, accuracy_false.shape[0] + 1), percentile_95, color='black', linewidth=4, label='95% Percentile')  
axs.set_title(data_types[0], fontsize=24, fontweight='bold')
axs.set_xlabel('Participants', fontsize=20)
axs.set_ylabel('Accuracy', fontsize=20)
axs.set_xticks([10, 20, 30])  
axs.set_xticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                        if tick != 0 else "0" for tick in [10, 20, 30]],fontsize=16, fontweight='bold')  
axs.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])  
axs.set_yticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                        if tick != 0 else "0" for tick in axs.get_yticks()],fontsize=16, fontweight='bold')
axs.set_ylim(0, 1)  
axs.tick_params(axis='both', labelsize=15) 
plt.show()
#%%
############## passive ####################
data = np.load(os.path.join(save_file_path, 'acc_perm_Local_SVM_block_compare_50.npz'))
# Access the arrays using the keys
accuracy_true = data['accuracy_true']
accuracy_false = data['accuracy_false']

n_perm = accuracy_true.shape[2]
data_types = ['Local I']
fig, axs = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
for j in range(accuracy_true.shape[0]):
    axs.scatter([j + 1] * n_perm, accuracy_true[j, 0, :], color=(1, 0, 0, 0.3), marker='s', s=20) 
    axs.scatter([j + 1] * n_perm, accuracy_false[j, 0, :], color=(0, 0, 1, 0.3), marker='s', s=20)
percentile_95 = np.percentile(accuracy_false[:, 0:1, :], 95, axis=2).flatten()  
# Initialize result array
result = np.zeros(30, dtype=int)  
# t test
for subject in range(30):
    t_stat, p_value = ttest_rel(
        accuracy_true[subject, 0, :],
        np.full_like(accuracy_true[subject, 0, :], percentile_95[subject,]),
        alternative='greater' 
    )
    if p_value < alpha_p: 
        result[subject] = 1

axs.plot(range(1, accuracy_false.shape[0] + 1), percentile_95, color='black', linewidth=4, label='95% Percentile') 
axs.set_title(data_types[0], fontsize=24, fontweight='bold')
axs.set_xlabel('Participants', fontsize=20)
axs.set_ylabel('Accuracy', fontsize=20)
axs.set_xticks([10, 20, 30])  
axs.set_xticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                        if tick != 0 else "0" for tick in [10, 20, 30]],fontsize=16, fontweight='bold')  
axs.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
axs.set_yticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                        if tick != 0 else "0" for tick in axs.get_yticks()],fontsize=16, fontweight='bold')
axs.set_ylim(0, 1)  
axs.tick_params(axis='both', labelsize=15) 
plt.show()
#%%
data_types = ['Local II']
fig, axs = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
for j in range(accuracy_true.shape[0]):
    axs.scatter([j + 1] * n_perm, accuracy_true[j, 1, :], color=(1, 0, 0, 0.3), marker='s', s=20)  
    axs.scatter([j + 1] * n_perm, accuracy_false[j, 1, :], color=(0, 0, 1, 0.3), marker='s', s=20)
percentile_95 = np.percentile(accuracy_false[:, 1:2, :], 95, axis=2).flatten()  
# Initialize result array
result = np.zeros(30, dtype=int)  
# t test
for subject in range(30):
    t_stat, p_value = ttest_rel(
        accuracy_true[subject, 1, :],
        np.full_like(accuracy_true[subject, 1, :], percentile_95[subject,]),
        alternative='greater' 
    )
    if p_value < alpha_p: 
        result[subject] = 1
axs.plot(range(1, accuracy_false.shape[0] + 1), percentile_95, color='black', linewidth=4, label='95% Percentile')  # 绘制 95% 位点曲线
axs.set_title(data_types[0], fontsize=24, fontweight='bold')
axs.set_xlabel('Participants', fontsize=20)
axs.set_ylabel('Accuracy', fontsize=20)
axs.set_xticks([10, 20, 30])  
axs.set_xticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                        if tick != 0 else "0" for tick in [10, 20, 30]],fontsize=16, fontweight='bold')  
axs.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
axs.set_yticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                        if tick != 0 else "0" for tick in axs.get_yticks()],fontsize=16, fontweight='bold')
axs.set_ylim(0, 1)  
axs.tick_params(axis='both', labelsize=15) 
plt.show()
# %%
alpha_p = 0.001/30
data = np.load(os.path.join(save_file_path, 'acc_perm_Global_400ms_SVM_block_compare_50_I.npz'))
# Access the arrays using the keys
accuracy_true = data['accuracy_true']
accuracy_false = data['accuracy_false']

n_perm = accuracy_true.shape[2]
data_types = ['Global I']
fig, axs = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
for j in range(accuracy_true.shape[0]):
    axs.scatter([j + 1] * n_perm, accuracy_true[j, 0, :], color=(1, 0, 0, 0.3), marker='s', s=20) 
    axs.scatter([j + 1] * n_perm, accuracy_false[j, 0, :], color=(0, 0, 1, 0.3), marker='s', s=20)
percentile_95 = np.percentile(accuracy_false[:, 0:1, :], 95, axis=2).flatten()
axs.plot(range(1, accuracy_false.shape[0] + 1), percentile_95, color='black', linewidth=4, label='95% Percentile')  

result_global_1 = np.zeros(30, dtype=int)
# t test
for subject in range(30):
    t_stat, p_value = ttest_rel(
        accuracy_true[subject, 0, :],
        np.full_like(accuracy_true[subject, 0, :], percentile_95[subject,]),
        alternative='greater'  
    )
    if p_value < alpha_p: 
        result_global_1[subject] = 1

axs.set_title(data_types[0], fontsize=24, fontweight='bold')
axs.set_xlabel('Participants', fontsize=20)
axs.set_ylabel('Accuracy', fontsize=20)
axs.set_xticks([10, 20, 30])  
axs.set_xticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                        if tick != 0 else "0" for tick in [10, 20, 30]],fontsize=16, fontweight='bold')  
axs.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
axs.set_yticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                        if tick != 0 else "0" for tick in axs.get_yticks()],fontsize=16, fontweight='bold')
axs.set_ylim(0, 1)  
axs.tick_params(axis='both', labelsize=15)  
plt.show()
#%%
##### Global II
alpha_p = 0.001/30
data = np.load(os.path.join(save_file_path, 'acc_perm_Global_400ms_SVM_block_compare_50_II.npz'))
# Access the arrays using the keys
accuracy_true = data['accuracy_true']
accuracy_false = data['accuracy_false']
mean_acc_pas_I = np.zeros(30)
std_acc_pas_I = np.zeros(30)
n_perm = accuracy_true.shape[2]
data_types = ['Global II']
fig, axs = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
for j in range(accuracy_true.shape[0]):
    axs.scatter([j + 1] * n_perm, accuracy_true[j, 1, :], color=(1, 0, 0, 0.3), marker='s', s=20)  
    axs.scatter([j + 1] * n_perm, accuracy_false[j, 1, :], color=(0, 0, 1, 0.3), marker='s', s=20)
    mean_acc_pas_I[j] = np.mean(accuracy_true[j, 1, :])
    std_acc_pas_I[j] = np.std(accuracy_true[j, 1, :])
mean_acc_pas_I_reduced = np.delete(mean_acc_pas_I,[19,21])
mean_1 = np.mean(mean_acc_pas_I_reduced)
std_1 = np.std(mean_acc_pas_I_reduced)    
percentile_95 = np.percentile(accuracy_false[:, 1:2, :], 95, axis=2).flatten()  
axs.plot(range(1, accuracy_false.shape[0] + 1), percentile_95, color='black', linewidth=4, label='95% Percentile')  
result_global_2 = np.zeros(30, dtype=int) 
for subject in range(30):
    t_stat, p_value = ttest_rel(
        accuracy_true[subject, 1, :],
        np.full_like(accuracy_true[subject, 1, :], percentile_95[subject,]),
        alternative='greater'  
    )
    if p_value < alpha_p:  
        result_global_2[subject] = 1

axs.set_title(data_types[0], fontsize=24, fontweight='bold')
axs.set_xlabel('Participants', fontsize=20)
axs.set_ylabel('Accuracy', fontsize=20)
axs.set_xticks([10, 20, 30])  
axs.set_xticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                        if tick != 0 else "0" for tick in [10, 20, 30]],fontsize=16, fontweight='bold')  
axs.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])  
axs.set_yticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                        if tick != 0 else "0" for tick in axs.get_yticks()],fontsize=16, fontweight='bold')
axs.set_ylim(0, 1)  
axs.tick_params(axis='both', labelsize=15)  
plt.show()
# %%

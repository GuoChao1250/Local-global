#%%
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests
from scipy.stats import spearmanr
#%%
save_file_path = f'.\\SVM\\accuracy'
data_passive = np.load(os.path.join(save_file_path, 'acc_perm_Global_400ms_SVM_compare_diff_chanAndepoch_1.npz'))
accuracy_true1 = data_passive['accuracy_true']
accuracy_false1 = data_passive['accuracy_false']
data_passive = np.load(os.path.join(save_file_path, 'acc_perm_Global_400ms_SVM_compare_diff_chanAndepoch_2.npz'))
accuracy_true2 = data_passive['accuracy_true']
accuracy_false2 = data_passive['accuracy_false']
combined_true = np.concatenate((accuracy_true1, accuracy_true2), axis=2)

mean_acc = np.mean(combined_true,axis=0)
all_acc = np.mean(mean_acc, axis=-1) # datatype*percentage*channels
datatype = ['Global_1','Global_2']
datatype_idx = 0
trial_percentages = [0.25, 0.50, 0.75, 1.0]
chan_mix = [8,16,32,64]

data = all_acc[datatype_idx, :, :]  # 4 percentages × 4 channels
sensors = np.tile([8, 16, 32, 64], 4)        # channels[8,16,32,64,8,16...]
epochs = np.repeat([0.25, 0.50, 0.75, 1.0], 4)       # epochs [0.25,0.25,0.25,0.25, 0.5,0.5,...]
performance = data.flatten()                  # all acc (16 points)

# --- Function: Use self-service method to calculate the confidence interval of rho ---
def bootstrap_ci(x, y, n_boot=1000):
    rhos = []
    n = len(x)
    for _ in range(n_boot):
        # Resampling (with replacement)
        indices = np.random.choice(n, n, replace=True)
        x_boot = x[indices]
        y_boot = y[indices]
        rho, _ = spearmanr(x_boot, y_boot)
        rhos.append(rho)
    ci_low = np.percentile(rhos, 2.5)
    ci_high = np.percentile(rhos, 97.5)
    return ci_low, ci_high

# --- 1. Number of sensors vs. performance ---
rho_sensor, p_sensor = spearmanr(sensors, performance)
ci_sensor_low, ci_sensor_high = bootstrap_ci(sensors, performance)
# --- 2. Epoch ratio vs. performance ---
rho_epoch, p_epoch = spearmanr(epochs, performance)
ci_epoch_low, ci_epoch_high = bootstrap_ci(epochs, performance)

print("=== Global Analysis ===")
print(f"Sensors vs. Performance:")
print(f"  rho = {rho_sensor:.3f}, 95% CI = [{ci_sensor_low:.3f}, {ci_sensor_high:.3f}], p = {p_sensor:.4f}")
print(f"Epochs vs. Performance:")
print(f"  rho = {rho_epoch:.3f}, 95% CI = [{ci_epoch_low:.3f}, {ci_epoch_high:.3f}], p = {p_epoch:.4f}")


plt.figure(figsize=(16, 7))

# 1：sensors vs. performance
plt.subplot(1, 2, 1)
plt.scatter(sensors, performance, alpha=0.6, color='blue')
# Add relevant curves
slope_sensor, intercept_sensor = np.polyfit(sensors, performance, 1)
x_values_sensor = np.linspace(min(sensors), max(sensors), 100)
y_values_sensor = slope_sensor * x_values_sensor + intercept_sensor
plt.plot(x_values_sensor, y_values_sensor, color='blue', linestyle='--', linewidth=3)
plt.title(f"Sensors vs. Performance", fontsize=18)
plt.xlabel("Number of Sensors", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
plt.xticks([8, 16, 32, 64], fontsize=14)
plt.yticks(np.arange(0.56, 0.72, 0.02), fontsize=14)
plt.ylim(0.56, 0.72)
# 2：Epochs vs. performance
plt.subplot(1, 2, 2)
plt.scatter(epochs, performance, alpha=0.6, color='red')
# Add relevant curves
slope_epoch, intercept_epoch = np.polyfit(epochs, performance, 1)
x_values_epoch = np.linspace(min(epochs), max(epochs), 100)
y_values_epoch = slope_epoch * x_values_epoch + intercept_epoch
plt.plot(x_values_epoch, y_values_epoch, color='red', linestyle='--', linewidth=3)
plt.title(f"Epochs vs. Performance", fontsize=18)
plt.xlabel("Percentage of Epochs", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
plt.xticks([0.25, 0.50, 0.75, 1.0], ['25%', '50%', '75%', '100%'], fontsize=14)
plt.yticks(np.arange(0.56, 0.72, 0.02), fontsize=14)
plt.ylim(0.56, 0.72)
# plt.show()
# Save the figure
save_path = (
    f"./Fig/individual/"
    f"Accuracy_corr.png"
)
plt.savefig(save_path, bbox_inches='tight')

# 32- and 64-channel data 
acc_32 = all_acc[:, :, 2]  # Global I/II × percentage (32)
acc_64 = all_acc[:, :, 3]  # Global I/II × percentage (64)

mean_32, std_32 = np.mean(acc_32), np.std(acc_32)
mean_64, std_64 = np.mean(acc_64), np.std(acc_64)

print(f"32导 | Mean = {mean_32:.4f} ± {std_32:.4f}")
print(f"64导 | Mean = {mean_64:.4f} ± {std_64:.4f}")

# 按Global I/II分别计算
for i, name in enumerate(['Global I', 'Global II']):
    m32, s32 = np.mean(acc_32[i]), np.std(acc_32[i])
    m64, s64 = np.mean(acc_64[i]), np.std(acc_64[i])
    print(f"{name}:\n  32导 = {m32:.4f} ± {s32:.4f}\n  64导 = {m64:.4f} ± {s64:.4f}\n")

# %%
# 数据加载部分保持不变
save_file_path = f'.\\SVM\\accuracy'
data_passive = np.load(os.path.join(save_file_path, 'acc_perm_Global_400ms_SVM_compare_diff_chanAndepoch_1.npz'))
accuracy_true1 = data_passive['accuracy_true']
accuracy_false1 = data_passive['accuracy_false']
data_passive = np.load(os.path.join(save_file_path, 'acc_perm_Global_400ms_SVM_compare_diff_chanAndepoch_2.npz'))
accuracy_true2 = data_passive['accuracy_true']
accuracy_false2 = data_passive['accuracy_false']
combined_true = np.concatenate((accuracy_true1, accuracy_true2), axis=2)

mean_acc = np.mean(combined_true, axis=0)
std_acc = np.std(combined_true, axis=0)
all_acc = np.mean(mean_acc, axis=-1)  # datatype*percentage*channels
all_std = np.mean(std_acc, axis=-1)   # std

datatype = ['Global_1','Global_2']
trial_percentages = [0.25, 0.50, 0.75, 1.0]
chan_mix = [8, 16, 32, 64]

# color
blue_shades = ['#ADD8E6', '#87CEFA', '#1E90FF', '#00008B']  # light to dark blue
red_shades = ['#FFB6C1', '#FF6347', '#DC143C', '#8B0000']   # light to dark red

plt.figure(figsize=(16, 7), dpi=600)
# plt.suptitle('Classification Accuracy with Error Bars (±1 SD)', fontsize=20, y=1.02)

# ========== Global I ==========
plt.subplot(1, 2, 1)
for chan_idx in range(4):
    # errorbar
    plt.errorbar(trial_percentages, 
                all_acc[0, :, chan_idx],
                yerr=all_std[0, :, chan_idx],
                color=blue_shades[chan_idx],
                linestyle='-',
                linewidth=2,
                marker='o',
                markersize=8,
                capsize=5,
                capthick=2,
                label=f'{chan_mix[chan_idx]} sens.')

plt.xlabel('Percentage of Epochs', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.title('Global I', fontsize=18)
plt.xticks(trial_percentages, ['25%', '50%', '75%', '100%'], fontsize=14)
plt.yticks(np.arange(0.4, 0.88, 0.1), fontsize=14)
plt.ylim(0.4, 0.88)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(ncol=1, fontsize=20, frameon=False)
# ========== Global II ==========
plt.subplot(1, 2, 2)
for chan_idx in range(4):
    plt.errorbar(trial_percentages, 
                all_acc[1, :, chan_idx],
                yerr=all_std[1, :, chan_idx],
                color=red_shades[chan_idx],
                linestyle='-',
                linewidth=2,
                marker='o',
                markersize=8,
                capsize=5,
                capthick=2,
                label=f'{chan_mix[chan_idx]} sens.')

plt.xlabel('Percentage of Epochs', fontsize=18)
plt.title('Global II', fontsize=18)
plt.xticks(trial_percentages, ['25%', '50%', '75%', '100%'], fontsize=14)
plt.yticks(np.arange(0.4, 0.88, 0.1), fontsize=14)
plt.ylim(0.4, 0.88)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(ncol=1, fontsize=20, frameon=False)
# plt.show()

# Adjust layout
# plt.tight_layout()

# Save the figure
save_path = (
    f"./Fig/individual/"
    f"Accuracy_Across_Different_Conditions_500.png"
)
plt.savefig(save_path, bbox_inches='tight')
#%%
alpha_p = 0.001/30

# 文件路径
save_file_path = './EEGNET/accuracy'

# 加载 passive 数据
data_passive = np.load(os.path.join(save_file_path, 'acc_perm_Global_400ms_SVM_block_pas_50_special.npz'))
accuracy_true_passive = data_passive['accuracy_true']
accuracy_false_passive = data_passive['accuracy_false']

# 加载 active 数据
data_active = np.load(os.path.join(save_file_path, 'acc_perm_Global_400ms_SVM_block_act_50_special.npz'))
accuracy_true_active = data_active['accuracy_true']
accuracy_false_active = data_active['accuracy_false']

n_perm_passive = accuracy_true_passive.shape[2]
n_perm_active = accuracy_true_active.shape[2]

fig, axs = plt.subplots(figsize=(12, 4), tight_layout=True)

# passive
for j in range(accuracy_true_passive.shape[0]):
    axs.scatter(-accuracy_true_passive[j, 0, :], [j + 1] * n_perm_passive, color=(1, 0, 0, 0.3), marker='s', s=20)  # 小红色方框
    axs.scatter(-accuracy_false_passive[j, 0, :], [j + 1] * n_perm_passive, color=(0, 0, 1, 0.3), marker='s', s=20)

#  passive  95% points
percentile_95_passive = np.percentile(accuracy_false_passive[:, 0:1, :], 95, axis=2).flatten()
axs.plot(-percentile_95_passive, range(1, accuracy_false_passive.shape[0] + 1), color='black', linewidth=4, label='Passive 95% Percentile')

# initialization
result_pas = np.zeros(30, dtype=int)

# Unilateral paired sample t-test
for subject in range(30):
    t_stat, p_value = ttest_rel(
        accuracy_true_passive[subject, 0, :],
        np.full_like(accuracy_true_passive[subject, 1, :], percentile_95_passive[subject,]),
        alternative='greater'  
    )
    if p_value < alpha_p:  
        result_pas[subject] = 1

#  active
for j in range(accuracy_true_active.shape[0]):
    axs.scatter(accuracy_true_active[j, 0, :], [j + 1 ] * n_perm_active, color=(1, 0, 0, 0.3), marker='s', s=20)  # 小红色方框
    axs.scatter(accuracy_false_active[j, 0, :], [j + 1 ] * n_perm_active, color=(0, 0, 1, 0.3), marker='s', s=20)

# active 95% points
percentile_95_active = np.percentile(accuracy_false_active[:, 0:1, :], 95, axis=2).flatten()
axs.plot(percentile_95_active, range(1, accuracy_false_active.shape[0] + 1), color='black', linewidth=4, label='Active 95% Percentile')

result_act = np.zeros(30, dtype=int)  

# Unilateral paired sample t-test
for subject in range(30):
    t_stat, p_value = ttest_rel(
        accuracy_true_active[subject, 0, :],
        np.full_like(accuracy_true_active[subject, 1, :], percentile_95_active[subject,]),
        alternative='greater'  
    )
    if p_value < alpha_p:  # 显著性水平为 0.05
        result_act[subject] = 1

# axs.set_title('Accuracy by Participants', fontsize=24, fontweight='bold')
axs.set_xlabel('Accuracy', fontsize=20)
axs.set_ylabel('Participants', fontsize=20)
axs.set_xlim(-1, 1)
axs.set_ylim(0, 30)
axs.set_xticks([-1, -0.5, 0, 0.5, 1])
axs.set_xticklabels([f"{tick:.1f}" for tick in [1, 0.5, 0, 0.5, 1]], fontsize=16, fontweight='bold')
axs.set_yticks([10, 20, 30]) 
axs.set_yticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                        if tick != 0 else "0" for tick in [10, 20, 30]],fontsize=16, fontweight='bold')  
axs.axvline(x=0, color='gray', linestyle='--', linewidth=2)

save_path = (
f"./plot_Fig/individual/"
f"Global_I_combine_400ms_length.png"
)      
plt.savefig(save_path, bbox_inches='tight')
#%%

fig, axs = plt.subplots(figsize=(12, 4), tight_layout=True)
# passive 
for j in range(accuracy_true_passive.shape[0]):
    axs.scatter(-accuracy_true_passive[j, 1, :], [j + 1] * n_perm_passive, color=(1, 0, 0, 0.3), marker='s', s=20)  # 小红色方框
    axs.scatter(-accuracy_false_passive[j, 1, :], [j + 1] * n_perm_passive, color=(0, 0, 1, 0.3), marker='s', s=20)

# passive 95% 
percentile_95_passive = np.percentile(accuracy_false_passive[:, 1:2, :], 95, axis=2).flatten()
axs.plot(-percentile_95_passive, range(1, accuracy_false_passive.shape[0] + 1), color='black', linewidth=4, label='Passive 95% Percentile')

# result_pas = np.zeros(30, dtype=int)  # 存储每个被试的结果，0 或 1
# for subject in range(30):
#     t_stat, p_value = ttest_rel(
#         accuracy_true_passive[subject, 1, :],
#         np.full_like(accuracy_true_passive[subject, 1, :], percentile_95_passive[subject,]),
#         alternative='greater'  
#     )
#     if p_value < 0.05:  
#         result_pas[subject] = 1

# active 
for j in range(accuracy_true_active.shape[0]):
    axs.scatter(accuracy_true_active[j, 1, :], [j + 1 ] * n_perm_active, color=(1, 0, 0, 0.3), marker='s', s=20)  # 小红色方框
    axs.scatter(accuracy_false_active[j, 1, :], [j + 1 ] * n_perm_active, color=(0, 0, 1, 0.3), marker='s', s=20)

#  active  95% 
percentile_95_active = np.percentile(accuracy_false_active[:, 1:2, :], 95, axis=2).flatten()
axs.plot(percentile_95_active, range(1, accuracy_false_active.shape[0] + 1), color='black', linewidth=4, label='Active 95% Percentile')
# result_act = np.zeros(30, dtype=int)  
# for subject in range(30):
#     t_stat, p_value = ttest_rel(
#         accuracy_true_active[subject, 1, :],
#         np.full_like(accuracy_true_active[subject, 1, :], percentile_95_active[subject,]),
#         alternative='greater'  
#     )
#     if p_value < 0.05: 
#         result_act[subject] = 1

# axs.set_title('Accuracy by Participants', fontsize=24, fontweight='bold')
axs.set_xlabel('Accuracy', fontsize=20)
axs.set_ylabel('Participants', fontsize=20)
axs.set_xlim(-1, 1)
axs.set_ylim(0, 30)
axs.set_xticks([-1, -0.5, 0, 0.5, 1])
axs.set_xticklabels([f"{tick:.1f}" for tick in [1, 0.5, 0, 0.5, 1]], fontsize=16, fontweight='bold')
axs.set_yticks([10, 20, 30]) 
axs.set_yticklabels([f"{tick:.1f}".rstrip("0").rstrip(".") 
                        if tick != 0 else "0" for tick in [10, 20, 30]],fontsize=16, fontweight='bold')  
axs.axvline(x=0, color='gray', linestyle='--', linewidth=2)

save_path = (
f"./plot_Fig/individual/"
f"Global_II_combine_400ms_length.png"
)      
plt.savefig(save_path, bbox_inches='tight')
import torch
import numpy as np

# 加载保存的结果
result_path = 'DeepSurv/result/result_BLCA_5_FOLD.pt'
checkpoint = torch.load(result_path)

fold_results = checkpoint['fold_c_indices']  # C-index 的折结果
fold_auc_values = checkpoint['fold_auc_values']  # AUC 的折结果

# 计算 C-index 和 AUC 的均值和 IQR
c_index_mean = np.mean(fold_results)
c_index_std = np.std(fold_results)
c_index_iqr = np.percentile(fold_results, 75) - np.percentile(fold_results, 25)

auc_mean = np.mean(fold_auc_values)
auc_std = np.std(fold_auc_values)
auc_iqr = np.percentile(fold_auc_values, 75) - np.percentile(fold_auc_values, 25)

# 输出均值、标准差和 IQR
print(f"C-index: {c_index_mean:.4f} ± {c_index_std:.4f} (Std), IQR: {c_index_iqr:.4f}")
print(f"AUC: {auc_mean:.4f} ± {auc_std:.4f} (Std), IQR: {auc_iqr:.4f}")

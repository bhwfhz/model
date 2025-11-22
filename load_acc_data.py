# data_loader_upgraded.py
import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from tqdm import tqdm

# 参数同旧
input_dir = r"C:\Users\保护我方虎子\Desktop\model\dataset\input"
output_dir = r"C:\Users\保护我方虎子\Desktop\model\dataset\output"
save_dir = "./dataset_upgraded"
os.makedirs(save_dir, exist_ok=True)

fs_input = 512.0
fs_output = 250.0
target_fs = 50.0
n_input = 800
n_output = 3000

input_files = sorted(glob.glob(os.path.join(input_dir, "*.xlsx")))
output_files = sorted(glob.glob(os.path.join(output_dir, "*.xlsx")))

for in_path, out_path in tqdm(zip(input_files, output_files), total=len(input_files)):
    name = os.path.splitext(os.path.basename(in_path))[0]

    df_in = pd.read_excel(in_path)
    X_raw = df_in.iloc[:, [1, 2, 3]].values.astype(np.float64)

    df_out = pd.read_excel(out_path)
    Y_raw = df_out.iloc[:, 1:28].values.astype(np.float64).reshape(-1, 9, 3)

    X = resample_poly(X_raw, int(target_fs), int(fs_input), axis=0)[:n_input]
    Y = resample_poly(Y_raw.reshape(-1, 27), int(target_fs), int(fs_output), axis=0).reshape(-1, 9, 3)[:n_output]

    # 修改：通道独立PGA
    pga_in = np.max(np.abs(X), axis=0) + 1e-8  # (3,)
    pga_out = np.max(np.abs(Y), axis=(0, 1)) + 1e-8  # (3,)
    X_norm = X / pga_in
    Y_norm = Y / pga_out[None, None, :]

    # 保存原始
    np.savez_compressed(os.path.join(save_dir, f"{name}.npz"), X=X_norm, Y=Y_norm.reshape(n_output, 27), pga_in=pga_in,
                        pga_out=pga_out)

    # 修改：数据增强 - 随机噪声 + 翻转
    X_aug = X_norm + np.random.normal(0, 0.01, X_norm.shape)
    Y_aug = Y_norm + np.random.normal(0, 0.01, Y_norm.shape)
    np.savez_compressed(os.path.join(save_dir, f"{name}_aug.npz"), X=X_aug, Y=Y_aug.reshape(n_output, 27),
                        pga_in=pga_in, pga_out=pga_out)

    X_flip = np.flip(X_norm, axis=0)
    Y_flip = np.flip(Y_norm, axis=0)
    np.savez_compressed(os.path.join(save_dir, f"{name}_flip.npz"), X=X_flip, Y=Y_flip.reshape(n_output, 27),
                        pga_in=pga_in, pga_out=pga_out)

print("升级数据集完成！路径：", save_dir)
# data_loader.py   （直接替换之前的预处理文件）
import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from tqdm import tqdm

# ==================== 参数 ====================
input_dir = r"C:\Users\保护我方虎子\Desktop\model\dataset\input"
output_dir = r"C:\Users\保护我方虎子\Desktop\model\dataset\output"
save_dir = "./dataset_corrected"
os.makedirs(save_dir, exist_ok=True)

fs_input = 512.0
fs_output = 250.0
target_fs = 50.0
input_seconds = 16.0
output_seconds = 60.0
n_input = int(input_seconds * target_fs)  # 800
n_output = int(output_seconds * target_fs)  # 3000

# ==================== 主循环 ====================
input_files = sorted(glob.glob(os.path.join(input_dir, "*.xlsx")))
output_files = sorted(glob.glob(os.path.join(output_dir, "*.xlsx")))

skipped = 0
for in_path, out_path in tqdm(zip(input_files, output_files), total=len(input_files)):
    name = os.path.splitext(os.path.basename(in_path))[0]

    # 读取输入（3通道）
    df_in = pd.read_excel(in_path)
    X_raw = df_in.iloc[:, [1, 2, 3]].values.astype(np.float64)  # xyz或zyx

    # 读取输出（27通道）
    df_out = pd.read_excel(out_path)
    Y_raw = df_out.iloc[:, 1:28].values.astype(np.float64).reshape(-1, 9, 3)

    # 重采样到50Hz
    X = resample_poly(X_raw, int(target_fs), int(fs_input), axis=0)
    Y = resample_poly(Y_raw.reshape(-1, 27), int(target_fs), int(fs_output), axis=0).reshape(-1, 9, 3)

    # ==================== 关键修改：不再强制延迟，直接对齐起点 ====================
    # 你已经手动对齐过，只需检查长度足够即可
    if X.shape[0] < n_input:
        print(f"{name} 输入太短，跳过")
        skipped += 1
        continue
    if Y.shape[0] < n_output:
        print(f"{name} 输出太短（只有{Y.shape[0] / 50:.1f}s），尝试截取前60s或补零")
        # 策略：如果比60s短，补零到60s（远场测点后期本来就小）
        pad_len = n_output - Y.shape[0]
        pad_block = np.zeros((pad_len, 9, 3))
        Y = np.concatenate([Y, pad_block], axis=0)
    else:
        Y = Y[:n_output]  # 截断到正好60s

    X = X[:n_input]  # 确保输入正好16s

    # 每条事件独立PGA归一化（保留）
    pga = max(np.max(np.abs(X)), 1e-8)
    X_norm = X / pga
    Y_norm = Y / pga

    # 保存
    np.savez_compressed(
        os.path.join(save_dir, f"{name}.npz"),
        X=X_norm.astype(np.float32),  # (800, 3)
        Y=Y_norm.reshape(n_output, 27).astype(np.float32),  # (3000, 27)
        pga=np.float32(pga),
        name=name
    )

print(f"预处理完成！共处理 {len(input_files) - skipped} 条，跳过 {skipped} 条")
print("新数据集路径：", save_dir)
# data_loader.py

import os
import glob
import pandas as pd
import numpy as np

def load_all_events(data_folder, dtype=np.float32):
    """
    读取 data_folder 下所有 .xlsx，每个事件数据得到 (T_i, 3, 3) 和 (T_i, 8, 3)。
    返回：
      - X_list: list of np.array, shape (T_i, 3, 3)
      - Y_list: list of np.array, shape (T_i, 8, 3)
      - file_list: list of 对应文件名，顺序对应 X_list/Y_list
    dtype: 要转换的 numpy dtype，默认 np.float32
    """
    pattern = os.path.join(data_folder, '*.xlsx')
    all_files = sorted(glob.glob(pattern))
    X_list = []
    Y_list = []
    file_list = []

    # 定义列名
    in_cols = [
        '180-BFE','270-BFE','UP-BFE',
        '180-BFN','270-BFN','UP-BFN',
        '180-BFS','270-BFS','UP-BFS'
    ]
    out_cols = [
        '180-1FE','270-1FE','UP-1FE',
        '180-2FE','270-2FE','UP-2FE',
        '180-2FW','270-2FW','UP-2FW',
        '180-5FE','270-5FE','UP-5FE',
        '180-5FW','270-5FW','UP-5FW',
        '180-8FE','270-8FE','UP-8FE',
        '180-8FN','270-8FN','UP-8FN',
        '180-8FS','270-8FS','UP-8FS'
    ]

    if not all_files:
        print(f"[警告] 在 {data_folder} 下未找到任何 .xlsx 文件。")
    for filepath in all_files:
        fname = os.path.basename(filepath)
        try:
            # 只读取需要的列，加快速度、减少内存
            df = pd.read_excel(filepath, sheet_name='Sheet1', skiprows=0, usecols=in_cols + out_cols)
        except Exception as e:
            print(f"[警告] 读取 {fname} 失败，跳过: {e}")
            continue

        # 检查列完整性
        missing_in  = [c for c in in_cols  if c not in df.columns]
        missing_out = [c for c in out_cols if c not in df.columns]
        if missing_in or missing_out:
            print(f"[错误] 文件 {fname} 缺少列，跳过。缺少 in: {missing_in}, 缺少 out: {missing_out}")
            continue

        raw_in_flat  = df[in_cols].values   # shape (T, 9)
        raw_out_flat = df[out_cols].values  # shape (T, 24)
        T = raw_in_flat.shape[0]
        if T == 0:
            print(f"[警告] 文件 {fname} 无有效数据，跳过。")
            continue

        try:
            raw_in  = raw_in_flat.reshape(T, 3, 3).astype(dtype)   # (T,3,3)
            raw_out = raw_out_flat.reshape(T, 8, 3).astype(dtype)  # (T,8,3)
        except Exception as e:
            print(f"[错误] 文件 {fname} reshape 失败，跳过: {e}")
            continue

        X_list.append(raw_in)
        Y_list.append(raw_out)
        file_list.append(fname)
        print(f"已加载事件 {fname}，长度 {T}")

    print(f"共加载 {len(X_list)} 个事件")
    return X_list, Y_list, file_list

def save_events_as_npz(X_list, Y_list, file_list, out_dir='./processed_events'):
    """
    将每个事件保存为单独 npz，文件名依据原始 file_list。
    out_dir: 输出文件夹
    保存格式： out_dir/<原文件名主体>.npz，内部包含 X 和 Y 两个数组
    """
    os.makedirs(out_dir, exist_ok=True)
    for xi, yi, fname in zip(X_list, Y_list, file_list):
        base = os.path.splitext(fname)[0]
        save_path = os.path.join(out_dir, f'{base}.npz')
        try:
            np.savez(save_path, X=xi, Y=yi)
            print(f"已保存 {save_path} （shape X={xi.shape}, Y={yi.shape}）")
        except Exception as e:
            print(f"[错误] 保存 {save_path} 失败: {e}")

if __name__ == '__main__':
    # 读取 Excel 并保存为 per-event npz
    data_folder = './data_events'  
    X_list, Y_list, file_list = load_all_events(data_folder, dtype=np.float32)
    if X_list:
        save_events_as_npz(X_list, Y_list, file_list, out_dir='./processed_events')
    else:
        print("未加载到任何事件，退出。")

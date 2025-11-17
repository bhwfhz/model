# eval_test.py (modified)
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# If you prefer, use keras.models.load_model when loading a full model; here we build architecture and load weights.
from model import build_hybrid_conv_lstm_lstm_se

# Optional: h5py used for diagnostics if weights fail to load
try:
    import h5py
except Exception:
    h5py = None

# 模型输入输出参数
window_size = 18
stride = 1
in_H, in_W, in_C = 3, 1, 3
out_H, out_W, out_C = 8, 1, 3

# ----------------------------
# 一、数据预处理
# ----------------------------
fs_orig = 100.0
lowcut, highcut = 0.2, 4.0
b, a = butter(4, [lowcut / (fs_orig / 2), highcut / (fs_orig / 2)], btype='band')

fs_new = 20.0
decim = int(fs_orig / fs_new)

# 加载训练时保存的标准化参数（脚本会在 hyperopt 训练时保存到 hyperopt_results/scalers.npz）
if not os.path.exists('scalers.npz'):
    raise FileNotFoundError("找不到 scalers.npz。请确认训练脚本已生成并保存 scalers.npz（通常在 train.py 中）。")

scalers = np.load('scalers.npz')
mean_X = scalers['mean_X']  # shape (1,1,3,1,3)
std_X = scalers['std_X']
mean_Y = scalers['mean_Y']  # shape (1,1,8,1,3)
std_Y = scalers['std_Y']

# 对 Y 的 mean/std 进行 squeeze，以匹配模型输出 shape (N,8,3)
mean_Y = np.squeeze(mean_Y, axis=(0, 1, 3))  # 结果形状 (8,3)
std_Y = np.squeeze(std_Y, axis=(0, 1, 3))  # 结果形状 (8,3)

# 与训练时保持一致的列名
in_cols = [
    '180-BFE', '270-BFE', 'UP-BFE',
    '180-BFN', '270-BFN', 'UP-BFN',
    '180-BFS', '270-BFS', 'UP-BFS'
]

out_cols = [
    '180-1FE', '270-1FE', 'UP-1FE',
    '180-2FE', '270-2FE', 'UP-2FE',
    '180-2FW', '270-2FW', 'UP-2FW',
    '180-5FE', '270-5FE', 'UP-5FE',
    '180-5FW', '270-5FW', 'UP-5FW',
    '180-8FE', '270-8FE', 'UP-8FE',
    '180-8FN', '270-8FN', 'UP-8FN',
    '180-8FS', '270-8FS', 'UP-8FS'
]

# ----------------------------
# Helper: load best_params saved by hyperopt_search.py
# ----------------------------
def load_best_params(search_dir="./hyperopt_results"):
    """
    尝试从多个可能的位置载入 best_params。
    支持： pickle (.pkl) 或 json 文件。
    返回字典或 None。
    """
    candidates = [
        os.path.join(search_dir, "best_params.pkl"),
        os.path.join(search_dir, "best_params.json"),
        os.path.join(search_dir, "best_raw.pkl"),
        os.path.join(search_dir, "best_params.pickle"),
        os.path.join(search_dir, "search_summary.pkl"),
        "best_params.pkl",
        "best_params.json",
    ]
    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            if p.endswith(".json"):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # if file contains wrapped summary
                if isinstance(data, dict) and 'best_hyperopt_mapped' in data:
                    return data['best_hyperopt_mapped']
                return data
            else:
                with open(p, "rb") as f:
                    data = pickle.load(f)
                # if it's a search_summary
                if isinstance(data, dict) and 'best_hyperopt_mapped' in data:
                    return data['best_hyperopt_mapped']
                # sometimes best_raw.pkl contains raw hyperopt encoding -> try to return as-is
                return data
        except Exception as e:
            print(f"Warning: failed to load {p}: {e}")
            continue
    return None

def normalize_params(raw):
    """
    将 raw params 转换为模型构建所需的标准字段并做类型转换。
    支持不同命名约定（conv_kh/conv_kw 或 conv_kernel）等。
    """
    if raw is None:
        return None

    # If raw is hyperopt 'best_raw' (indexes), it's difficult to map here.
    # We expect a dict mapping parameter names to numeric values.
    params = {}

    # helper to extract key ignoring case
    def get_key(k1, k2=None, default=None):
        if k1 in raw:
            return raw[k1]
        if k2 and k2 in raw:
            return raw[k2]
        # search case-insensitive
        for kk in raw.keys():
            if kk.lower() == k1.lower():
                return raw[kk]
            if k2 and kk.lower() == k2.lower():
                return raw[kk]
        return default

    # conv_filters
    cf = get_key('conv_filters', 'conv_filter', None)
    if cf is not None:
        params['conv_filters'] = int(cf)
    # conv_kernel: either two separate keys or a tuple/list under 'conv_kernel'
    ck = get_key('conv_kernel', None, None)
    if ck is not None:
        # ck may be "(9, 5)" string or list/tuple
        if isinstance(ck, str):
            try:
                ck_eval = eval(ck)
                ck = ck_eval
            except Exception:
                pass
        if isinstance(ck, (list, tuple)) and len(ck) >= 2:
            params['conv_kh'] = int(ck[0])
            params['conv_kw'] = int(ck[1])
    else:
        kh = get_key('conv_kh', 'conv_k', None)
        kw = get_key('conv_kw', 'conv_w', None)
        if kh is not None and kw is not None:
            params['conv_kh'] = int(kh)
            params['conv_kw'] = int(kw)

    # lstm_units
    lu = get_key('lstm_units', 'lstm_unit', None)
    if lu is not None:
        params['lstm_units'] = int(lu)

    # dropout, l2_reg, lr, batch_size
    dr = get_key('dropout', None, 0.3)
    params['dropout'] = float(dr)
    l2 = get_key('l2_reg', 'l2', 3e-4)
    params['l2_reg'] = float(l2)
    lr = get_key('lr', None, None)
    if lr is not None:
        params['lr'] = float(lr)
    bs = get_key('batch_size', 'batch', 16)
    params['batch_size'] = int(bs)

    return params

def inspect_weights_file_shapes(weights_path):
    """
    If h5py available, print weight dataset names and shapes inside the weights file.
    This helps debug shape mismatch.
    """
    if h5py is None:
        print("h5py is not available; cannot inspect weights file shapes.")
        return
    if not os.path.exists(weights_path):
        print(f"Weights file {weights_path} not found.")
        return
    print(f"\nInspecting weight file: {weights_path}")
    try:
        with h5py.File(weights_path, 'r') as f:
            # Recursively visit datasets
            def print_item(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"{name}: shape={obj.shape}")
            f.visititems(print_item)
    except Exception as e:
        print("Failed to inspect weights file:", e)

# ----------------------------
# process_test_file (unchanged except uses global mean/std)
# ----------------------------
def process_test_file(file_path, model):
    fname = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\nProcessing test file: {fname}")

    # 1. 读表 & 检查列名
    df = pd.read_excel(file_path, sheet_name='Sheet1', skiprows=0)
    missing_in = [c for c in in_cols if c not in df.columns]
    missing_out = [c for c in out_cols if c not in df.columns]

    if missing_in or missing_out:
        raise KeyError(f"{fname} 列缺失, in:{missing_in}, out:{missing_out}")

    # 2. reshape 原始数据
    data_in = df[in_cols].values.reshape(-1, 3, 3).astype(np.float32)
    data_out = df[out_cols].values.reshape(-1, 8, 3).astype(np.float32)

    # 3. 滤波 + 降采样
    data_in = filtfilt(b, a, data_in, axis=0)[::decim]
    data_out = filtfilt(b, a, data_out, axis=0)[::decim]

    T = data_in.shape[0]
    if T < window_size:
        raise ValueError(f"{fname} 降采样后长度 {T} < window_size")

    # 4. 滑窗
    X_list, Y_list = [], []
    for i in range(0, T - window_size + 1, stride):
        X_list.append(data_in[i:i + window_size])
        Y_list.append(data_out[i])

    X_arr = np.stack(X_list, axis=0)  # (N,18,3,3)
    Y_true_arr = np.stack(Y_list, axis=0)  # (N,8,3)
    N = X_arr.shape[0]
    print(f" Total windows: {N}")

    # 5. 标准化输入
    X_reshaped = X_arr.reshape(N, window_size, in_H, in_W, in_C).astype(np.float32)  # (N,18,3,1,3)

    # 读取并应用 axis_scale（如果 scalers.npz 中存在）
    axis_scale = None
    if 'axis_scale' in scalers:
        axis_scale = scalers['axis_scale']
        # 兼容不同存储形式（比如 list/ndarray）
        axis_scale = np.array(axis_scale, dtype=np.float32).reshape(1, 1, 1, 1, -1)  # (1,1,1,1,3) 可广播
        X_scaled = X_reshaped * axis_scale
    else:
        # 如果没有 axis_scale，我们打印警告并继续（但结果可能与训练不一致）
        print("Warning: 'axis_scale' not found in scalers.npz. Test input will NOT be channel-scaled.")
        X_scaled = X_reshaped

    # 最后用训练期计算的 mean_X/std_X 做标准化（mean_X/std_X 都是基于缩放后数据计算得到的）
    # 注意 mean_X/std_X 的 shape 应为 (1,1,in_H,1,in_C) 与 X_scaled shape 对齐
    X_std = (X_scaled - mean_X) / std_X

    # 6. 预测 & 反标准化输出
    Y_pred_std = model.predict(X_std, batch_size=16)
    Y_pred_std = Y_pred_std.reshape(N, 8, 3)
    Y_pred_real = Y_pred_std * std_Y + mean_Y

    # 7. flatten
    Y_pred_flat = Y_pred_real.reshape(N, 24)
    Y_true_flat = Y_true_arr.reshape(N, 24)

    # 8. 计算指标
    y_true_all = Y_true_flat.flatten()
    y_pred_all = Y_pred_flat.flatten()

    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    mae = mean_absolute_error(y_true_all, y_pred_all)
    r2 = r2_score(y_true_all, y_pred_all)

    print(f" Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    # 9. 输出目录
    out_dir = os.path.join("test_results", fname)
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame(Y_pred_flat, columns=out_cols) \
        .to_excel(os.path.join(out_dir, fname + "_pred.xlsx"), index=False)

    # 10. 保存所有 24 列对比图
    time_axis = np.arange(N) / fs_new
    for idx, col in enumerate(out_cols):
        plt.figure(figsize=(6, 3))
        plt.plot(time_axis, Y_true_flat[:, idx], label='True')
        plt.plot(time_axis, Y_pred_flat[:, idx], '--', label='Pred')
        plt.title(f"{fname} - {col}")
        plt.xlabel("Time (s)")
        plt.ylabel("Accel")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{col}.png"))
        plt.close()

    # 11. 弹出最佳拟合图
    mse_cols = np.mean((Y_pred_flat - Y_true_flat) ** 2, axis=0)
    best_idx = int(np.argmin(mse_cols))
    best_col = out_cols[best_idx]

    plt.figure(figsize=(6, 3))
    plt.plot(time_axis, Y_true_flat[:, best_idx], label='True')
    plt.plot(time_axis, Y_pred_flat[:, best_idx], '--', label='Pred')
    plt.title(f"{fname} - Best: {best_col}")
    plt.xlabel("Time (s)")
    plt.ylabel("Accel")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------
# main: build model using best_params (if found), load weights and test
# ----------------------------
def main():
    # 1) 尝试加载 best_params 并映射为具体构建参数
    best_raw = load_best_params(search_dir="./hyperopt_results")
    best_params = normalize_params(best_raw)
    if best_params is None:
        print("Warning: 未找到 best_params，使用默认超参构建模型（可能导致 load_weights shape mismatch）。")
        best_params = {
            'conv_filters': 16,
            'conv_kh': 9,
            'conv_kw': 5,
            'lstm_units': 128,
            'dropout': 0.3,
            'l2_reg': 3e-4,
            'batch_size': 16
        }
    else:
        print("Loaded best_params (mapped):", best_params)

    # 2) 构建模型（使用 best_params 中的 conv_filters / conv_kh / conv_kw / lstm_units 等）
    conv_filters = int(best_params.get('conv_filters', 16))
    conv_kh = int(best_params.get('conv_kh', 9))
    conv_kw = int(best_params.get('conv_kw', 5))
    lstm_units = int(best_params.get('lstm_units', 128))
    dropout_rate = float(best_params.get('dropout', 0.3))
    l2_reg = float(best_params.get('l2_reg', 3e-4))

    print(f"Building model with conv_filters={conv_filters}, conv_kernel=({conv_kh},{conv_kw}), lstm_units={lstm_units}, dropout={dropout_rate}, l2_reg={l2_reg}")

    model = build_hybrid_conv_lstm_lstm_se(
        n_timesteps=window_size,
        in_H=in_H, in_W=in_W, in_C=in_C,
        out_H=out_H, out_W=out_W, out_C=out_C,
        conv_filters=conv_filters,
        conv_kernel=(conv_kh, conv_kw),
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        se_ratio=8
    )

    # 打印 summary，便于检查每层的参数
    model.summary()

    # 3) load weights (with diagnostics on failure)
    weights_path = "best_model.h5"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到权重文件 {weights_path}。请确保 hyperopt_search 已生成 best_model.h5。")

    try:
        model.load_weights(weights_path)
        print("Model and SE weights loaded successfully.")
    except ValueError as e:
        print("Error loading weights:", e)
        print("可能的原因：当前模型结构/超参与权重文件不匹配。")
        # 打印模型所需的 layer weight shapes（帮助诊断）
        print("\n--- Model expected weight shapes (per layer) ---")
        for layer in model.layers:
            try:
                weights = layer.get_weights()
                if not weights:
                    continue
                shapes = [w.shape for w in weights]
                print(f"{layer.name}: {shapes}")
            except Exception:
                pass

        # 试图打印权重文件内部的 shapes（需要 h5py）
        if h5py is not None:
            inspect_weights_file_shapes(weights_path)
        else:
            print("h5py not installed — cannot inspect weight file shapes. Consider installing h5py to get detailed shapes.")
        # 结束并抛出异常，避免继续运行
        raise

    # 4) 测试列表（按需修改）
    test_files = [
        r"F:\ConvLSTM - model4\data_events\test\201212071718ANX.xlsx",
        r"F:\ConvLSTM - model4\data_events\test\201612282138ANX.xlsx"
    ]

    # 5) 逐文件处理
    for fp in test_files:
        try:
            process_test_file(fp, model)
        except Exception as e:
            print(f"Error processing {fp}: {e}")

if __name__ == "__main__":
    main()
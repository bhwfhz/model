# data_loader_resample.py
import os
import glob
from fractions import Fraction
import re

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import resample_poly
from tqdm import tqdm

# ---------- 用户可修改的参数 ----------
input_dir = r"C:\Users\保护我方虎子\Desktop\model\dataset\input"
output_dir = r"C:\Users\保护我方虎子\Desktop\model\dataset\output"
out_npz_dir = r"./processed_resampled_events"
os.makedirs(out_npz_dir, exist_ok=True)

# 原始采样率
fs_input = 512.0
fs_output = 250.0
# 目标（统一）采样率，训练时可快速调整
target_fs = 50.0

# 期望时长（秒）
expected_in_seconds = 16.0
expected_out_seconds = 60.0

# 输入 Excel: 前7列有数据，第一列是时间，接下 3 列是加速度 (假设列索引 1..3 为 x,y,z)
INPUT_TIME_COL_IDX = 0
INPUT_CHANNEL_COLS = [1, 2, 3]   # 原始的 x,y,z 列索引
# 你要求把 x 与 z 互换（即把列 1 和 3 交换） -> 之后的顺序为 z, y, x
SWAP_COL_A = INPUT_CHANNEL_COLS[0]  # 1
SWAP_COL_B = INPUT_CHANNEL_COLS[2]  # 3

# 输出 Excel: 前28列有数据，第一列时间，接下来的 27 列为 9 个测点每点3通道（顺序假定是 pt1(x,y,z), pt2(x,y,z), ...）
OUTPUT_TIME_COL_IDX = 0
OUTPUT_CHANNEL_COLS = list(range(1, 28))  # 1..27

# ---------- 辅助函数 ----------
def list_xlsx(folder):
    """返回 folder 下按 name 排序的 .xlsx 文件列表，过滤临时文件。"""
    pattern = os.path.join(folder, "*.xlsx")
    files = [f for f in glob.glob(pattern) if os.path.isfile(f)]
    # 过滤掉 Excel 临时文件 ~$ 开头
    files = [f for f in files if not os.path.basename(f).startswith("~$")]
    # 排序
    files.sort()
    return files

def read_and_proc_input(fp):
    """
    读取输入 Excel，交换第2列和第4列（x 与 z），返回 shape (T_in, 3)
    假设前 7 列含有需要的数据（第一列时间，接下三个是 accel）。
    """
    df = pd.read_excel(fp, header=0)
    cols = df.columns.tolist()
    if max(INPUT_CHANNEL_COLS) >= len(cols):
        raise ValueError(f"输入文件 {fp} 列数不足，期望至少包含索引 {INPUT_CHANNEL_COLS}")
    # 做列顺序的局部交换：复制前7列并交换位置
    arr = df.iloc[:, :7].copy()
    arr_cols = arr.columns.tolist()
    # 交换索引位置 SWAP_COL_A 与 SWAP_COL_B
    arr_cols[SWAP_COL_A], arr_cols[SWAP_COL_B] = arr_cols[SWAP_COL_B], arr_cols[SWAP_COL_A]
    arr = arr[arr_cols]
    # 提取三通道（现在仍位于 INPUT_CHANNEL_COLS 位置）
    ch = arr.iloc[:, INPUT_CHANNEL_COLS].values.astype(np.float32)  # (T_in, 3)
    return ch

def read_and_proc_output(fp):
    """
    读取输出 Excel，提取列索引 OUTPUT_CHANNEL_COLS 并 reshape 成 (T_out, 9, 3)
    """
    df = pd.read_excel(fp, header=0)
    cols = df.columns.tolist()
    if max(OUTPUT_CHANNEL_COLS) >= len(cols):
        raise ValueError(f"输出文件 {fp} 列数不足，期望至少包含索引 {OUTPUT_CHANNEL_COLS}")
    out_flat = df.iloc[:, OUTPUT_CHANNEL_COLS].values.astype(np.float32)  # (T_out, 27)
    out3 = out_flat.reshape((-1, 9, 3))
    return out3

def _rational_approx(u, v, max_den=1000):
    frac = Fraction(str(u)) / Fraction(str(v))
    frac = frac.limit_denominator(max_den)
    return frac.numerator, frac.denominator

def resample_event(x_in, fs_in, fs_target, y_out=None, fs_out=None, max_denominator=1024):
    """
    用 resample_poly 对输入 x_in 和可选的 y_out 做重采样到 fs_target。
    fallback 使用线性插值以避免超大 up/down。
    返回 (x_res, y_res)
    """
    # --- 处理 x_in ---
    if fs_in == fs_target:
        x_res = x_in.astype(np.float32)
    else:
        up, down = _rational_approx(fs_target, fs_in, max_den=max_denominator)
        if up > 2000 or down > 2000:
            T_in = x_in.shape[0]
            t_old = np.linspace(0.0, (T_in - 1) / fs_in, T_in)
            T_new = int(round((T_in - 1) * fs_target / fs_in)) + 1
            t_new = np.linspace(0.0, (T_new - 1) / fs_target, T_new)
            f = interp1d(t_old, x_in, axis=0, kind='linear', fill_value='extrapolate')
            x_res = f(t_new).astype(np.float32)
        else:
            x_res = resample_poly(x_in, up, down, axis=0).astype(np.float32)

    # --- 处理 y_out (可选) ---
    y_res = None
    if y_out is not None and fs_out is not None:
        if fs_out == fs_target:
            y_res = y_out.astype(np.float32)
        else:
            up_y, down_y = _rational_approx(fs_target, fs_out, max_den=max_denominator)
            if up_y > 2000 or down_y > 2000:
                T_y = y_out.shape[0]
                t_old = np.linspace(0.0, (T_y - 1) / fs_out, T_y)
                T_new = int(round((T_y - 1) * fs_target / fs_out)) + 1
                t_new = np.linspace(0.0, (T_new - 1) / fs_target, T_new)
                f_y = interp1d(t_old, y_out, axis=0, kind='linear', fill_value='extrapolate')
                y_res = f_y(t_new).astype(np.float32)
            else:
                y_res = resample_poly(y_out, up_y, down_y, axis=0).astype(np.float32)

    return x_res, y_res

# ---------- 主流程 ----------
input_files = list_xlsx(input_dir)
output_files = list_xlsx(output_dir)

# 打印被忽略的临时文件（便于调试）
temp_inputs = [os.path.basename(f) for f in glob.glob(os.path.join(input_dir, "~$*.xlsx"))]
temp_outputs = [os.path.basename(f) for f in glob.glob(os.path.join(output_dir, "~$*.xlsx"))]
if temp_inputs:
    print("Ignored temp input files:", temp_inputs)
if temp_outputs:
    print("Ignored temp output files:", temp_outputs)

if len(input_files) == 0:
    raise RuntimeError("在 input_dir 未找到任何 .xlsx 文件")
if len(input_files) != len(output_files):
    print("警告: input 与 output 文件数量不一致。继续但请确认文件对应关系（按排序顺序配对）。")

# 期望长度（以样点计）
expected_in_len = int(round(expected_in_seconds * target_fs))
expected_out_len = int(round(expected_out_seconds * target_fs))

for i, inp_fp in enumerate(tqdm(input_files, desc="Processing events")):
    try:
        base = os.path.splitext(os.path.basename(inp_fp))[0]
        # 找对应输出文件：优先同名，否则按索引位置配对
        candidate = os.path.join(output_dir, base + ".xlsx")
        if os.path.exists(candidate):
            out_fp = candidate
        else:
            if i < len(output_files):
                out_fp = output_files[i]
            else:
                print(f"[跳过] 找不到配对输出文件给 {inp_fp}")
                continue

        # 读取原始（未重采样）数据
        x_raw = read_and_proc_input(inp_fp)     # (T_in, 3)
        y_raw = read_and_proc_output(out_fp)    # (T_out, 9, 3)

        # ---------- 对特定 event id 的输入通道做幅值校正（只修改输入三通道） ----------
        scale = 1.0
        try:
            event_id = int(base)
        except Exception:
            m = re.search(r'\d+', base)
            event_id = int(m.group(0)) if m else None

        if event_id is not None and 23 <= event_id <= 56:
            scale = 1.1
            x_raw = (x_raw * scale).astype(np.float32)
            print(f"{base}: applied INPUT scale={scale} to channels for event id {event_id}")

        # ---------- 重采样（input 与 output） ----------
        x_res, y_res = resample_event(x_raw, fs_input, target_fs, y_out=y_raw, fs_out=fs_output)

        # ---------- 强制统一长度（截断或填充） ----------
        # 处理输入长度
        x_res = x_res.astype(np.float32)
        if x_res.shape[0] > expected_in_len:
            x_res = x_res[:expected_in_len]
            print(f"{base}: x_res truncated to {expected_in_len}")
        elif x_res.shape[0] < expected_in_len:
            pad_len_x = expected_in_len - x_res.shape[0]
            last_x = x_res[-1:]
            pad_block_x = np.repeat(last_x, pad_len_x, axis=0)
            x_res = np.concatenate([x_res, pad_block_x], axis=0)
            print(f"{base}: x_res padded from {expected_in_len - pad_len_x} to {expected_in_len} (pad_len={pad_len_x})")

        # 处理输出长度
        if y_res is None:
            raise RuntimeError(f"{base}: y_res 为 None —— 重采样或读取失败")
        y_res = y_res.astype(np.float32)
        applied_pad = False
        if y_res.shape[0] > expected_out_len:
            y_res = y_res[:expected_out_len]
            print(f"{base}: y_res truncated to {expected_out_len}")
        elif y_res.shape[0] < expected_out_len:
            pad_len = expected_out_len - y_res.shape[0]
            last = y_res[-1:]  # shape (1,9,3)
            pad_block = np.repeat(last, pad_len, axis=0)
            y_res = np.concatenate([y_res, pad_block], axis=0)
            applied_pad = True
            print(f"{base}: y_res padded from {expected_out_len - pad_len} to {expected_out_len} (pad_len={pad_len})")

        # 打印长度对比（便于检查）
        print(f"{base}: input {x_raw.shape[0]} -> {x_res.shape[0]} samples ({x_res.shape[0]/target_fs:.2f}s @ {target_fs}Hz), "
              f"output {y_raw.shape[0]} -> {y_res.shape[0]} samples ({y_res.shape[0]/target_fs:.2f}s @ {target_fs}Hz)")

        # ---------- 保存 .npz（使用压缩并记录 meta） ----------
        save_path = os.path.join(out_npz_dir, f"{base}.npz")
        meta = {
            "orig_input_len": int(x_raw.shape[0]),
            "orig_output_len": int(y_raw.shape[0]),
            "resampled_input_len": int(x_res.shape[0]),
            "resampled_output_len": int(y_res.shape[0]),
            "fs_target": float(target_fs),
            "applied_input_scale": float(scale) if scale != 1.0 else None,
            "applied_output_pad": bool(applied_pad)
        }
        np.savez_compressed(save_path, X=x_res, Y=y_res, meta=meta)
        print(f"Saved {save_path}  X:{x_res.shape}  Y:{y_res.shape}")

    except Exception as e:
        print(f"[错误] 处理 {inp_fp} 失败: {e}")

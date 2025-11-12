# data_loader_resample.py
import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import resample
from tqdm import tqdm

# ---------- 用户可修改的参数 ----------
input_dir = r"C:\Users\保护我方虎子\Desktop\model\dataset\input"
output_dir = r"C:\Users\保护我方虎子\Desktop\model\dataset\output"
out_npz_dir = r"./processed_resampled_events"
os.makedirs(out_npz_dir, exist_ok=True)

# 原始采样率（你的描述）
fs_input = 512.0   # 输入原始采样率
fs_output = 250.0  # 输出原始采样率
# 目标（统一）采样率，训练时可快速调整
target_fs = 50.0

# 输入/输出的列说明（基于你描述）
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
def read_and_proc_input(fp):
    df = pd.read_excel(fp, header=0)
    # 选出前三通道（按照文件结构）
    cols = df.columns.tolist()
    # safety: ensure column indices exist
    if max(INPUT_CHANNEL_COLS) >= len(cols):
        raise ValueError(f"输入文件 {fp} 列数不足，期望至少包含索引 {INPUT_CHANNEL_COLS}")
    # swap x and z columns (index positions)
    cols_list = list(df.columns)
    a_col = cols_list[SWAP_COL_A]
    b_col = cols_list[SWAP_COL_B]
    # make copy and swap
    arr = df.iloc[:, :7].copy()  # keep first 7 cols for safety
    arr_cols = arr.columns.tolist()
    arr_cols[SWAP_COL_A], arr_cols[SWAP_COL_B] = arr_cols[SWAP_COL_B], arr_cols[SWAP_COL_A]
    arr = arr[arr_cols]  # reorder columns in this local df
    # now extract the 3 channels (after swap, they are at positions 1,2,3 still)
    ch = arr.iloc[:, INPUT_CHANNEL_COLS].values.astype(np.float32)  # shape (T_in, 3)
    return ch

def read_and_proc_output(fp):
    df = pd.read_excel(fp, header=0)
    cols = df.columns.tolist()
    if max(OUTPUT_CHANNEL_COLS) >= len(cols):
        raise ValueError(f"输出文件 {fp} 列数不足，期望至少包含索引 {OUTPUT_CHANNEL_COLS}")
    out_flat = df.iloc[:, OUTPUT_CHANNEL_COLS].values.astype(np.float32)  # (T_out, 27)
    # reshape to (T_out, 9, 3)
    out3 = out_flat.reshape((-1, 9, 3))
    return out3

def resample_event(x_in, fs_in, fs_target, y_out=None, fs_out=None):
    """
    x_in: (T_in, Cx)
    y_out: (T_out, Cy) or None
    returns: x_res (T_in_new, Cx), y_res (T_out_new, Cy) or None
    """
    T_in = x_in.shape[0]
    duration = T_in / fs_in
    T_in_new = int(round(duration * fs_target))
    x_res = resample(x_in, T_in_new, axis=0)

    y_res = None
    if y_out is not None and fs_out is not None:
        # We assume y_out length corresponds to same absolute time origin (start at 0)
        T_out = y_out.shape[0]
        duration_out = T_out / fs_out
        # prefer using same duration (may be longer than input)
        T_out_new = int(round(duration_out * fs_target))
        y_res = resample(y_out, T_out_new, axis=0)
        # if resample returns floats, ensure float32
        y_res = y_res.astype(np.float32)
    return x_res.astype(np.float32), y_res

# ---------- 主流程 ----------
input_files = sorted(glob.glob(os.path.join(input_dir, "*.xlsx")))
output_files = sorted(glob.glob(os.path.join(output_dir, "*.xlsx")))

if len(input_files) == 0:
    raise RuntimeError("在 input_dir 未找到任何 .xlsx 文件")
if len(input_files) != len(output_files):
    print("警告: input 与 output 文件数量不一致。继续但请确认文件对应关系（按排序顺序配对）。")

for i, inp_fp in enumerate(tqdm(input_files, desc="Processing events")):
    try:
        base = os.path.splitext(os.path.basename(inp_fp))[0]
        # try to find matching output by same name or by index
        # default: pair by sorted order
        out_fp = None
        # first try same base name in output dir
        candidate = os.path.join(output_dir, base + ".xlsx")
        if os.path.exists(candidate):
            out_fp = candidate
        else:
            # fallback: by index if available
            if i < len(output_files):
                out_fp = output_files[i]
            else:
                print(f"找不到配对输出文件给 {inp_fp}, 跳过")
                continue

        x_raw = read_and_proc_input(inp_fp)     # (T_in, 3)
        y_raw = read_and_proc_output(out_fp)    # (T_out, 9, 3)

        x_res, y_res = resample_event(x_raw, fs_input, target_fs, y_out=y_raw, fs_out=fs_output)

        save_path = os.path.join(out_npz_dir, f"{base}.npz")
        np.savez(save_path, X=x_res, Y=y_res, meta={"orig_input_len": x_raw.shape[0], "orig_output_len": y_raw.shape[0], "fs_target": target_fs})
        print(f"Saved {save_path}  X:{x_res.shape}  Y:{None if y_res is None else y_res.shape}")

    except Exception as e:
        print(f"[错误] 处理 {inp_fp} 失败: {e}")

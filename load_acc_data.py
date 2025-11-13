# data_loader_resample.py
import os
import glob
from fractions import Fraction

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
    df = pd.read_excel(fp, header=0)   # 读取excel文件，第一列标为表头
    # 选出前三通道（按照文件结构）
    cols = df.columns.tolist()         # 读取列名，记成列表
    # safety: ensure column indices exist
    if max(INPUT_CHANNEL_COLS) >= len(cols):
        raise ValueError(f"输入文件 {fp} 列数不足，期望至少包含索引 {INPUT_CHANNEL_COLS}")
    # swap x and z columns (index positions)
    cols_list = list(df.columns)
    a_col = cols_list[SWAP_COL_A]
    b_col = cols_list[SWAP_COL_B]
    # make copy and swap
    arr = df.iloc[:, :7].copy()
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

def _rational_approx(u, v, max_den=1000):
    frac = Fraction(str(u)) / Fraction(str(v))  # 以字符串避免 float 精度问题
    frac = frac.limit_denominator(max_den)
    return frac.numerator, frac.denominator

def resample_event(x_in, fs_in, fs_target, y_out=None, fs_out=None, max_denominator=1024):
    """
    用 resample_poly 对输入 x_in 和可选的 y_out 做重采样到 fs_target。
    - x_in: shape (T_in, Cx)
    - fs_in: 原始采样率 (float)
    - fs_target: 目标采样率 (float)
    - y_out: 可选, shape (T_out, ...)，对 axis=0 做重采样
    - fs_out: y_out 的原始采样率 (float)
    返回 (x_res, y_res)
    注：resample_poly 对非周期信号比 FFT resample 更稳健（自带抗混叠滤波器）。
    """
    # --- 处理 x_in ---
    if fs_in == fs_target:
        x_res = x_in.astype(np.float32)
    else:
        # 近似为有理比 up/down，限制分母避免特别大滤波器
        up, down = _rational_approx(fs_target, fs_in, max_den=max_denominator)
        # 如果 up/down 非常大（例如 > 2000），退回到插值方法以避免超大滤波器
        if up > 2000 or down > 2000:
            # 时间向量插值（线性），作为后备
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
                # fallback to interpolation
                T_y = y_out.shape[0]
                t_old = np.linspace(0.0, (T_y - 1) / fs_out, T_y)
                T_new = int(round((T_y - 1) * fs_target / fs_out)) + 1
                t_new = np.linspace(0.0, (T_new - 1) / fs_target, T_new)
                f_y = interp1d(t_old, y_out, axis=0, kind='linear', fill_value='extrapolate')
                y_res = f_y(t_new).astype(np.float32)
            else:
                # resample_poly supports multi-dim arrays; axis=0 is time
                y_res = resample_poly(y_out, up_y, down_y, axis=0).astype(np.float32)

    return x_res, y_res
# ---------- 主流程 ----------
def list_xlsx(folder):
    pattern = os.path.join(folder, "*.xlsx")
    files = [f for f in glob.glob(pattern) if os.path.isfile(f)]
    # 过滤掉 Excel 临时文件 ~$
    files = [f for f in files if not os.path.basename(f).startswith("~$")]
    # 也可排除其他以点开头的隐藏文件（可选）
    files = [f for f in files if not os.path.basename(f).startswith(".")]
    files.sort()
    return files

input_files = list_xlsx(input_dir)
output_files = list_xlsx(output_dir)

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

# check_processed_lengths.py
import os, glob, numpy as np
import json

processed_dir = "./processed_resampled_events"
files = sorted(glob.glob(os.path.join(processed_dir, "*.npz")))
print("num files:", len(files))
if len(files) == 0:
    raise SystemExit("no files found")

# 以下数值要与 train.py 中一致（请确认）
target_fs = 50.0
window_seconds = 16.0
out_seconds = 60.0
stride_seconds = 1.0

window_size = int(round(window_seconds * target_fs))   # e.g. 800
out_steps = int(round(out_seconds * target_fs))        # e.g. 3000
stride = max(1, int(round(stride_seconds * target_fs)))# e.g. 50

print("Expect window_size =", window_size, "out_steps =", out_steps, "stride =", stride)

total_possible_windows = 0
file_info = []
for fp in files:
    d = np.load(fp, allow_pickle=True)
    X = d['X']; Y = d['Y']
    T_in = X.shape[0]; T_out = Y.shape[0]
    max_start = T_in - window_size
    count = 0
    if max_start >= 0:
        for start in range(0, max_start + 1, stride):
            y_start = start + window_size
            # require full future of length out_steps
            if y_start + out_steps <= T_out:
                count += 1
    file_info.append((os.path.basename(fp), T_in, T_out, count))
    total_possible_windows += count

# summarize
print(f"Total possible windows across files: {total_possible_windows}")
print("Files with non-zero windows (file, T_in, T_out, windows):")
for item in file_info:
    if item[3] > 0:
        print(item)
print("Files with zero windows (first 20 shown):")
zeros = [it for it in file_info if it[3] == 0]
for it in zeros[:20]:
    print(it)

# quick stats
T_ins = [it[1] for it in file_info]
T_outs = [it[2] for it in file_info]
print("T_in min/max/mean:", min(T_ins), max(T_ins), sum(T_ins)/len(T_ins))
print("T_out min/max/mean:", min(T_outs), max(T_outs), sum(T_outs)/len(T_outs))
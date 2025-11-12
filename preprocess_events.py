# preprocess_events.py
import os, numpy as np
from scipy.signal import butter, filtfilt

fs_orig = 100.0
lowcut, highcut = 0.2, 4.0
b, a = butter(4, [lowcut/(fs_orig/2), highcut/(fs_orig/2)], btype='band')
fs_new = 20.0
decim = int(fs_orig / fs_new)

src_dir = './processed_events'      # reshape 后但未滤波下采样的 per-event npz
dst_dir = './preprocessed_events'   # 滤波+下采样后存放位置
os.makedirs(dst_dir, exist_ok=True)
for fname in os.listdir(src_dir):
    if not fname.endswith('.npz'):
        continue
    fp = os.path.join(src_dir, fname)
    data = np.load(fp)
    X = data['X']  # shape (T,3,3)
    Y = data['Y']  # shape (T,8,3)
    try:
        X_f = filtfilt(b, a, X, axis=0)
        Y_f = filtfilt(b, a, Y, axis=0)
    except Exception as e:
        print(f"[警告] 事件 {fname} 滤波失败: {e}, 跳过")
        continue
    X_ds = X_f[::decim]
    Y_ds = Y_f[::decim]
    out_path = os.path.join(dst_dir, fname)
    np.savez(out_path, X=X_ds.astype(np.float32), Y=Y_ds.astype(np.float32))
    print(f"已保存预处理事件 {out_path}, shape X={X_ds.shape}, Y={Y_ds.shape}")

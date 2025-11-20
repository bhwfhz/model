# eval_resampled.py
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import re

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# import your model constructor (must match training)
from model import build_hybrid_conv_lstm_seq2seq

# ------------------- Config -------------------
processed_dir = "./processed_resampled_events"
scalers_path = "scalers_seq2seq_resampled.npz"   # from train_resampled.py
weights_path = "best_model_seq2seq.h5"           # saved weights
target_fs = 50.0

TEST_IDS = [43, 38, 33, 28, 23, 18, 13]
# same as training settings
window_seconds = 16.0
out_seconds = 60.0
window_size = int(round(window_seconds * target_fs))   # 800
out_steps = int(round(out_seconds * target_fs))        # 3000
stride_seconds = 1.0
stride = max(1, int(round(stride_seconds * target_fs)))   # 50

in_H, in_W, in_C = 3, 1, 1
out_H, out_W, out_C = 9, 1, 3

# prediction batch size for speed / memory
pred_batch_size = 4

# output dir for test results
out_dir_root = "test_results_resampled"
os.makedirs(out_dir_root, exist_ok=True)

# ------------------------------------
def load_best_params_local(path="./best_params_seq2seq.json"):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f'加载模型参数出错: {e}')
        return None

def build_and_load_model(params=None):
    # default params (should match training defaults)
    defaults = {
        'conv_filters': 16, 'conv_kh': 3, 'conv_kw': 1,
        'encoder_lstm_units': 128, 'decoder_lstm_units': 128,
        'dropout': 0.3, 'l2_reg': 3e-4, 'lr': 1e-3
    }
    if params:
        defaults.update(params)

    model = build_hybrid_conv_lstm_seq2seq(
        n_timesteps=window_size,
        in_H=in_H, in_W=in_W, in_C=in_C,
        out_steps=out_steps,
        out_H=out_H, out_W=out_W, out_C=out_C,
        conv_filters=int(defaults['conv_filters']),
        conv_kernel=(int(defaults.get('conv_kh', 3)), int(defaults.get('conv_kw', 1))),
        encoder_lstm_units=int(defaults.get('encoder_lstm_units', 128)),
        decoder_lstm_units=int(defaults.get('decoder_lstm_units', 128)),
        dropout_rate=float(defaults.get('dropout', 0.3)),
        l2_reg=float(defaults.get('l2_reg', 3e-4))
    )
    model.summary()
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print(f"Loaded weights from {weights_path}")
        except Exception as e:
            print(f"Failed to load weights: {e}")
            raise
    else:
        raise FileNotFoundError(f"Weights file {weights_path} not found.")
    return model

def make_windows(X_event, Y_event, window_size=window_size, stride=stride, out_steps=out_steps):
    """
    Produce X_windows and Y_true windows aligned by start (same as training).
    Returns:
      Xw: shape (N, window_size, in_H, in_W, in_C)
      Yw: shape (N, out_steps, out_H, out_W, out_C)
      starts: list of start indices
    """
    T = X_event.shape[0]
    T_out = Y_event.shape[0]
    Xs = []
    Ys = []
    starts = []
    if T < window_size:
        return (np.zeros((0, window_size, in_H, in_W, in_C), dtype=np.float32),
               np.zeros((0, out_steps, out_H, out_W, out_C), dtype=np.float32),
                starts)

    max_start = T - window_size
    for start in range(0, max_start + 1, stride):
        x_win = X_event[start:start + window_size]
        # label aligned at start
        y_start = start
        if y_start >= T_out:
            # no overlap, skip
            continue
        available = Y_event[y_start: min(T_out, y_start + out_steps)]
        if available.shape[0] < out_steps:
            # pad by repeating last frame
            if available.shape[0] == 0:
                # fallback: use last frame of Y_event if exists
                if T_out > 0:
                    pad_block = np.repeat(Y_event[-1:], out_steps, axis=0)
                    y_seq = pad_block
                else:
                    y_seq = np.zeros((out_steps, out_H, out_W, out_C), dtype=np.float32)
            else:
                pad_len = out_steps - available.shape[0]
                last = available[-1:]
                y_seq = np.concatenate([available, np.repeat(last, pad_len, axis=0)], axis=0)
        else:
            y_seq = available[:out_steps]

        x5d = x_win.reshape(window_size, in_H, in_W, in_C).astype(np.float32)
        y5d = y_seq.reshape(out_steps, out_H, out_W, out_C).astype(np.float32)
        Xs.append(x5d)
        Ys.append(y5d)
        starts.append(start)

    if len(Xs) == 0:
        return np.zeros((0, window_size, in_H, in_W, in_C), dtype=np.float32), \
               np.zeros((0, out_steps, out_H, out_W, out_C), dtype=np.float32), starts

    Xw = np.stack(Xs, axis=0)
    Yw = np.stack(Ys, axis=0)
    return Xw, Yw, starts

# --------------- 主评估流程 ---------------
def evaluate_all(processed_dir=processed_dir):
    # load scalers
    if not os.path.exists(scalers_path):
        raise FileNotFoundError(f"Scalers file not found: {scalers_path}")
    sc = np.load(scalers_path)
    mean_X = sc['mean_X']  # shape (1,1,in_H,in_W,in_C)
    std_X = sc['std_X']
    mean_Y = sc['mean_Y']  # shape (1,1,out_H,out_W,out_C)
    std_Y = sc['std_Y']

    # load best params if exist to build model
    best_params = load_best_params_local("./best_params_seq2seq.json")
    model = build_and_load_model(best_params)

    def basename_id(fp):
        bn = os.path.splitext(os.path.basename(fp))[0]
        m = re.search(r'\d+', bn)
        return int(m.group(0)) if m else None

    all_files = sorted(glob.glob(os.path.join(processed_dir, "*.npz")))
    files = []
    for fp in all_files:
        eid = basename_id(fp)
        if eid in TEST_IDS:
            files.append(fp)

    print(f"Evaluating {len(files)} test files: {[os.path.basename(f) for f in files]}")
    all_y_true = []
    all_y_pred = []
    summary = []
    for fp in files:
        base = os.path.splitext(os.path.basename(fp))[0]  # 提取基础文件名（不含扩展名）
        print(f"\nProcessing {base} ...")
        d = np.load(fp, allow_pickle=True)
        X_event = d['X'].astype(np.float32)   # (T_in, 3)
        Y_event = d['Y'].astype(np.float32)   # (T_out, 9, 3)

        Xw, Yw, starts = make_windows(X_event, Y_event)
        if Xw.shape[0] == 0:                             # 检查是否产生了窗口？
            print(f"  {base}: no windows generated, skipping.")
            continue

        # normalize using training mean/std
        Xw_s = (Xw - mean_X) / std_X

        # predict in batches
        preds = []
        for i in range(0, Xw_s.shape[0], pred_batch_size):
            batch = Xw_s[i:i+pred_batch_size]
            p = model.predict(batch, verbose=0)
            # p shape => (B, out_steps, out_H, out_W, out_C)
            preds.append(p.astype(np.float32))
        preds = np.concatenate(preds, axis=0)  # (N, out_steps, out_H, out_W, out_C)

        # de-normalize predictions
        Y_pred_real = preds * std_Y + mean_Y  # broadcast
        # ensure shapes: (N, out_steps, out_H, out_W, out_C)
        if Y_pred_real.shape != Yw.shape:
            raise RuntimeError(f"Shape mismatch for {base}: pred {Y_pred_real.shape} vs true {Yw.shape}. "
                               "Check mean/std shapes and model output dims.")

        # flatten for metrics: (N*out_steps, out_H*out_W*out_C)
        N = Yw.shape[0]
        y_true_flat = Yw.reshape(N * out_steps, out_H * out_W * out_C)
        y_pred_flat = Y_pred_real.reshape(N * out_steps, out_H * out_W * out_C)

        # compute metrics for this file
        rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        try:
            r2 = r2_score(y_true_flat, y_pred_flat)
        except Exception:
            r2 = float('nan')
        summary.append((base, rmse, mae, r2))
        print(f"{base}: windows={N}  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

        all_y_true.append(y_true_flat)
        all_y_pred.append(y_pred_flat)

        # save predictions for this file
        ev_out = os.path.join(out_dir_root, base)
        os.makedirs(ev_out, exist_ok=True)
        np.save(os.path.join(ev_out, f"{base}_pred.npy"), Y_pred_real)  # shape (N, out_steps, out_H, out_W, out_C)

        # Also save flattened tabular (N*out_steps rows, channels columns)
        # columns names: pt1_x, pt1_y, pt1_z, ..., pt9_z
        chan_names = []
        for p in range(out_H):
            chan_names += [f"pt{p+1}_x", f"pt{p+1}_y", f"pt{p+1}_z"]
        import pandas as pd
        df_pred = pd.DataFrame(y_pred_flat, columns=chan_names)
        df_true = pd.DataFrame(y_true_flat, columns=chan_names)
        df_all = pd.concat([df_true.add_suffix("_true"), df_pred.add_suffix("_pred")], axis=1)
        df_all.to_excel(os.path.join(ev_out, f"{base}_pred_flat.xlsx"), index=False)

        # Plot first 6 channels of first window as quick check
        times = np.arange(out_steps) / target_fs
        plot_n = min(6, out_H * out_C)
        for ch in range(plot_n):
            plt.figure(figsize=(8, 3))
            plt.plot(times, y_true_flat[:out_steps, ch], label='true')
            plt.plot(times, y_pred_flat[:out_steps, ch], '--', label='pred')
            plt.title(f"{base} - ch {chan_names[ch]}")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(ev_out, f"ch{ch}_{chan_names[ch]}.png"))
            plt.close()

    # overall metrics across files
    if all_y_true:
        all_true = np.concatenate(all_y_true, axis=0)
        all_pred = np.concatenate(all_y_pred, axis=0)
        print("\n=== Overall metrics ===")
        print("Total samples:", all_true.shape[0])
        rmse_all = np.sqrt(mean_squared_error(all_true, all_pred))
        mae_all = mean_absolute_error(all_true, all_pred)
        try:
            r2_all = r2_score(all_true, all_pred)
        except Exception:
            r2_all = float('nan')
        print(f"Overall RMSE={rmse_all:.6f}, MAE={mae_all:.6f}, R2={r2_all:.6f}")

        # save summary
        summary_path = os.path.join(out_dir_root, "summary.csv")
        import csv
        with open(summary_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["file", "rmse", "mae", "r2"])
            for row in summary:
                writer.writerow(row)
        print("Saved per-file summary to", summary_path)
    else:
        print("No predictions were generated (no windows).")



if __name__ == "__main__":
    evaluate_all()

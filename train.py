# train.py (modified)
import os
import glob
import datetime
import time
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# 确保可重复
np.random.seed(42)
tf.random.set_seed(42)

# 导入用户的 model 构建函数（与你的代码一致）
from model4 import build_hybrid_conv_lstm_lstm_se

# ----------------------------
# 配置部分（可在外部覆盖）
# ----------------------------
processed_dir = './preprocessed_events'
if not os.path.isdir(processed_dir):
    raise RuntimeError(f"{processed_dir} 不存在，请先预处理并保存 per-event npz")

# 模型输入输出参数
window_size = 18
stride = 1
in_H, in_W, in_C = 3, 1, 3
out_H, out_W, out_C = 8, 1, 3

# 训练参数（默认）
batch_size = 16
epochs = 50
ckpt_filepath = "best_model.h5"
log_base = "logs"

# ----------------------------
# 缩放因子（针对三个加速度方向 x,y,z）
# ----------------------------
AXIS_SCALE = np.array([1.0, 0.85, 0.65], dtype=np.float32)  # shape (3,)

# ----------------------------
# 辅助函数
# ----------------------------

def list_event_files(processed_dir):
    files = sorted([
        os.path.join(processed_dir, f)
        for f in os.listdir(processed_dir)
        if f.endswith('.npz')
    ])
    if not files:
        raise RuntimeError(f"在 {processed_dir} 未找到任何 .npz 文件")
    return files


def generate_xy_from_files(file_list, window_size=window_size, stride=stride):
    """
    从 per-event npz 里生成滑窗样本。
    返回:
      X_arr: (N, window_size, in_H, in_W, in_C)
      Y_arr: (N, out_H, out_W, out_C)  注意：之前你用 y5d = y_target.reshape(1,out_H,out_W,out_C) 并堆叠成 (N,1,out_H,out_W,out_C)
             为了与原 pipeline 一致，我们这里保持 Y_arr 为 (N, out_H, out_W, out_C)
    """
    Xs = []
    Ys = []
    for fp in file_list:
        data = np.load(fp)
        X_event = data['X']  # shape (T_ds,3,3)
        Y_event = data['Y']  # shape (T_ds,8,3)
        T = X_event.shape[0]
        if T < window_size:
            continue
        for start in range(0, T - window_size + 1, stride):
            x_win = X_event[start: start + window_size]           # (window_size, 3, 3)
            y_target = Y_event[start]                              # (8, 3)
            x5d = x_win.reshape(window_size, in_H, in_W, in_C)     # (window_size,3,1,3)
            y5d = y_target.reshape(out_H, out_W, out_C)           # (8,1,3)
            Xs.append(x5d)
            Ys.append(y5d)
    if not Xs:
        raise RuntimeError("未生成任何样本，请检查数据和滑窗参数")
    X_arr = np.stack(Xs, axis=0).astype(np.float32)  # (N, window_size, in_H, in_W, in_C)
    Y_arr = np.stack(Ys, axis=0).astype(np.float32)  # (N, out_H, out_W, out_C)
    return X_arr, Y_arr


def setup_tensorboard(log_base):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    abs_base = os.path.abspath(log_base)
    if os.path.exists(abs_base) and not os.path.isdir(abs_base):
        print(f"[警告] 日志根路径 {abs_base} 存在且不是目录，跳过 TensorBoard")
        return None
    os.makedirs(abs_base, exist_ok=True)
    log_dir = os.path.join(abs_base, ts)
    os.makedirs(log_dir, exist_ok=True)
    return os.path.abspath(log_dir)


# ----------------------------
# 可复用的数据准备接口（包含缩放因子）
# ----------------------------
def load_and_prepare_data(processed_dir, val_names, axis_scale=AXIS_SCALE):
    """
    返回标准化后的训练/验证集 (X_train_s, Y_train_s, X_val_s, Y_val_s) 以及 scalers 字典
    val_names: 验证集文件名列表（例如 ['200410062340ANX.npz', ...]）
    axis_scale: array_like shape (3,) 对最后一个通道(axis=-1)的缩放因子 [sx, sy, sz]
    """
    event_files = sorted(glob.glob(os.path.join(processed_dir, '*.npz')))
    if not event_files:
        raise RuntimeError(f"在 {processed_dir} 未找到任何 .npz 文件")

    train_files = [f for f in event_files if os.path.basename(f) not in val_names]
    val_files = [f for f in event_files if os.path.basename(f) in val_names]

    print(f"Train events: {len(train_files)}, Val events: {len(val_files)}")

    X_train, Y_train = generate_xy_from_files(train_files)
    X_val, Y_val = generate_xy_from_files(val_files)

    # apply axis scaling: X shape (N, window_size, in_H, in_W, in_C)
    # axis_scale shape (3,) -> reshape to (1,1,1,1,3) for broadcast
    scale_resh = np.reshape(np.array(axis_scale, dtype=np.float32), (1, 1, 1, 1, -1))
    X_train = X_train * scale_resh
    X_val = X_val * scale_resh

    # 标准化（按原脚本行为：按训练集统计量）
    mean_X = X_train.mean(axis=(0, 1), keepdims=True)
    std_X = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    X_train_s = (X_train - mean_X) / std_X
    X_val_s = (X_val - mean_X) / std_X

    mean_Y = Y_train.mean(axis=(0, 1), keepdims=True)
    std_Y = Y_train.std(axis=(0, 1), keepdims=True) + 1e-6
    Y_train_s = (Y_train - mean_Y) / std_Y
    Y_val_s = (Y_val - mean_Y) / std_Y

    scalers = {'mean_X': mean_X, 'std_X': std_X, 'mean_Y': mean_Y, 'std_Y': std_Y, 'axis_scale': axis_scale}

    return X_train_s, Y_train_s, X_val_s, Y_val_s, scalers


# ----------------------------
# 模型构建与训练封装
# ----------------------------
def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse


def build_and_compile_model(params, window_size=window_size,
                            in_H=in_H, in_W=in_W, in_C=in_C,
                            out_H=out_H, out_W=out_W, out_C=out_C):
    """
    params: dict 包含 'conv_filters','conv_kh','conv_kw','lstm_units','dropout','l2_reg','lr'
    """
    conv_filters = int(params.get('conv_filters', 16))
    conv_kh = int(params.get('conv_kh', 9))
    conv_kw = int(params.get('conv_kw', 5))
    lstm_units = int(params.get('lstm_units', 128))
    dropout_rate = float(params.get('dropout', 0.3))
    l2_reg = float(params.get('l2_reg', 3e-4))
    lr = float(params.get('lr', 1e-3))

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
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=combined_loss, metrics=['mae'])
    return model


def train_one_run(params, X_train, Y_train, X_val, Y_val, epochs, batch_size, ckpt_path, verbose=0):
    """
    单次训练接口：保存最优权重到 ckpt_path，返回 (val_rmse, model, history)
    """
    model = build_and_compile_model(params)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)
    ]

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=verbose
    )

    # 使用模型预测验证集并计算 RMSE
    pred = model.predict(X_val, batch_size=max(1, int(batch_size // 2)), verbose=0)
    val_rmse = float(np.sqrt(np.mean((Y_val - pred) ** 2)))

    return val_rmse, model, history


# ----------------------------
# Helper: load best params from hyperopt results (if present)
# ----------------------------
def load_best_params(search_dir="./hyperopt_results"):
    """
    尝试读取 hyperopt 保存的 best_params（pickle 或 json）
    返回 dict 或 None
    """
    candidates = [
        os.path.join(search_dir, "best_params.pkl"),
        os.path.join(search_dir, "best_params.json"),
        os.path.join(search_dir, "best_raw.pkl"),
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
                return data
            else:
                with open(p, "rb") as f:
                    data = pickle.load(f)
                return data
        except Exception as e:
            print(f"Warning: failed to load {p}: {e}")
            continue
    return None


def normalize_params(raw):
    """
    将 raw params 转换为模型构建所需的标准字段并做类型转换。
    """
    if raw is None:
        return None
    params = {}
    def get_key(k):
        for kk in raw.keys():
            if kk == k or kk.lower() == k.lower():
                return raw[kk]
        return None

    # straightforward mapping
    for key in ['conv_filters','conv_kh','conv_kw','lstm_units','dropout','l2_reg','lr','batch_size']:
        val = get_key(key)
        if val is not None:
            params[key] = val

    # ensure types
    if 'conv_filters' in params: params['conv_filters'] = int(params['conv_filters'])
    if 'conv_kh' in params: params['conv_kh'] = int(params['conv_kh'])
    if 'conv_kw' in params: params['conv_kw'] = int(params['conv_kw'])
    if 'lstm_units' in params: params['lstm_units'] = int(params['lstm_units'])
    if 'batch_size' in params: params['batch_size'] = int(params['batch_size'])
    if 'dropout' in params: params['dropout'] = float(params['dropout'])
    if 'l2_reg' in params: params['l2_reg'] = float(params['l2_reg'])
    if 'lr' in params: params['lr'] = float(params['lr'])

    return params


# ----------------------------
# 主流程（保留一键训练）
# ----------------------------
def main():
    # 验证集文件名（和你原脚本保持一致）
    val_names = [
        '200410062340ANX.npz',
        '200510192044ANX.npz',
        '201103230734ANX.npz',
        '201409161228ANX.npz',
        '201505251428ANX.npz',
        '201901141323ANX.npz',
    ]

    # ---------- 1. 加载 & 预处理数据（含 axis scaling） ----------
    X_train, Y_train, X_val, Y_val, scalers = load_and_prepare_data(processed_dir, val_names, axis_scale=AXIS_SCALE)

    # 保存 scalers（包含 axis_scale 以便 eval 使用）
    np.savez('scalers.npz',
             mean_X=scalers['mean_X'], std_X=scalers['std_X'],
             mean_Y=scalers['mean_Y'], std_Y=scalers['std_Y'],
             axis_scale=scalers['axis_scale'])
    print("Saved scalers.npz with mean/std for X and Y and axis_scale.")

    # ---------- 2. 读取 hyperopt 最优超参（如存在）并决定使用哪些超参 ----------
    raw_best = load_best_params(search_dir="./hyperopt_results")
    mapped = normalize_params(raw_best) if raw_best is not None else None

    default_params = {
        'conv_filters': 16, 'conv_kh': 9, 'conv_kw': 5,
        'lstm_units': 128, 'dropout': 0.3, 'l2_reg': 3e-4, 'lr': 1e-3,
    }

    if mapped:
        # 合并 mapped 到 default_params（mapped 覆盖 default）
        used_params = default_params.copy()
        used_params.update(mapped)
        print("Loaded best hyperopt params and will use them for final training:")
        print(json.dumps(used_params, indent=2))
    else:
        used_params = default_params
        print("No hyperopt best params found — using default params:")
        print(json.dumps(used_params, indent=2))

    # 保存最终使用的超参以便复现
    with open('best_params_used.pkl', 'wb') as f:
        pickle.dump(used_params, f)
    with open('best_params_used.json', 'w', encoding='utf-8') as f:
        json.dump(used_params, f, indent=2)

    # 如果 mapped 中没有 batch_size，使用全局 batch_size 变量
    used_batch_size = int(used_params.get('batch_size', batch_size))
    used_epochs = int(epochs)

    # ---------- 3. 训练 ----------
    print("Starting final training with params:", used_params)
    start_time = time.time()
    val_rmse, model, history = train_one_run(used_params, X_train, Y_train, X_val, Y_val,
                                             epochs=used_epochs, batch_size=used_batch_size,
                                             ckpt_path=ckpt_filepath, verbose=1)
    elapsed = time.time() - start_time
    print(f"Final training done. Best val RMSE: {val_rmse:.6f}. Elapsed: {elapsed/3600:.2f} hours")

    # 保存训练曲线图片（与原脚本行为类似）
    loss_vals = history.history.get('loss', [])
    val_loss_vals = history.history.get('val_loss', [])
    mae_vals = history.history.get('mae', [])
    val_mae_vals = history.history.get('val_mae', [])
    epochs_range = range(1, len(loss_vals) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss_vals, 'b-', label='train loss')
    plt.plot(epochs_range, val_loss_vals, 'r--', label='val loss')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, mae_vals, 'b-', label='train MAE')
    plt.plot(epochs_range, val_mae_vals, 'r--', label='val MAE')
    plt.title('MAE vs Epoch')
    plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print('Saved training_curves.png')


if __name__ == '__main__':
    print('Current working dir:', os.getcwd())
    main()

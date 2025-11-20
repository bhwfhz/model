# predict.py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model import build_model

model = tf.keras.models.load_model("best_model_fixed.keras", compile=False)


def autoregressive_predict(filepath, horizon=100):
    data = np.load(filepath)
    X = data['X']  # (800,3)
    pga = float(data['pga'])
    Y_true = data['Y'] * pga  # (3000,27)

    pred_all = []
    window = X.copy()
    for i in range(0, 3000, horizon):
        pred = model.predict(window[np.newaxis, ...], verbose=0)[0]  # (100,27)
        take = min(horizon, 3000 - i)
        pred_all.append(pred[:take])
        window = np.concatenate([window[take:], pred[:take]], axis=0)

    Y_pred = np.concatenate(pred_all, axis=0) * pga
    return Y_true, Y_pred


# 测试
true, pred = autoregressive_predict("./dataset_corrected/1.npz")  # 改成你的文件名
plt.figure(figsize=(15, 5))
plt.plot(true[:2000, 0], label='True pt1_x', alpha=0.8)
plt.plot(pred[:2000, 0], label='Pred pt1_x', alpha=0.8)
plt.legend();
plt.title("终于不是直线了！");
plt.show()
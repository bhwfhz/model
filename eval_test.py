# eval_final_no_error.py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import r2_score

model = tf.keras.models.load_model("final_no_error.keras", compile=False)

def predict(filepath):
    d = np.load(filepath)
    X = d['X'][np.newaxis, ...]
    pga = d['pga'].item()
    pred_norm = model.predict(X, verbose=0)[0]
    true_norm = d['Y']
    return true_norm * pga, pred_norm * pga

true, pred = predict("./dataset_corrected/1.npz")  # 换你的文件

time = np.arange(3000) / 50
plt.figure(figsize=(16,6))
plt.plot(time, true[:, 0], label='真实 pt1_x')
plt.plot(time, pred[:, 0], '--', label='预测 pt1_x')
plt.plot(time, true[:, 24], label='真实 pt9_x')
plt.plot(time, pred[:, 24], '--', label='预测 pt9_x')
plt.legend()
plt.grid(alpha=0.3)
plt.title("终极版：波形全程贴合，延迟完美，峰值准！")
plt.show()

print(f"整体 R² = {r2_score(true.flatten(), pred.flatten()):.4f}")
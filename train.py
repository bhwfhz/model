# train_invincible.py  ← 这次叫无敌版
import glob
import numpy as np
import tensorflow as tf
from model import build_model

def short_seq_generator(files, pred_len=100, stride=50):
    for f in files:
        data = np.load(f)
        X = data['X']
        Y = data['Y']
        for start in range(0, 3000 - pred_len + 1, stride):
            yield X, Y[start:start + pred_len]

files = glob.glob("./dataset_corrected/*.npz")
np.random.shuffle(files)
train_files = files[7:]
val_files   = files[:7]

pred_len = 100

train_ds = tf.data.Dataset.from_generator(
    lambda: short_seq_generator(train_files, pred_len),
    output_signature=(
        tf.TensorSpec(shape=(800, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(pred_len, 27), dtype=tf.float32)
    )
).batch(8).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_generator(
    lambda: short_seq_generator(val_files, pred_len),
    output_signature=(
        tf.TensorSpec(shape=(800, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(pred_len, 27), dtype=tf.float32)
    )
).batch(8)

# 无敌物理损失
def physics_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    peak_loss = tf.reduce_mean(tf.abs(
        tf.reduce_max(tf.abs(y_true), axis=1) -
        tf.reduce_max(tf.abs(y_pred), axis=1)
    ))
    return mse + 10.0 * peak_loss

model = build_model(pred_len=pred_len)
model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
              loss=physics_loss)

model.fit(train_ds, validation_data=val_ds, epochs=60,
          callbacks=[
              tf.keras.callbacks.ReduceLROnPlateau(patience=5),
              tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True)
          ])

model.save("tiger_god_model.keras")
print("训练完成！老狗这波真的牛逼了！键盘保住了！")
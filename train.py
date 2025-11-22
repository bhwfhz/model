# train_final_no_error.py
import glob
import numpy as np
import tensorflow as tf
from model import build_model

files = glob.glob("./dataset_corrected/*.npz")  # 用你之前成功的那个数据集
np.random.shuffle(files)
train_files = files[7:]
val_files = files[:7]

def gen(files):
    for f in files:
        d = np.load(f)
        yield d['X'], d['Y']

ds_train = tf.data.Dataset.from_generator(lambda: gen(train_files), (tf.float32, tf.float32)).batch(4).prefetch(2)
ds_val = tf.data.Dataset.from_generator(lambda: gen(val_files), (tf.float32, tf.float32)).batch(4)

# 最简单的损失（绝对不会广播错）
def simple_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss=simple_loss)  # 去掉 metrics!!!

model.fit(ds_train, validation_data=ds_val, epochs=100,
          callbacks=[
              tf.keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5),
              tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
          ])

model.save("final_no_error.keras")
print("训练完成！这回绝对不报错了！")
# model.py
import tensorflow as tf


def build_model(input_len=800, pred_len=100, in_ch=3, out_ch=27):
    inputs = tf.keras.Input(shape=(input_len, in_ch))
    x = inputs

    # 膨胀卷积骨干（感受野 > 16s）
    for dilation_rate in [1, 2, 4, 8, 16]:
        x = tf.keras.layers.Conv1D(64, kernel_size=3, padding='causal',
                                   dilation_rate=dilation_rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    # 轻量Transformer（2层）
    for _ in range(2):
        attn = tf.keras.layers.MultiHeadAttention(8, 64)(x, x)
        x = tf.keras.layers.Add()([x, attn])
        x = tf.keras.layers.LayerNormalization()(x)

        ff = tf.keras.layers.Dense(256, activation='relu')(x)
        ff = tf.keras.layers.Dense(64)(ff)
        x = tf.keras.layers.Add()([x, ff])
        x = tf.keras.layers.LayerNormalization()(x)

    # 关键修复：先把 x 从 (batch, 800, 64) 取最后时刻，再挤压成 2D
    last_step = x[:, -1, :]  # (batch, 64)
    repeated = tf.keras.layers.RepeatVector(pred_len)(last_step)  # (batch, pred_len, 64)

    # 再接几层解码器，把64维映射到27维
    x = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu')(repeated)
    x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    outputs = tf.keras.layers.Conv1D(out_ch, 1, activation='linear')(x)  # (batch, pred_len, 27)

    model = tf.keras.Model(inputs, outputs)
    return model


# 直接跑一下看形状对不对
if __name__ == "__main__":
    model = build_model(pred_len=100)
    model.summary()
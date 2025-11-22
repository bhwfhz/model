# model_final_no_error.py
import tensorflow as tf


def positional_encoding(length, depth):
    depth = depth // 2
    positions = tf.range(length, dtype=tf.float32)[:, tf.newaxis]  # (seq_len, 1)
    depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, :] / depth  # (1, depth//2)
    angle_rates = 1 / tf.pow(10000.0, depths)  # (1, depth//2)
    angle_rads = positions * angle_rates  # (seq_len, depth//2)
    sin = tf.sin(angle_rads)
    cos = tf.cos(angle_rads)
    pos_encoding = tf.concat([sin, cos], axis=-1)  # (seq_len, depth)
    return tf.cast(pos_encoding, tf.float32)


def build_model():
    inputs = tf.keras.Input(shape=(800, 3))
    x = tf.keras.layers.Dense(64)(inputs)
    x = x + positional_encoding(800, 64)

    for d in [1, 2, 4, 8, 16]:
        x = tf.keras.layers.Conv1D(64, 3, padding='causal', dilation_rate=d, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)

    for _ in range(3):
        attn = tf.keras.layers.MultiHeadAttention(8, 64)(x, x)
        x = tf.keras.layers.Add()([x, attn])
        x = tf.keras.layers.LayerNormalization()(x)
        ff = tf.keras.layers.Dense(256, activation='relu')(x)
        ff = tf.keras.layers.Dense(64)(ff)
        x = tf.keras.layers.Add()([x, ff])
        x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Conv1D(128, 1, activation='relu')(x)
    x = tf.keras.layers.UpSampling1D(3000 // 800)(x)  # 上采样到3000
    outputs = tf.keras.layers.Conv1D(27, 1)(x)

    return tf.keras.Model(inputs, outputs)
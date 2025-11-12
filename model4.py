import tensorflow as tf
from tensorflow.keras.layers import (
    Input,ConvLSTM2D,BatchNormalization,Reshape,LSTM,Dense,Dropout,GlobalAveragePooling3D,Multiply,Activation,Reshape as KReshape,Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2



def squeeze_excitation_block(input_tensor, ratio=8):
    # Channel count
    channel_axis = -1
    filters = input_tensor.shape[channel_axis]
    se_shape = (1, 1, 1, filters)

    # Squeeze: global spatial & temporal pooling
    se = GlobalAveragePooling3D()(input_tensor)
    se = KReshape((1, 1, 1, filters))(se)

    # Excitation: two fully-connected layers
    se = Dense(
        filters // ratio,
        activation='relu',
        kernel_initializer='he_normal',
        use_bias=False
    )(se)
    se = Dense(
        filters,
        activation='sigmoid',
        kernel_initializer='he_normal',
        use_bias=False
    )(se)

    # Scale
    x = Multiply()([input_tensor, se])
    return x


def build_hybrid_conv_lstm_lstm_se(
        n_timesteps,in_H,in_W,in_C,out_H,out_W,out_C,
        conv_filters=16,
        conv_kernel=(9, 5),
        lstm_units=128,
        dropout_rate=0.3,
        l2_reg=3e-4,
        se_ratio=8
):
    """
    Hybrid ConvLSTM + LSTM model with Squeeze-and-Excitation channel attention.
    """
    inputs = Input(shape=(n_timesteps, in_H, in_W, in_C), name='input_window')

    # 1. ConvLSTM + BatchNorm
    x = ConvLSTM2D(
        filters=conv_filters,
        kernel_size=conv_kernel,
        padding='same',
        return_sequences=True,
        kernel_regularizer=l2(l2_reg),
        name='conv_lstm'
    )(inputs)
    x = BatchNormalization(name='bn_after_conv')(x)

    # 1.5. SE attention on ConvLSTM output
    x = squeeze_excitation_block(x, ratio=se_ratio)

    # 2. reshape to (batch, time, features)
    feat_dim = in_H * in_W * conv_filters
    x = Reshape(target_shape=(n_timesteps, feat_dim), name='reshape_for_lstm')(x)

    # 3. first LSTM
    x = LSTM(
        units=lstm_units,
        return_sequences=True,
        kernel_regularizer=l2(l2_reg),
        name='lstm1'
    )(x)
    x = Dropout(rate=dropout_rate, name='dropout1')(x)

    # 4. second LSTM
    x = LSTM(
        units=lstm_units,
        return_sequences=False,
        kernel_regularizer=l2(l2_reg),
        name='lstm2'
    )(x)
    x = Dropout(rate=dropout_rate, name='dropout2')(x)

    # 5. Dense to output dims
    dense_units = out_H * out_C
    x = Dense(units=dense_units, activation=None, name='dense_output')(x)

    # 6. reshape to (batch, 1, out_H, out_W, out_C)
    outputs = Reshape(target_shape=(1, out_H, out_W, out_C), name='final_output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='hybrid_conv_lstm_lstm_se')
    return model


if __name__ == '__main__':
    model = build_hybrid_conv_lstm_lstm_se(
        n_timesteps=18,
        in_H=3, in_W=1, in_C=3,
        out_H=8, out_W=1, out_C=3
    )
    model.summary()
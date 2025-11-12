# model4.py
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, ConvLSTM2D, BatchNormalization, Reshape, LSTM, Dense, Dropout,
    GlobalAveragePooling3D, Multiply, Reshape as KReshape, RepeatVector,
    TimeDistributed
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def squeeze_excitation_block(input_tensor, ratio=8):
    channel_axis = -1
    filters = int(input_tensor.shape[channel_axis])
    se = GlobalAveragePooling3D()(input_tensor)               # (batch, filters)
    se = KReshape((1,1,1,filters))(se)                       # (batch,1,1,1,filters)
    se = Dense(filters // ratio, activation='relu', use_bias=False, kernel_initializer='he_normal')(se)
    se = Dense(filters, activation='sigmoid', use_bias=False, kernel_initializer='he_normal')(se)
    x = Multiply()([input_tensor, se])
    return x

def build_hybrid_conv_lstm_seq2seq(
        n_timesteps, in_H, in_W, in_C,
        out_steps, out_H, out_W, out_C,
        conv_filters=16,
        conv_kernel=(3,1),
        encoder_lstm_units=128,
        decoder_lstm_units=128,
        dropout_rate=0.3,
        l2_reg=3e-4,
        se_ratio=8
):
    """
    Seq2Seq variant:
      Encoder: ConvLSTM2D -> BN -> SE -> reshape -> LSTM(s) -> encoding vector
      Decoder: RepeatVector(out_steps) -> LSTM(return_sequences=True) -> TimeDistributed(Dense) -> reshape output (out_steps, out_H, out_W, out_C)
    Inputs:
      n_timesteps: input window length (time)
      in_H,in_W,in_C: spatial dims for ConvLSTM input
      out_steps: number of future time steps to predict
      out_H,out_W,out_C: spatial dims for output at each time step
    """
    # clamp conv kernel to spatial dims (safe-guard)
    kh, kw = conv_kernel
    kh = int(min(kh, max(1, in_H)))
    kw = int(min(kw, max(1, in_W)))
    conv_kernel = (kh, kw)

    inputs = Input(shape=(n_timesteps, in_H, in_W, in_C), name='encoder_input')  # (batch, T, H, W, C)

    # Encoder: ConvLSTM -> BN -> SE
    x = ConvLSTM2D(filters=conv_filters, kernel_size=conv_kernel, padding='same',
                   return_sequences=True, kernel_regularizer=l2(l2_reg), name='conv_lstm')(inputs)
    x = BatchNormalization(name='bn_after_conv')(x)
    x = squeeze_excitation_block(x, ratio=se_ratio)

    # Flatten time+space to sequence of features for LSTM
    feat_dim = in_H * in_W * conv_filters
    x = Reshape(target_shape=(n_timesteps, feat_dim), name='reshape_for_encoder_lstm')(x)

    # Encoder LSTM(s)
    x = LSTM(units=encoder_lstm_units, return_sequences=False, kernel_regularizer=l2(l2_reg), name='encoder_lstm')(x)
    x = Dropout(rate=dropout_rate, name='encoder_dropout')(x)   # x is (batch, encoder_lstm_units)

    # Decoder: repeat encoded vector out_steps times -> LSTM (return_sequences=True) -> TimeDistributed Dense
    d = RepeatVector(out_steps, name='repeat_for_decoder')(x)      # (batch, out_steps, encoder_lstm_units)

    d = LSTM(units=decoder_lstm_units, return_sequences=True, name='decoder_lstm')(d)  # (batch, out_steps, decoder_lstm_units)
    d = Dropout(rate=dropout_rate, name='decoder_dropout')(d)

    # Map each time-step to output channels (out_H * out_W * out_C)
    per_step_units = out_H * out_W * out_C
    d = TimeDistributed(Dense(units=per_step_units, activation=None), name='time_distributed_dense')(d)  # (batch, out_steps, per_step_units)

    # reshape to (batch, out_steps, out_H, out_W, out_C)
    outputs = Reshape(target_shape=(out_steps, out_H, out_W, out_C), name='final_output')(d)

    model = Model(inputs=inputs, outputs=outputs, name='hybrid_conv_lstm_seq2seq')
    return model


if __name__ == "__main__":
    # quick sanity-check model summary with example dims
    model = build_hybrid_conv_lstm_seq2seq(
        n_timesteps=800, in_H=3, in_W=1, in_C=1,
        out_steps=3000, out_H=9, out_W=1, out_C=3,
        conv_filters=16, conv_kernel=(3,1),
        encoder_lstm_units=128, decoder_lstm_units=128
    )
    model.summary()

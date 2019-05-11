import keras.backend as K
from keras.layers import (
    TimeDistributed, 
    Dense, 
    Concatenate, 
    Conv1D, 
    Add, 
    BatchNormalization
)
import numpy as np


def scaled_dot_prod_attention(q, k, v):
    #k -> [batch_size, seq_len, d_k]
    d_k = k.output_shape[-1]
    k_T = K.permute_dimensions(k, pattern=[0, 2, 1])
    
    return K.batch_dot(K.softmax(K.batch_dot(q, k_T) / K.sqrt(d_k)), v)


def multi_head_attention(q, k, v, d_model, h=8):
    d_k = d_v = d_model // h

    heads = []
    for i in range(h):
        q_proj = TimeDistributed(Dense(units=d_k, activation='linear', use_bias=False)(q))
        k_proj = TimeDistributed(Dense(units=d_k, activation='linear', use_bias=False)(k))
        v_proj = TimeDistributed(Dense(units=d_v, activation='linear', use_bias=False)(v))

        heads.append(
            scaled_dot_prod_attention(q_proj, k_proj, v_proj))

    concat = Concatenate()(heads)
    proj = TimeDistributed(Dense(d_model, activation='linear', use_bias=False)(concat))

    return proj


def feed_forward_net(x, d_ff=2048):
    #x -> [batch_size, seq_len, d_model]
    d_model = x.output_shape[-1]

    transform1 = Conv1D(filters=d_ff, kernel_size=1, activation='relu')(x)
    transform2 = Conv1D(filters=d_model, kernel_size=1, activation='linear')(transform1)

    return transform2


def get_pos_encoding(d_model, seq_len):
    pos = np.arange(d_model)

    return [np.sin(pos / (10000 ** (2 * i / d_model))) if i % 2 == 0 else np.cos(pos / (10000 ** (2 * i / d_model))) 
        for i in range(seq_len)]


def encoder(x):
    d_model = x.output_shape[-1]
    mha = multi_head_attention(x, x, x, d_model)

    add = Add()([mha, x])
    norm = BatchNormalization()(add)

    ffn = feed_forward_net(norm)

    add = Add()([ffn, norm])
    norm = BatchNormalization()(add)

    return norm
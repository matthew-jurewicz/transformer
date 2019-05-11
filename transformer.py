import keras.backend as K
from keras.layers import (
    Dense, 
    TimeDistributed, 
    Concatenate
)


def scaled_dot_prod_attention(q, k, v):
    d_k = k.output_shape[0]
    
    return K.dot(K.softmax(K.dot(q, K.transpose(k)) / K.sqrt(d_k)), v)


def multi_head_attention(q, k, v, d_model=512, h=8):
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
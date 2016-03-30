# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops, math_ops, nn_ops
from tensorflow.python.ops import variable_scope as vs

import cells

VINYALS_KAISER = 'vinyals_kayser'
LUONG_GENERAL = 'luong_general'
LUONG_DOT = 'luong_dot'
MOD_VINYALS_KAISER = 'modified_vinyals_kayser'
MOD_BAHDANAU = 'modified_bahdanau'
BAHDANAU_NMT = 'bahdanau_nmt'
DECODER_TYPE_1 = 'decoder_type_1'
DECODER_TYPE_2 = 'decoder_type_2'


def get_decoder_content_f(name):
    if name == DECODER_TYPE_1:
        return decoder_type_1
    else:
        return decoder_type_2


def get_content_f(name):
    if name == LUONG_GENERAL:
        return luong_general
    elif name == LUONG_DOT:
        return luong_dot
    elif name == MOD_BAHDANAU:
        return mod_bahdanau
    elif name == MOD_VINYALS_KAISER:
        return mod_vinyals_kayser
    elif name == BAHDANAU_NMT:
        return bahdanau_nmt
    else:
        return mod_vinyals_kayser

def decoder_type_1(decoder_hidden, attn_size, initializer=None):

    with vs.variable_scope("decoder_type_1", initializer=initializer):

        k = vs.get_variable("AttnDecW_%d" % 0, [1, 1, attn_size, 1], initializer=initializer)
        hidden_features = nn_ops.conv2d(decoder_hidden, k, [1, 1, 1, 1], "SAME")

        # s will be (?, timesteps)
        s = math_ops.reduce_sum(math_ops.tanh(hidden_features), [2, 3])

    return s


def decoder_type_2(decoder_hidden, attn_size, initializer=None):

    with vs.variable_scope("decoder_type_2", initializer=initializer):

        k = vs.get_variable("AttnDecW_%d" % 0, [1, 1, attn_size, attn_size], initializer=initializer)
        hidden_features = nn_ops.conv2d(decoder_hidden, k, [1, 1, 1, 1], "SAME")
        v = vs.get_variable("AttnDecV_%d" % 0, [attn_size])

        # s will be (?, timesteps)
        s = math_ops.reduce_sum((v * math_ops.tanh(hidden_features)), [2, 3])

    return s


def bahdanau_nmt(hidden, decoder_previous_state, initializer=None):
    # size of decoder layers
    attention_vec_size = hidden.get_shape()[3].value
    decoder_size = decoder_previous_state.get_shape()[1].value

    with vs.variable_scope("bahdanau_nmt", initializer=initializer):
        # here we calculate the W_a * s_i-1 (W1 * h_1) part of the attention alignment
        k = vs.get_variable("AttnW_%d" % 0, [1, 1, attention_vec_size, attention_vec_size], initializer=initializer)
        hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
        va = vs.get_variable("AttnV_%d" % 0, [attention_vec_size], initializer=initializer)

        y = cells.linear(decoder_previous_state, decoder_size, True)
        y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])

        # Attention mask is a softmax of v^T * tanh(...).
        s = math_ops.reduce_sum(va * math_ops.tanh(hidden_features + y), [2, 3])

    return s


def luong_dot(hidden, decoder_hidden_state, initializer=None):

    with vs.variable_scope("luong_dot", initializer=initializer):

        s = math_ops.reduce_sum((hidden * decoder_hidden_state), [2, 3])

    return s


def luong_general(hidden, decoder_hidden_state, initializer=None):

    # size of decoder layers
    attention_vec_size = hidden.get_shape()[3].value

    with vs.variable_scope("luong_general", initializer=initializer):

        # here we calculate the W_a * s_i-1 (W1 * h_1) part of the attention alignment
        k = vs.get_variable("AttnW_%d" % 0, [1, 1, attention_vec_size, attention_vec_size], initializer=initializer)
        hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
        s = math_ops.reduce_sum((hidden_features * decoder_hidden_state), [2, 3])

    return s


def mod_bahdanau(hidden, decoder_hidden_state, initializer=None):

    # size of decoder layers
    attention_vec_size = hidden.get_shape()[3].value

    with vs.variable_scope("mod_bahdanau", initializer=initializer):

        k = vs.get_variable("AttnW_%d" % 0, [1, 1, attention_vec_size, 1], initializer=initializer)
        hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")

        y = cells.linear(decoder_hidden_state, 1, True)
        y = array_ops.reshape(y, [-1, 1, 1, 1])

        # Attention mask is a softmax of v^T * tanh(...).
        s = math_ops.reduce_sum(math_ops.tanh(hidden_features + y), [2, 3])

    return s


def mod_vinyals_kayser(hidden, decoder_hidden_state, initializer=None):

    # size of decoder layers
    attention_vec_size = hidden.get_shape()[3].value

    with vs.variable_scope("mod_vinyals_kayser", initializer=initializer):

        k = vs.get_variable("AttnW_%d" % 0, [1, 1, attention_vec_size, 1], initializer=initializer)
        hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")

        y = cells.linear(decoder_hidden_state, 1, True)
        y = array_ops.reshape(y, [-1, 1, 1, 1])

        # Attention mask is a softmax of v^T * tanh(...).
        s = math_ops.reduce_sum(math_ops.tanh(hidden_features + y), [2, 3])

    return s


def vinyals_kaiser(hidden, decoder_hidden_state, initializer=None):

    # size of decoder layers
    attention_vec_size = hidden.get_shape()[3].value

    with vs.variable_scope("vinyals_kaiser", initializer=initializer):

        # here we calculate the W_a * s_i-1 (W1 * h_1) part of the attention alignment
        k = vs.get_variable("AttnW_%d" % 0, [1, 1, attention_vec_size, attention_vec_size], initializer=initializer)
        hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
        va = vs.get_variable("AttnV_%d" % 0, [attention_vec_size], initializer=initializer)

        y = cells.linear(decoder_hidden_state, attention_vec_size, True)
        y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])

        # Attention mask is a softmax of v^T * tanh(...).
        s = math_ops.reduce_sum(va * math_ops.tanh(hidden_features + y), [2, 3])

    return s


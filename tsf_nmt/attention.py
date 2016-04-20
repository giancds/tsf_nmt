# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import array_ops, math_ops, nn_ops
from tensorflow.python.ops import variable_scope as vs

import cells
from content_functions import vinyals_kaiser

GLOBAL = "global"
LOCAL = "local"
HYBRID = "hybrid"


def get_attention_f(name):
    if name == LOCAL:
        return local_attention
    elif name == HYBRID:
        return hybrid_attention
    else:
        return global_attention


def hybrid_attention(decoder_hidden_state, hidden_attn, initializer, window_size=10,
                     content_function=vinyals_kaiser, dtype=tf.float32):
    """Put hybrid attention (mix of global and local attention) on hidden using decoder hidden states
    and the hidden states of encoder (hidden_attn).

        Parameters
        ----------
        decoder_hidden_state : 2-D Tensor
            Tensor representing the current hidden state of the decoder (output of the recurrent layers).
            Shape is (?, decoder_size).
        hidden_attn : 4-D Tensor
            Tensor representing the hidden states of the encoder (output of the recurrent layers). It has
            shape (?, timesteps, 1, decoder_sdize) so it is possible to apply a 1-D convolution to calculate
            the attention score more efficiently.
        initializer : function
            Function to use when initializing variables within the variables context.
        window_size : int
            Size of each side of the window to use when applying local attention. Not relevant to global
            attention. Default to 10.
        content_function : function
            Content function to score the decoder hidden states and encoder hidden states to extract their
            weights. Default to 'vinyals_kaiser'.
        dtype : tensorflow dtype
            Type of tensors. Default to tf.float32

        Returns
        -------
        ds : 2-D Tensor
            Tensor representing the context vector generated after scoring the encoder and decoder hidden
            states. Has shape (?, decoder_size), i.e., one context vector per batch sample.

    """
    assert content_function is not None

    attention_vec_size = hidden_attn.get_shape()[3].value

    local_attn = local_attention(decoder_hidden_state=decoder_hidden_state,
                                 hidden_attn=hidden_attn,
                                 content_function=content_function,
                                 window_size=window_size, initializer=initializer, dtype=dtype)

    global_attn = global_attention(decoder_hidden_state=decoder_hidden_state,
                                   hidden_attn=hidden_attn,
                                   content_function=content_function,
                                   window_size=window_size, initializer=initializer, dtype=dtype)

    with vs.variable_scope("FeedbackGate_%d" % 0, initializer=initializer):
        y = cells.linear(decoder_hidden_state, attention_vec_size, True)
        y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])

        vb = vs.get_variable("FeedbackVb_%d" % 0, [attention_vec_size], initializer=initializer)

        # tanh(Wp*ht)
        tanh = math_ops.tanh(y)
        beta = math_ops.sigmoid(math_ops.reduce_sum((vb * tanh), [2, 3]))

        _ = tf.histogram_summary('hybrid_beta_weights', beta)

        attns = beta * global_attn + (1 - beta) * local_attn

    return attns


def global_attention(decoder_hidden_state, hidden_attn, initializer, window_size=10,
                     content_function=vinyals_kaiser, dtype=tf.float32):

    """Put global attention on hidden using decoder hidden states and the hidden states of encoder (hidden_attn).

    Parameters
    ----------
    decoder_hidden_state : 2-D Tensor
        Tensor representing the current hidden state of the decoder (output of the recurrent layers).
        Shape is (?, decoder_size).
    hidden_attn : 4-D Tensor
        Tensor representing the hidden states of the encoder (output of the recurrent layers). It has
        shape (?, timesteps, 1, decoder_sdize) so it is possible to apply a 1-D convolution to calculate
        the attention score more efficiently.
    initializer : function
        Function to use when initializing variables within the variables context.
    window_size : int
        Size of each side of the window to use when applying local attention. Not relevant to global
        attention. Default to 10.
    content_function : function
        Content function to score the decoder hidden states and encoder hidden states to extract their
        weights. Default to 'vinyals_kaiser'.
    dtype : tensorflow dtype
        Type of tensors. Default to tf.float32

    Returns
    -------
    ds : 2-D Tensor
        Tensor representing the context vector generated after scoring the encoder and decoder hidden
        states. Has shape (?, decoder_size), i.e., one context vector per batch sample.

    """
    assert content_function is not None

    attention_vec_size = hidden_attn.get_shape()[3].value
    attn_length = hidden_attn.get_shape()[1].value

    with vs.variable_scope("AttentionGlobal", initializer=initializer):

        # apply content function to score the hidden states from the encoder
        s = content_function(hidden_attn, decoder_hidden_state)

        alpha = nn_ops.softmax(s)

        _ = tf.histogram_summary('global_alpha_weights', alpha)

        # Now calculate the attention-weighted vector d.
        d = math_ops.reduce_sum(array_ops.reshape(alpha, [-1, attn_length, 1, 1]) * hidden_attn, [1, 2])
        ds = array_ops.reshape(d, [-1, attention_vec_size])#

    _ = tf.histogram_summary('global_attention_context', ds)

    return ds


def local_attention(decoder_hidden_state, hidden_attn, initializer, window_size=10,
                    content_function=vinyals_kaiser, dtype=tf.float32):
    """Put local attention on hidden using decoder hidden states and the hidden states of encoder (hidden_attn).

    Parameters
    ----------
    decoder_hidden_state : 2-D Tensor
        Tensor representing the current hidden state of the decoder (output of the recurrent layers).
        Shape is (?, decoder_size).
    hidden_attn : 4-D Tensor
        Tensor representing the hidden states of the encoder (output of the recurrent layers). It has
        shape (?, timesteps, 1, decoder_sdize) so it is possible to apply a 1-D convolution to calculate
        the attention score more efficiently.
    initializer : function
        Function to use when initializing variables within the variables context.
    window_size : int
        Size of each side of the window to use when applying local attention. Not relevant to global
        attention. Default to 10.
    content_function : function
        Content function to score the decoder hidden states and encoder hidden states to extract their
        weights. Default to 'vinyals_kaiser'.
    dtype : tensorflow dtype
        Type of tensors. Default to tf.float32

    Returns
    -------
    ds : 2-D Tensor
        Tensor representing the context vector generated after scoring the encoder and decoder hidden
        states. Has shape (?, decoder_size), i.e., one context vector per batch sample.

    """
    assert content_function is not None
    sigma = window_size / 2
    denominator = sigma ** 2

    attention_vec_size = hidden_attn.get_shape()[3].value
    attn_length = hidden_attn.get_shape()[1].value

    batch_size = array_ops.shape(hidden_attn)[0]

    with vs.variable_scope("AttentionLocal", initializer=initializer):

        # apply content function to score the hidden states from the encoder
        s = content_function(hidden_attn, decoder_hidden_state)

        with vs.variable_scope("WindowPrediction", initializer=initializer):
            ht = cells.linear([decoder_hidden_state], attention_vec_size, True)

        # get the parameters (vp)
        vp = vs.get_variable("AttnVp_%d" % 0, [attention_vec_size], initializer=initializer)

        # tanh(Wp*ht)
        tanh = math_ops.tanh(ht)
        # S * sigmoid(vp * tanh(Wp*ht))  - this is going to return a number
        # for each sentence in the batch - i.e., a tensor of shape batch x 1
        S = attn_length
        pt = math_ops.reduce_sum((vp * tanh), [2, 3])
        pt = math_ops.sigmoid(pt) * S

        # now we get only the integer part of the values
        pt = tf.floor(pt)

        _ = tf.histogram_summary('local_window_predictions', pt)

        # we now create a tensor containing the indices representing each position
        # of the sentence - i.e., if the sentence contain 5 tokens and batch_size is 3,
        # the resulting tensor will be:
        # [[0, 1, 2, 3, 4]
        #  [0, 1, 2, 3, 4]
        #  [0, 1, 2, 3, 4]]
        #
        indices = []
        for pos in xrange(attn_length):
            indices.append(pos)
        indices = indices * batch_size
        idx = tf.convert_to_tensor(tf.to_float(indices), dtype=dtype)
        idx = tf.reshape(idx, [-1, attn_length])

        # here we calculate the boundaries of the attention window based on the ppositions
        low = pt - window_size + 1  # we add one because the floor op already generates the first position
        high = pt + window_size

        # here we check our positions against the boundaries
        mlow = tf.to_float(idx < low)
        mhigh = tf.to_float(idx > high)

        # now we combine both into a pre-mask that has 0s and 1s switched
        # i.e, at this point, True == 0 and False == 1
        m = mlow + mhigh  # batch_size

        # here we switch the 0s to 1s and the 1s to 0s
        # we correct the values so True == 1 and False == 0
        mask = tf.to_float(tf.equal(m, 0.0))

        # here we switch off all the values that fall outside the window
        # first we switch off those in the truncated normal
        alpha = s * mask
        masked_soft = nn_ops.softmax(alpha)

        _ = tf.histogram_summary('local_alpha_weights', alpha)

        # here we calculate the 'truncated normal distribution'
        numerator = -tf.pow((idx - pt), tf.convert_to_tensor(2, dtype=dtype))
        div = tf.truediv(numerator, denominator)
        e = math_ops.exp(div)  # result of the truncated normal distribution

        at = masked_soft * e

        # Now calculate the attention-weighted vector d.
        d = math_ops.reduce_sum(
                array_ops.reshape(at, [-1, attn_length, 1, 1]) * hidden_attn,
                [1, 2])
        ds = array_ops.reshape(d, [-1, attention_vec_size])

    _ = tf.histogram_summary('local_attention_context', ds)

    return ds
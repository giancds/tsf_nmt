# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn.rnn_cell import RNNCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs


class GRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, input_size, num_units, initializer=None):
        self._num_units = num_units
        self._input_size = input_size
        self._initializer = initializer

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, initializer=None, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope(scope or type(self).__name__, initializer=self._initializer):  # "GRUCell"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not udpate.
                r, u = tf.split(1, 2, linear([inputs, state], 2 * self._num_units, True, 1.0))
                r, u = tf.sigmoid(r), tf.sigmoid(u)
            with tf.variable_scope("Candidate"):
                c = tf.tanh(linear([inputs, r * state], self._num_units, True))
            new_h = u * state + (1 - u) * c
        return new_h, new_h


def linear(args, output_size, bias, bias_start=0.0, initializer=None, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      initializer: weight initializer. If None, random_uniform_initializer will be used.
      scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    assert args
    if not isinstance(args, (list, tuple)):
        args = [args]

    if initializer is None:
        initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1, seed=1234)

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with vs.variable_scope(scope or "Linear", initializer=initializer):
        matrix = vs.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = math_ops.matmul(args[0], matrix)
        else:
            res = math_ops.matmul(array_ops.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = vs.get_variable(
            "Bias", [output_size],
            initializer=init_ops.constant_initializer(bias_start))
    return res + bias_term


def build_nmt_multicell_rnn(num_layers_encoder, num_layers_decoder, encoder_size, decoder_size,
                                source_proj_size, target_proj_size, use_lstm=True, input_feeding=True,
                                dropout=0.0):


        if use_lstm:
            cell_class = rnn_cell.LSTMCell
        else:
            cell_class = GRUCell

        initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1, seed=1234)

        encoder_cell = cell_class(num_units=encoder_size, input_size=source_proj_size, initializer=initializer)
        if input_feeding:
            decoder_cell0 = cell_class(num_units=decoder_size, input_size=decoder_size * 2, initializer=initializer)
        else:
            decoder_cell0 = cell_class(num_units=decoder_size, input_size=decoder_size, initializer=initializer)
        decoder_cell1 = cell_class(num_units=decoder_size, input_size=decoder_size, initializer=initializer)

        # if dropout > 0.0:  # if dropout is 0.0, it is turned off
        encoder_cell = rnn_cell.DropoutWrapper(encoder_cell, output_keep_prob=1.0 - dropout)
        decoder_cell0 = rnn_cell.DropoutWrapper(decoder_cell0, output_keep_prob=1.0 - dropout)
        decoder_cell1 = rnn_cell.DropoutWrapper(decoder_cell1, output_keep_prob=1.0 - dropout)

        encoder_rnncell = rnn_cell.MultiRNNCell([encoder_cell] * num_layers_encoder)
        decoder_rnncell = rnn_cell.MultiRNNCell([decoder_cell0] + [decoder_cell1] * (num_layers_decoder - 1))

        return encoder_rnncell, decoder_rnncell


def build_lm_multicell_rnn(num_layers, hidden_size, proj_size, use_lstm=True, dropout=0.0):

    if use_lstm:
        cell_class = rnn_cell.LSTMCell
    else:
        cell_class = GRUCell

    if num_layers > 1:

        lm_cell0 = cell_class(num_units=hidden_size, input_size=proj_size)
        lm_cell1 = cell_class(num_units=hidden_size, input_size=hidden_size)

        if dropout > 0.0:  # if dropout is 0.0, it is turned off
            lm_cell0 = rnn_cell.DropoutWrapper(lm_cell0, output_keep_prob=1.0-dropout)
            lm_cell1 = rnn_cell.DropoutWrapper(lm_cell1, output_keep_prob=1.0-dropout)

        lm_rnncell = rnn_cell.MultiRNNCell([lm_cell0] + [lm_cell1] * num_layers)
    else:

        lm_cell0 = cell_class(num_units=hidden_size, input_size=proj_size)

        if dropout > 0.0:  # if dropout is 0.0, it is turned off
            lm_cell0 = rnn_cell.DropoutWrapper(lm_cell0, output_keep_prob=1.0-dropout)

        lm_rnncell = lm_cell0

    return lm_rnncell
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.models.rnn.rnn_cell import RNNCell, linear


class GRU(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, input_size, num_units):
        self._num_units = num_units
        self._input_size = input_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not udpate.
                r, u = tf.split(1, 2, linear([inputs, state], 2 * self._num_units, True, 1.0))
                r, u = tf.sigmoid(r), tf.sigmoid(u)
            with tf.variable_scope("Candidate"):
                c = tf.tanh(linear([inputs, r * state], self._num_units, True))
            new_h = u * state + (1 - u) * c
        return new_h, new_h

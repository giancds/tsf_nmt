# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn.rnn_cell import RNNCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs


def _reverse_seq(input_seq, lengths):
    """Reverse a list of Tensors up to specified lengths.

    Args:
      input_seq: Sequence of seq_len tensors of dimension (batch_size, depth)
      lengths:   A tensor of dimension batch_size, containing lengths for each
                 sequence in the batch. If "None" is specified, simply reverses
                 the list.

    Returns:
      time-reversed sequence
    """
    if lengths is None:
        return list(reversed(input_seq))

    for input_ in input_seq:
        input_.set_shape(input_.get_shape().with_rank(2))

    # Join into (time, batch_size, depth)
    s_joined = array_ops.pack(input_seq)

    # Reverse along dimension 0
    s_reversed = array_ops.reverse_sequence(s_joined, lengths, 0, 1)
    # Split again into list
    result = array_ops.unpack(s_reversed)
    return result


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


class GRUCellCond(RNNCell):
    """Gated Recurrent Unit cell conditioned on the context (cf. http://arxiv.org/abs/1406.1078)."""

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

    def __call__(self, inputs, state, context=None, initializer=None, scope=None):
        """Gated recurrent unit conditioned on the context (GRU) with nunits cells."""
        with tf.variable_scope(scope or type(self).__name__, initializer=self._initializer):  # "GRUCell"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not udpate.
                r, u = tf.split(1, 2, linear([inputs, state, context], 2 * self._num_units, True, 1.0))
                r, u = tf.sigmoid(r), tf.sigmoid(u)
            with tf.variable_scope("Candidate"):
                c = tf.tanh(linear([inputs, r * state, context], self._num_units, True))
            new_h = u * state + (1 - u) * c
        return new_h, new_h


class DropoutWrapperCond(RNNCell):
    """Operator adding dropout to inputs and outputs of the given cell."""

    def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
                 seed=None):
        """Create a cell with added input and/or output dropout.

        Dropout is never used on the state.

        Args:
          cell: an RNNCell, a projection to output_size is added to it.
          input_keep_prob: unit Tensor or float between 0 and 1, input keep
            probability; if it is float and 1, no input dropout will be added.
          output_keep_prob: unit Tensor or float between 0 and 1, output keep
            probability; if it is float and 1, no output dropout will be added.
          seed: (optional) integer, the randomness seed.

        Raises:
          TypeError: if cell is not an RNNCell.
          ValueError: if keep_prob is not between 0 and 1.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")
        if (isinstance(input_keep_prob, float) and
                not (input_keep_prob >= 0.0 and input_keep_prob <= 1.0)):
            raise ValueError("Parameter input_keep_prob must be between 0 and 1: %d"
                             % input_keep_prob)
        if (isinstance(output_keep_prob, float) and
                not (output_keep_prob >= 0.0 and output_keep_prob <= 1.0)):
            raise ValueError("Parameter input_keep_prob must be between 0 and 1: %d"
                             % output_keep_prob)
        self._cell = cell
        self._input_keep_prob = input_keep_prob
        self._output_keep_prob = output_keep_prob
        self._seed = seed

    @property
    def input_size(self):
        return self._cell.input_size

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def state_size(self):
        return self._cell.state_size

    def __call__(self, inputs, state, context=None, scope=None):
        """Run the cell with the declared dropouts."""
        if (not isinstance(self._input_keep_prob, float) or
                    self._input_keep_prob < 1):
            inputs = nn_ops.dropout(inputs, self._input_keep_prob, seed=self._seed)
        output, new_state = self._cell(inputs, state, context)
        if (not isinstance(self._output_keep_prob, float) or
                    self._output_keep_prob < 1):
            output = nn_ops.dropout(output, self._output_keep_prob, seed=self._seed)
        return output, new_state


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
    assert args is not None
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


def bidirectional_rnn(cell_fw, cell_bw, inputs,
                      initial_state_fw=None, initial_state_bw=None,
                      dtype=None, sequence_length=None, scope=None):
    """Creates a bidirectional recurrent neural network.
    Similar to the unidirectional case above (rnn) but takes input and builds
    independent forward and backward RNNs with the final forward and backward
    outputs depth-concatenated, such that the output will have the format
    [time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
    forward and backward cell must match. The initial state for both directions
    is zero by default (but can be set optionally) and no intermediate states are
    ever returned -- the network is fully unrolled for the given (passed in)
    length(s) of the sequence(s) or completely unrolled if length(s) is not given.
    Args:
      cell_fw: An instance of RNNCell, to be used for forward direction.
      cell_bw: An instance of RNNCell, to be used for backward direction.
      inputs: A length T list of inputs, each a tensor of shape
        [batch_size, cell.input_size].
      initial_state_fw: (optional) An initial state for the forward RNN.
        This must be a tensor of appropriate type and shape
        [batch_size x cell.state_size].
      initial_state_bw: (optional) Same as for initial_state_fw.
      dtype: (optional) The data type for the initial state.  Required if either
        of the initial states are not provided.
      sequence_length: (optional) An int32/int64 vector, size [batch_size],
        containing the actual lengths for each of the sequences.
      scope: VariableScope for the created subgraph; defaults to "BiRNN"
    Returns:
      A tuple (outputs, output_state_fw, output_state_bw) where:
        outputs is a length T list of outputs (one for each input), which
        are depth-concatenated forward and backward outputs
        output_state_fw is the final state of the forward rnn
        output_state_bw is the final state of the backward rnn
    Raises:
      TypeError: If "cell_fw" or "cell_bw" is not an instance of RNNCell.
      ValueError: If inputs is None or an empty list.
    """

    if not isinstance(cell_fw, rnn_cell.RNNCell):
        raise TypeError("cell_fw must be an instance of RNNCell")
    if not isinstance(cell_bw, rnn_cell.RNNCell):
        raise TypeError("cell_bw must be an instance of RNNCell")
    if not isinstance(inputs, list):
        raise TypeError("inputs must be a list")
    if not inputs:
        raise ValueError("inputs must not be empty")

    name = scope or "BiRNN"
    # Forward direction
    with vs.variable_scope(name + "_FW") as fw_scope:
        output_fw, output_state_fw = rnn.rnn(cell_fw, inputs, initial_state_fw, dtype,
                                         sequence_length, scope=fw_scope)

    # Backward direction
    with vs.variable_scope(name + "_BW") as bw_scope:
        tmp, output_state_bw = rnn.rnn(cell_bw, _reverse_seq(inputs, sequence_length),
                                   initial_state_bw, dtype, sequence_length, scope=bw_scope)
    output_bw = _reverse_seq(tmp, sequence_length)
    # Concat each of the forward/backward outputs
    outputs = [array_ops.concat(1, [fw, bw])
               for fw, bw in zip(output_fw, output_bw)]

    return (outputs, output_state_fw, output_state_bw)


def build_nmt_multicell_rnn(num_layers_encoder, num_layers_decoder, encoder_size, decoder_size,
                            source_proj_size, use_lstm=True, input_feeding=True,
                            dropout=0.0):

        if use_lstm:
            print("I'm building the model with LSTM cells")
            cell_class = rnn_cell.LSTMCell
        else:
            print("I'm building the model with GRU cells")
            cell_class = GRUCell

        initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1, seed=1234)

        encoder_cell = cell_class(num_units=encoder_size, input_size=source_proj_size, initializer=initializer)

        if input_feeding:
            decoder_cell0 = cell_class(num_units=decoder_size, input_size=decoder_size * 2, initializer=initializer)
        else:
            decoder_cell0 = cell_class(num_units=decoder_size, input_size=decoder_size, initializer=initializer)

        # if dropout > 0.0:  # if dropout is 0.0, it is turned off
        encoder_cell = rnn_cell.DropoutWrapper(encoder_cell, output_keep_prob=1.0 - dropout)
        encoder_rnncell = rnn_cell.MultiRNNCell([encoder_cell] * num_layers_encoder)

        decoder_cell0 = rnn_cell.DropoutWrapper(decoder_cell0, output_keep_prob=1.0 - dropout)
        if num_layers_decoder > 1:
            decoder_cell1 = cell_class(num_units=decoder_size, input_size=decoder_size, initializer=initializer)
            decoder_cell1 = rnn_cell.DropoutWrapper(decoder_cell1, output_keep_prob=1.0 - dropout)
            decoder_rnncell = rnn_cell.MultiRNNCell([decoder_cell0] + [decoder_cell1] * (num_layers_decoder - 1))

        else:

            decoder_rnncell = rnn_cell.MultiRNNCell([decoder_cell0])

        return encoder_rnncell, decoder_rnncell


def build_nmt_bidirectional_cell(encoder_size, decoder_size, source_proj_size, target_proj_size, dropout=0.0):

        initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1, seed=1234)

        encoder_cell_fw = GRUCell(num_units=encoder_size, input_size=source_proj_size, initializer=initializer)
        encoder_cell_bw = GRUCell(num_units=encoder_size, input_size=source_proj_size, initializer=initializer)

        decoder_cell = GRUCellCond(num_units=decoder_size, input_size=target_proj_size, initializer=initializer)

        # if dropout > 0.0:  # if dropout is 0.0, it is turned off
        encoder_cell_fw = DropoutWrapperCond(encoder_cell_fw, output_keep_prob=1.0 - dropout)
        encoder_cell_bw = DropoutWrapperCond(encoder_cell_bw, output_keep_prob=1.0 - dropout)
        decoder_cell = DropoutWrapperCond(decoder_cell, output_keep_prob=1.0 - dropout)

        return encoder_cell_fw, encoder_cell_bw, decoder_cell
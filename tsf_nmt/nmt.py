import data_utils
import tensorflow as tf
from tensorflow.models.rnn import rnn, linear, seq2seq
from tensorflow.models.rnn.rnn_cell import RNNCell

flags = tf.app.flags
FLAGS = flags.FLAGS


@tf.ops.RegisterGradient("Reverse")
def _tf_reverse_grad(op, grad):
    """
    Method to calculate gradients for the Reverse op. Taken from a suggestion to fix the issue
    #58 of TensorFlow. Link

        https://github.com/tensorflow/tensorflow/issues/58

    Parameters
    ----------
    op
    grad

    Returns
    -------

    """
    reverse_dims = op.inputs[1]
    return tf.array_ops.reverse(grad, reverse_dims), None


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
                r, u = tf.split(1, 2, linear.linear([inputs, state],
                                                    2 * self._num_units, True, 1.0))
                r, u = tf.sigmoid(r), tf.sigmoid(u)
            with tf.variable_scope("Candidate"):
                c = tf.tanh(linear.linear([inputs, r * state], self._num_units, True))
            new_h = u * state + (1 - u) * c
        return new_h, new_h


def bidirectional_model_with_buckets(encoder_inputs, encoder_inputs_r, decoder_inputs, targets,
                                     weights, buckets, num_decoder_symbols, seq2seq_f,
                                     softmax_loss_function=None, name=None):
    """
    Parameters
    ----------
        encoder_inputs: a list of Tensors to feed the encoder; first seq2seq input.

        encoder_inputs_r: a list of Tensors to feed the encoder; second seq2seq input.

        decoder_inputs: a list of Tensors to feed the decoder; third seq2seq input.

        targets: a list of 1D batch-sized int32-Tensors (desired output sequence).

        weights: list of 1D batch-sized float-Tensors to weight the targets.

        buckets: a list of pairs of (input size, output size) for each bucket.

        num_decoder_symbols: integer, number of decoder symbols (output classes).

        seq2seq_f: a sequence-to-sequence model function; it takes 2 input that
            agree with encoder_inputs and decoder_inputs, and returns a pair
            consisting of outputs and states (as, e.g., basic_rnn_seq2seq).

        softmax_loss_function: function (inputs-batch, labels-batch) -> loss-batch
            to be used instead of the standard softmax (the default if this is None).

        name: optional name for this operation, defaults to "model_with_buckets".

    Returns
    -------
        outputs: The outputs for each bucket. Its j'th element consists of a list
            of 2D Tensors of shape [batch_size x num_decoder_symbols] (j'th outputs).

        losses: List of scalar Tensors, representing losses for each bucket.

    Raises
    ------
        ValueError: if length of encoder_inputsut, targets, or weights is smaller
            than the largest (last) bucket.
    """
    if len(encoder_inputs) < buckets[-1][0]:
        raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                         "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
    if len(targets) < buckets[-1][1]:
        raise ValueError("Length of targets (%d) must be at least that of last"
                         "bucket (%d)." % (len(targets), buckets[-1][1]))
    if len(weights) < buckets[-1][1]:
        raise ValueError("Length of weights (%d) must be at least that of last"
                         "bucket (%d)." % (len(weights), buckets[-1][1]))

    all_inputs = encoder_inputs + encoder_inputs_r + decoder_inputs + targets + weights
    losses = []
    outputs = []
    with tf.op_scope(all_inputs, name, "model_with_buckets"):
        for j in xrange(len(buckets)):
            if j > 0:
                tf.get_variable_scope().reuse_variables()

            bucket_encoder_inputs = [encoder_inputs[i] for i in xrange(buckets[j][0])]

            bucket_encoder_inputs_r = [encoder_inputs_r[i] for i in xrange(buckets[j][0])]

            bucket_decoder_inputs = [decoder_inputs[i] for i in xrange(buckets[j][1])]

            bucket_outputs, _ = seq2seq_f(bucket_encoder_inputs,
                                          bucket_encoder_inputs_r,
                                          bucket_decoder_inputs)

            outputs.append(bucket_outputs)

            bucket_targets = [targets[i] for i in xrange(buckets[j][1])]
            bucket_weights = [weights[i] for i in xrange(buckets[j][1])]
            losses.append(seq2seq.sequence_loss(
                outputs[-1], bucket_targets, bucket_weights, num_decoder_symbols,
                softmax_loss_function=softmax_loss_function))

    return outputs, losses


def bidirectional_encoder(source, source_r, source_vocab_size, source_proj_size,
                          encoder_cell, encoder_cell_r, encoder_size, batch_size,
                          dtype=tf.float32):
    """

    Parameters
    ----------
    source
    source_r
    source_vocab_size
    source_proj_size
    encoder_cell
    encoder_cell_r
    encoder_size
    batch_size
    dtype

    Returns
    -------

    """
    # source_r = [tf.reverse(s, [False, True]) for s in source]

    with tf.device("/cpu:0"):
        src_embedding = tf.Variable(
            tf.truncated_normal(
                [source_vocab_size, source_proj_size], stddev=0.01
            ),
            name='embedding_src'
        )

        # with tf.Graph.device("/cpu:0"):
        # get the embeddings
        emb_inp = [tf.nn.embedding_lookup(src_embedding, s) for s in source]
        emb_inpr = [tf.nn.embedding_lookup(src_embedding, r) for r in source_r]

    initial_state = encoder_cell.zero_state(batch_size=batch_size, dtype=dtype)
    initial_state_r = encoder_cell_r.zero_state(batch_size=batch_size, dtype=dtype)

    fwd_outputs, fwd_states = rnn.rnn(encoder_cell, emb_inp,
                                      initial_state=initial_state,
                                      dtype=dtype,
                                      scope='fwd_encoder')

    bkw_outputs, bkw_states = rnn.rnn(encoder_cell_r, emb_inpr,
                                      initial_state=initial_state_r,
                                      dtype=dtype,
                                      scope='bkw_encoder')

    # revert the reversed sentence states to concatenate
    # the second parameter is a list of dimensions to revert - False means does not change
    # True means revert -We are reverting just the data dimension while time/bach stay equal
    bkw_outputs = [tf.reverse(b, [False, True]) for b in bkw_outputs]

    # concatenates the forward and backward annotations
    context = []
    for f, b in zip(fwd_outputs, bkw_outputs):
        context.append(tf.concat(0, [f, b]))

    # define the initial state for the decoder
    # first create the weights and biases
    Ws = tf.Variable(
        tf.truncated_normal(
            [encoder_size, encoder_size], stddev=0.01
        ),
        name='Ws'
    )

    bs = tf.Variable(
        tf.truncated_normal([encoder_size], stddev=0.01),
        name='bs'
    )

    # get the last hidden state of the encoder in the backward process
    h1 = bkw_states[-1]

    # perform tanh((Ws * h) + b0) step
    decoder_initial_state = tf.nn.tanh(tf.nn.xw_plus_b(h1, Ws, bs))

    return context, decoder_initial_state


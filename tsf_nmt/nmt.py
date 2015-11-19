import data_utils
import tensorflow as tf
from tensorflow.models.rnn import rnn, seq2seq
from tensorflow.models.rnn.rnn_cell import GRUCell

flags = tf.app.flags
FLAGS = flags.FLAGS

_DTYPE = tf.float32

# flags definition
flags.DEFINE_integer('source_v_size', 30000, 'Source vocabulary size.')
flags.DEFINE_integer('target_v_size', 30000, 'Target vocabulary size.')

flags.DEFINE_integer('encoder_size', 100, 'Number of hidden units in the decoder.')
flags.DEFINE_integer('decoder_size', 100, 'NUmber of hidden units in the encoder.')

flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_float('learning_rate_decay', 0.001, 'Decay learning_rate by this much when needed.')

flags.DEFINE_integer('source_projection_size', 10, 'Dimension of words projection in the encoder.')
flags.DEFINE_integer('target_projection_size', 10, 'Dimension of words projection in the decoder.')

# encoder/decoder cell types
_ENCODER_CELL = GRUCell(FLAGS.encoder_size)
_DECODER_CELL = GRUCell(FLAGS.decoder_size)

# learning rate definitions
learning_rate = tf.Variable(float(FLAGS.learning_rate), trainable=False)
learning_rate_decay_op = learning_rate.assign(FLAGS.learning_rate * FLAGS.learning_rate_decay)

#
global_step = tf.Variable(0, trainable=False)

# data_utils.prepare_data(en_v_size, pt_v_size) - only needed if the data is not ready


def inference(source, target):

    source_r = tf.reverse(source, [False, True])

    # encoder embedding layer
    with tf.name_scope('source_embedding'):
        with tf.device("/cpu:0"):
            src_embedding = tf.Variable(
                tf.truncated_normal(
                    [FLAGS.source_v_size, FLAGS.source_projection_size], stddev=0.01
                ),
                name='Emb_src'
            )

            # get the embeddings
            emb_src_fwd = tf.nn.embedding_lookup(src_embedding, source)
            emb_src_bkw = tf.nn.embedding_lookup(src_embedding, source_r)

    # encoder bi-directional recurrent layer
    with tf.name_scope('encoder_hidden'):
        fwd_outputs, fwd_annotations = rnn.rnn(_ENCODER_CELL, emb_src_fwd)
        bkw_outputs, bkw_annotations = rnn.rnn(_DECODER_CELL, emb_src_bkw)

        # revert the reversed sentence states to concatenate
        # the second parameter is a list of dimensions to revert - False means does not change
        # True means revert - We are reverting just the data dimension while time/bach stay equal
        bkw_annotations = tf.reverse(bkw_annotations, [False, False, True])
        bkw_outputs = tf.reverse(bkw_outputs, [False, False, True])

        # concatenates the forward and backward annotations
        context = tf.concat(0, [fwd_annotations, bkw_annotations])
        encoder_outputs = tf.concat(0, [fwd_outputs, bkw_outputs])

    # define the initial state for the decoder
    with tf.name_scope('decoder_initial_state'):
        Winit = tf.Variable(
            tf.truncated_normal([FLAGS.source_v_size, FLAGS.source_projection_size], stddev=0.01),
            name='W_s'
        )

        # get the last hidden state of the encoder in the backward process
        h1 = bkw_annotations[-1]

        # perform tanh(Ws * h) step
        decoder_initial_state = tf.nn.tanh(tf.matmul(h1, Winit))

    with tf.name_scope('target_embedding'):
        with tf.device("/cpu:0"):
            tgt_embedding = tf.Variable(
                tf.truncated_normal(
                    [FLAGS.source_v_size, FLAGS.source_projection_size], stddev=0.01
                ),
                name='Emb_tgt'
            )
            emb_tgt = tf.nn.embedding_lookup(tgt_embedding, target)

    # start the decoder
    with tf.name_scope('decoder_with_attention'):
        W = tf.Variable(
            tf.truncated_normal([FLAGS.target_v_size, FLAGS.source_projection_size], stddev=0.01),
            name='W_out'
        )

        b = tf.Variable(tf.zeros([FLAGS.target_v_size]), name='b_out')

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [tf.reshape(e, [-1, 1, FLAGS.decoder_size]) for e in encoder_outputs]
        attention_states = tf.concat(1, top_states)

        # run the recurrent network
        outputs, states = seq2seq.attention_decoder(
            context, decoder_initial_state, attention_states,
            _DECODER_CELL, FLAGS.target_v_size, num_heads=1,
            output_size=None, output_projection=(W, b),
            feed_previous=False, dtype=_DTYPE, scope=None
        )

    return outputs


def loss(encoder_out, y_pred, y_true, n_samples=512):
    w = tf.get_variable('W_out')
    w_t = tf.transpose(w)
    b = tf.get_variable('b_out')
    cost = tf.nn.sampled_softmax_loss(w_t, b, encoder_out, y_true, n_samples, FLAGS.target_v_size)
    return cost
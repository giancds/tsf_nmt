import pkg_resources
import tensorflow as tf

from tensorflow.models.rnn import rnn
from tensorflow.python.ops import embedding_ops
from tensorflow.python.framework import ops

import cells


def reverse_encoder(source, src_embedding, encoder_cell, batch_size,
                    dropout=None, dtype=tf.float32):
    """

    Parameters
    ----------
    source
    src_embedding
    encoder_cell
    batch_size
    dtype

    Returns
    -------

    """
    # get the embeddings
    with ops.device("/cpu:0"):
        emb_inp = [embedding_ops.embedding_lookup(src_embedding, s) for s in source]

    initial_state = encoder_cell.zero_state(batch_size=batch_size, dtype=dtype)

    if dropout is not None:

        for cell in encoder_cell._cells:
            cell.input_keep_prob = 1.0 - dropout

    outputs, state = rnn.rnn(encoder_cell, emb_inp,
                              initial_state=initial_state,
                              dtype=dtype,
                              scope='reverse_encoder')

    hidden_states = outputs

    tf_version = pkg_resources.get_distribution("tensorflow").version

    if tf_version == '0.6.0':

        decoder_initial_state = state[-1]
    else:

        decoder_initial_state = state

    return hidden_states, decoder_initial_state


def bidirectional_encoder(source, src_embedding, encoder_cell_fw, encoder_cell_bw,
                          dropout=None, dtype=tf.float32):
    """

    Parameters
    ----------
    source
    src_embedding
    encoder_cell
    batch_size
    dtype

    Returns
    -------

    """
    # get the embeddings
    with ops.device("/cpu:0"):
        emb_inp = [embedding_ops.embedding_lookup(src_embedding, s) for s in source]

    if dropout is not None:

        encoder_cell_fw.input_keep_prob = 1.0 - dropout
        encoder_cell_bw.input_keep_prob = 1.0 - dropout

    outputs, _, output_state_bw = cells.bidirectional_rnn(
        encoder_cell_fw, encoder_cell_bw, emb_inp, dtype=dtype, scope='bidirectional_encoder'
    )

    tf_version = pkg_resources.get_distribution("tensorflow").version

    if tf_version == '0.6.0':

        decoder_initial_state = output_state_bw[-1]
    else:

        decoder_initial_state = output_state_bw

    return outputs, decoder_initial_state
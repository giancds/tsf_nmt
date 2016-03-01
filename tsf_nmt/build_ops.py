import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.python.platform import gfile

import lm_models
import nmt_models
import cells


def get_optimizer(name='sgd', lr_rate=0.1, decay=0.9):
    """

    Parameters
    ----------
    name
    lr_rate
    decay

    Returns
    -------

    """
    optimizer = None
    if name is 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr_rate)
    elif name is 'adagrad':
        optimizer = tf.train.AdagradOptimizer(lr_rate)
    elif name is 'adam':
        optimizer = tf.train.AdamOptimizer(lr_rate, epsilon=1e-8)
    elif name is 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(lr_rate, decay)
    else:
        raise ValueError('Optimizer not found.')
    return optimizer


def build_nmt_multicell_rnn(num_layers_encoder, num_layers_decoder, encoder_size, decoder_size,
                            source_proj_size, target_proj_size, use_lstm=True, input_feeding=True,
                            dropout=0.0):

    if use_lstm:
        cell_class = rnn_cell.LSTMCell
    else:
        cell_class = cells.GRUCell

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
        cell_class = rnn_cell.GRUCell

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


def create_nmt_model(session, forward_only, model_path=None, use_best=False, FLAGS=None, buckets=None, translate=False):
    """Create translation model and initialize or load parameters in session."""

    assert FLAGS is not None
    assert buckets is not None

    decode_input = FLAGS.decode_input
    decode_file = FLAGS.decode_file

    assert (decode_input is True and decode_file is False) \
           or (decode_input is False and decode_file is True) \
           or (decode_input is False and decode_file is False), \
        'Cannot decode from input AND from file. Please choose just one option.'

    # we should set batch to 1 when decoding
    if decode_input or decode_file:
        batch = 1
    else:
        batch = FLAGS.batch_size

    dropout_rate = FLAGS.dropout

    if translate:
        dropout_rate = 0.0

    model = nmt_models.Seq2SeqModel(source_vocab_size=FLAGS.src_vocab_size,
                                    target_vocab_size=FLAGS.tgt_vocab_size,
                                    buckets=buckets,
                                    source_proj_size=FLAGS.proj_size,
                                    target_proj_size=FLAGS.proj_size,
                                    encoder_size=FLAGS.hidden_size,
                                    decoder_size=FLAGS.hidden_size,
                                    num_layers_encoder=FLAGS.num_layers,
                                    num_layers_decoder=FLAGS.num_layers,
                                    max_gradient_norm=FLAGS.max_gradient_norm,
                                    batch_size=batch,
                                    learning_rate=FLAGS.learning_rate,
                                    learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
                                    optimizer=FLAGS.optimizer,
                                    use_lstm=FLAGS.use_lstm,
                                    input_feeding=FLAGS.input_feeding,
                                    dropout=dropout_rate,
                                    attention_type=FLAGS.attention_type,
                                    content_function=FLAGS.content_function,
                                    output_attention=FLAGS.output_attention,
                                    num_samples=FLAGS.num_samples_loss,
                                    forward_only=forward_only,
                                    beam_size=FLAGS.beam_size,
                                    max_len=FLAGS.max_len,
                                    cpu_only=FLAGS.cpu_only,
                                    early_stop_patience=FLAGS.early_stop_patience)

    if model_path is None:

        if use_best:
            ckpt = tf.train.get_checkpoint_state(FLAGS.best_models_dir)

        else:
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

        if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
            print('Reading model parameters from %s' % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print('Created model with fresh parameters.')
            session.run(tf.initialize_all_variables())

    else:
        print('Reading model parameters from %s' % model_path)
        model.saver.restore(session, model_path)

    return model


def create_lm_model(session, is_training=True, FLAGS=None, initializer=None, model_path=None):

    assert FLAGS is not None
    assert initializer is not None

    with tf.variable_scope("model", reuse=None, initializer=initializer):

        if is_training:

            model = lm_models.LMModel(is_training=is_training,
                                      learning_rate=FLAGS.learning_rate,
                                      max_grad_norm=FLAGS.max_grad_norm,
                                      num_layers=FLAGS.num_layers,
                                      num_steps=FLAGS.num_steps,
                                      proj_size=FLAGS.proj_size,
                                      hidden_size=FLAGS.hidden_size,
                                      use_lstm=FLAGS.use_lstm,
                                      early_stop_patience=FLAGS.early_stop_patience,
                                      dropout_rate=FLAGS.dropout_rate,
                                      lr_decay=FLAGS.lr_decay,
                                      batch_size=FLAGS.batch_size,
                                      vocab_size=FLAGS.src_vocab_size)
        else:
            model = lm_models.LMModel(is_training=is_training,
                                      learning_rate=FLAGS.learning_rate,
                                      max_grad_norm=FLAGS.max_grad_norm,
                                      num_layers=FLAGS.num_layers,
                                      num_steps=1,
                                      proj_size=FLAGS.proj_size,
                                      hidden_size=FLAGS.hidden_size,
                                      use_lstm=FLAGS.use_lstm,
                                      early_stop_patience=FLAGS.early_stop_patience,
                                      dropout_rate=FLAGS.dropout_rate,
                                      lr_decay=FLAGS.lr_decay,
                                      batch_size=1,
                                      vocab_size=FLAGS.src_vocab_size)

        if model_path is None:

            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

            if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
                print('Reading model parameters from %s' % ckpt.model_checkpoint_path)
                model.saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print('Created model with fresh parameters.')
                session.run(tf.initialize_all_variables())

        else:
            print('Reading model parameters from %s' % model_path)
            model.saver.restore(session, model_path)

    return model


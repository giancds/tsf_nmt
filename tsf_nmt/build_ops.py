# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.platform import gfile

import attention
import content_functions
import decoders
import nmt_models


def create_seq2seq_model(session, forward_only, model_path=None, use_best=False, FLAGS=None, buckets=None, translate=False):
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

    if FLAGS.output_attention == "None":
        if FLAGS.informed_decoder:
            decoder = decoders.attention_decoder_informed
        else:
            decoder = decoders.attention_decoder
    else:
        if FLAGS.informed_decoder:
            decoder = decoders.attention_decoder_output_informed
        else:
            decoder = decoders.attention_decoder_output

    attention_f = attention.get_attention_f(FLAGS.attention_type)
    content_function = content_functions.get_content_f(FLAGS.content_function)
    decoder_attention_f = content_functions.get_decoder_content_f(FLAGS.output_attention)

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
                                    decoder=decoder,
                                    optimizer=FLAGS.optimizer,
                                    use_lstm=FLAGS.use_lstm,
                                    input_feeding=FLAGS.input_feeding,
                                    dropout=dropout_rate,
                                    attention_f=attention_f,
                                    window_size=FLAGS.window_size,
                                    content_function=content_function,
                                    decoder_attention_f=decoder_attention_f,
                                    num_samples=FLAGS.num_samples_loss,
                                    forward_only=forward_only,
                                    max_len=FLAGS.max_len,
                                    cpu_only=FLAGS.cpu_only,
                                    early_stop_patience=FLAGS.early_stop_patience,
                                    save_best_model=FLAGS.save_best_model,
                                    log_tensorboard=FLAGS.log_tensorboard)

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

    attention_f = attention.get_attention_f(FLAGS.attention_type)
    content_function = content_functions.get_content_f(FLAGS.content_function)
    decoder_attention_f = content_functions.get_decoder_content_f(FLAGS.output_attention)

    model = nmt_models.NMTModel(source_vocab_size=FLAGS.src_vocab_size,
                                target_vocab_size=FLAGS.tgt_vocab_size,
                                buckets=buckets,
                                source_proj_size=FLAGS.proj_size,
                                target_proj_size=FLAGS.proj_size,
                                encoder_size=FLAGS.hidden_size,
                                decoder_size=FLAGS.hidden_size,
                                max_gradient_norm=FLAGS.max_gradient_norm,
                                batch_size=batch,
                                learning_rate=FLAGS.learning_rate,
                                learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
                                optimizer=FLAGS.optimizer,
                                input_feeding=FLAGS.input_feeding,
                                dropout=dropout_rate,
                                attention_f=attention_f,
                                window_size=FLAGS.window_size,
                                content_function=content_function,
                                decoder_attention_f=decoder_attention_f,
                                num_samples=FLAGS.num_samples_loss,
                                forward_only=forward_only,
                                max_len=FLAGS.max_len,
                                cpu_only=FLAGS.cpu_only,
                                early_stop_patience=FLAGS.early_stop_patience,
                                save_best_model=FLAGS.save_best_model)

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

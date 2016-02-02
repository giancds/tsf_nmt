# -*- coding: utf-8 -*-
import data_utils
import math
import nmt_models
import numpy
import os
import tensorflow as tf
import time
import sys
from tensorflow.python.platform import gfile
# from six.moves import xrange

def read_data(source_path, target_path, FLAGS=None, buckets=None, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:lse
      source_path: path to the files with token-ids for the source language.
      target_path: path to the file with token-ids for the target language;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """

    assert FLAGS is not None
    assert buckets is not None

    data_set = [[] for _ in buckets]
    counter = 0
    with gfile.GFile(source_path, mode='r') as source_file:
        with gfile.GFile(target_path, mode='r') as target_file:
            source, target = source_file.readline(), target_file.readline()

            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 10000 == 0:
                    print('  reading data line %d' % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


def create_model(session, forward_only, model_path=None, FLAGS=None, buckets=None, translate=False):
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
                                    forward_only=forward_only,
                                    early_stop_patience=FLAGS.early_stop_patience)

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


def train(FLAGS=None, buckets=None, save_before_training=False):
    """Train a source->target translation model using some bilingual data."""

    assert FLAGS is not None
    assert buckets is not None

    # Prepare WMT data.
    print('Preparing data in %s' % FLAGS.data_dir)
    src_train, tgt_train, src_dev, tgt_dev, src_test, tgt_test = data_utils.prepare_data(FLAGS)

    # summary_op = tf.merge_all_summaries()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

        # Create model.
        print('Creating layers.')
        model = create_model(sess, False, FLAGS=FLAGS, buckets=buckets)

        if save_before_training:
            # Save checkpoint
            checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)

        # tf.train.write_graph(sess.graph_def, '/home/gian/train2', 'graph.pbtxt')

        # Read data into buckets and compute their sizes.
        print('Reading development and training data (limit: %d).'
              % FLAGS.max_train_data_size)
        dev_set = read_data(src_dev, tgt_dev, FLAGS=FLAGS, buckets=buckets)
        train_set = read_data(src_train, tgt_train, max_size=FLAGS.max_train_data_size, FLAGS=FLAGS, buckets=buckets)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        print("Total number of updates per epoch: %d" % (train_total_size / FLAGS.batch_size))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time = 0.0
        words_time = 0.0
        previous_losses = []
        n_target_words = 0

        while model.epoch.eval() < FLAGS.max_epochs:

            start_time = time.time()

            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = numpy.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            encoder_inputs, decoder_inputs, target_weights, n_words = model.get_train_batch(
                    train_set, bucket_id
            )

            n_target_words += n_words

            # session, encoder_inputs, decoder_inputs, target_weights, bucket_id
            _, step_loss, _ = model.train_step(session=sess, encoder_inputs=encoder_inputs,
                                               decoder_inputs=decoder_inputs,
                                               target_weights=target_weights,
                                               bucket_id=bucket_id)

            currloss = model.current_loss.eval()
            sess.run(model.current_loss.assign(currloss + step_loss))

            # increase the number of seen samples
            sess.run(model.samples_seen_update_op)
            # sess.run(model.current_loss_update_op)

            # update epoch number
            if model.samples_seen.eval() >= train_total_size:
                sess.run(model.epoch_update_op)
                ep = model.epoch.eval()
                print("Epoch %d finished..." % (ep - 1))

                # Save checkpoint
                checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                if ep >= FLAGS.max_epochs:
                    break

                print("Epoch %d started..." % ep)
                sess.run(model.samples_seen_reset_op)

                if FLAGS.start_decay > 0:
                    if model.epoch.eval() >= FLAGS.start_decay:
                        sess.run(model.learning_rate_decay_op)

            current_step = model.global_step.eval()

            if current_step % FLAGS.steps_verbosity == 0:

                closs = model.current_loss.eval()
                gstep = model.global_step.eval()
                avgloss = closs / gstep
                sess.run(model.avg_loss.assign(avgloss))

                target_words_speed = n_target_words / words_time

                loss = model.avg_loss.eval()
                ppx = math.exp(loss) if loss < 300 else float('inf')

                if ppx == float('inf'):
                    pass

                if numpy.isnan(loss) or numpy.isinf(loss):
                    print 'NaN detected'
                    break

                print('epoch %d gl.step %d lr.rate %.4f steps-time %.2f avg.loss %.8f avg.ppx %.8f - avg. %.2f K target words/sec' %
                      (model.epoch.eval(), model.global_step.eval(), model.learning_rate.eval(),
                       step_time, loss, ppx, (target_words_speed / 1000.0)))

                n_target_words = 0
                step_time = 0.0
                words_time = 0.0

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Save checkpoint
                checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)

            if current_step % FLAGS.steps_per_validation == 0:

                if FLAGS.dropout > 0.0:
                    _turn_dropout(model=model, rate=0.0)

                total_eval_loss = 0.0
                total_ppx = 0.0

                print('\n')

                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(buckets)):

                    encoder_inputs, decoder_inputs, target_weights, _ = model.get_train_batch(dev_set, bucket_id)

                    _, eval_loss, _ = model.train_step(session=sess, encoder_inputs=encoder_inputs,
                                                       decoder_inputs=decoder_inputs, target_weights=target_weights,
                                                       bucket_id=bucket_id)

                    total_eval_loss += eval_loss

                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    total_ppx += eval_ppx
                    print('  eval: bucket %d perplexity %.2f' % (bucket_id, eval_ppx))

                avg_eval_loss = total_eval_loss / len(buckets)
                avg_ppx = math.exp(avg_eval_loss) if avg_eval_loss < 300 else float('inf')
                print('\n  eval: averaged perplexity %.8f' % avg_ppx)
                print('  eval: averaged loss %.8f\n' % avg_eval_loss)

                sys.stdout.flush()

                estop = FLAGS.early_stop_patience

                if FLAGS.dropout > 0.0:
                    _turn_dropout(model=model, rate=FLAGS.dropout)

                # check early stop - if early stop patience is greater than 0, test it
                if estop > 0:
                    if avg_eval_loss < model.best_eval_loss.eval():
                        sess.run(model.best_eval_loss.assign(avg_eval_loss))
                        sess.run(model.estop_counter_reset_op)
                        # Save checkpoint
                        print('Saving the best model so far...')
                        best_model_path = os.path.join(FLAGS.train_dir, FLAGS.model_name + '-best')
                        model.saver.save(sess, best_model_path, global_step=0)
                    else:
                        sess.run(model.estop_counter_update_op)
                        if model.estop_counter.eval() >= FLAGS.early_stop_patience:
                            print('\nEARLY STOP!\n')
                            break

            step_time += (time.time() - start_time) / FLAGS.steps_verbosity
            words_time += (time.time() - start_time)

        # # Save checkpoint
        # checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
        # model.saver.save(sess, checkpoint_path, global_step=model.global_step)

        print("\nTraining finished!!\n")


# TODO: there is a better way of turning dropout off during validation? - this is a little bit "hacky"
def _turn_dropout(model, rate=0.0):

    for cell in model.encoder_cell._cells:
        cell.input_keep_prob = rate

    for cell in model.decoder_cell._cells:
        cell.input_keep_prob = rate


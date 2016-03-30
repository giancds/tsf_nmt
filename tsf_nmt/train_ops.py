# -*- coding: utf-8 -*-
import data_utils
import math
import numpy
import random
import os
import tensorflow as tf
import time
import sys
import build_ops
from data_utils import read_nmt_data
# from six.moves import xrange


def train_nmt(FLAGS=None, buckets=None, save_before_training=False):
    """Train a source->target translation model using some bilingual data."""

    assert FLAGS is not None
    assert buckets is not None

    # Prepare data for training
    print('Preparing data in %s' % FLAGS.data_dir)
    src_train, tgt_train, src_dev, tgt_dev, _, _ = data_utils.prepare_nmt_data(FLAGS)

    # summary_op = tf.merge_all_summaries()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:

        nan_detected = False

        # Create model.
        print('Creating layers.')

        if FLAGS.model == "seq2seq":
            model = build_ops.create_seq2seq_model(sess, False, FLAGS=FLAGS, buckets=buckets)
        else:
            model = build_ops.create_nmt_model(sess, False, FLAGS=FLAGS, buckets=buckets)

        if save_before_training:
            # Save checkpoint
            checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)

        # tf.train.write_graph(sess.graph_def, '/home/gian/train2', 'graph.pbtxt')

        # Read data into buckets and compute their sizes.
        print('Reading development and training data (limit: %d).' % FLAGS.max_train_data_size)
        dev_set = read_nmt_data(src_dev, tgt_dev, FLAGS=FLAGS, buckets=buckets)
        train_set = read_nmt_data(src_train, tgt_train, max_size=FLAGS.max_train_data_size, FLAGS=FLAGS,
                                  buckets=buckets)
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
        n_target_words = 0

        print("Optimization started...")
        while model.epoch.eval() < FLAGS.max_epochs:

            saved = False

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
            # note: step loss is averaged across the batch
            gradient_norm, step_loss, _ = model.train_step(session=sess, encoder_inputs=encoder_inputs,
                                                           decoder_inputs=decoder_inputs,
                                                           target_weights=target_weights,
                                                           bucket_id=bucket_id,
                                                           validation_step=False)
            # step_loss = numpy.nan

            if numpy.isnan(step_loss) or numpy.isinf(step_loss):

                numpy.set_printoptions(linewidth=200)

                print('\nNaN detected\n')
                nan_detected = True

                print("\nStep loss:")
                print(step_loss)

                print("\nEncoder inputs: ")
                print(encoder_inputs)

                print("\nDecoder inputs: ")
                print(decoder_inputs)

                print("\nTarget weights inputs: ")
                print(target_weights)

                print("\nGradient norm: ")
                print(gradient_norm)

                break

            currloss = model.current_loss.eval()
            sess.run(model.current_loss.assign(currloss + step_loss))

            # increase the number of seen samples
            sess.run(model.samples_seen_update_op)

            current_step = model.global_step.eval()

            if current_step % FLAGS.steps_verbosity == 0:

                closs = model.current_loss.eval()
                gstep = model.global_step.eval()
                avgloss = closs / gstep
                sess.run(model.avg_loss.assign(avgloss))

                target_words_speed = n_target_words / words_time

                loss = model.avg_loss.eval()
                ppx = math.exp(loss) if loss < 300 else float('inf')

                if ppx > 1000.0:
                    print(
                    'epoch %d gl.step %d lr.rate %.4f steps-time %.2f avg.loss %.8f avg.ppx > %.8f - avg. %.2f K target words/sec' %
                    (model.epoch.eval(), model.global_step.eval(), model.learning_rate.eval(),
                     step_time, loss, 1000.0, (target_words_speed / 1000.0)))
                else:
                    print(
                    'epoch %d gl.step %d lr.rate %.4f steps-time %.2f avg.loss %.8f avg.ppx %.8f - avg. %.2f K target words/sec' %
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
                saved = True

                # update epoch number
            if model.samples_seen.eval() >= train_total_size:
                sess.run(model.epoch_update_op)
                ep = model.epoch.eval()
                print("Epoch %d finished..." % (ep - 1))

                # Save checkpoint
                checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                if ep >= FLAGS.max_epochs:
                    if not saved:
                        # Save checkpoint
                        checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
                        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                    finished = True
                    break

                print("Epoch %d started..." % ep)
                sess.run(model.samples_seen_reset_op)

                if FLAGS.start_decay > 0:

                    if FLAGS.stop_decay > 0:

                        if FLAGS.start_decay <= model.epoch.eval() <= FLAGS.stop_decay:
                            sess.run(model.learning_rate_decay_op)

                    else:

                        if FLAGS.start_decay <= model.epoch.eval():
                            sess.run(model.learning_rate_decay_op)

            if current_step % FLAGS.steps_per_validation == 0:

                total_eval_loss = 0.0
                total_ppx = 0.0

                print('\n')

                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(buckets)):

                    n_steps = len(dev_set[bucket_id]) / model.batch_size

                    bucket_loss = 0.0

                    for _ in xrange(n_steps):
                        encoder_inputs, decoder_inputs, target_weights, _ = model.get_train_batch(dev_set,
                                                                                                  bucket_id)

                        _, eval_loss, _ = model.train_step(session=sess, encoder_inputs=encoder_inputs,
                                                           decoder_inputs=decoder_inputs, target_weights=target_weights,
                                                           bucket_id=bucket_id, validation_step=True)

                        bucket_loss += eval_loss

                    bucket_avg_loss = bucket_loss / n_steps
                    total_eval_loss += bucket_avg_loss

                    eval_ppx = math.exp(bucket_avg_loss) if eval_loss < 300 else float('inf')
                    total_ppx += eval_ppx
                    print('  eval: bucket %d perplexity %.4f' % (bucket_id, eval_ppx))

                avg_eval_loss = total_eval_loss / len(buckets)
                avg_ppx = math.exp(avg_eval_loss) if avg_eval_loss < 300 else float('inf')

                if avg_ppx > 1000.0:
                    print('\n  eval: averaged perplexity > 1000.0')
                else:
                    print('\n  eval: averaged perplexity %.8f' % avg_ppx)
                print('  eval: averaged loss %.8f\n' % avg_eval_loss)

                sys.stdout.flush()

                estop = FLAGS.early_stop_patience

                # check early stop - if early stop patience is greater than 0, test it
                if estop > 0:

                    if avg_eval_loss < model.best_eval_loss.eval():
                        sess.run(model.best_eval_loss.assign(avg_eval_loss))
                        sess.run(model.estop_counter_reset_op)
                        # Save checkpoint
                        print('Saving the best model so far...')
                        best_model_path = os.path.join(FLAGS.best_models_dir, FLAGS.model_name + '-best')
                        model.saver_best.save(sess, best_model_path, global_step=model.global_step)

                    else:

                        # if FLAGS.early_stop_after_epoch is equal to 0, it will monitor from the beginning
                        if model.epoch.eval() >= FLAGS.early_stop_after_epoch:

                            sess.run(model.estop_counter_update_op)

                            if model.estop_counter.eval() >= estop:
                                print('\nEARLY STOP!\n')
                                finished = True
                                break

                    print('\n   best valid. loss: %.8f' % model.best_eval_loss.eval())
                    print('early stop patience: %d - max %d\n' % (int(model.estop_counter.eval()), estop))

            step_time += (time.time() - start_time) / FLAGS.steps_verbosity
            words_time += (time.time() - start_time)

        print("\nTraining finished!!\n")

        if not nan_detected:

            # # Save checkpoint
            checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)

            print("Final validation:")

            total_eval_loss = 0.0
            total_ppx = 0.0

            print('\n')

            # Run evals on development set and print their perplexity.
            for bucket_id in xrange(len(buckets)):

                n_steps = len(dev_set[bucket_id]) / model.batch_size

                bucket_loss = 0.0

                for _ in xrange(n_steps):
                    encoder_inputs, decoder_inputs, target_weights, _ = model.get_train_batch(dev_set,
                                                                                              bucket_id)

                    _, eval_loss, _ = model.train_step(session=sess, encoder_inputs=encoder_inputs,
                                                       decoder_inputs=decoder_inputs, target_weights=target_weights,
                                                       bucket_id=bucket_id)

                    bucket_loss += eval_loss

                bucket_avg_loss = bucket_loss / n_steps
                total_eval_loss += bucket_avg_loss

                eval_ppx = math.exp(bucket_avg_loss) if eval_loss < 300 else float('inf')
                total_ppx += eval_ppx
                print('  eval: bucket %d perplexity %.4f' % (bucket_id, eval_ppx))

            avg_eval_loss = total_eval_loss / len(buckets)
            avg_ppx = math.exp(avg_eval_loss) if avg_eval_loss < 300 else float('inf')

            if avg_ppx > 1000.0:
                print('\n  eval: averaged perplexity > 1000.0')
            else:
                print('\n  eval: averaged perplexity %.8f' % avg_ppx)
            print('  eval: averaged loss %.8f\n' % avg_eval_loss)

            print('\n   best valid. loss during training: %.8f' % model.best_eval_loss.eval())

            sys.stdout.flush()

"""
"""

from __future__ import print_function
import time
import random
import numpy
import tensorflow as tf
from tensorflow.models.rnn import seq2seq, rnn
from tensorflow.python.ops import array_ops, nn_ops

import data_utils
import build_ops


class LMModel(object):
    """The PTB model."""

    def __init__(self,
                 is_training,
                 learning_rate=1.0,
                 max_grad_norm=5,
                 num_layers=2,
                 use_lstm=True,
                 num_steps=35,
                 proj_size=650,
                 hidden_size=650,
                 num_samples=512,
                 early_stop_patience=0,
                 dropout_rate=0.0,
                 lr_decay=0.8,
                 batch_size=20,
                 vocab_size=10000):

        self.batch_size = batch_size = batch_size
        self.num_steps = num_steps = num_steps
        size = hidden_size
        vocab_size = vocab_size
        #
        # self._input_data = tf.placeholder(tf.int32, [None, num_steps], name='input_data')
        # self._targets = tf.placeholder(tf.int32, [None, num_steps], name='targets')

        self.input_data = []
        self.targets = []
        self.mask = []

        for i in xrange(num_steps):  # Last bucket is the biggest one.
            self.input_data.append(tf.placeholder(tf.int32, shape=[None], name="input{0}".format(i)))
            self.targets.append(tf.placeholder(tf.int32, shape=[None], name="target{0}".format(i)))
            self.mask.append(tf.placeholder(tf.float32, shape=[None], name="mask{0}".format(i)))

        self.cell = build_ops.build_lm_multicell_rnn(num_layers, hidden_size, proj_size, use_lstm=use_lstm, dropout=dropout_rate)

        # self._initial_state = tf.placeholder(tf.float32, [None], name='initial_state')

        # learning rate ops
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * lr_decay)

        # epoch ops
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_update_op = self.epoch.assign(self.epoch + 1)

        # samples seen ops
        self.samples_seen = tf.Variable(0, trainable=False)
        self.samples_seen_update_op = self.samples_seen.assign(self.samples_seen + batch_size)
        self.samples_seen_reset_op = self.samples_seen.assign(0)

        # global step variable - controled by the model
        self.global_step = tf.Variable(0.0, trainable=False)

        # average loss ops
        self.current_loss = tf.Variable(0.0, trainable=False)
        self.current_loss_update_op = None
        self.avg_loss = tf.Variable(0.0, trainable=False)
        self.avg_loss_update_op = self.avg_loss.assign(tf.div(self.current_loss, self.global_step))

        if early_stop_patience > 0:
            self.best_eval_loss = tf.Variable(numpy.inf, trainable=False)
            self.estop_counter = tf.Variable(0, trainable=False)
            self.estop_counter_update_op = self.estop_counter.assign(self.estop_counter + 1)
            self.estop_counter_reset_op = self.estop_counter.assign(0)
        else:
            self.best_eval_loss = None
            self.estop_counter = None
            self.estop_counter_update_op = None
            self.estop_counter_reset_op = None

        with tf.device("/cpu:0"):
            # input come as one big tensor so we have to split it into a list of tensors to run the rnn cell
            embedding = tf.get_variable("embedding", [vocab_size, proj_size])

            inputs = [tf.nn.embedding_lookup(embedding, i) for i in self.input_data]
            # inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        with tf.variable_scope("RNN"):
            # initial_state = self.cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            # initial state will be all zeros
            outputs, states = rnn.rnn(self.cell, inputs, dtype=tf.float32)

            with tf.device("/cpu:0"):
                w = tf.get_variable("softmax_w", [size, vocab_size])
                w_t = tf.transpose(w)
                b = tf.get_variable("softmax_b", [vocab_size])

            def sampled_loss(logits, labels):
                with tf.device("/cpu:0"):
                    labels = tf.reshape(labels, [-1, 1])
                    losses = tf.nn.sampled_softmax_loss(w_t, b, logits, labels, num_samples, vocab_size)
                    return losses

        softmax_loss_function = sampled_loss

        if is_training:
            loss = seq2seq.sequence_loss_by_example(outputs,
                                                    self.targets,
                                                    self.mask,
                                                    vocab_size,
                                                    softmax_loss_function=softmax_loss_function)

            b_size = array_ops.shape(self.input_data[0])[0]
            self._cost = cost = tf.reduce_sum(loss) / tf.to_float(b_size)
            self._final_state = states[-1]

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                              max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        else:
            logit = tf.nn.xw_plus_b(outputs[-1], w, b)
            self.logits = nn_ops.softmax(logit)

        self.saver = tf.train.Saver(tf.all_variables())
        self.saver_best = tf.train.Saver(tf.all_variables())

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self.learning_rate

    @property
    def train_op(self):
        return self._train_op

    def get_train_batch(self, data, batch=None):
        """
        """
        max_n_steps = self.num_steps
        batch_inputs, batch_targets = [], []

        n_target_words = 0

        if batch is None:
            batch = self.batch_size

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in xrange(batch):
            encoder_input = random.choice(data)

            # Encoder inputs are padded and then reversed.
            lm_pad = [data_utils.PAD_ID] * (max_n_steps - len(encoder_input))
            batch_inputs.append(list(encoder_input + lm_pad))

            n_target_words += len(encoder_input)

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            batch_targets.append([data_utils.GO_ID] + list(encoder_input + lm_pad))

        # Now we create batch-major vectors from the data selected above.
        batch_lm_inputs, batch_lm_targets, batch_weights = [], [], []

        for length_idx in xrange(self.num_steps):
            batch_lm_inputs.append(
                    numpy.array([batch_inputs[batch_idx][length_idx]
                              for batch_idx in xrange(batch)], dtype=numpy.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(self.num_steps):
            batch_lm_targets.append(
                    numpy.array([batch_targets[batch_idx][length_idx]
                              for batch_idx in xrange(batch)], dtype=numpy.int32))

            batch_weight = numpy.ones(batch, dtype=numpy.float32)
            for batch_idx in xrange(batch):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < self.num_steps - 1:
                    target = batch_targets[batch_idx][length_idx + 1]
                if length_idx == self.num_steps - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        return batch_lm_inputs, batch_lm_targets, batch_weights, n_target_words

    def train_step(self, session, lm_inputs, lm_targets, mask, op=None):

        # the op define if we do a parametyer update or not - when validation, op must be tf.no_op()
        if op is None:
            op = self.train_op

        input_feed = {}
        for l in xrange(self.num_steps):
            input_feed[self.input_data[l].name] = lm_inputs[l]
            input_feed[self.targets[l].name] = lm_targets[l]
            input_feed[self.mask[l].name] = mask[l]

        output_feed = [self.cost, self.final_state, op]

        cost_, state_, _ = session.run(output_feed, feed_dict=input_feed)

        return cost_, state_

    def get_translate_batch(self, data):
        """Get a random batch of data from the specified bucket, prepare for step.
        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.
        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.
        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = (len(data), 1)
        lm_inputs, lm_targets = [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in xrange(self.batch_size):
            # encoder_input, _, decoder_input = random.choice(d)
            lm_input, lm_target = random.choice(data)

            # Encoder inputs are padded and then reversed.
            lm_pad = [data_utils.PAD_ID] * (encoder_size - len(lm_input))
            lm_inputs.append(list(reversed(lm_input + lm_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(lm_target) - 1
            lm_targets.append([data_utils.GO_ID] + lm_target +
                                  [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs = [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                    numpy.array([lm_inputs[batch_idx][length_idx]
                              for batch_idx in xrange(self.batch_size)], dtype=numpy.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                    numpy.array([lm_targets[batch_idx][length_idx]
                              for batch_idx in xrange(self.batch_size)], dtype=numpy.int32))

        return batch_encoder_inputs, batch_decoder_inputs

    def decode_step(self, session, lm_inputs, lm_targets, mask):

        input_feed = {}

        for l in xrange(self.num_steps):
            input_feed[self.input_data[l].name] = lm_inputs[l]
            input_feed[self.targets[l].name] = lm_targets[l]
            input_feed[self.mask[l].name] = mask[l]

        output_feed = [self.logits]

        return output_feed[0]

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

To compile on CPU:
  bazel build -c opt tensorflow/models/rnn/ptb:ptb_word_lm
To compile on GPU:
  bazel build -c opt tensorflow --config=cuda \
    tensorflow/models/rnn/ptb:ptb_word_lm
To run:
  ./bazel-bin/.../ptb_word_lm \
    --data_path=/tmp/simple-examples/data/ --alsologtostderr

"""
from __future__ import print_function
import time
import random
import numpy
import tensorflow as tf
from tensorflow.models.rnn import seq2seq, rnn

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
                 early_stop_patience=0,
                 dropout_rate=0.0,
                 lr_decay=0.8,
                 batch_size=20,
                 vocab_size=10000):

        self.batch_size = batch_size = batch_size
        self.num_steps = num_steps = num_steps
        size = hidden_size
        vocab_size = vocab_size

        self._input_data = tf.placeholder(tf.int32, [None, num_steps], name='input_data')
        self._targets = tf.placeholder(tf.int32, [None, num_steps], name='targets')

        cell = build_ops.build_lm_multicell_rnn(num_layers, hidden_size, proj_size, use_lstm=use_lstm, dropout=dropout_rate)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

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
            embedding = tf.get_variable("embedding", [vocab_size, proj_size])
            inputs = tf.split(
                1, num_steps, tf.nn.embedding_lookup(embedding, self._input_data))
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        with tf.variable_scope("RNN"):
            outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)

            output = tf.reshape(tf.concat(1, outputs), [-1, size])

            logits = tf.nn.xw_plus_b(output,
                                     tf.get_variable("softmax_w", [size, vocab_size]),
                                     tf.get_variable("softmax_b", [vocab_size]))

        loss = seq2seq.sequence_loss_by_example([logits],
                                                [tf.reshape(self._targets, [-1])],
                                                [tf.ones([batch_size * num_steps])],
                                                vocab_size)
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = states[-1]

        if not is_training:
            return

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.all_variables())


    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

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
            encoder_pad = [data_utils.PAD_ID] * (max_n_steps - len(encoder_input))
            batch_inputs.append(list(encoder_input + encoder_pad))

            n_target_words += len(encoder_input)

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            batch_targets.append([data_utils.GO_ID] + list(encoder_input + encoder_pad))

        # Now we create batch-major vectors from the data selected above.
        batch_lm_inputs, batch_lm_targets = [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for batch_idx in xrange(batch):

            batch_lm_inputs.append(numpy.array(batch_inputs[batch_idx], dtype=numpy.int32)[0:max_n_steps])
            batch_lm_targets.append(numpy.array(batch_targets[batch_idx], dtype=numpy.int32)[0:max_n_steps])


        return batch_lm_inputs, batch_lm_targets, n_target_words

    def train_step(self, session, lm_inputs, lm_targets, op=None):

        if op is None:
            op = self.train_op

        cost_, state_, _ = session.run(
            [self.cost, self.final_state, op],
            {self.input_data: lm_inputs, self.targets: lm_targets}
        )

        return cost_, state_
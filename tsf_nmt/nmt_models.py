# -*- coding: utf-8 -*-
"""
    Sequence-to-sequence model with bi-directional encoder and the attention mechanism described in

        arxiv.org/abs/1412.2007

    and support to buckets.

"""
import copy
import random
import numpy
import pkg_resources
import tensorflow as tf

from tensorflow.models.rnn import seq2seq
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope

import data_utils
import cells
import encoders
import optimization_ops
from decoders import attention_decoder_nmt

# from six.moves import xrange


def model_with_buckets(encoder_inputs, decoder_inputs, targets, weights,
                       buckets, seq2seq_f, softmax_loss_function=None,
                       per_example_loss=False, name=None):
    """Create a sequence-to-sequence model with support for bucketing.

    The seq2seq argument is a function that defines a sequence-to-sequence model,
    e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))

    Args:
      encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
      decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
      targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
      weights: List of 1D batch-sized float-Tensors to weight the targets.
      buckets: A list of pairs of (input size, output size) for each bucket.
      seq2seq_f: A sequence-to-sequence model function; it takes 2 input that
        agree with encoder_inputs and decoder_inputs, and returns a pair
        consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
      softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
      per_example_loss: Boolean. If set, the returned loss will be a batch-sized
        tensor of losses for each sequence in the batch. If unset, it will be
        a scalar with the averaged loss from all examples.
      name: Optional name for this operation, defaults to "model_with_buckets".

    Returns:
      A tuple of the form (outputs, losses), where:
        outputs: The outputs for each bucket. Its j'th element consists of a list
          of 2D Tensors of shape [batch_size x num_decoder_symbols] (jth outputs).
        losses: List of scalar Tensors, representing losses for each bucket, or,
          if per_example_loss is set, a list of 1D batch-sized float Tensors.

    Raises:
      ValueError: If length of encoder_inputsut, targets, or weights is smaller
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

    all_inputs = encoder_inputs + decoder_inputs + targets + weights
    losses = []
    outputs = []
    with ops.op_scope(all_inputs, name, "model_with_buckets"):
        for j, bucket in enumerate(buckets):
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                               reuse=True if j > 0 else None):
                bucket_outputs, _ = seq2seq_f(encoder_inputs[:bucket[0]],
                                            decoder_inputs[:bucket[1]])
                outputs.append(bucket_outputs)

                if per_example_loss:
                    losses.append(seq2seq.sequence_loss_by_example(
                        outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
                        average_across_timesteps=True,
                        softmax_loss_function=softmax_loss_function))
                else:
                    losses.append(seq2seq.sequence_loss(
                        outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
                        average_across_timesteps=True,
                        softmax_loss_function=softmax_loss_function))

    return outputs, losses


class TranslationModel(object):

    def __init__(self):
        self.buckets = []
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        self.dropout_feed = None
        self.updates = None
        self.gradient_norms = None
        self.losses = None
        self.dropout = 0.0
        self.max_len = 120
        self.batch_size = 32
        self.ret0, self.ret1, self.ret2 = [], [], []
        self.decoder_size = 100
        self.logits, self.states, self.decoder_states = [], [], []
        self.step_num = None
        self.decoder_init_plcholder = None
        self.attn_plcholder = None
        self.decoder_states_holders = None
        self.decoder_attention_f = None

    def inference(self, source, target):
        raise NotImplementedError

    def encode(self, source, batch_size, translate=False):
        raise NotImplementedError

    def get_train_batch(self, data, bucket_id, batch_size=None):
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
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        n_target_words = 0
        #
        if batch_size is None:
            batch_size = self.batch_size

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in xrange(batch_size):
            # encoder_input, _, decoder_input = random.choice(d)
            encoder_input, decoder_input = random.choice(data[bucket_id])

            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            n_target_words += len(decoder_input)

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                  [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                numpy.array([encoder_inputs[batch_idx][length_idx]
                             for batch_idx in xrange(batch_size)], dtype=numpy.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                numpy.array([decoder_inputs[batch_idx][length_idx]
                             for batch_idx in xrange(batch_size)], dtype=numpy.int32))

            # Create target_weights to be 0 for targets that are padding.
            # batch_weight = numpy.ones(self.batch_size, dtype=numpy.float32)
            batch_weight = numpy.ones(batch_size, dtype=numpy.float32)
            # for batch_idx in xrange(self.batch_size):
            for batch_idx in xrange(batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights, n_target_words

    def train_step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, validation_step=False):
        """Run a step of the model feeding the given inputs.
        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.
          validation_step: whether to do the backward step or only forward.
          softmax: whether to apply softmax to the output_logits before returning them
        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        Raises:
          ValueError: if length of enconder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = numpy.zeros([len(encoder_inputs[0])], dtype=numpy.int32)

        # Output feed: depends on whether we do a backward step or not.
        if validation_step:
            input_feed[self.dropout_feed.name] = 0.0
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.

        else:
            input_feed[self.dropout_feed.name] = self.dropout
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]  # Loss for this batch.

        outputs = session.run(output_feed, feed_dict=input_feed)

        # function return: depends on whether we do a backward step or not.
        if validation_step:
            # No gradient norm, loss, no outputs.
            return None, outputs[0], None

        else:
            # Gradient norm, loss, no outputs.
            return outputs[1], outputs[2], None

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
        encoder_size, decoder_size = (self.max_len, 1)
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in xrange(self.batch_size):
            # encoder_input, _, decoder_input = random.choice(d)
            encoder_input, decoder_input = random.choice(data)

            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                  [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs = [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                numpy.array([encoder_inputs[batch_idx][length_idx]
                             for batch_idx in xrange(self.batch_size)], dtype=numpy.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                numpy.array([decoder_inputs[batch_idx][length_idx]
                             for batch_idx in xrange(self.batch_size)], dtype=numpy.int32))

        return batch_encoder_inputs, batch_decoder_inputs

    def translation_step(self, session, token_ids, beam_size=5, normalize=True, dump_remaining=True):

        sample = []
        sample_score = []

        live_hyp = 1
        dead_hyp = 0

        hyp_samples = [[]] * live_hyp
        hyp_scores = numpy.zeros(live_hyp).astype('float32')

        # Get a 1-element batch to feed the sentence to the model
        encoder_inputs, decoder_inputs = self.get_translate_batch([(token_ids, [])])
        decoder_inputs = decoder_inputs[-1]

        # here we encode the input sentence
        encoder_input_feed = {}
        for l in xrange(self.max_len):
            encoder_input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]

        # we select the last element of ret0 to keep as it is a list of hidden_states
        encoder_output_feed = [self.ret0[-1], self.ret1, self.ret2]

        # get the return of encoding step: hidden_states, decoder_initial_states, attention_states
        ret = session.run(encoder_output_feed, encoder_input_feed)

        # here we get info to the decode step
        attention_states = ret[2]
        shape = ret[1][0].shape
        # decoder_init = numpy.tile(ret[1][0].reshape(1, shape[0]), (12, 1))
        decoder_init = ret[1][0].reshape(1, shape[0])
        decoder_states = numpy.zeros((1, 1, 1, self.decoder_size))

        # we must retrieve the last state to feed the decoder run
        decoder_output_feed = [self.logits, self.states, self.decoder_states]

        for ii in xrange(self.max_len):

            session.run(self.step_num.assign(ii + 2))

            # we must feed decoder_initial_state and attention_states to run one decode step
            decoder_input_feed = {self.decoder_inputs[0].name: decoder_inputs,
                                  self.decoder_init_plcholder.name: decoder_init,
                                  self.attn_plcholder.name: attention_states}
            # print ii
            if self.decoder_attention_f:
                # if ii == 1:
                #     decoder_states = numpy.tile(decoder_states, (12, 1, 1, 1))
                decoder_input_feed[self.decoder_states_holders.name] = decoder_states

                # print "Step %d - States shape %s - Input shape %s" % (ii, decoder_states.shape, decoder_inputs.shape)

            ret = session.run(decoder_output_feed, decoder_input_feed)

            next_p = ret[0]
            next_state = ret[1]
            decoder_states = ret[2]

            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(beam_size - dead_hyp)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(beam_size - dead_hyp).astype('float32')
            new_hyp_states = []
            new_dec_states = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti] + [wi])
                new_hyp_scores[idx] = copy.copy(costs[ti])
                new_hyp_states.append(copy.copy(next_state[ti]))
                new_dec_states.append(copy.copy(decoder_states[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            dec_states = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == data_utils.EOS_ID:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_hyp += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    dec_states.append(new_dec_states[idx])

            dec_states = [d.reshape(1, d.shape[0], d.shape[1], d.shape[2]) for d in dec_states]
            dec_states = numpy.concatenate(dec_states, axis=0)

            hyp_scores = numpy.array(hyp_scores)
            live_hyp = new_live_k

            if new_live_k < 1:
                break
            if dead_hyp >= beam_size:
                break

            decoder_inputs = numpy.array([w[-1] for w in hyp_samples])
            decoder_init = numpy.array(hyp_states)
            decoder_states = dec_states

        # dump every remaining one
        if dump_remaining:
            if live_hyp > 0:
                for idx in xrange(live_hyp):
                    sample.append(hyp_samples[idx])
                    sample_score.append(hyp_scores[idx])

        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            sample_score = sample_score / lengths

        # sort the samples by score (it is in log-scale, therefore lower is better)
        sidx = numpy.argsort(sample_score)
        sample = numpy.array(sample)[sidx]
        sample_score = numpy.array(sample_score)[sidx]

        return sample.tolist(), sample_score.tolist()


class Seq2SeqModel(TranslationModel):
    """Sequence-to-sequence model with attention and for multiple buckets.
    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
    or into the seq2seq library for complete model implementation.
    This class also allows to use GRU cells in addition to LSTM cells, and
    sampled softmax to handle large output vocabulary size. A single-layer
    version of this model, but with bi-directional encoder, was presented in
      http://arxiv.org/abs/1409.0473
    and sampled softmax is described in Section 3 of the following paper.
      http://arxiv.org/pdf/1412.2007v2.pdf
    """

    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 buckets,
                 source_proj_size,
                 target_proj_size,
                 encoder_size,
                 decoder_size,
                 num_layers_encoder,
                 num_layers_decoder,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 decoder=None,
                 optimizer='sgd',
                 use_lstm=False,
                 input_feeding=False,
                 combine_inp_attn=False,
                 dropout=0.0,
                 attention_f=None,
                 window_size=10,
                 content_function=None,
                 decoder_attention_f="None",
                 num_samples=512,
                 forward_only=False,
                 max_len=100,
                 cpu_only=False,
                 early_stop_patience=0,
                 save_best_model=True,
                 log_tensorboard=False,
                 dtype=tf.float32):
        """Create the model.
        Args:

        """
        super(Seq2SeqModel, self).__init__()
        assert decoder is not None

        if cpu_only:
            device = "/cpu:0"
        else:
            device = "/gpu:0"

        with tf.device(device):

            self.source_vocab_size = source_vocab_size
            self.target_vocab_size = target_vocab_size
            self.buckets = buckets
            self.batch_size = batch_size
            self.attention_f = attention_f
            self.content_function = content_function
            self.window_size = window_size

            self.combine_inp_attn = combine_inp_attn

            if decoder_attention_f == "None":
                self.decoder_attention_f = None
            else:
                self.decoder_attention_f = decoder_attention_f

            self.decoder = decoder

            # learning rate ops
            self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

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

            if early_stop_patience > 0 or save_best_model:
                self.best_eval_loss = tf.Variable(numpy.inf, trainable=False)
                self.estop_counter = tf.Variable(0, trainable=False)
                self.estop_counter_update_op = self.estop_counter.assign(self.estop_counter + 1)
                self.estop_counter_reset_op = self.estop_counter.assign(0)
            else:
                self.best_eval_loss = None
                self.estop_counter = None
                self.estop_counter_update_op = None
                self.estop_counter_reset_op = None

            self.source_proj_size = source_proj_size
            self.target_proj_size = target_proj_size
            self.encoder_size = encoder_size
            self.decoder_size = decoder_size

            self.input_feeding = input_feeding

            self.max_len = max_len
            self.dropout = dropout
            self.dropout_feed = tf.placeholder(tf.float32, name="dropout_rate")
            self.step_num = tf.Variable(0, trainable=False)

            self.dtype = dtype

            # If we use sampled softmax, we need an output projection.
            loss_function = None

            with tf.device("/cpu:0"):
                w = tf.get_variable("proj_w", [decoder_size, self.target_vocab_size])
                w_t = tf.transpose(w)
                b = tf.get_variable("proj_b", [self.target_vocab_size])
            self.output_projection = (w, b)

            self.sampled_softmax = False

            # Sampled softmax only makes sense if we sample less than vocabulary size.
            if 0 < num_samples < self.target_vocab_size:
                self.sampled_softmax = True
                def sampled_loss(inputs, labels):
                    with tf.device("/cpu:0"):
                        labels = tf.reshape(labels, [-1, 1])
                        return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                                                          self.target_vocab_size)

                loss_function = sampled_loss

            # create the embedding matrix - this must be done in the CPU for now
            with tf.device("/cpu:0"):
                self.src_embedding = tf.Variable(
                        tf.truncated_normal(
                                [source_vocab_size, source_proj_size], stddev=0.01
                        ),
                        name='embedding_src'
                )

                # decoder with attention
                with tf.name_scope('decoder_with_attention') as scope:
                    # create this variable to be used inside the embedding_attention_decoder
                    self.tgt_embedding = tf.Variable(
                            tf.truncated_normal(
                                    [target_vocab_size, target_proj_size], stddev=0.01
                            ),
                            name='embedding'
                    )

            # Create the internal multi-layer cell for our RNN.
            self.encoder_cell, self.decoder_cell = cells.build_nmt_multicell_rnn(
                    num_layers_encoder, num_layers_decoder, encoder_size, decoder_size,
                    source_proj_size, use_lstm=use_lstm, dropout=dropout,
                    input_feeding=input_feeding)

            # The seq2seq function: we use embedding for the input and attention.
            def seq2seq_f(encoder_inputs, decoder_inputs):
                return self.inference(encoder_inputs, decoder_inputs)

            # Feeds for inputs.
            self.encoder_inputs = []
            self.decoder_inputs = []
            self.target_weights = []

            for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))

            for i in xrange(buckets[-1][1] + 1):
                self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None, ], name="decoder{0}".format(i)))
                self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

            # Our targets are decoder inputs shifted by one.
            targets = [self.decoder_inputs[i + 1]
                       for i in xrange(len(self.decoder_inputs) - 1)]

            self.decoder_states_holders = None

            # Training outputs and losses.
            if forward_only:

                # self.batch_size = beam_size

                for i in xrange(len(self.encoder_inputs), self.max_len):
                    self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))

                b_size = array_ops.shape(self.encoder_inputs[0])[0]

                # context, decoder_initial_state, attention_states, input_length
                self.ret0, self.ret1, self.ret2 = self.encode(self.encoder_inputs, b_size)

                if use_lstm:

                    self.decoder_init_plcholder = tf.placeholder(tf.float32,
                                                             shape=[None, (target_proj_size) * 2 * num_layers_decoder],
                                                             name="decoder_init")
                else:

                    # GRU has hidden state with half the size of the LSTM because it does noe have a memory cell
                    self.decoder_init_plcholder = tf.placeholder(tf.float32,
                                                             shape=[None, (target_proj_size) * num_layers_decoder],
                                                             name="decoder_init")

                # shape of this placeholder: the first None indicate the batch size and the second the input length
                self.attn_plcholder = tf.placeholder(tf.float32,
                                                     shape=[None, self.ret2.get_shape()[1], target_proj_size],
                                                     name="attention_states")

                # decoder_states = None
                if self.decoder_attention_f is not None:
                    self.decoder_states_holders = tf.placeholder(tf.float32, shape=[None, None, 1, decoder_size],
                                                                 name="decoder_state")
                decoder_states = self.decoder_states_holders

                self.logits, self.states, self.decoder_states = decoder(
                    decoder_inputs=[self.decoder_inputs[0]], initial_state=self.decoder_init_plcholder,
                    attention_states=self.attn_plcholder, cell=self.decoder_cell,
                    num_symbols=target_vocab_size, attention_f=attention_f,
                    window_size=window_size, content_function=content_function,
                    decoder_attention_f=decoder_attention_f, combine_inp_attn=combine_inp_attn,
                    input_feeding=input_feeding, dropout=self.dropout_feed, initializer=None,
                    decoder_states=decoder_states, step_num=self.step_num, dtype=dtype
                )

                # If we use output projection, we need to project outputs for decoding.
                self.logits = tf.nn.xw_plus_b(self.logits[-1], self.output_projection[0], self.output_projection[1])
                self.logits = nn_ops.softmax(self.logits)

            else:

                tf_version = pkg_resources.get_distribution("tensorflow").version

                if tf_version == "0.6.0" or tf_version == "0.5.0":

                    self.outputs, self.losses = seq2seq.model_with_buckets(
                        encoder_inputs=self.encoder_inputs, decoder_inputs=self.decoder_inputs,
                        targets=targets, weights=self.target_weights, num_decoder_symbols=self.target_vocab_size,
                        buckets=buckets, seq2seq=lambda x, y: seq2seq_f(x, y), softmax_loss_function=loss_function)

                else:

                    self.outputs, self.losses = model_with_buckets(
                        encoder_inputs=self.encoder_inputs, decoder_inputs=self.decoder_inputs,
                        targets=targets, weights=self.target_weights, buckets=buckets,
                        seq2seq_f=lambda x, y: seq2seq_f(x, y), softmax_loss_function=loss_function)

            # Gradients and SGD update operation for training the model.
            params = tf.trainable_variables()
            if not forward_only:
                self.gradient_norms = []
                self.updates = []
                self.gradients = []
                # opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                opt = optimization_ops.get_optimizer(optimizer, learning_rate)
                for b in xrange(len(buckets)):
                    grads = tf.gradients(self.losses[b], params)
                    self.gradients.append(grads)
                    clipped_gradients, norm = tf.clip_by_global_norm(grads,
                                                                     max_gradient_norm)
                    self.gradient_norms.append(norm)
                    self.updates.append(opt.apply_gradients(
                            zip(clipped_gradients, params), global_step=self.global_step))

            self.saver = tf.train.Saver(tf.all_variables())
            self.saver_best = tf.train.Saver(tf.all_variables())

            if log_tensorboard:

                # include everything to log here:
                _ = tf.histogram_summary('W_output_proj', self.output_projection[0])
                _ = tf.histogram_summary('b_output_proj', self.output_projection[1])

                _ = tf.histogram_summary('logits', self.outputs)

                for b in xrange(len(buckets)):
                    _ = tf.histogram_summary('gradient_norm_bucket_{0}'.format(b), self.gradient_norms[b])
                    _ = tf.histogram_summary('update_bucket_{0}'.format(b), self.updates[b])
                    _ = tf.histogram_summary('gradient_bucket_{0}'.format(b), self.gradients[b])
                    _ = tf.scalar_summary('loss_bucket_{0}', self.losses[b])

                # merge the summary ops into one big op
                self.summary_op = tf.merge_all_summaries()

    def inference(self, source, target):
        """
        Function to be used together with the 'model_with_buckets' function from Tensorflow's
            seq2seq module.

        Parameters
        ----------
        source: Tensor
            a Tensor corresponding to the source sentence
        target: Tensor
            A Tensor corresponding to the target sentence
        do_decode: boolean
            Flag indicating whether or not to use the feed_previous parameter of the
                seq2seq.embedding_attention_decoder function.

        Returns
        -------

        """
        b_size = array_ops.shape(source[0])[0]

        # encode source
        context, decoder_initial_state, attention_states = self.encode(source, b_size)

        # decode target - note that we pass decoder_states as None when training the model
        outputs, state, _ = self.decoder(
            decoder_inputs=target, initial_state=decoder_initial_state,
            attention_states=attention_states, cell=self.decoder_cell,
            num_symbols=self.target_vocab_size, attention_f=self.attention_f,
            window_size=self.window_size,  content_function=self.content_function,
            decoder_attention_f=self.decoder_attention_f, combine_inp_attn=self.combine_inp_attn,
            input_feeding=self.input_feeding, dropout=self.dropout_feed,
            initializer=None, decoder_states=None, step_num=self.step_num, dtype=self.dtype
        )

        if self.sampled_softmax is False:
            outputs = [tf.nn.xw_plus_b(o, self.output_projection[0], self.output_projection[1]) for o in outputs]

        # return the output (logits) and internal states
        return outputs, state

    def encode(self, source, batch_size, translate=False):

        # encoder embedding layer and recurrent layer
        # with tf.name_scope('bidirectional_encoder') as scope:
        with tf.name_scope('reverse_encoder') as scope:
            if translate:
                scope.reuse_variables()
            context, decoder_initial_state = encoders.reverse_encoder(
                    source, self.src_embedding, self.encoder_cell,
                    batch_size, dropout=self.dropout_feed, dtype=self.dtype)

            # First calculate a concatenation of encoder outputs to put attention on.
            top_states = [
                tf.reshape(e, [-1, 1, self.encoder_size]) for e in context
                ]
            attention_states = tf.concat(1, top_states)

        return context, decoder_initial_state, attention_states


class NMTModel(TranslationModel):

    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 buckets,
                 source_proj_size,
                 target_proj_size,
                 encoder_size,
                 decoder_size,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 optimizer='sgd',
                 input_feeding=False,
                 combine_inp_attn=False,
                 dropout=0.0,
                 attention_f=None,
                 window_size=10,
                 content_function=None,
                 decoder_attention_f="None",
                 num_samples=512,
                 forward_only=False,
                 max_len=100,
                 cpu_only=False,
                 early_stop_patience=0,
                 save_best_model=True,
                 dtype=tf.float32):
        super(NMTModel, self).__init__()

        if cpu_only:
            device = "/cpu:0"
        else:
            device = "/gpu:0"

        with tf.device(device):

            self.source_vocab_size = source_vocab_size
            self.target_vocab_size = target_vocab_size
            self.buckets = buckets
            self.batch_size = batch_size
            self.attention_f = attention_f
            self.content_function = content_function
            self.window_size = window_size

            self.combine_inp_attn = combine_inp_attn

            if decoder_attention_f == "None":
                self.decoder_attention_f = None
            else:
                self.decoder_attention_f = decoder_attention_f

            # learning rate ops
            self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

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

            if early_stop_patience > 0 or save_best_model:
                self.best_eval_loss = tf.Variable(numpy.inf, trainable=False)
                self.estop_counter = tf.Variable(0, trainable=False)
                self.estop_counter_update_op = self.estop_counter.assign(self.estop_counter + 1)
                self.estop_counter_reset_op = self.estop_counter.assign(0)
            else:
                self.best_eval_loss = None
                self.estop_counter = None
                self.estop_counter_update_op = None
                self.estop_counter_reset_op = None

            self.source_proj_size = source_proj_size
            self.target_proj_size = target_proj_size
            self.encoder_size = encoder_size
            self.decoder_size = decoder_size

            self.input_feeding = input_feeding

            self.max_len = max_len
            self.dropout = dropout
            self.dropout_feed = tf.placeholder(tf.float32, name="dropout_rate")
            self.step_num = tf.Variable(0, trainable=False)

            self.dtype = dtype

            # If we use sampled softmax, we need an output projection.
            loss_function = None

            with tf.device("/cpu:0"):
                w = tf.get_variable("proj_w", [decoder_size, self.target_vocab_size])
                w_t = tf.transpose(w)
                b = tf.get_variable("proj_b", [self.target_vocab_size])
            self.output_projection = (w, b)

            self.sampled_softmax = False

            # Sampled softmax only makes sense if we sample less than vocabulary size.
            if 0 < num_samples < self.target_vocab_size:
                self.sampled_softmax = True
                def sampled_loss(inputs, labels):
                    with tf.device("/cpu:0"):
                        labels = tf.reshape(labels, [-1, 1])
                        return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                                                          self.target_vocab_size)

                loss_function = sampled_loss

            # create the embedding matrix - this must be done in the CPU for now
            with tf.device("/cpu:0"):
                self.src_embedding = tf.Variable(
                    tf.truncated_normal(
                        [source_vocab_size, source_proj_size], stddev=0.01
                    ),
                    name='embedding_src'
                )

                # decoder with attention
                with tf.name_scope('decoder_with_attention') as scope:
                    # create this variable to be used inside the embedding_attention_decoder
                    self.tgt_embedding = tf.Variable(
                        tf.truncated_normal(
                            [target_vocab_size, target_proj_size], stddev=0.01
                        ),
                        name='embedding'
                    )

            # Create the internal multi-layer cell for our RNN.
            self.encoder_cell_fw, self.encoder_cell_bw, self.decoder_cell = cells.build_nmt_bidirectional_cell(
                encoder_size, decoder_size, source_proj_size, target_proj_size, dropout=dropout)

            # The seq2seq function: we use embedding for the input and attention.
            def seq2seq_f(encoder_inputs, decoder_inputs):
                return self.inference(encoder_inputs, decoder_inputs)

            # Feeds for inputs.
            self.encoder_inputs = []
            self.decoder_inputs = []
            self.target_weights = []

            for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))

            for i in xrange(buckets[-1][1] + 1):
                self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None, ], name="decoder{0}".format(i)))
                self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

            # Our targets are decoder inputs shifted by one.
            targets = [self.decoder_inputs[i + 1]
                       for i in xrange(len(self.decoder_inputs) - 1)]

            self.decoder_states_holders = None

            # Training outputs and losses.
            if forward_only:

                # self.batch_size = beam_size

                for i in xrange(len(self.encoder_inputs), self.max_len):
                    self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))

                b_size = array_ops.shape(self.encoder_inputs[0])[0]

                # context, decoder_initial_state, attention_states, input_length
                self.ret0, self.ret1, self.ret2 = self.encode(self.encoder_inputs, b_size)

                self.decoder_init_plcholder = tf.placeholder(tf.float32,
                                                             shape=[None, (target_proj_size) * 2],
                                                             name="decoder_init")

                # shape of this placeholder: the first None indicate the batch size and the second the input length
                self.attn_plcholder = tf.placeholder(tf.float32,
                                                     shape=[None, self.ret2.get_shape()[1], target_proj_size],
                                                     name="attention_states")

                # decoder_states = None
                if self.decoder_attention_f is not None:
                    self.decoder_states_holders = tf.placeholder(tf.float32, shape=[None, None, 1, decoder_size],
                                                                 name="decoder_state")
                decoder_states = self.decoder_states_holders

                self.logits, self.states = attention_decoder_nmt(
                    decoder_inputs=[self.decoder_inputs[0]], initial_state=self.decoder_init_plcholder,
                    attention_states=self.attn_plcholder, cell=self.decoder_cell,
                    num_symbols=target_vocab_size, attention_f=attention_f,
                    window_size=window_size, content_function=content_function,
                    decoder_attention_f=decoder_attention_f, combine_inp_attn=combine_inp_attn,
                    input_feeding=input_feeding, dropout=self.dropout_feed, initializer=None,
                    dtype=dtype
                )

                # If we use output projection, we need to project outputs for decoding.
                self.logits = tf.nn.xw_plus_b(self.logits[-1], self.output_projection[0], self.output_projection[1])
                self.logits = nn_ops.softmax(self.logits)

            else:

                tf_version = pkg_resources.get_distribution("tensorflow").version

                if tf_version == "0.6.0" or tf_version == "0.5.0":

                    self.outputs, self.losses = seq2seq.model_with_buckets(
                        encoder_inputs=self.encoder_inputs, decoder_inputs=self.decoder_inputs,
                        targets=targets, weights=self.target_weights, num_decoder_symbols=self.target_vocab_size,
                        buckets=buckets, seq2seq=lambda x, y: seq2seq_f(x, y), softmax_loss_function=loss_function)

                else:

                    self.outputs, self.losses = model_with_buckets(
                        encoder_inputs=self.encoder_inputs, decoder_inputs=self.decoder_inputs,
                        targets=targets, weights=self.target_weights, buckets=buckets,
                        seq2seq_f=lambda x, y: seq2seq_f(x, y), softmax_loss_function=loss_function)

            # Gradients and SGD update operation for training the model.
            params = tf.trainable_variables()
            if not forward_only:
                self.gradient_norms = []
                self.updates = []
                # opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                opt = optimization_ops.get_optimizer(optimizer, learning_rate)
                for b in xrange(len(buckets)):
                    gradients = tf.gradients(self.losses[b], params)
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                     max_gradient_norm)
                    self.gradient_norms.append(norm)
                    self.updates.append(opt.apply_gradients(
                        zip(clipped_gradients, params), global_step=self.global_step))

            self.saver = tf.train.Saver(tf.all_variables())
            self.saver_best = tf.train.Saver(tf.all_variables())

    def inference(self, source, target):
        """
        Function to be used together with the 'model_with_buckets' function from Tensorflow's
            seq2seq module.

        Parameters
        ----------
        source: Tensor
            a Tensor corresponding to the source sentence
        target: Tensor
            A Tensor corresponding to the target sentence
        do_decode: boolean
            Flag indicating whether or not to use the feed_previous parameter of the
                seq2seq.embedding_attention_decoder function.

        Returns
        -------

        """
        b_size = array_ops.shape(source[0])[0]

        # encode source
        context, decoder_initial_state, attention_states = self.encode(source, b_size)

        # decode target - note that we pass decoder_states as None when training the model
        outputs, state = attention_decoder_nmt(
            decoder_inputs=target, initial_state=decoder_initial_state,
            attention_states=attention_states, cell=self.decoder_cell,
            num_symbols=self.target_vocab_size, attention_f=self.attention_f,
            window_size=self.window_size, content_function=self.content_function,
            decoder_attention_f=self.decoder_attention_f, combine_inp_attn=self.combine_inp_attn,
            input_feeding=self.input_feeding, dropout=self.dropout_feed,
            initializer=None, dtype=self.dtype
        )

        if self.sampled_softmax is False:
            outputs = [tf.nn.xw_plus_b(o, self.output_projection[0], self.output_projection[1]) for o in outputs]

        # return the output (logits) and internal states
        return outputs, state

    def encode(self, source, batch_size, translate=False):

        # encoder embedding layer and recurrent layer
        # with tf.name_scope('bidirectional_encoder') as scope:
        with tf.name_scope('reverse_encoder') as scope:
            if translate:
                scope.reuse_variables()
            context, decoder_initial_state = encoders.bidirectional_encoder(
                source, self.src_embedding, self.encoder_cell_fw, self.encoder_cell_bw,
                dropout=self.dropout_feed, dtype=self.dtype)

            # First calculate a concatenation of encoder outputs to put attention on.
            top_states = [
                tf.reshape(e, [-1, 1, self.encoder_size * 2]) for e in context
                ]
            attention_states = tf.concat(1, top_states)

        return context, decoder_initial_state, attention_states
# -*- coding: utf-8 -*-
"""
    Sequence-to-sequence model with bi-directional encoder and the attention mechanism described in

        arxiv.org/abs/1412.2007

    and support to buckets.

"""
import copy
import random
import numpy
import tensorflow as tf

from tensorflow.models.rnn import seq2seq, rnn
from tensorflow.python.ops import nn_ops, embedding_ops  #, rnn
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import data_utils
import attention
import build_ops
# from six.moves import xrange


def _reverse_encoder(source,
                    src_embedding,
                    encoder_cell,
                    batch_size,
                    dtype=tf.float32):
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

    outputs, state = rnn.rnn(encoder_cell, emb_inp,
                              initial_state=initial_state,
                              dtype=dtype,
                              scope='reverse_encoder')

    hidden_states = outputs

    decoder_initial_state = state

    return hidden_states, decoder_initial_state


def _decode(target,
            decoder_cell,
            decoder_initial_state,
            attention_states,
            target_vocab_size,
            output_projection,
            batch_size,
            do_decode=False,
            input_feeding=False,
            attention_type=None,
            content_function='vinyals_kayser',
            output_attention=False,
            translate=False,
            beam_size=12,
            dtype=tf.float32):
    """

    Parameters
    ----------
    target
    decoder_cell
    decoder_initial_state
    attention_states
    target_vocab_size
    output_projection
    batch_size
    do_decode
    input_feeding
    attention_type
    content_function
    output_attention
    translate
    beam_size
    dtype

    Returns
    -------

    """
    assert attention_type is not None

    assert attention_type is 'local' or attention_type is 'global' or attention_type is 'hybrid'

    # decoder with attention
    with tf.name_scope('decoder_with_attention') as scope:

        if translate:

            b_symbols, log_probs, b_path = attention.embedding_attention_decoder(
                target, decoder_initial_state, attention_states,
                decoder_cell, batch_size, target_vocab_size,
                output_size=None, output_projection=output_projection,
                feed_previous=do_decode, input_feeding=input_feeding,
                attention_type=attention_type, dtype=dtype,
                content_function=content_function,
                output_attention=output_attention,
                translate=translate, beam_size=beam_size,
                scope='decoder_with_attention'
            )

            return b_symbols, log_probs, b_path

        else:

            # run the decoder with attention
            outputs, states = attention.embedding_attention_decoder(
                target, decoder_initial_state, attention_states,
                decoder_cell, batch_size, target_vocab_size,
                output_size=None, output_projection=output_projection,
                feed_previous=do_decode, input_feeding=input_feeding,
                attention_type=attention_type, dtype=dtype,
                content_function=content_function,
                output_attention=output_attention,
                translate=translate, beam_size=beam_size,
                scope='decoder_with_attention'
            )

            return outputs, states


class Seq2SeqModel(object):
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
                 optimizer='sgd',
                 use_lstm=False,
                 input_feeding=False,
                 dropout=0.0,
                 attention_type='global',
                 content_function='vinyals_kayser',
                 num_samples=512,
                 forward_only=False,
                 max_len=100,
                 cpu_only=False,
                 output_attention=False,
                 early_stop_patience=0,
                 beam_size=12,
                 dtype=tf.float32):
        """Create the model.
        Args:
          source_vocab_size: size of the source vocabulary.
          target_vocab_size: size of the target vocabulary.
          buckets: a list of pairs (I, O), where I specifies maximum input length
            that will be processed in that bucket, and O specifies maximum output
            length. Training instances that have inputs longer than I or outputs
            longer than O will be pushed to the next bucket and padded accordingly.
            We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
          size: number of units in each layer of the model.
          num_layers_encoder: number of layers in the model.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          use_lstm: if true, we use LSTM cells instead of GRU cells.
          num_samples: number of samples for sampled softmax.
          forward_only: if set, we do not construct the backward pass in the model.
        """
        if cpu_only:
            device = "/cpu:0"
        else:
            device = "/gpu:0"

        with tf.device(device):

            self.source_vocab_size = source_vocab_size
            self.target_vocab_size = target_vocab_size
            self.buckets = buckets
            self.batch_size = batch_size
            self.attention_type = attention_type
            self.content_function = content_function

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

            self.source_proj_size = source_proj_size
            self.target_proj_size = target_proj_size
            self.encoder_size = encoder_size
            self.decoder_size = decoder_size

            self.input_feeding = input_feeding
            self.output_attention = output_attention
            self.max_len = max_len
            self.dropout = dropout

            self.dtype = dtype

            # If we use sampled softmax, we need an output projection.
            self.output_projection = None
            loss_function = None

            # Sampled softmax only makes sense if we sample less than vocabulary size.
            if 0 < num_samples < self.target_vocab_size:
                with tf.device("/cpu:0"):
                    w = tf.get_variable("proj_w", [decoder_size, self.target_vocab_size])
                    w_t = tf.transpose(w)
                    b = tf.get_variable("proj_b", [self.target_vocab_size])
                self.output_projection = (w, b)

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
            self.encoder_cell, self.decoder_cell = build_ops.build_nmt_multicell_rnn(
                    num_layers_encoder, num_layers_decoder, encoder_size, decoder_size,
                    source_proj_size, target_proj_size, use_lstm=use_lstm, dropout=dropout,
                    input_feeding=input_feeding)

            # The seq2seq function: we use embedding for the input and attention.
            def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                return self.inference(encoder_inputs, decoder_inputs, do_decode)

            # Feeds for inputs.
            self.encoder_inputs = []
            self.decoder_inputs = []
            self.target_weights = []
            # dropout feed
            dropout_feed = tf.placeholder(tf.float32, name="dropout_feed")

            for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))

            for i in xrange(buckets[-1][1] + 1):
                self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None, ], name="decoder{0}".format(i)))
                self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

            # Our targets are decoder inputs shifted by one.
            targets = [self.decoder_inputs[i + 1]
                       for i in xrange(len(self.decoder_inputs) - 1)]

            # Training outputs and losses.
            if forward_only:

                for i in xrange(len(self.encoder_inputs), self.max_len):
                    self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))

                b_size = array_ops.shape(self.encoder_inputs[0])[0]

                # context, decoder_initial_state, attention_states
                self.context, self.decoder_initial_state, self.attention_states = self.encode(self.encoder_inputs, b_size)

                self.outputs, self.scores, self.hypothesis_path = _decode(
                    [self.decoder_inputs[0]], self.decoder_cell, self.decoder_initial_state, self.attention_states,
                    self.target_vocab_size, self.output_projection, batch_size=b_size,
                    attention_type=self.attention_type, content_function=self.content_function, do_decode=True,
                    input_feeding=self.input_feeding, dtype=self.dtype, output_attention=self.output_attention,
                    translate=forward_only, beam_size=beam_size
                )

            else:

                self.outputs, self.losses = seq2seq.model_with_buckets(
                    encoder_inputs=self.encoder_inputs, decoder_inputs=self.decoder_inputs,
                    targets=targets, weights=self.target_weights, buckets=buckets,
                    seq2seq=lambda x, y: seq2seq_f(x, y, False), softmax_loss_function=loss_function)

            # Gradients and SGD update operation for training the model.
            params = tf.trainable_variables()
            if not forward_only:
                self.gradient_norms = []
                self.updates = []
                # opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                opt = build_ops.get_optimizer(optimizer, learning_rate)
                for b in xrange(len(buckets)):
                    gradients = tf.gradients(self.losses[b], params)
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                     max_gradient_norm)
                    self.gradient_norms.append(norm)
                    self.updates.append(opt.apply_gradients(
                            zip(clipped_gradients, params), global_step=self.global_step))

            self.saver = tf.train.Saver(tf.all_variables())
            self.saver_best = tf.train.Saver(tf.all_variables())

    def inference(self, source, target, do_decode=False):
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
        outputs, state = _decode(target, self.decoder_cell, decoder_initial_state, attention_states,
                                                  self.target_vocab_size, self.output_projection,
                                                  batch_size=b_size, attention_type=self.attention_type,
                                                  do_decode=do_decode, input_feeding=self.input_feeding,
                                                  content_function=self.content_function, dtype=self.dtype,
                                                  output_attention=self.output_attention)

        # return the output (logits) and internal states
        return outputs, state

    def encode(self, source, batch_size, translate=False):

        # encoder embedding layer and recurrent layer
        # with tf.name_scope('bidirectional_encoder') as scope:
        with tf.name_scope('reverse_encoder') as scope:
            if translate:
                scope.reuse_variables()
            context, decoder_initial_state = _reverse_encoder(
                    source, self.src_embedding, self.encoder_cell,
                    batch_size, dtype=self.dtype)

            # First calculate a concatenation of encoder outputs to put attention on.
            top_states = [
                tf.reshape(e, [-1, 1, self.encoder_size]) for e in context
                ]
            attention_states = tf.concat(1, top_states)

        return context, decoder_initial_state, attention_states

    def get_train_batch(self, data, bucket_id, batch=None):
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

        if batch is None:
            batch = self.batch_size

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in xrange(batch):
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
                              for batch_idx in xrange(batch)], dtype=numpy.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                    numpy.array([decoder_inputs[batch_idx][length_idx]
                              for batch_idx in xrange(batch)], dtype=numpy.int32))

            # Create target_weights to be 0 for targets that are padding.
            # batch_weight = numpy.ones(self.batch_size, dtype=numpy.float32)
            batch_weight = numpy.ones(batch, dtype=numpy.float32)
            # for batch_idx in xrange(self.batch_size):
            for batch_idx in xrange(batch):
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
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.

        else:
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

    def translation_step(self, session, token_ids, normalize=True):

        # Get a 1-element batch to feed the sentence to the model
        encoder_inputs, decoder_inputs = self.get_translate_batch([(token_ids, [])])
        decoder_inputs = decoder_inputs[-1]

        # here we encode the input sentence
        input_feed = {}
        for l in xrange(self.max_len):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]

        input_feed[self.decoder_inputs[0].name] = decoder_inputs[0],

        # we select the last element of ret0 to keep as it is a list of hidden_states
        output_feed = [self.outputs, self.scores, self.hypothesis_path]

        # outputs will be a list of length beam_size containing the best generated hypothesis
        symbols, probs, path = session.run(output_feed, input_feed)

        hypotheses = [[]]
        probabilities = [[]]
        tempH = []
        tempS = []
        for time in xrange(len(symbols)):
            # if time is 0, we add all the symbols to the temp list
            c = 0
            for p in path[time]:
                tempH.append(hypotheses[p] + [symbols[time][c]])
                tempS.append(probabilities[p] + [probs[time][c]])
                c += 1
            #
            hypotheses = copy.copy(tempH)
            probabilities = copy.copy(tempS)
            tempH = []
            tempS = []

        sample, sample_score = [], []

        for hypothesis, hyp_score in zip(hypotheses, probabilities):

            try:
                # ge the index of the EOS symbol
                idx = hypothesis.index(data_utils.EOS_ID)
            except ValueError:
                idx = len(token_ids)

            hyp = hypothesis[0:idx]

            if idx > 0:
                score = hyp_score[idx-1]
            else:
                score = hyp_score[0]

            length = len(hyp)
            if normalize:
                score /= length

            sample.append(hypothesis)
            sample_score.append(hyp_score)

        n_best = numpy.array(sample)
        n_best_scores = numpy.array(sample_score)

        order = n_best_scores.argsort()[::-1]

        n_best = n_best[order]
        n_best_scores = n_best_scores[order]

        return n_best.tolist(), n_best_scores.tolist()

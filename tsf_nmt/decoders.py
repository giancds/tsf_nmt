# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import cells
from attention import global_attention
from content_functions import decoder_type_2, vinyals_kaiser, mod_bahdanau
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, embedding_ops, math_ops, nn_ops
from tensorflow.python.ops import variable_scope as vs

# from six.moves import xrange

_SEED = 1234


# TODO: finish pydocs


def _embed_inputs(decoder_inputs, num_symbols, input_size, input_feeding=False):

    with ops.device("/cpu:0"):

        if input_feeding:
            embedding = vs.get_variable("embedding", [num_symbols, input_size / 2])

        else:
            embedding = vs.get_variable("embedding", [num_symbols, input_size])

        emb_inp = [embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]

    return emb_inp


def attention_decoder(decoder_inputs, initial_state, attention_states, cell, num_symbols,
                      attention_f=global_attention, window_size=10, content_function=vinyals_kaiser,
                      decoder_attention_f=decoder_type_2, combine_inp_attn=False, input_feeding=False,
                      dropout=None, initializer=None, decoder_states=None, step_num=None,
                      dtype=tf.float32, scope=None):
    """

    Helper function implementing a RNN decoder with global, local or hybrid attention for the sequence-to-sequence
        model.

    Parameters
    ----------

    decoder_inputs: list
            a list of 2D Tensors [batch_size x cell.input_size].

    initial_state: tensor
            2d Tensor [batch_size x (number of decoder layers * hidden_layer_size * 2)] if LSTM or
            [batch_size x (number of decoder layers * hidden_layer_size)] if GRU representing the initial
                state (usually, we take the states of the encoder) to be used when running the decoder. The '2' on
                the LSTM formula mean that we have to set the hidden state and the cell state.

    attention_states: tensor
            3D tensor [batch_size x attn_length (time) x attn_size (hidden_layer_size)] representing the encoder
                hidden states that will be used to derive the context (attention) vector.

    cell: RNNCell
            rnn_cell.RNNCell defining the cell function and size.

    batch_size: tensor
            tensor representing the batch size used when training the model

    attention_f: function
            function indicating which type of attention to use. Default to global_attention.

    loop_function:
            if not None, this function will be applied to i-th output
                in order to generate i+1-th input, and decoder_inputs will be ignored,
                except for the first element ("GO" symbol). This can be used for decoding,
                but also for training to emulate http://arxiv.org/pdf/1506.03099v2.pdf.
            Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x cell.output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x cell.input_size].

    window_size: int
            size of the window to apply on local attention.Default to 10.

    input_feeding : boolean
            Flag indicating where to use the "input feeding approach" proposed by Luong et al. (2015).
                Default to False.

    content_function: string

    dtype:
            The dtype to use for the RNN initial state (default: tf.float32).

    scope:
            VariableScope for the created subgraph; default: "attention_decoder".

    Returns
    -------

    outputs:
            A list of the same length as decoder_inputs of 2D Tensors of shape
                [batch_size x output_size]. These represent the generated outputs.
                Output i is computed from input i (which is either i-th decoder_inputs or
                loop_function(output {i-1}, i)) as follows. First, we run the cell
                on a combination of the input and previous attention masks:
                    cell_output, new_state = cell(linear(input, prev_attn), prev_state).
                Then, we calculate new attention masks:
                    new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
                and then we calculate the output:
                    output = linear(cell_output, new_attn).

    states:
            The state of each decoder cell in each time-step. This is a list
                with length len(decoder_inputs) -- one item for each time-step.
                Each item is a 2D Tensor of shape [batch_size x cell.state_size].

    """
    assert attention_f is not None

    output_size = cell.output_size

    if dropout is not None:

        for c in cell._cells:
            c.input_keep_prob = 1.0 - dropout

    if initializer is None:
        initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1, seed=_SEED)

    with vs.variable_scope(scope or "embedding_attention_decoder", initializer=initializer):

        emb_inp = _embed_inputs(decoder_inputs, num_symbols, cell.input_size, input_feeding=input_feeding)

        batch = array_ops.shape(emb_inp[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(attention_states, [-1, attn_length, 1, attn_size])

        cell_states = initial_state
        cell_outputs = []
        outputs = []
        batch_attn_size = array_ops.pack([batch, attn_size])

        # initial attention state
        ct = array_ops.zeros(batch_attn_size, dtype=dtype)
        ct.set_shape([None, attn_size])

        for i in xrange(len(emb_inp)):
            if i > 0:
                vs.get_variable_scope().reuse_variables()

            if input_feeding:
                # if using input_feeding, concatenate previous attention with input to layers
                inp = array_ops.concat(1, [emb_inp[i], ct])
            else:
                inp = emb_inp[i]

            if combine_inp_attn:
                # Merge input and previous attentions into one vector of the right size.
                x = cells.linear([inp] + [ct], cell.input_size, True)
            else:
                x = inp

            # Run the RNN.
            cell_output, new_state = cell(x, cell_states)
            cell_states = new_state
            cell_outputs.append(cell_output)

            # dt = new_state
            if content_function is mod_bahdanau:
                dt = cell_outputs[-2]
            else:
                dt = cell_output

            ct = attention_f(decoder_hidden_state=dt, hidden_attn=hidden,
                             initializer=initializer, window_size=window_size,
                             content_function=content_function, dtype=dtype)

            #
            with vs.variable_scope("AttnOutputProjection", initializer=initializer):

                # if we pass a list of tensors, linear will first concatenate them over axis 1
                output = cells.linear([ct] + [cell_output], output_size, True)

                output = tf.tanh(output)

            outputs.append(output)

    return outputs, cell_states, None


def attention_decoder_output(decoder_inputs, initial_state, attention_states, cell, num_symbols,
                             attention_f=global_attention, window_size=10, content_function=vinyals_kaiser,
                             decoder_attention_f=decoder_type_2, combine_inp_attn=False, input_feeding=False,
                             dropout=None, initializer=None, decoder_states=None, step_num=None,
                             dtype=tf.float32, scope=None):
    """

    Helper function implementing a RNN decoder with global, local or hybrid attention for the sequence-to-sequence
        model.

    Parameters
    ----------

    decoder_inputs: list
            a list of 2D Tensors [batch_size x cell.input_size].

    initial_state: tensor
            3D Tensor [batch_size x attn_length x attn_size].

    attention_states:

    cell: RNNCell
            rnn_cell.RNNCell defining the cell function and size.

    batch_size: int
            batch size when training the model

    attention_f: function
            function indicating which type of attention to use. Default to global_attention.

    output_size: int
            size of the output vectors; if None, we use cell.output_size.

    loop_function:
            if not None, this function will be applied to i-th output
                in order to generate i+1-th input, and decoder_inputs will be ignored,
                except for the first element ("GO" symbol). This can be used for decoding,
                but also for training to emulate http://arxiv.org/pdf/1506.03099v2.pdf.
            Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x cell.output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x cell.input_size].

    window_size: int
            size of the window to apply on local attention

    input_feeding: boolean
            whether or not to use the input feeding approach by Luong et al., 2015.

    content_function: string

    dtype:
            The dtype to use for the RNN initial state (default: tf.float32).

    scope:
            VariableScope for the created subgraph; default: "attention_decoder".

    Returns
    -------

    outputs:
            A list of the same length as decoder_inputs of 2D Tensors of shape
                [batch_size x output_size]. These represent the generated outputs.
                Output i is computed from input i (which is either i-th decoder_inputs or
                loop_function(output {i-1}, i)) as follows. First, we run the cell
                on a combination of the input and previous attention masks:
                    cell_output, new_state = cell(linear(input, prev_attn), prev_state).
                Then, we calculate new attention masks:
                    new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
                and then we calculate the output:
                    output = linear(cell_output, new_attn).

    states:
            The state of each decoder cell in each time-step. This is a list
                with length len(decoder_inputs) -- one item for each time-step.
                Each item is a 2D Tensor of shape [batch_size x cell.state_size].

    """
    assert attention_f is not None

    output_size = cell.output_size

    if dropout is not None:

        for c in cell._cells:
            c.input_keep_prob = 1.0 - dropout

    if initializer is None:
        initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1, seed=_SEED)

    with vs.variable_scope(scope or "attention_decoder", initializer=initializer):

        emb_inp = _embed_inputs(decoder_inputs, num_symbols, cell.input_size, input_feeding=input_feeding)

        batch = array_ops.shape(emb_inp[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(attention_states, [-1, attn_length, 1, attn_size])

        cell_state = initial_state

        outputs = []
        batch_attn_size = array_ops.pack([batch, attn_size])

        # initial attention state
        ct = array_ops.zeros(batch_attn_size, dtype=dtype)
        ct.set_shape([None, attn_size])

        if decoder_states is None:
            cell_outputs = []
        else:
            cell_outputs = decoder_states

        for i in xrange(len(emb_inp)):
            if i > 0:
                vs.get_variable_scope().reuse_variables()

            if input_feeding:
                # if using input_feeding, concatenate previous attention with input to layers
                inp = array_ops.concat(1, [emb_inp[i], ct])
            else:
                inp = emb_inp[i]

            if combine_inp_attn:
                # Merge input and previous attentions into one vector of the right size.
                x = cells.linear([inp] + [ct], cell.input_size, True)
            else:
                x = inp

            # Run the RNN.
            cell_output, new_state = cell(x, cell_state)
            cell_state = new_state

            if decoder_states is None:

                # states.append(new_state)  # new_state = dt#
                cell_outputs.append(cell_output)

            else:
                reshaped = tf.reshape(cell_output, [-1, 1, 1, attn_size])
                decoder_states = tf.concat(1, [decoder_states, reshaped])

            # dt = new_state
            if content_function is mod_bahdanau:
                dt = cell_outputs[-2]
            else:
                dt = cell_output

            ct = attention_f(decoder_hidden_state=dt, hidden_attn=hidden,
                             initializer=initializer, window_size=window_size,
                             content_function=content_function, dtype=dtype)

            with vs.variable_scope("AttnOutputProjection", initializer=initializer):

                if decoder_states is None:

                    shape1 = len(cell_outputs)

                    top_states = [tf.reshape(o, [-1, 1, attn_size]) for o in cell_outputs]

                    output_attention_states = tf.concat(1, top_states)

                    decoder_hidden = array_ops.reshape(output_attention_states, [-1, shape1, 1, attn_size])

                    ht_hat = decoder_output_attention(decoder_hidden,
                                                      attn_size,
                                                      decoder_attention_f,
                                                      initializer=initializer)
                else:

                    decoder_hidden = decoder_states

                    ht_hat = decoder_output_attention(decoder_hidden,
                                                      attn_size,
                                                      decoder_attention_f,
                                                      initializer=initializer,
                                                      step_num=step_num)

                output = cells.linear([ct] + [ht_hat], output_size, True)

                output = tf.tanh(output)

            outputs.append(output)

    if decoder_states is None:

        cell_outs = [tf.reshape(o, [-1, 1, 1, attn_size]) for o in cell_outputs]

        cell_outputs = tf.concat(1, cell_outs)

    else:

        cell_outputs = decoder_states

    return outputs, cell_state, cell_outputs


def decoder_output_attention(decoder_hidden, attn_size, decoder_attention_f, initializer=None, step_num=None):
    """

    Parameters
    ----------
    decoder_states
    attn_size

    Returns
    -------

    """
    assert initializer is not None

    with vs.variable_scope("decoder_output_attention", initializer=initializer):

        s = decoder_attention_f(decoder_hidden, attn_size)

        # beta will be (?, timesteps)
        beta = nn_ops.softmax(s)

        if step_num is None:  # step_num is None when training

            shape = decoder_hidden.get_shape()
            timesteps = shape[1].value
            b = array_ops.reshape(beta, [-1, timesteps, 1, 1])

        else:

            b = array_ops.reshape(beta, tf.pack([-1, step_num, 1, 1]))

        # b  and decoder_hidden will be (?, timesteps, 1, 1)
        d = math_ops.reduce_sum(b * decoder_hidden, [1, 2])

        # d will be (?, decoder_size)
        ds = tf.reshape(d, [-1, attn_size])

    # ds is (?, decoder_size)
    return ds


def attention_decoder_nmt(decoder_inputs, initial_state, attention_states, cell, num_symbols,
                          attention_f=global_attention, window_size=10, content_function=vinyals_kaiser,
                          decoder_attention_f=decoder_type_2, combine_inp_attn=False, input_feeding=False,
                          dropout=None, initializer=None, dtype=tf.float32, scope=None):
    """

    Helper function implementing a RNN decoder with global, local or hybrid attention for the sequence-to-sequence
        model.

    Parameters
    ----------

    decoder_inputs: list
            a list of 2D Tensors [batch_size x cell.input_size].

    initial_state: tensor
            2d Tensor [batch_size x (number of decoder layers * hidden_layer_size * 2)] if LSTM or
            [batch_size x (number of decoder layers * hidden_layer_size)] if GRU representing the initial
                state (usually, we take the states of the encoder) to be used when running the decoder. The '2' on
                the LSTM formula mean that we have to set the hidden state and the cell state.

    attention_states: tensor
            3D tensor [batch_size x attn_length (time) x attn_size (hidden_layer_size)] representing the encoder
                hidden states that will be used to derive the context (attention) vector.

    cell: RNNCell
            rnn_cell.RNNCell defining the cell function and size.

    batch_size: tensor
            tensor representing the batch size used when training the model

    attention_f: function
            function indicating which type of attention to use. Default to global_attention.

    loop_function:
            if not None, this function will be applied to i-th output
                in order to generate i+1-th input, and decoder_inputs will be ignored,
                except for the first element ("GO" symbol). This can be used for decoding,
                but also for training to emulate http://arxiv.org/pdf/1506.03099v2.pdf.
            Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x cell.output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x cell.input_size].

    window_size: int
            size of the window to apply on local attention.Default to 10.

    input_feeding : boolean
            Flag indicating where to use the "input feeding approach" proposed by Luong et al. (2015).
                Default to False.

    content_function: string

    dtype:
            The dtype to use for the RNN initial state (default: tf.float32).

    scope:
            VariableScope for the created subgraph; default: "attention_decoder".

    Returns
    -------

    outputs:
            A list of the same length as decoder_inputs of 2D Tensors of shape
                [batch_size x output_size]. These represent the generated outputs.
                Output i is computed from input i (which is either i-th decoder_inputs or
                loop_function(output {i-1}, i)) as follows. First, we run the cell
                on a combination of the input and previous attention masks:
                    cell_output, new_state = cell(linear(input, prev_attn), prev_state).
                Then, we calculate new attention masks:
                    new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
                and then we calculate the output:
                    output = linear(cell_output, new_attn).

    states:
            The state of each decoder cell in each time-step. This is a list
                with length len(decoder_inputs) -- one item for each time-step.
                Each item is a 2D Tensor of shape [batch_size x cell.state_size].

    """
    assert attention_f is not None

    output_size = cell.output_size

    if dropout is not None:

        cell.input_keep_prob = 1.0 - dropout

    if initializer is None:
        initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1, seed=_SEED)

    with vs.variable_scope(scope or "embedding_attention_decoder", initializer=initializer):
        emb_inp = _embed_inputs(decoder_inputs, num_symbols, cell.input_size, input_feeding=input_feeding)

        batch = array_ops.shape(emb_inp[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(attention_states, [-1, attn_length, 1, attn_size])

        cell_states = initial_state
        initial_state_decoder = tf.zeros_like(initial_state)
        initial_state_decoder.set_shape([None, initial_state.get_shape()[1].value])
        cell_outputs = [initial_state_decoder]
        outputs = []
        batch_attn_size = array_ops.pack([batch, attn_size])

        # initial attention state
        ct = array_ops.zeros(batch_attn_size, dtype=dtype)
        ct.set_shape([None, attn_size])

        for i in xrange(len(emb_inp)):
            if i > 0:
                vs.get_variable_scope().reuse_variables()

            if input_feeding:
                # if using input_feeding, concatenate previous attention with input to layers
                inp = array_ops.concat(1, [emb_inp[i], ct])
            else:
                inp = emb_inp[i]

            if combine_inp_attn:
                # Merge input and previous attentions into one vector of the right size.
                x = cells.linear([inp] + [ct], cell.input_size, True)
            else:
                x = inp

            dt = cell_outputs[-1]

            ct = attention_f(decoder_hidden_state=dt, hidden_attn=hidden,
                             initializer=initializer, window_size=window_size,
                             content_function=content_function, dtype=dtype)

            # Run the RNN.
            cell_output, new_state = cell(x, cell_states, context=ct)
            cell_states = new_state
            cell_outputs.append(cell_output)

            #
            with vs.variable_scope("AttnOutputProjection", initializer=initializer):

                with vs.variable_scope("AttnOutputProjection_logit_lstm", initializer=initializer):

                    # if we pass a list of tensors, linear will first concatenate them over axis 1
                    logit_lstm = cells.linear([cell_output], num_symbols, True)

                with vs.variable_scope("AttnOutputProjection_logit_ctx", initializer=initializer):

                    # if we pass a list of tensors, linear will first concatenate them over axis 1
                    logit_ctx = cells.linear([ct], num_symbols, True)

                with vs.variable_scope("AttnOutputProjection_logit_emb", initializer=initializer):

                    # if we pass a list of tensors, linear will first concatenate them over axis 1
                    logit_prev = cells.linear([x], num_symbols, True)

                output = tf.tanh(logit_lstm + logit_prev + logit_ctx)

            outputs.append(output)

    return outputs, cell_states
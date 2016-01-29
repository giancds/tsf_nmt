# -*- coding: utf-8 -*-

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import print_function
import tensorflow as tf

from train_ops import train
from translate_ops import decode_from_stdin, decode_from_file
import attention

# flags related to the model optimization
tf.app.flags.DEFINE_float('learning_rate', 1.0, 'Learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5, 'Learning rate decays by this much.')
tf.app.flags.DEFINE_integer('start_decay', 4, 'Start learning rate decay at this epoch. Set to 0 to use patience.')
tf.app.flags.DEFINE_string('optimizer', 'sgd',
                           'Name of the optimizer to use (adagrad, adam, rmsprop or sgd')

tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'Clip gradients to this norm.')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size to use during training.')
tf.app.flags.DEFINE_integer('beam_size', 12, 'Max size of the beam used for decoding.')
tf.app.flags.DEFINE_integer('max_epochs', 10, 'Max number of epochs to use during training. The actual value will be (max_epochs-1) as it is 0-based.')
tf.app.flags.DEFINE_integer('max_train_data_size', 0,
                            'Limit on the size of training data (0: no limit).')

# flags related to model architecture
tf.app.flags.DEFINE_string('model', 'seq2seq', 'one of these 2 models: seq2seq or bidirectional')
tf.app.flags.DEFINE_string('attention_type', 'local', 'Which type of attention to use. One of \'local\', \'global\' and \'hybrid\'.')
tf.app.flags.DEFINE_string('content_function', attention.VINYALS_KAISER, 'Type of content-based function to define the attention. One of vinyals_kayser, luong_general and luong_dot')
tf.app.flags.DEFINE_boolean('use_lstm', True, 'Whether to use LSTM units. Default to False.')
tf.app.flags.DEFINE_boolean('input_feeding', True, 'Whether to input the attention states as part of input to the decoder at each timestep. Default to False.')
tf.app.flags.DEFINE_boolean('output_attention', False, 'Whether to pay attention on the decoder outputs. Default to False.')
tf.app.flags.DEFINE_integer('proj_size', 100, 'Size of words projection.')
tf.app.flags.DEFINE_integer('hidden_size', 100, 'Size of each layer.')
tf.app.flags.DEFINE_integer('num_layers', 4, 'Number of layers in each component of the model.')
tf.app.flags.DEFINE_float('dropout', 0.0, 'Dropout rate. When the value is 0.0 dropout is turned off. Optimal should be 0.2 as indicated by Zaremba et al. (2014)')

# flags related to the source and target vocabularies
tf.app.flags.DEFINE_integer('src_vocab_size', 30000, 'Source language vocabulary size.')
tf.app.flags.DEFINE_integer('tgt_vocab_size', 30000, 'Target vocabulary size.')

# information about the datasets and their location
tf.app.flags.DEFINE_string('model_name', 'model_local_lstm_hid200_proj100_en30000_pt30000_sgd1.0.ckpt', 'Data directory')
tf.app.flags.DEFINE_string('data_dir', '/home/gian/data/', 'Data directory')
tf.app.flags.DEFINE_string('train_dir', '/home/gian/train_local/', 'Train directory')
tf.app.flags.DEFINE_string('train_data', 'fapesp-v2.pt-en.train.tok.%s', 'Data for training.')
tf.app.flags.DEFINE_string('valid_data', 'fapesp-v2.pt-en.dev.tok.%s', 'Data for validation.')
tf.app.flags.DEFINE_string('test_data', 'fapesp-v2.pt-en.test-a.tok.%s', 'Data for testing.')
tf.app.flags.DEFINE_string('vocab_data', '', 'Training directory.')
tf.app.flags.DEFINE_string('source_lang', 'en', 'Source language extension.')
tf.app.flags.DEFINE_string('target_lang', 'pt', 'Target language extension.')

# verbosity and checkpoints
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 100,
                            'How many training steps to do per checkpoint.')
tf.app.flags.DEFINE_integer('steps_per_validation', 1000,
                            'How many training steps to do between each validation.')
tf.app.flags.DEFINE_integer('steps_verbosity', 10,
                            'How many training steps to do between each information print.')

# pacience flags (learning_rate decay and early stop)
tf.app.flags.DEFINE_integer('lr_rate_patience', 3, 'How many training steps to monitor.')
tf.app.flags.DEFINE_integer('early_stop_patience', 10, 'How many training steps to monitor.')

# decoding/testing flags
tf.app.flags.DEFINE_boolean('decode_file', False, 'Set to True for decoding sentences in a file.')
tf.app.flags.DEFINE_boolean('decode_input', False, 'Set to True for interactive decoding.')

tf.app.flags.DEFINE_boolean('self_test', False, 'Run a self-test if this is set to True.')

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

# _buckets = [(40, 50)]

def main(_):
    if FLAGS.decode_input:
        decode_from_stdin(show_all_n_best=True, FLAGS=FLAGS, buckets=_buckets)
    elif FLAGS.decode_file:
        decode_from_file('/home/gian/data/fapesp-v2.pt-en.test-a.tok.en', FLAGS=FLAGS, buckets=_buckets)
    else:
        train(FLAGS=FLAGS, buckets=_buckets)


if __name__ == '__main__':
    tf.app.run()

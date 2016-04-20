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
from tensorflow.python.platform import gfile

import content_functions
import attention
import nmt_models
import decoders
from train_ops import train_nmt
from translate_ops import decode_from_stdin, decode_from_file

flags = tf.flags

# flags related to the model optimization
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_float('learning_rate_decay_factor', 1.0, 'Learning rate decays by this much. Setting it to 1.0 will not affect the learning rate.')
flags.DEFINE_integer('start_decay', 0, 'Start learning rate decay at this epoch. Set to 0 to use patience.')
flags.DEFINE_integer('stop_decay', 0, 'Stop learning rate decay at this epoch. Set to 0 to use patience.')
flags.DEFINE_string('optimizer', 'adam', 'Name of the optimizer to use (adagrad, adam, rmsprop or sgd')

flags.DEFINE_float('max_gradient_norm', 5.0, 'Clip gradients to this norm.')
flags.DEFINE_integer('batch_size', 32, 'Batch size to use during training.')
flags.DEFINE_integer('beam_size', 12, 'Max size of the beam used for decoding.')
flags.DEFINE_integer('num_samples_loss', 0, 'Number of samples to use in sampled softmax. Set to 0 to use regular loss.')
flags.DEFINE_integer('max_len', 120, 'Max size of the beam used for decoding.')
flags.DEFINE_integer('max_epochs', 23,  'Max number of epochs to use during training. The actual value will be (max_epochs-1) as it is 0-based.')
flags.DEFINE_integer('max_train_data_size', 0, 'Limit on the size of training data (0: no limit).')

flags.DEFINE_boolean('cpu_only', False, 'Whether or not to use GPU only.')

# flags related to model architecture
flags.DEFINE_string('model', 'nmt', 'one of these models: seq2seq or nmt')
flags.DEFINE_string('attention_type', attention.GLOBAL, 'Which type of attention to use. One of local, global and hybrid.')
flags.DEFINE_integer('window_size', 10, 'Size of each size of the window to use when applying local attention. Not relevant to global attention')
flags.DEFINE_string('content_function', content_functions.BAHDANAU_NMT, 'Type of content-based function to define the attention. One of vinyals_kayser, luong_general and luong_dot')
flags.DEFINE_boolean('use_lstm', False, 'Whether to use LSTM units. Default to False.')
flags.DEFINE_boolean('input_feeding', False, 'Whether to input the attention states as part of input to the decoder at each timestep. Default to False.')
flags.DEFINE_string('output_attention', 'None', 'Whether to pay attention on the decoder outputs. Default to False.')
flags.DEFINE_integer('proj_size', 500, 'Size of words projection.')
flags.DEFINE_integer('hidden_size', 500, 'Size of each layer.')
flags.DEFINE_integer('num_layers', 1, 'Number of layers in each component of the model.')

flags.DEFINE_float('dropout', 0.0, 'Dropout rate. When the value is 0.0 dropout is turned off. Optimal should be 0.2 as indicated by Zaremba et al. (2014)')

# flags related to the source and target vocabularies
flags.DEFINE_integer('src_vocab_size', 30000, 'Source language vocabulary size.')
flags.DEFINE_integer('tgt_vocab_size', 30000, 'Target vocabulary size.')

# information about the datasets and their location
flags.DEFINE_string('model_name', 'model_nmt_global_output_None_bahdanau_1lr_hid500_proj500_en30000_pt30000_maxNrm5_adam_dropout-off_input-feed-off_att.ckpt',
                           'Model name')
flags.DEFINE_string('data_dir', '/home/gian/data/', 'Data directory')
flags.DEFINE_string('train_dir', '/home/gian/train_global/model_nmt_global_output_none_bahdanau_1lr_hid500_proj500_en30000_pt30000_maxNrm5_adam_dropout-off_input-feed-off_att/', 'Train directory')
flags.DEFINE_string('best_models_dir', '/home/gian/train_global/', 'Train directory')
flags.DEFINE_string('train_data', 'fapesp-v2.pt-en.train.tok.%s', 'Data for training.')
flags.DEFINE_string('valid_data', 'fapesp-v2.pt-en.dev.tok.%s', 'Data for validation.')
flags.DEFINE_string('test_data', 'fapesp-v2.pt-en.test-a.tok.%s', 'Data for testing.')
flags.DEFINE_string('vocab_data', '', 'Training directory.')
flags.DEFINE_string('source_lang', 'en', 'Source language extension.')
flags.DEFINE_string('target_lang', 'pt', 'Target language extension.')

# verbosity and checkpoints
flags.DEFINE_integer('steps_per_checkpoint', 500, 'How many training steps to do per checkpoint.')
flags.DEFINE_integer('steps_per_validation', 1000, 'How many training steps to do between each validation.')
flags.DEFINE_integer('steps_verbosity', 10, 'How many training steps to do between each information print.')
flags.DEFINE_boolean('log_tensorboard', True, 'Whether or not to use Tensorboard to log info about training. Default to False.')

# pacience flags (learning_rate decay and early stop)
flags.DEFINE_integer('lr_rate_patience', 3, 'How many training steps to monitor.')
flags.DEFINE_integer('early_stop_patience', 20, 'How many training steps to monitor.')
flags.DEFINE_integer('early_stop_after_epoch', 20, 'Start monitoring early_stop after this epoch.')
flags.DEFINE_boolean('save_best_model', True, 'Set to True to save the best model even if not using early stop.')

# decoding/testing flags
flags.DEFINE_boolean('decode_file', False, 'Set to True for decoding sentences in a file.')
flags.DEFINE_boolean('decode_input', False, 'Set to True for interactive decoding.')

FLAGS = flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
# _buckets = [(5, 10), (10, 15), (20, 25), (40, 50), (50, 50)]
_buckets = [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 50)]
# _buckets = [(50, 50)]


def main(_):

    if FLAGS.decode_input:
        decode_from_stdin(show_all_n_best=True, FLAGS=FLAGS, buckets=_buckets)

    elif FLAGS.decode_file:

        # model_path = FLAGS.best_models_dir + FLAGS.model_name + '-best-0'
        model_path = None
        decode_from_file(['/home/gian/data/fapesp-v2.pt-en.test-a.tok.en',
                          '/home/gian/data/fapesp-v2.pt-en.test-b.tok.en'],
                         model_path=model_path, use_best=True, FLAGS=FLAGS,
                         buckets=_buckets)

    else:
        train_nmt(FLAGS=FLAGS, buckets=_buckets)


if __name__ == '__main__':
    tf.app.run()

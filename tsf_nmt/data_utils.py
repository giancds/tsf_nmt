# -*- coding: utf-8 -*-
"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import print_function
import collections
import os
import re
import sys
from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = '_PAD'
_GO = '_GO'
_EOS = '_EOS'
_UNK = '_UNK'
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

_DIGIT_RE = re.compile(r'\d')


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for ss in sentence.strip().split():
        words.append(ss)
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      normalize_digits=True):
    """
    Create vocabulary file (if it does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(vocabulary_path):
        print('Creating vocabulary %s from data %s' % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode='r') as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 10000 == 0:
                    print("  processing line %d" % counter)
                tokens = basic_tokenizer(line)
                for w in tokens:
                    word = re.sub(_DIGIT_RE, '0', w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode='w') as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + '\n')


def initialize_vocabulary(vocabulary_path):
    """
    Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {'dog': 0, 'cat': 1}, and this function will
    also return the reversed-vocabulary ['dog', 'cat'].

    Args:
      vocabulary_path: path to the file containing the vocabulary.

    Returns:
      a pair: the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).

    Raises:
      ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode='r') as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError('Vocabulary file %s not found.', vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          normalize_digits=True):
    """
    Convert a string to list of integers representing token-ids.

    For example, a sentence 'I have a dog' may become tokenized into
    ['I', 'have', 'a', 'dog'] and with vocabulary {'I': 1, 'have': 2,
    'a': 4, 'dog': 7'} this function will return [1, 2, 4, 7].

    Args:
      sentence: a string, the sentence to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
      a list of integers, the token-ids for the sentence.
    """
    words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(re.sub(_DIGIT_RE, '0', w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
    """
    Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(target_path):
        print('Tokenizing data in %s' % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode='r') as data_file:
            with gfile.GFile(target_path, mode='w') as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 10000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, normalize_digits)
                    tokens_file.write(' '.join([str(tok) for tok in token_ids]) + '\n')


def prepare_nmt_data(FLAGS):
    """Get WMT data into data_dir, create vocabularies and tokenize data.

    Args:
      src_vocabulary_size: size of the source vocabulary to create and use.
      tgt_vocabulary_size: size of the target vocabulary to create and use.

    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for source training data-set,
        (2) path to the token-ids for target training data-set,
        (3) path to the token-ids for source development data-set,
        (4) path to the token-ids for target development data-set,
        (5) path to the token-ids for source test data-set,
        (6) path to the token-ids for target test data-set,
    """
    # setting relevant info:
    data_dir = FLAGS.data_dir
    train_data = data_dir + FLAGS.train_data
    valid_data = data_dir + FLAGS.valid_data
    test_data = data_dir + FLAGS.test_data

    source_lang = FLAGS.source_lang
    target_lang = FLAGS.target_lang

    src_vocabulary_size = FLAGS.src_vocab_size
    tgt_vocabulary_size = FLAGS.tgt_vocab_size

    # Create vocabularies of the appropriate sizes.
    src_vocab_path = (train_data % str(src_vocabulary_size)) + ('.vocab.%s' % source_lang)
    tgt_vocab_path = (train_data % str(tgt_vocabulary_size)) + ('.vocab.%s' % target_lang)

    create_vocabulary(src_vocab_path, train_data % source_lang, src_vocabulary_size)
    create_vocabulary(tgt_vocab_path, train_data % target_lang, tgt_vocabulary_size)

    # Create token ids for the training data.
    src_train_ids_path = (train_data % str(src_vocabulary_size)) + ('.ids.%s' % source_lang)
    tgt_train_ids_path = (train_data % str(tgt_vocabulary_size)) + ('.ids.%s' % target_lang)

    data_to_token_ids(train_data % source_lang, src_train_ids_path, src_vocab_path)
    data_to_token_ids(train_data % target_lang, tgt_train_ids_path, tgt_vocab_path)

    # Create token ids for the development data.
    src_dev_ids_path = (valid_data % str(src_vocabulary_size)) + ('.ids.%s' % source_lang)
    tgt_dev_ids_path = (valid_data % str(tgt_vocabulary_size)) + ('.ids.%s' % target_lang)

    data_to_token_ids(valid_data % source_lang, src_dev_ids_path, src_vocab_path)
    data_to_token_ids(valid_data % target_lang, tgt_dev_ids_path, tgt_vocab_path)

    # Create token ids for the test data.
    src_test_ids_path = (test_data % str(src_vocabulary_size)) + ('.ids.%s' % source_lang)
    tgt_test_ids_path = (test_data % str(tgt_vocabulary_size)) + ('.ids.%s' % target_lang)

    data_to_token_ids(test_data % source_lang, src_test_ids_path, src_vocab_path)
    data_to_token_ids(test_data % target_lang, tgt_test_ids_path, tgt_vocab_path)

    return (src_train_ids_path, tgt_train_ids_path,
            src_dev_ids_path, tgt_dev_ids_path,
            src_test_ids_path, tgt_test_ids_path)


def prepare_lm_data(FLAGS):
    """Get WMT data into data_dir, create vocabularies and tokenize data.

    Args:
      src_vocabulary_size: size of the source vocabulary to create and use.
      tgt_vocabulary_size: size of the target vocabulary to create and use.

    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for source training data-set,
        (2) path to the token-ids for source development data-set,
        (3) path to the token-ids for source test data-set,
    """
    # setting relevant info:
    data_dir = FLAGS.data_dir
    train_data = data_dir + FLAGS.train_data
    valid_data = data_dir + FLAGS.valid_data
    test_data = data_dir + FLAGS.test_data

    source_lang = FLAGS.source_lang

    src_vocabulary_size = FLAGS.src_vocab_size

    # Create vocabularies of the appropriate sizes.
    src_vocab_path = (train_data % str(src_vocabulary_size)) + ('.vocab.%s' % source_lang)
    create_vocabulary(src_vocab_path, train_data % source_lang, src_vocabulary_size)

    # Create token ids for the training data.
    src_train_ids_path = (train_data % str(src_vocabulary_size)) + ('.ids.%s' % source_lang)
    data_to_token_ids(train_data % source_lang, src_train_ids_path, src_vocab_path)

    # Create token ids for the development data.
    src_dev_ids_path = (valid_data % str(src_vocabulary_size)) + ('.ids.%s' % source_lang)
    data_to_token_ids(valid_data % source_lang, src_dev_ids_path, src_vocab_path)

    # Create token ids for the test data.
    src_test_ids_path = (test_data % str(src_vocabulary_size)) + ('.ids.%s' % source_lang)
    data_to_token_ids(test_data % source_lang, src_test_ids_path, src_vocab_path)

    return src_train_ids_path, src_dev_ids_path, src_test_ids_path


def read_nmt_data(source_path, target_path, FLAGS=None, buckets=None, max_size=None):
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
                target_ids.append(EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


def read_lm_data(source_path, FLAGS=None, max_size=None):
    """Read data from source and shift by one to be the LM target.


    """
    assert FLAGS is not None

    data_set = []
    counter = 0
    with gfile.GFile(source_path, mode='r') as source_file:
            source = source_file.readline()

            while source and (not max_size or counter < max_size):
                counter += 1

                if counter % 10000 == 0:
                    print('  reading data line %d' % counter)
                    sys.stdout.flush()

                source_ids = [int(x) for x in source.split()]
                data_set.append(source_ids)

                source = source_file.readline()

    return data_set
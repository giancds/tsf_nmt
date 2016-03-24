# -*- coding: utf-8 -*-
import tensorflow as tf


def get_optimizer(name='sgd', lr_rate=0.1, decay=0.9):
    """

    Parameters
    ----------
    name
    lr_rate
    decay

    Returns
    -------

    """
    optimizer = None
    if name is 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr_rate)
    elif name is 'adagrad':
        optimizer = tf.train.AdagradOptimizer(lr_rate)
    elif name is 'adam':
        optimizer = tf.train.AdamOptimizer(lr_rate, epsilon=1e-8)
    elif name is 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(lr_rate, decay)
    else:
        raise ValueError('Optimizer not found.')
    return optimizer
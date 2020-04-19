"""
Python module defining various metrics.

"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.layers import BatchNormalization, MaxPool2D, ReLU
from tensorflow.keras.layers import Lambda, Dropout
from tensorflow.keras import Model
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras import backend as K
import numpy as np

VECTOR_SIZE = 512*4*4 # resnet18
# VECTOR_SIZE = 512*8*8 # mobilenetv2

def compute_accuracy(y_true, y_pred, mode='dist', margin=0.5):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    if mode == 'dot':
        pred = y_pred.ravel() > margin
        return np.mean(pred == y_true)
    elif mode == 'dist':
        pred = y_pred.ravel() < margin
        return np.mean(pred == y_true)

def accuracy(y_true, y_pred, mode='dist', margin=0.5):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    if mode == 'dot':
        return K.mean(K.equal(y_true, K.cast(y_pred > margin, y_true.dtype)))
    elif mode == 'dist':
        return K.mean(K.equal(y_true, K.cast(y_pred < margin, y_true.dtype)))

def get_label(y_true, y_pred, mode='dist', margin=0.5):

    if y_true is None:
        dtype = tf.float32
    else:
        dtype = y_true.dtype

    if mode == 'dist':
        return K.cast(y_pred < 0.5, dtype)
    elif mode == 'dot':
        return K.cast(y_pred > 0.5, dtype)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def target_category_loss(x, category_index, num_dims=2048):
    return tf.multiply(x, K.one_hot([category_index], num_dims))


def cos_sim_pos(y_true, y_pred, concat=True, vec_size=VECTOR_SIZE):
    """ Cosine similarity between anchor and positive sample
        Higher value is better.
    """

    if concat:
        anchor_vec = y_pred[:, :vec_size]
        positive_vec = y_pred[:, vec_size:2*vec_size]
    else:
        anchor_vec = y_pred[0]
        positive_vec = y_pred[1]

    d1 = tf.keras.losses.cosine_similarity(anchor_vec, positive_vec)
    return d1


def cos_sim_neg(y_true, y_pred, concat=True, vec_size=VECTOR_SIZE):
    """ Cosine similarity between anchor and negative sample
        Lower value is better.
    """

    if concat:
        anchor_vec = y_pred[:, :vec_size]
        negative_vec = y_pred[:, 2*vec_size:]
    else:
        anchor_vec = y_pred[0]
        negative_vec = y_pred[1]

    d2 = tf.keras.losses.cosine_similarity(anchor_vec, negative_vec)
    return d2

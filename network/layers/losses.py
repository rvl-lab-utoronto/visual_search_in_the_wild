"""
Python module defining losses.

"""

import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


VECTOR_SIZE = 512*4*4 # resnet18
# VECTOR_SIZE = 512*88 # mobnetv2

def contrastive_loss(y_true, y_pred, margin=0.7):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    This loss encourages the embedding to be close to each other for
    the samples of the same label and the embedding to be far apart at least
    by the margin constant for the samples of different labels.

    (from Keras/Tf addd-on)

    '''
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0.))
    return K.mean(y_true * square_pred + (1. - y_true) * margin_square)

def triplet_loss(y_true, y_pred, margin=0.5):
    anchor = y_pred[:, :VECTOR_SIZE]
    positive = y_pred[:, VECTOR_SIZE:2*VECTOR_SIZE]
    negative = y_pred[:, 2*VECTOR_SIZE:]

    d1 = tf.keras.losses.cosine_similarity(anchor, positive)
    d2 = tf.keras.losses.cosine_similarity(anchor, negative)

    return tf.keras.backend.clip(d1 - d2 + margin, 0, None)

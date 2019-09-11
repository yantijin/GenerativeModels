import tensorflow as tf
from tensorflow import keras as ks
import numpy as np

def cosineSimilarity(labels, pred, axis):
    labelNorm = tf.linalg.norm(labels, axis=axis)
    predNorm = tf.linalg.norm(pred, axis=axis)
    tmp = tf.multiply(pred, labels) / (labelNorm, predNorm)
    return 1-tmp

def relative_euclidean_distance(laebls, pred, axis):
    t1 = tf.linalg.norm(laebls-pred, axis=axis)
    return t1 / tf.linalg.norm(laebls, axis=axis)


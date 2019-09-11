import tensorflow as tf
from tensorflow import keras as ks
from utils import *
import numpy as np

class DAGMM():

    def __init__(self, seq_len, hidden_size, z_dim):
        self.inputs = tf.placeholder(shape=[None, seq_len], dtype=tf.float32)
        self._seq_len = seq_len
        self._hidden_size = hidden_size
        self._z_dim = z_dim

    @property
    def seq_len(self):
        return self._seq_len

    @property
    def hidden_size(self):
        return self.hidden_size

    @property
    def z_dim(self):
        return self._z_dim

    def encoder(self):
        with tf.variable_scope('Encoder', reuse=tf.compat.v1.AUTO_REUSE):
            layer1 = ks.layers.Dense(self.hidden_size,
                                     activation=tf.nn.relu,
                                     use_bias=True,
                                     name='layer1')(self.inputs)
            layer2 = ks.layers.Dense(self.hidden_size,
                                     activation=tf.nn.relu,
                                     use_bias=True,
                                     name='layer2')(layer1)
            layer3 = ks.layers.Dense(self.z_dim,
                                     activation=tf.nn.relu,
                                     use_bias=True,
                                     name='layer2')(layer2)
            return layer3

    def decoder(self, z):
        with tf.variable_scope('Decoder', tf.compat.v1.AUTO_REUSE):
            layer1 = ks.layers.Dense(self.hidden_size,
                                     activation=tf.nn.relu,
                                     use_bias=True,
                                     name='layer1')(z)
            layer2 = ks.layers.Dense(self.hidden_size,
                                     activation=tf.nn.relu,
                                     use_bias=True,
                                     name='layer1')(layer1)
            layer3 = ks.layers.Dense(self.seq_len,
                                     activation=tf.nn.relu,
                                     use_bias=True,
                                     name='layer1')(layer2)
            return layer3

    def recon(self):
        z = self.encoder()
        out = self.decoder(z)
        # reconErr = self.inputs - out
        return out

    def EstiInput(self):
        zc = self.encoder()
        recon = self.decoder(zc)
        term1 = cosineSimilarity(self.inputs, recon, axis=1)
        term2 = relative_euclidean_distance(self.inputs, recon, axis=1)
        out = tf.concat([zc, tf.concat([term1, term2], axis=-1)], axis=-1)
        return out

    def EstimateNet(self):







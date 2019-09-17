import tensorflow as tf
from tensorflow import keras as ks
from .utils import *
import numpy as np

class DAGMM():

    def __init__(self, seq_len, hidden_size, z_dim, estimate_hidden_size,
                 n_gmm=4, lr=1e-4, batch_size=1024, max_epoch=200,
                 lambda1=0.1, lambda2=0.005, data_path='./kdd_cup.npz', ):
        self.inputs = tf.placeholder(shape=[None, seq_len], dtype=tf.float32)
        self._seq_len = seq_len
        self._hidden_size = hidden_size
        self._z_dim = z_dim
        self._estimate_hidden_size = estimate_hidden_size
        self._n_gmm = n_gmm # number of Gaussian components
        self._lambda1 = lambda1
        self._lambda2 = lambda2
        self._data_path = data_path
        self._batch_size = batch_size
        self._max_epoch = max_epoch
        self._lr = lr

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def max_epoch(self):
        return self.max_epoch

    @property
    def seq_len(self):
        return self._seq_len

    @property
    def hidden_size(self):
        return self.hidden_size

    @property
    def z_dim(self):
        return self._z_dim

    @property
    def estimate_hidden_size(self):
        return self._estimate_hidden_size

    @property
    def n_gmm(self):
        return self._n_gmm

    @property
    def lambda1(self):
        return self._lambda1

    @property
    def lambda2(self):
        return self._lambda2

    @property
    def data_path(self):
        return self._data_path

    @property
    def lr(self):
        return self._lr

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
        with tf.variable_scope('estimateNet'):
            out = self.EstiInput()
            layer1 = ks.layers.Dense(self.estimate_hidden_size,
                                     activation=tf.nn.tanh,
                                     use_bias=True,
                                     name='layer1')(out)
            layer1 = tf.nn.dropout(layer1, rate=0.5)

            gamma = ks.layers.Dense(self.n_gmm,
                                     activation=tf.nn.softmax,
                                     use_bias=False,
                                     name='out')(layer1)
            return gamma

    def EZ(self, gamma, z):
        '''
        :param gamma: estimate output [N, K]
        :param z: shape [N, D]
        :return:
        '''
        with tf.variable_scope('EZ'):
            phi = tf.reduce_sum(gamma, 0) / tf.shape(gamma)[0] # [K,1]
            mu = tf.reduce_sum(tf.expand_dims(gamma,-1)*tf.expand_dims(z,1),0)/tf.reduce_sum(gamma,0)  # [K,D]
            sigma = tf.reduce_sum(
                tf.expand_dims(gamma,-1) *
                tf.expand_dims(
                    tf.matmul(
                        tf.expand_dims(z-mu, -1),
                        tf.expand_dims(z-mu, 1)
                    )
                    ,1)
                ,0)\
                    /tf.reduce_sum(gamma,0)  # [K, D, D]
            residual = tf.expand_dims(z,1)-mu  # [N, K, D]
            den = tf.matmul(
                tf.matmul(
                tf.expand_dims(residual, 2),
                tf.linalg.inv(sigma)),
                tf.expand_dims(residual, -1))  # [N, K, 1, 1]
            EZ = -tf.log(
                tf.reduce_sum(
                    tf.multiply(
                        phi,
                        tf.squeeze(den)
                    ) / tf.linalg.norm(2 * np.pi * sigma, axis=(1,2)),
                1)
            )  # [N , 1]
            lossSigma = tf.reduce_sum(
                1 / tf.linalg.diag_part(sigma)[:, self.z_dim])
            return EZ, lossSigma

    def loss(self):
        z = self.EstiInput()
        gamma = self.EstimateNet()
        EZ, lossSigma = self.EZ(gamma, z)
        reconError = tf.reduce_mean(tf.square(z, self.inputs))
        EZError = tf.reduce_mean(EZ)
        loss = reconError + \
               self.lambda1 * EZError + \
               self.lambda2 * lossSigma
        return loss, reconError, EZError, lossSigma

    def train(self, data):
        train, test, train_labels, test_labels = data_loader(self.data_path)

        # build tf graph
        loss, reconError, EZError, lossSigma = self.loss()
        train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        # train step
        len_train = train.shape[0]
        iterNum = len_train // self.batch_size
        TrainIndex = list(range(len_train))
        np.random.shuffle(TrainIndex)

        testIterNum = test.shape[0] // self.batch_size
        TestIndex = list(range(test.shape[0]))
        np.random.shuffle(TestIndex)

        with tf.Session() as sess:
            for epoch in range(self.max_epoch):
                for i in range(iterNum):
                    inputData = train[TrainIndex[i*self.batch_size:(i+1)*self.batch_size]]
                    _, loss_, recon_, EZ_, SIGMA_ = sess.run([train_op, loss, reconError, EZError, lossSigma],
                                                             feed_dict={self.inputs: inputData})
            if epoch % 5 == 0:
                print("recon: {}, EZ: {}, SIGMA {} loss {}".format(recon_, EZ_, SIGMA_, loss_))

            if epoch % 20 == 0:
                for i in range(testIterNum):
                    inputData = test[TestIndex[i*self.batch_size:(i+1)*self.batch_size]]
                    _, loss_, recon_, EZ_, SIGMA_ = sess.run([train_op, loss, reconError, EZError, lossSigma],
                                                             feed_dict={self.inputs: inputData})
                print("Test loss:{}".format(loss_))


















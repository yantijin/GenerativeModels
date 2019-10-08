import tensorflow as tf
import numpy as np
from .utils import *

def sample_Z(batch_size, z_dim):
    return np.random.normal(0., 1., size = [batch_size, z_dim])


class GAN():
    def __init__(self, input_dim, hidden_dim, z_dim):
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._z_dim = z_dim
        self._input = tf.placeholder([None, input_dim], dtype=tf.float32, name='input')
        self._z = tf.placeholder([None, z_dim], dtype=tf.float32, name='noise_input')

    def input_dim(self):
        return self._input_dim

    def hidden_dim(self):
        return self._hidden_dim

    def z_dim(self):
        return self._z_dim

    def generator(self):
        with tf.variable_scope('generator'):
            G1 = fcLayer(self._z, 'gen1layer', self.hidden_dim, activation_func=tf.nn.relu)
            G2 = fcLayer(G1, 'gen2layer', self.z_dim, activation_func=tf.nn.sigmoid)

            return G2

    def discriminator(self, input):
        with tf.variable_scope('discriminator', reuse=tf.compat.v1.AUTO_REUSE):
            G1 = fcLayer(input, 'dis1layer', self.hidden_dim, activation_func=tf.nn.relu)
            G2 = fcLayer(G1, 'dis2layer', 1, activation_func=tf.nn.sigmoid)

            return G2

    def loss(self):
        with tf.variable_scope('loss'):
            # generate fake data
            x_fake = self.generator()

            xreal_prob = self.discriminator(self._input)
            xfake_prob = self.discriminator(x_fake)
            term1 = tf.reduce_mean(tf.log(xreal_prob + 1e-8))
            term2 = tf.reduce_mean(tf.log(1 - xfake_prob + 1e-8))
            Dloss = -(term1+term2)

            # Gloss
            Gloss = -tf.reduce_mean(tf.log(xfake_prob + 1e-8))

            return Dloss, Gloss


    def train(self, batch_size, learning_rate, n_epochs):

        Dloss, Gloss = self.loss()
        solver_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Dloss)
        solver_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Gloss)

        train_total_data, train_size, _, _, test_data, test_labels = prepare_MNIST_data()
        x_test = test_data[:100, :]
        total_batch = int(train_size / batch_size)


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            index = 0
            # writer = tf.summary.FileWriter('./logs', sess.graph)

            for epoch in range(n_epochs):
                np.random.shuffle(train_total_data)
                train_data_ = train_total_data[:, :-10]

                for batch in range(total_batch):
                    index += 1

                    with tf.Session() as sess:
                        _, loss_D = sess.run([solver_D, Dloss],
                                             feed_dict={
                                                 self._input: train_data_[batch*batch_size:(batch+1)*batch_size],
                                                 self._z: sample_Z(batch_size, z_dim=self.z_dim)
                                             })
                        _, loss_G = sess.run([solver_G, Gloss],
                                             feed_dict={
                                                 self._z: sample_Z(batch_size, self.z_dim)
                                             })

                    if index % 500 == 0:
                        print("index:{}, lossD:{}, lossG:{}, ".format(index, loss_D, loss_G))
                if epoch +1 == n_epochs:
                    test_data_ = test_data[:100, :]









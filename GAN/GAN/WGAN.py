import tensorflow as tf
import numpy as np
from scipy.misc import imresize, imsave
from utils import *
from mnist_data import *


def sample_Z(batch_size, z_dim):
    res = np.random.uniform(-1, 1., size = [batch_size, z_dim])
    return res


class GAN():
    def __init__(self, input_dim, hidden_dim, z_dim):
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._z_dim = z_dim
        self._input = tf.placeholder(tf.float32, [None, input_dim], name='input')
        self._z = tf.placeholder(tf.float32, [None, z_dim], name='noise_input')

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @property
    def z_dim(self):
        return self._z_dim

    @property
    def input(self):
        return self._input

    @property
    def z(self):
        return self._z

    def generator(self):
        with tf.variable_scope('generator', reuse=tf.compat.v1.AUTO_REUSE):
            G1 = fcLayer(self.z, 'gen1layer', self.hidden_dim, activation_func=tf.nn.relu)
            G2 = fcLayer(G1, 'gen2layer', self.input_dim, activation_func=tf.nn.sigmoid)
            GVarList = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

            return G2, GVarList

    def discriminator(self, input):
        with tf.variable_scope('discriminator', reuse=tf.compat.v1.AUTO_REUSE):
            G1 = fcLayer(input, 'dis1layer', self.hidden_dim, activation_func=tf.nn.relu)
            G2 = fcLayer(G1, 'dis2layer', 1, activation_func=None)
            DVarList = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')

            return G2, DVarList

    def loss(self):
        # with tf.variable_scope('loss'):
        # generate fake data
        """
         与GAN相比，有四点改进
        （1）Dis为判别器，此时为W——dis，应去掉sigmoid
        （2）Dloss和Gloss中去掉对数
        （3）为满足Lipschitz限制，将dis参数控制在一定范围内
        （4）尽量不使用给予动量的优化算法，推荐RMSProp,SGD也可以
        """

        x_fake, GVarList = self.generator()

        xreal_prob, DVarList = self.discriminator(self.input)
        xfake_prob, DVarList = self.discriminator(x_fake)

        # Dloss
        term1 = tf.reduce_mean(xreal_prob)
        term2 = tf.reduce_mean(xfake_prob)
        Dloss = -(term1-term2)

        # Gloss
        Gloss = -tf.reduce_mean(xfake_prob)

        # clip
        clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in DVarList]

        return x_fake, Dloss, Gloss, GVarList, DVarList, clip_D


    def train(self, batch_size, learning_rate, n_epochs):

        xfake, Dloss, Gloss, GVarList, DVarList, clip_D = self.loss()
        solver_D = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(Dloss, var_list=DVarList)
        solver_G = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(Gloss, var_list=GVarList)

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

                    # 此处是采用5:1频率比进行优化的

                    noise = sample_Z(batch_size, z_dim=self._z_dim)
                    for ind in range(5):
                        _, loss_D, _ = sess.run([solver_D, Dloss, clip_D],
                                               feed_dict={
                                                 self._input: train_data_[batch*batch_size:(batch+1)*batch_size],
                                                 self._z: noise
                                               })
                    _, loss_G = sess.run([solver_G, Gloss],
                                         feed_dict={
                                             self._z: sample_Z(batch_size, self.z_dim)
                                         })


                    if index % 500 == 0:
                        print("index:{}, lossD:{}, lossG:{}, ".format(index, loss_D, loss_G))
                if epoch +1 == n_epochs:
                    # get fake data
                    fakex = sess.run(xfake,  feed_dict={self._z: sample_Z(batch_size, z_dim=self.z_dim)})
                    fakex = fakex.reshape(batch_size, 28, 28)

                    img = np.zeros((28*4, int(28*batch_size/4)))

                    for idx, image in enumerate(fakex):

                        i = int(idx % int(batch_size / 4))
                        j = int(idx / int(batch_size / 4))
                        image_ = imresize(image, size=(28, 28), interp='bicubic')
                        img[j * 28: j * 28 + 28, i * 28: i * 28 + 28] = image_
                    imsave('./WGAN' + str(epoch) + '.jpg', img)


if __name__ == '__main__':
    tf.reset_default_graph()
    batch_size = 100
    learning_rate = 1e-3
    n_epochs = 100
    gan = GAN(input_dim=28*28, hidden_dim=128, z_dim=100)
    gan.train(batch_size, learning_rate, n_epochs)







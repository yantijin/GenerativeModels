import tensorflow as tf
from utils import *
from mnist_data import *
import numpy as np
from scipy.misc import imresize, imsave
class FCAE():
    def __init__(self, inputs, hidden_dim, z_dim, activation_func=tf.nn.relu):
        self._inputs = inputs
        self._hidden_dim = hidden_dim
        self._z_dim = z_dim
        self._activation_func = activation_func

    @property
    def inputs(self):
        return self._inputs

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @property
    def z_dim(self):
        return self._z_dim

    @property
    def activation_func(self):
        return self._activation_func

    @property
    def output_dim(self):
        ShapeList = self.inputs.get_shape().as_list() #[batch_szie, len]
        return ShapeList[-1]

    def fcEncoder(self):
        with tf.variable_scope('fcEncoder'):
            # ShapeList = inputs.get_shape().as_list()  # [batch, len]
            fclayer1 = fcLayer(self.inputs, 'fclayer1', self.hidden_dim, activation_func=self.activation_func)
            fclayer2 = fcLayer(fclayer1, 'fclayer2', self.z_dim, activation_func=self.activation_func)
            return fclayer2

    def fcDecoder(self, res):
        with tf.variable_scope('fcDecoder'):
            fclayer1 = fcLayer(res, 'fclayer1', self.hidden_dim, activation_func=self.activation_func)
            fclayer2 = fcLayer(fclayer1, 'fclayer2', self.output_dim, activation_func=self.activation_func)

            return fclayer2

    def _reconstruct(self):
        z = self.fcEncoder()
        reconX = self.fcDecoder(z)

        return reconX

    def _loss(self):
        reconX = self._reconstruct()
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(reconX-self.inputs)))
        return loss

    def reconstruction(self):
        return self._reconstruct()

    def get_loss(self):
        return self._loss()




class FCVAE():
    def __init__(self, seq_len, hidden_dim, z_dim, activation_func=tf.nn.relu):
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, seq_len])
        self._hidden_dim = hidden_dim
        self._z_dim = z_dim
        self._activation_func = activation_func

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @property
    def z_dim(self):
        return self._z_dim

    @property
    def output_dim(self):
        ShapeList = self.inputs.get_shape().as_list()
        return ShapeList[-1]

    @property
    def activation_func(self):
        return self._activation_func

    def FCEcoder(self):
        with tf.variable_scope('Ecnoder', reuse=tf.compat.v1.AUTO_REUSE):
            fclayer1 = fcLayer(self.inputs, 'fclayer1', self.hidden_dim, activation_func=tf.nn.elu)#self.activation_func)
            fclayer2 = fcLayer(fclayer1, 'fclayer2', self.hidden_dim, activation_func=tf.nn.tanh)#self.activation_func)

            mean = fcLayer(fclayer2, 'mean', self.z_dim) #, activation_func=self.activation_func)
            std = fcLayer(fclayer2, 'std', self.z_dim, activation_func=tf.nn.softplus)
            return mean, std

    def FCDecoder(self, z=None):
        with tf.variable_scope('Decoder', reuse=tf.compat.v1.AUTO_REUSE):
            # get samples from standard Gaussian distribution
            if z == None:
                z = tf.random_normal((tf.shape(self.inputs)[0], self.z_dim))

            fclayer1 = fcLayer(z, 'fclayer1', self.hidden_dim, activation_func=tf.nn.tanh)#self.activation_func)
            fclayer2 = fcLayer(fclayer1, 'fclayer2', self.hidden_dim, activation_func=tf.nn.elu)#self.activation_func)

            # mean = fcLayer(fclayer2, 'mean', self.output_dim)#, activation_func=self.activation_func)
            # std = fcLayer(fclayer2, 'std', self.output_dim, activation_func=tf.nn.softplus)

            # epsilon = tf.random_normal(tf.shape(mean))
            # out = tf.nn.sigmoid(mean + tf.multiply(epsilon, std))
            fclayer2 = fcLayer(fclayer2, 'output', self.output_dim, activation_func=tf.nn.sigmoid)
            return fclayer2

    def reconstruction(self):
        mean, std = self.FCEcoder()
        epislon = tf.random_normal(tf.shape(mean))
        z = mean + tf.multiply(epislon, std)
        reconX = self.FCDecoder(z)
        # logits = self.FCDecoder(z)
        # alpha = tf.random_uniform(tf.shape(logits), minval=0, maxval=1, dtype=tf.float32)
        # reconX = tf.cast(tf.less(alpha, tf.nn.sigmoid(logits)), dtype=tf.float32)

        # epsilon = tf.random_normal(tf.shape(outMean))
        # reconX = outMean + epsilon * outStd
        return reconX

    def _loss(self):
        mean, std = self.FCEcoder()
        epislon = tf.random_normal(tf.shape(mean))
        z = mean + tf.multiply(epislon, std)
        out = self.FCDecoder(z)
        out = tf.clip_by_value(out, 1e-8, 1-1e-8)
        # logits = self.FCDecoder(z)

        # alpha = tf.random_uniform(tf.shape(logits), minval=0, maxval=1, dtype=tf.float32)
        # out = tf.cast(tf.less(alpha, tf.nn.sigmoid(logits)), dtype=tf.float32)


        # epsilon = tf.random_normal(tf.shape(outMean))
        # out = outMean + epsilon * outStd

        # print(out.get_shape().as_list())
        KL = 0.5 * tf.reduce_mean(
            tf.reduce_sum(tf.square(std)
                          + tf.square(mean)-1
                          - tf.log(tf.square(std)+1e-8), 1))
        first_term = 0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + tf.log(1e-8 + tf.square(std)) - tf.square(mean) - tf.square(std), 1))
        # entropy_term = tf.losses.mean_squared_error(self.inputs, out)
        # c = -0.5 * np.exp(2 * np.log(outStd))
        # precision = tf.exp(-2*tf.log(outStd))
        # entropy_term = tf.reduce_mean(c - outStd - 0.5 * precision * tf.square(out - outMean))
        # entropy_term = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=out))
        entropy_term = tf.reduce_mean(tf.reduce_sum(self.inputs * tf.log(out) + (1-self.inputs) * tf.log(1-out), 1))

        second_term = tf.reduce_mean(tf.reduce_sum(self.inputs * tf.log(out) + (1 - self.inputs) * tf.log(1 - out), 1))

        loss =  entropy_term - KL
        return -loss, KL, entropy_term, first_term, second_term

    def get_loss(self):
        return self._loss()

    def optimiser(self, learning_rate):
        return tf.train.AdamOptimizer(learning_rate)

    def train(self, n_epochs, batch_size, learning_rate):
        loss, KL, entropy, FT, ST = self.get_loss()
        train_op = self.optimiser(learning_rate).minimize(loss)

        # prepare data
        train_total_data, train_size, _, _, test_data, test_labels = prepare_MNIST_data()
        x_test = test_data[:100, :]
        total_batch = int(train_size / batch_size)


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # writer = tf.summary.FileWriter('./logs', sess.graph)


            for epoch in range(n_epochs):
                np.random.shuffle(train_total_data)
                train_data_ = train_total_data[:,:-10]

                for i in range(total_batch):
                    offset = (i*batch_size) % train_size
                    batch_xs_input = train_data_[offset:offset+batch_size, :]

                    _,loss_, KL_, entropy_, ft, st = sess.run([train_op, loss, KL, entropy, FT, ST],
                                                              feed_dict={self.inputs:batch_xs_input})
                print('loss: {}, KL:{}, entropy:{} ft:{} st:{}'.format(loss_, KL_, entropy_, ft, st))

                if epoch+1 == n_epochs:
                    y_PRR = sess.run(self.reconstruction(), feed_dict={self.inputs: x_test})
                    y_PRR_img = y_PRR.reshape(100, 28, 28)
                    #                f.write(y_PRR_img)
                    #                f.close()
                    # y_PRR_img.reshape(100, 28, 28)
                    img = np.zeros((280, 280))
                    for idx, image in enumerate(y_PRR_img):
                        i = int(idx % 10)
                        j = int(idx / 10)
                        image_ = imresize(image, size=(28, 28), interp='bicubic')
                        img[j * 28: j * 28 + 28, i * 28: i * 28 + 28] = image_
                    imsave('./result' + str(epoch) + '.jpg', img)


if __name__ == '__main__':
    seq_len = 28*28
    hidden_dim = 500
    z_dim = 20
    n_epochs = 20
    learning_rate = 0.001
    batch_size = 128
    Model = FCVAE(seq_len,hidden_dim, z_dim)
    Model.train(n_epochs, batch_size, learning_rate)

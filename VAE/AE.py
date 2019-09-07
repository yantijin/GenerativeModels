import tensorflow as tf
from .utils import *

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

    # @property
    # def inputs(self):
    #     return self._inputs

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
        with tf.variable_scope('Ecnoder'):
            fclayer1 = fcLayer(self.inputs, 'fclayer1', self.hidden_dim, activation_func=tf.nn.elu)#self.activation_func)
            fclayer2 = fcLayer(fclayer1, 'fclayer2', self.hidden_dim, activation_func=tf.nn.tanh)#self.activation_func)

            mean = fcLayer(fclayer2, 'mean', self.z_dim) #, activation_func=self.activation_func)
            std = fcLayer(fclayer2, 'std', self.z_dim, activation_func=tf.nn.softplus)
            return mean, std

    def FCDecoder(self, z=None):
        with tf.variable_scope('Decoder'):
            # get samples from standard Gaussian distribution
            if z == None:
                z = tf.random_normal((tf.shape(self.inputs)[0], self.z_dim))

            fclayer1 = fcLayer(z, 'fclayer1', self.hidden_dim, activation_func=tf.nn.tanh)#self.activation_func)
            fclayer2 = fcLayer(fclayer1, 'fclayer2', self.hidden_dim, activation_func=tf.nn.elu)#self.activation_func)

            # mean = fcLayer(fclayer2, 'mean', self.output_dim)#, activation_func=self.activation_func)
            # std = fcLayer(fclayer2, 'std', self.output_dim, activation_func=tf.nn.softplus)
            #
            # epsilon = tf.random_normal(tf.shape(mean))
            # out = tf.nn.sigmoid(mean + tf.multiply(epsilon, std))
            fclayer2 = fcLayer(fclayer2, 'output', self.output_dim, activation_func=tf.nn.sigmoid)
            return fclayer2

    def reconstruction(self):
        mean, std = self.FCEcoder()
        epislon = tf.random_normal(tf.shape(mean))
        z = mean + tf.multiply(epislon, std)
        reconX = self.FCDecoder(z)
        return reconX

    def _loss(self):
        mean, std = self.FCEcoder()
        epislon = tf.random_normal(tf.shape(mean))
        z = mean + tf.multiply(epislon, std)
        out = self.FCDecoder(z)
        KL = 0.5 * tf.reduce_mean(tf.reduce_sum(std**2+mean**2-1-tf.log(std**2+1e-8), 1))
        entropy_term = tf.losses.mean_squared_error(self.inputs, out)
        loss =  entropy_term - KL
        return -loss, KL, entropy_term

    def get_loss(self):
        return self._loss()











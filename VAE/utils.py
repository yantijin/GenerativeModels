import tensorflow as tf
from tensorflow import keras as ks

"""
    construction of common layers of NN
"""

def conv2dLayer(inputs, name, output_num, kernel_size, stride_size, padding, activation_func=None, concatTerm=None):
    '''
    :param inputs: shape [batch_size, height, width, channels]
    :param name: the variable_scope name for this conv layer
    :param output_num: the output channels of the figure
    :param kernel_size: make the filter size as [kernel_size, kernel_size, input_channels, output_channels]
    :param stride_size: a list of int(len:4)
    :param padding: 'SAME' or 'VALID'
    :param activation_func: the activation function of the layer output
    :param concatTerm: if exists, will concat this term into the results after the activation_func
    '''
    with tf.variable_scope(name):
        ShapeList = inputs.get_shape().as_list()
        weights = tf.get_variable('convWeights', [kernel_size, kernel_size, ShapeList[-1], output_num], dtype=tf.float32)
        bias = tf.get_variable('convBias', [output_num], dtype=tf.float32)

        convLayerRes = tf.nn.conv2d(inputs, weights, strides=stride_size, padding=padding)
        convLayerRes = tf.nn.bias_add(convLayerRes, bias)

        if activation_func:
            convLayerRes = activation_func(convLayerRes)
        if concatTerm:
            convLayerRes = tf.concat([convLayerRes, concatTerm], axis=-1)
        return convLayerRes


def deconv2dLayer(inputs, name, output_num, kernel_size, stride_size, padding, activation_func=None, concatTerm=None):
    '''
    :param inputs: a 4d tensor [bs, height, width, channels]
    :param name: vs name for this layer
    :param output_num: output channels, 1D
    :param kernel_size: 1D/2D integers [kernel, kernel, output_num, input_channels]
    :param stride_size: 1D/2D integers [1, stride, stride, 1]
    :param padding: 'same' or 'valid'
    :param activation_func: if exists,
    :param concatTerm: if exists concat the term into the result after layer
    '''
    with tf.variable_scope(name):
        deConvRes = tf.layers.conv2d_transpose(inputs, output_num, kernel_size, stride_size, padding=padding,
                                               activation=activation_func,
                                               use_bias=True)

        if concatTerm:
            deConvRes = tf.concat([deConvRes, concatTerm], -1)

        return deConvRes


def fcLayer(inputs, name, output_num, activation_func=None):
    '''
    :param inputs:
    :param name:
    :param output_num:  output dimension of the fully connect layers
    :param activation_func:  if exits
    '''
    with tf.variable_scope(name):
        res = ks.layers.Dense(output_num,
                        activation=activation_func,
                        use_bias=True)(inputs)
        return res



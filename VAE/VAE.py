import tensorflow as tf
import numpy as np
from mnist_data import *
from scipy.misc import imsave, imresize
from utils import fcLayer

tf.reset_default_graph()

'''
n_hidden 隐藏层维度
X: 输入数据
'''


def GaussianEncoder(X, n_hidden, n_output, dropout):
    with tf.variable_scope('GaussianEncoder'):
        # w_0 = tf.contrib.layers.variance_scaling_initializer()
        # b_0 = tf.constant_initializer(0.)
        #
        # # 1st layer
        # w_1 = tf.get_variable('w_1', shape=[X.get_shape()[1], n_hidden], initializer=w_0)
        # b_1 = tf.get_variable('b_1', shape=[n_hidden], initializer=b_0)
        # h1 = tf.matmul(X, w_2) + b_1
        # h1 = tf.nn.elu(h1)
        # h1 = tf.nn.dropout(h1, keep_prob=dropout)
        #
        # # 2nd layer
        # w_2 = tf.get_variable('w_2', shape=[h1.get_shape()[1], n_hidden], initializer=w_0)
        # b_2 = tf.get_variable('b_2', shape=[n_hidden], initializer=b_0)
        # h2 = tf.matmul(h1, w_2) + b_2
        # h2 = tf.nn.tanh(h2)
        # h2 = tf.nn.dropout(h2, keep_prob=dropout)
        #
        # # output layer
        # w_o = tf.get_variable('w_o', shape=[h2.get_shape()[1], 2 * n_output], initializer=w_0)
        # b_o = tf.get_variable('b_o', shape=[2 * n_output], initializer=b_0)
        #
        # total_Out = tf.matmul(h2, w_o) + b_o
        #
        # mean = total_Out[:, :n_output]
        # logstd = tf.nn.softplus(total_Out[:, n_output:]) + 1e-6
        layer1 = fcLayer(X, 'layer1', n_hidden, activation_func=tf.nn.elu)
        layer2 = fcLayer(layer1, 'layer2', n_hidden, activation_func=tf.nn.tanh)

        mean = fcLayer(layer2, 'mean', output_num=n_output, activation_func=None)
        logstd = fcLayer(layer2, 'std', output_num=n_output, activation_func=tf.nn.softplus)
        return mean, logstd


"""
伯努利解码器
Z 隐藏层采样变量
n_hidden 中间隐藏层的维度
n_output 最终输出维度
dropout 丢弃率
"""


def Bernoulli_decoder(Z, n_hidden, n_output, dropout):
    with tf.variable_scope('Bernoulli_decoder'):
        # w_0 = tf.contrib.layers.variance_scaling_initializer()
        # b_0 = tf.constant_initializer(0.)
        #
        # # 1st layer
        # w_1 = tf.get_variable('w_1', shape=[Z.get_shape()[1], n_hidden], initializer=w_0)
        # b_1 = tf.get_variable('b_1', shape=[n_hidden], initializer=b_0)
        # h1 = tf.matmul(Z, w_1) + b_1
        # h1 = tf.nn.tanh(h1)
        # h1 = tf.nn.dropout(h1, keep_prob=dropout)
        #
        # # 2nd layer
        # w_2 = tf.get_variable('w_2', shape=[h1.get_shape()[1], n_hidden], initializer=w_0)
        # b_2 = tf.get_variable('b_2', shape=[n_hidden], initializer=b_0)
        # h2 = tf.matmul(h1, w_2) + b_2
        # h2 = tf.nn.elu(h2)
        # h2 = tf.nn.dropout(h2, keep_prob=dropout)
        #
        # # output layer
        # w_o = tf.get_variable('w_o', shape=[h2.get_shape()[1], n_output], initializer=w_0)
        # b_o = tf.get_variable('b_o', shape=[n_output], initializer=b_0)
        #
        # total_Out = tf.matmul(h2, w_o) + b_o
        # y = tf.sigmoid(total_Out)
        layer1 = fcLayer(Z, 'layer1', n_hidden, activation_func=tf.nn.tanh)
        layer2 = fcLayer(layer1, 'layer2', n_hidden, activation_func=tf.nn.elu)

        y = fcLayer(layer2, 'out', n_output,activation_func=tf.nn.sigmoid)

        return y


'''
X 为输入
dimX为X的维度
n_hidden为隐藏层的维度
dropout 为遗弃概率
L 为采样次数
'''


def GauBernou_VAE(X, dimX, n_hidden, n_output, dropout, L=1):
    mean, std = GaussianEncoder(X, n_hidden, n_output, dropout)

    Z = tf.zeros(tf.shape(mean), dtype=tf.float32)
    for i in range(L):
        ll = tf.random_normal(tf.shape(mean), 0, 1, dtype=tf.float32)
        Z += mean + std * ll  # 注意此处是点乘
    Z = Z / L

    y = Bernoulli_decoder(Z, n_hidden, dimX, dropout)
    y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

    first_term = 0.5 * tf.reduce_mean(
        tf.reduce_sum(1 + tf.log(1e-8 + tf.square(std)) - tf.square(mean) - tf.square(std), 1))
    second_term = tf.reduce_mean(tf.reduce_sum(X * tf.log(y) + (1 - X) * tf.log(1 - y), 1))
    entropy_term = tf.losses.mean_squared_error(X, y)
    # variationalLowerBound = first_term + second_term
    variationalLowerBound = first_term + entropy_term
    #    print(first_term.get_shape())
    #    print(second_term.get_shape())

    return variationalLowerBound, first_term, entropy_term, y

def loss(X, dimX, n_hidden, n_output, dropout, L=1):
    mean, std = GaussianEncoder(X, n_hidden, n_output, dropout)

    Z = tf.zeros(tf.shape(mean), dtype=tf.float32)
    for i in range(L):
        ll = tf.random_normal(tf.shape(mean), 0, 1, dtype=tf.float32)
        Z += mean + std * ll  # 注意此处是点乘
    Z = Z / L

    y = Bernoulli_decoder(Z, n_hidden, dimX, dropout)
    y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

    KL = 0.5 * tf.reduce_mean(
        tf.reduce_sum(tf.square(std)
                      + tf.square(mean) - 1
                      - tf.log(tf.square(std) + 1e-8), 1))

    entropy_term = tf.losses.mean_squared_error(X, y)
    elbo = entropy_term-KL
    return elbo, KL, entropy_term, y


def main():
    # network architecture
    n_hidden = 500
    IMAGE_SIZE = 28
    dim_img = IMAGE_SIZE ** 2
    dim_z = 20

    # train
    n_epochs = 20
    batch_size = 128
    learn_rate = 1e-3

    # prepare data
    train_total_data, train_size, _, _, test_data, test_labels = prepare_MNIST_data()
    #    print(train_total_data.shape)
    x_test = test_data[:100, :]

    X = tf.placeholder(tf.float32, shape=[None, dim_img], name='Mnist')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    ELBO, KL_divergence, marginal_likelihood, Xhat = GauBernou_VAE(X, dim_img, n_hidden, dim_z, keep_prob)
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(-ELBO)

    total_batch = int(train_size / batch_size)
    with tf.Session() as sess:
        min_loss = 1e10
        sess.run(tf.global_variables_initializer(), feed_dict={keep_prob: 0.9})

        for epoch in range(n_epochs):
            np.random.shuffle(train_total_data)
            train_data_ = train_total_data[:, :-10]
            print(train_data_.shape)

            for i in range(total_batch):
                #                print(i)
                offset = (i * batch_size) % train_size
                batch_xs_input = train_data_[offset:offset + batch_size, :]

                _, tot_loss, marginal_likelihood_re, KL_divergence_re = sess.run(
                    (train_op, -ELBO, marginal_likelihood, KL_divergence),
                    feed_dict={X: batch_xs_input, keep_prob: 0.9})

            print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                epoch, tot_loss, marginal_likelihood_re, KL_divergence_re))

            if epoch + 1 == n_epochs or tot_loss < min_loss:
                min_loss = tot_loss
                y_PRR = sess.run(Xhat, feed_dict={X: x_test, keep_prob: 1})
                print(y_PRR.shape)
                #                f = open('re','w')

                y_PRR_img = y_PRR.reshape(100, IMAGE_SIZE, IMAGE_SIZE)
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

    # training

    # X = tf.placeholder(tf.float32,shape=[None, 1024], name='Mnist')
    # n_hidden = 512
    # dimX = 1024
    # dropout = 0.9
    # n_output = 368 # 隐变量维度
    # ELBO, KL_divergence, marginal_likelihood, Xhat = GauBernou_VAE(X, dimX, n_hidden, n_output, dropout)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(-ELBO)

    # epoches = 100
    # batch_size = 128
    # total_batch = train_data_size / batch_size

    # with tf.Session() as sess:
    #   sess.run(tf.global_variables_initializer())
    #   for epoch in range(epoches):
    #       for i in range(total_batch):
    #           offset = ( i * batch_size) % train_data_size
    #           data = train_data[offset: offset + batch_size, :]
    #           sess.run((optimizer, -ELBO, marginal_likelihood, KL_divergence), feed_dict={X:data})
    #       print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (epoch, -ELBO, marginal_likelihood, KL_divergence))

    #       if epoch == epoches - 1:
    #           reproduce_data = sess.run(Xhat, feed_dict={X: testData})
    # 将生成的datareshape成图像的形式


if __name__ == "__main__":
    main()











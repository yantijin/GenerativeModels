import tensorflow as tf
import numpy as np
from AE import *
import matplotlib.pyplot as plt
from scipy.misc import imsave, imresize
from mnist_data import *
tf.reset_default_graph()
def get_train_data(value, seq_len):
    """
    Create x, y train data windows.
    """
    data_x = []
    data_y = []
    # value = (value-np.mean(value))/np.std(value)
    for i in range(len(value) - seq_len):
        x = value[i:i + seq_len + 1]
        y = value[i:i + seq_len + 1]
        data_x.append(x[:-1])
        data_y.append(y[-1])
    return np.array(data_x), np.array(data_y)


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

    # X = tf.placeholder(tf.float32, shape=[None, dim_img], name='Mnist')

    Model1 = FCVAE(dim_img, hidden_dim=n_hidden, z_dim=dim_z, activation_func=tf.nn.sigmoid)

    loss1, KL, entropy = Model1.get_loss()
    recon = Model1.reconstruction()
    train_op1 = tf.train.AdamOptimizer(0.001).minimize(loss1)

    total_batch = int(train_size / batch_size)
    lossList = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            np.random.shuffle(train_total_data)
            train_data_ = train_total_data[:, :-10]
            print(train_data_.shape)

            for i in range(total_batch):
                #                print(i)
                offset = (i * batch_size) % train_size
                batch_xs_input = train_data_[offset:offset + batch_size, :]


                _,loss_, KL_, entropy_ = sess.run([train_op1, loss1,KL, entropy], feed_dict={Model1.inputs:batch_xs_input})
                lossList.append(loss_)
            print('*********')
            print('loss:{} KL:{} entropy:{}'.format(loss_, KL_, entropy_))

            if epoch + 1 == n_epochs:
                y_PRR = sess.run(recon, feed_dict={X: x_test})
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

    plt.plot(lossList)
    plt.show()
# seq_len = 64
# z_dim = 5
# epochs = 100
# batch_size = 64
# latent_dim = 64
# learning_rate = 0.001
# decay_factor = 0.9
#
# data = [5+np.sin(0.025*i) for i in range(10000)]
# data = (data-np.min(data))/(np.max(data)-np.min(data))
# x_train, y_train = get_train_data(data, seq_len)
# inputs = tf.placeholder(dtype=tf.float32, shape=[None, seq_len])
# # FCAE
# Model1 = FCAE(inputs,hidden_dim=latent_dim,z_dim=z_dim,activation_func=tf.nn.softplus)
#
# loss1 = Model1.get_loss()
# recon = Model1.reconstruction()
#
# train_op1 = tf.train.AdamOptimizer(0.001).minimize(loss1)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# lossList = []
# reconList = np.zeros((1,len(data)))
# for epoch in range(epochs):
#     for i in range(x_train.shape[0] // batch_size):
#         _, loss_=  sess.run([train_op1, loss1], feed_dict={inputs: x_train[i*batch_size:(i+1)*batch_size]})
#         lossList.append(loss_)


# reconstruction
# for i in range(x_train.shape[0] // batch_size):
#     tmpList = sess.run(recon, feed_dict={inputs:x_train})
#     for j in range(batch_size):
#         reconList[0, i*batch_size+j:i*batch_size+j+seq_len] = tmpList[j]
#
# plt.figure()
# plt.plot(data)
# plt.plot(reconList[0],'r')
# plt.figure()
# plt.plot(lossList)
# plt.show()
if __name__ == '__main__':
    main()







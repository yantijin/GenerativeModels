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


def data_loader(data_path):
    data = np.load(data_path)

    labels = data['kdd'][:,-1]
    features = data["kdd"][:,:-1]
    N, D = features.shape

    normal_data = features[labels==1]
    normal_labels = labels[labels==1]

    N_normal = normal_data.shape[0]

    attack_data = features[labels==0]
    attack_labels = labels[labels==0]

    N_attack = attack_data.shape[0]

    randIdx = np.arange(N_attack)
    np.random.shuffle(randIdx)
    N_train = N_attack // 2

    train = attack_data[randIdx[:N_train]]
    train_labels = attack_labels[randIdx[:N_train]]

    test = attack_data[randIdx[N_train:]]
    test_labels = attack_labels[randIdx[N_train:]]

    test = np.concatenate((test,normal_data), axis=0)
    test_labels = np.concatenate((test_labels, normal_labels), axis=0)

    return train, test, train_labels, test_labels


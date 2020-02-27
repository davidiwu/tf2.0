'''
https://stackoverflow.com/questions/47034888/how-to-choose-cross-entropy-loss-in-tensorflow

    H(p,q) = - SUM_x(p(x) * log(q(x)))
    H(p,q) = - SUM_X(p(x) * log(q(x)) + (1 - p(x)) * log(1-q(x)))

cross entropy in keras:

    categorical_crossentropy:
        tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output)
        is for multi-class classification (classes are exclusive).
        when using the categorical_crossentropy loss, your targets should be in categorical format 
        (e.g. if you have 10 classes, the target for each sample should be a 10-dimensional vector 
        that is all-zeros expect for a 1 at the index corresponding to the class of the sample).

    binary_crossentropy:
        tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)
        is for binary multi-label classification (labels are independent). 
        The labels must be one-hot encoded or can contain soft class probabilities.
    
'''
import tensorflow as tf
import numpy as np

def categorical_cross_entropy_from_probability(real, preds):

    logs = np.log(preds)
    cross_value = real * logs

    mean = tf.reduce_mean(cross_value, axis=-1)
    sum = tf.reduce_sum(mean)

    return -sum

def categorical_cross_entropy_from_logits(real, preds):

    preds = tf.nn.softmax(preds, axis=-1)

    cross_value = categorical_cross_entropy_from_probability(real, preds)
    
    return cross_value

def binary_cross_entropy(real, preds):

    ones_real = tf.ones_like(real)
    ones_preds = tf.ones_like(preds)
    cross_extra = (ones_real - real) * np.log(ones_preds - preds)

    logs = np.log(preds)
    cross_value = real * logs + cross_extra

    mean = tf.reduce_mean(cross_value, axis=-1)
    sum = tf.reduce_sum(mean)

    return -sum

def test_binary_cross_entropy_compare():

    real = [0., 0., 1., 1.]
    preds = [0.09, 0.9, 0.99, 0.01]

    loss = binary_cross_entropy(real, preds)

    bce = tf.keras.losses.BinaryCrossentropy()
    loss_keras = bce(real, preds)

    print(f'binary cross entropy loss: {loss.numpy()}, from keras loss: {loss_keras.numpy()}')

def test_categorical_cross_entropy():

    real = [[1.0, 0, 0]]
    preds = [[0.8, 0.1, 0.1]]

    cross_entropy = categorical_cross_entropy_from_probability(real, preds)

    print(cross_entropy.numpy())

def test_categorical_cross_entropy_compare():

    real = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    preds = [[.9, .05, .05], [.05, .89, .06], [.05, .01, .94]]

    loss = categorical_cross_entropy_from_probability(real, preds)

    cce = tf.keras.losses.CategoricalCrossentropy()
    loss_keras = cce(real, preds)

    print(f'categorical cross entropy loss: {loss.numpy()}, from keras loss: {loss_keras.numpy()}')


if __name__ == '__main__':

    test_categorical_cross_entropy_compare()
    test_binary_cross_entropy_compare()
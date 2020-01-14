import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
from helper import plot_training_history, plot_cnn_activation


def get_mnist_data_ready_for_training():

    mnist = keras.datasets.mnist
    (train_image, train_label), (test_image, test_label) = mnist.load_data()

    print(train_image.shape)

    train_image, test_image = train_image / 255.0, test_image / 255.0  # normalnization

    train_image = train_image[..., tf.newaxis]  # add channel axis
    test_image = test_image[..., tf.newaxis]
    print(train_image.shape)

    return (train_image, train_label), (test_image, test_label)

'''
https://keras.io/getting-started/faq/#why-is-the-training-loss-much-higher-than-the-testing-loss
A Keras model has two modes:
    training
    testing

Regularization mechanisms, such as Dropout and L1/L2 weight regularization, are turned off at testing/prediction time.

Note: Also, Batch Normalization is a much-preferred technique for regularization, in my opinion, as compared to Dropout. 
Consider using it.
'''
def define_training_model():
    model = keras.models.Sequential(
        [
            keras.layers.Conv2D(filters=32, kernel_size=[3,3], padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=[2,2], strides=2),
            keras.layers.Conv2D(filters=64, kernel_size=[3,3], padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=[2,2], strides=2),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(10, activation='softmax')
        ]
    )

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_the_model(train_image, train_label, test_image, test_label):

    model = define_training_model()
 
    history = model.fit(train_image, train_label, validation_data=(test_image, test_label), epochs=5, batch_size=32)

    plot_training_history(history)

    test_loss, test_acc = model.evaluate(test_image, test_label, verbose=0)
    print(f'evaulation results: loss: {test_loss}, with accuracy: {test_acc}')

    return model

def show_layer_activation(model, test_im):

    cnn_layer = model.get_layer(index=0)

    model2 = keras.models.Sequential([cnn_layer])

    cnn_activation = model2.predict(test_im.reshape(1, 28, 28, 1))

    plot_cnn_activation(cnn_activation)


if __name__ == '__main__':
    (train_image, train_label), (test_image, test_label) = get_mnist_data_ready_for_training()
    model = train_the_model(train_image, train_label, test_image, test_label)

    test_im = train_image[156]
    show_layer_activation(model, test_im)


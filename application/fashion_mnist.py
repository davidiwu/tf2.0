import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


def load_mnist_fashion_data():

    fashion_mnist = keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    return (x_train, y_train), (x_test, y_test), class_names


def build_fashion_mnist_model():

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])

    # default from_logits=False, means from probability
    # usually after a softmax or sigmoid activation
    loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer='adam',
        loss = loss_object,
        metrics=['accuracy']
    )

    return model

def make_predictions(model, test_images):

    probability_model = keras.Sequential([
        model,
        keras.layers.Softmax()
    ])

    predictions = probability_model(test_images)

    return predictions

def validate_predictions(predictions, images, labels, class_names):

    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(predictions[i], labels[i], images[i], class_names)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_values(predictions[i], labels[i])

    plt.tight_layout()
    plt.show()

def plot_image(prediction, label, image, class_names):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image, cmap=plt.cm.binary)
    pred_label = np.argmax(prediction)

    pred_name = class_names[pred_label]
    real_name = class_names[label]

    color = 'blue' if pred_label == label else 'red'

    plt.xlabel(f'{pred_name} {100*np.max(prediction):2.0f}% ({real_name})', color=color)

def plot_values(prediction, label):
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])

    thisplot = plt.bar(range(10), prediction, color='#777777')
    plt.ylim([0, 1])

    pred_label = np.argmax(prediction)
    thisplot[pred_label].set_color('red')
    thisplot[label].set_color('blue')

if __name__ == '__main__':

    model = build_fashion_mnist_model()

    (x_train, y_train), (x_test, y_test), class_names = load_mnist_fashion_data()

    model.fit(x_train, y_train, epochs = 10)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'test loss: {test_loss}, test accuracy: {test_acc}')

    predictions = make_predictions(model, x_test)

    validate_predictions(predictions, x_test, y_test, class_names)
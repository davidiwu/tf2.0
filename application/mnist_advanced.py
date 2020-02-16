import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import Model

def print_data_shape_type(name, data):

    if isinstance(data, list):
        for index, item in enumerate(data):
            print(f'{name} {index} shape: {item.shape}, type is :{type(item)}')
    else:
        print(f'{name} shape: {data.shape}, type is :{type(data)}')


def load_mnist_data():

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    return train_ds, test_ds

class MyModel(Model):

    def __init__(self):

        super(MyModel, self).__init__()

        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):

        print_data_shape_type('model input', x)
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        print_data_shape_type('model output', x)
        return x


@tf.function
def train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):

    with tf.GradientTape() as tape:

        predictions = model(images)
        print_data_shape_type('model output', predictions)

        loss = loss_object(labels, predictions)
        print_data_shape_type('model loss', loss)

    print_data_shape_type('model trainable variables', model.trainable_variables)

    gradients = tape.gradient(target=loss, sources=model.trainable_variables)

    print_data_shape_type('gradients', gradients)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):

    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


def train_mnist_model(model, train_ds, test_ds, epochs = 5):

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        count = 0
        for images, labels in train_ds:
            count = count + 1
            train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)

        print(f'there are {count} batchs trained.')
        
        for t_images, t_labels in test_ds:
            test_step(model, t_images, t_labels, loss_object, test_loss, test_accuracy)

        print(f'Epoch {epoch+1}, loss: {train_loss.result()}, accuracy: {train_accuracy.result()*100}')
        print(f'Epoch {epoch+1}, test loss: {test_loss.result()}, test accuracy: {test_accuracy.result()*100}')


if __name__ == '__main__':

    model = MyModel()    

    train_ds, test_ds = load_mnist_data()

    train_mnist_model(model, train_ds, test_ds, epochs=5)

    model.summary()

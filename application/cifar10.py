import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


def load_cifar10_data():

    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    return (x_train, y_train), (x_test, y_test), class_names

def build_cifar10_model():

    model = models.Sequential()

    # default padding='valid' for Conv2D, so output width = (32 - 3) /strides + 1 = 30
    # if padding ='same', output width = 32 /strides = 32
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

    # after MaxPooling2D, output width= floor(30 / 2) = 15
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer='adam',
        loss = loss_object,
        metrics=['accuracy']
    )

    model.summary()
    return model

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test), class_names = load_cifar10_data()

    model = build_cifar10_model()

    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'test loss: {test_loss}, test accuracy: {test_acc}')
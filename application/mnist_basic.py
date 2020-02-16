
import tensorflow as tf
import numpy as np

def load_mnist_data():

    mnist = tf.keras.datasets.mnist

    print('type(mnist): ', type(mnist))

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print('type(x_train): ', type(x_train))
    print('type(y_train): ', type(y_train))

    print('shape of x_train: ', x_train.shape)
    print('shape of y_train: ', y_train.shape)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return (x_train, y_train), (x_test, y_test)

def define_keras_model():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


def evaluate_predictions(preds, actuals):

    # type of keras model output:  <class 'numpy.ndarray'>
    print('type of model output: ', type(preds))  
    print('shape of model output: ', preds.shape)

    preds_argmax = np.argmax(preds, axis=-1)

    print('type of model argmax output: ', type(preds_argmax))
    print('shape of model argmax output: ', preds_argmax.shape)

    pred_stat = actuals == preds_argmax

    true_pred = np.where(pred_stat == True)
    print('shape of true_pred: ', true_pred[0].shape)

    correct_preds = np.extract(pred_stat, preds_argmax)

    correct_percentage = correct_preds.size / preds_argmax.size

    print('predict correct with percentage: ', correct_percentage)


if __name__ == '__main__':

    model = define_keras_model()

    (x_train, y_train), (x_test, y_test) = load_mnist_data()    

    model.fit(x_train, y_train, epochs=5, validation_split=0.2)

    model.evaluate(x_test, y_test, verbose=2)

    preds = model.predict(x_test)

    evaluate_predictions(preds, y_test)

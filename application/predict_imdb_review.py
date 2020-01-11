
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

vocab_size = 10000

def get_training_data():

    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    word_index = imdb.get_word_index()

    word_index = {k:(v+3) for k, v in word_index.items()}
    word_index['<PAD>'] = 0
    word_index['<START>'] = 1
    word_index['<UNK>'] = 2
    word_index['<UNUSED>'] = 3

    sample_training_data(word_index, train_data[0])

    # pad the sequences since max length of a entry is 256
    train_data = keras.preprocessing.sequence.pad_sequences(
        train_data,
        value=word_index['<PAD>'],
        padding='post',
        maxlen=256
    )

    test_data = keras.preprocessing.sequence.pad_sequences(
        test_data,
        value=word_index['<PAD>'],
        padding='post',
        maxlen=256
    )

    return train_data, train_labels, test_data, test_labels


def sample_training_data(word_index, text):

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode_review(reverse_word_index, text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    test = decode_review(reverse_word_index, text)
    print(test)

# define the model with embedding
def get_model():
    # using Word2Vec for embedding will improve the result
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 128))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True, recurrent_dropout=0.1)))
    model.add(keras.layers.GlobalAveragePooling1D())
    #model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(20)) #, activation='relu'))
    model.add(keras.layers.Dropout(0.05))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_the_model(train_data, train_labels, test_data, test_labels):

    model = get_model()

    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]

    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(partial_x_train, partial_y_train, epochs=5, 
                        batch_size=512, validation_data=(x_val, y_val), verbose=1,
                        callbacks=[early_stopping])

    results = model.evaluate(test_data, test_labels, verbose=2)
    print(results)

    return history

def plot_training_history(history):

    history_dict = history.history
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(2*6, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

if __name__ == '__main__':

    train_data, train_labels, test_data, test_labels = get_training_data()

    his = train_the_model(train_data, train_labels, test_data, test_labels)

    plot_training_history(his)
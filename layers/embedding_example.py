import numpy as np
from numpy import array
import tensorflow as tf

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Embedding, Dense

VOCAB_SIZE = 50
MAX_LENGTH = 4

def get_training_data():

    # Define 10 resturant reviews.
    reviews = [
        'Never coming back!',
        'Horrible service',
        'Rude waitress',
        'Cold food.',
        'Horrible food!',
        'Awesome',
        'Awesome service!',
        'Rocks!',
        'poor work',
        'Couldn\'t have done better']

    # Define labels (1=negative, 0=positive)
    labels = array([1,1,1,1,1,0,0,0,0,0])
    
    encoded_reviews = [one_hot(d, VOCAB_SIZE) for d in reviews]
    print(f"Encoded reviews: {encoded_reviews}")
    
    padded_reviews = pad_sequences(encoded_reviews, maxlen=MAX_LENGTH, padding='post')
    print(padded_reviews)

    return padded_reviews, labels


def get_model():

    model = Sequential()
    embedding_layer3 = Embedding(VOCAB_SIZE, 8, input_length=MAX_LENGTH)
    model.add(embedding_layer3)
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    return model


def train_embedding_model(padded_reviews, labels):
    model = get_model()

    pred = model.predict(padded_reviews)
    embedding_layer = model.get_layer(index=0)

    el_output = embedding_layer.output
    print(el_output.shape)

    # fit the model
    model.fit(padded_reviews, labels, epochs=10, verbose=1)

    embedding_table = embedding_layer.get_weights()[0]
    print(embedding_table.shape)

    return model, embedding_table


def reverse_embedding(input_word_embedding, embedding_table):

    input_shape = input_word_embedding.shape
    table_shape = embedding_table.shape

    assert len(input_shape) == len(table_shape) == 2
    assert input_shape[1] == table_shape[1]

    embedding_table = tf.cast(embedding_table, dtype = tf.float64)

    matrix_multiple = tf.matmul(input_word_embedding, embedding_table, transpose_b=True)

    print(matrix_multiple.shape)

    #softmax = tf.nn.log_softmax(matrix_multiple, axis=-1)
    softmax = tf.nn.softmax(matrix_multiple, axis=-1)
    print(softmax.shape)

    index_in_embedding_table = tf.argmax(softmax, axis=-1)
 
    return index_in_embedding_table


def reuse_embedding_result(model, padded_reviews):

    model2 = Model(inputs = model.input, outputs = model.layers[0].output)
    activation = model2.predict(padded_reviews)

    print(activation.shape)
    print(activation[3][1])


if __name__ == "__main__":

    padded_reviews, labels = get_training_data()

    model, embedding_table = train_embedding_model(padded_reviews, labels)

    input_word_embedding = np.array([[-0.03867966,  0.01225551, -0.05507576, -0.01467555, -0.03268626, -0.01057478, -0.03672233, -0.02771262]])
    index_in_embedding_table = reverse_embedding(input_word_embedding, embedding_table)
    print(index_in_embedding_table.numpy())

    reuse_embedding_result(model, padded_reviews)
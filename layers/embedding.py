
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding

import numpy as np

print(tf.__version__)

def get_embedding_model():

    model = Sequential()
    embedding_layer = Embedding(input_dim=10, output_dim=4, input_length=2)
    model.add(embedding_layer)

    return model

def test_embedding_parameters():

    model = get_embedding_model()
    model.compile('adam', 'mse')

    input_data = np.array([1,2])
    input_data = input_data[np.newaxis, :]  # add batch_size axis for model
    print(input_data.shape)

    pred = model.predict(input_data)
    print(pred.shape)
    print(pred)

    embedding_table = model.layers[0].get_weights()[0]
    print(embedding_table.shape)
    print(embedding_table)

    embedding = tf.gather(embedding_table, [1,2])
    pred = tf.squeeze(pred, axis=0)

    assert (pred.numpy() == embedding.numpy()).all()  # embedding results should be the same as the embedding weights


if __name__ == "__main__":
    test_embedding_parameters()

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub

import tokenization


# # Load and Preprocess
# 
# - Load BERT from the Tensorflow Hub
# - Load CSV files containing training data
# - Load tokenizer from the bert layer
# - Encode the text into tokens, masks, and segment flags
def get_bert_layer_from_tfhub():

    module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
    bert_layer = hub.KerasLayer(module_url, trainable=True)    

    return bert_layer

def get_tokenizer(bert_layer):
    
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
    print(vocab_file, do_lower_case)

    return tokenizer

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def build_model(bert_layer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def load_csv_data():

    train = pd.read_csv("/data/train.csv")
    test = pd.read_csv("/data/test.csv")
    submission = pd.read_csv("/data/sample_submission.csv")

    # train.head()

    train.keyword.fillna("Null", inplace=True)
    train.location.fillna("Empty", inplace=True)
    train['tweets_length'] = train.text.str.len() + train.keyword.str.len() + train.location.str.len() + 2
    # train.head()

    # sorted_pd = train.sort_values('tweets_length', ascending=False)
    # sorted_pd.head()
    return train, test, submission


def train_the_model(bert_layer, train_input, train_labels):

    # Model: Build, Train, Predict, Submit
    model = build_model(bert_layer, max_len=160)
    model.summary()

    checkpoint = ModelCheckpoint('bert_model.h5', monitor='val_loss', save_best_only=True)

    train_history = model.fit(
        train_input, train_labels,
        validation_split=0.2,
        epochs=3,
        callbacks=[checkpoint],
        batch_size=32
    )

    return model, train_history


if __name__ == '__main__':

    bert_layer = get_bert_layer_from_tfhub()
    tokenizer = get_tokenizer(bert_layer)

    train, test, submission = load_csv_data()

    train_input = bert_encode(train.text.values, tokenizer, max_len=160)
    test_input = bert_encode(test.text.values, tokenizer, max_len=160)
    train_labels = train.target.values

    model, _ = train_the_model(bert_layer, train_input, train_labels)

    model.load_weights('bert_model.h5')
    test_pred = model.predict(test_input)

    submission['target'] = test_pred.round().astype(int)
    submission.to_csv('submission.csv', index=False)

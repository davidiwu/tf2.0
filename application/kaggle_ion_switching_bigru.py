'''
kaggle competition:
    https://www.kaggle.com/c/liverpool-ion-switching/notebooks?sortBy=hotness&group=everyone&pageSize=20&competitionId=18045
    
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras.layers as L
from tensorflow.keras import Model
from sklearn.metrics import f1_score
from tensorflow.keras import callbacks

def load_train_test_data(seq_len=5000):

    train = pd.read_csv("./application/iondata/train.csv")
    test = pd.read_csv("./application/iondata/test.csv")
    sub = pd.read_csv("./application/iondata/sample_submission.csv", dtype=dict(time=str))

    n_classes = train.open_channels.unique().shape[0]

    train_input = train.signal.values.reshape(-1, seq_len, 1)
    train_target = train.open_channels.values.reshape(-1, seq_len, 1)

    X_test = test.signal.values.reshape(-1, seq_len, 1)

    return n_classes, train_input, train_target, X_test, sub


def add_extra_data(train_input, train_target, seq_len=5000, groups=1000):

    extra_i = []
    extra_t = []
    for i in range(groups -1):
        
        if i > 0 and (i+1) % (groups//10) == 0:
            continue
            
        delta = seq_len // 4
        for j in range(3):            
            start = delta * j
            end = seq_len - start
            extra_input = np.zeros((seq_len,1))
            extra_input[:end]=train_input[i, start:] 
            extra_input[end:]=train_input[i+1, :start] 
            extra_i.append(extra_input)            
        
            extra_target = np.zeros((seq_len,1))
            extra_target[:end] = train_target[i, start:]
            extra_target[end:] = train_target[i+1, :start]
            extra_t.append(extra_target)

    train_input = np.concatenate((train_input, extra_i), axis=0)
    train_target = np.concatenate((train_target, extra_t), axis=0)

    X_train, X_valid, y_train, y_valid = train_test_split(train_input, train_target, test_size=0.2)

    return X_train, X_valid, y_train, y_valid


def build_model(n_classes, seq_len=500, n_units=256):

    inputs = L.Input(shape=(seq_len, 1))
    x = L.Dense(n_units, activation='linear')(inputs)
    
    x = L.Bidirectional(L.GRU(n_units, return_sequences=True))(x)
    x = L.Bidirectional(L.GRU(n_units, return_sequences=True))(x)
    x = L.AveragePooling1D(data_format='channels_first')(x)
    x = L.Dense(22)(x)
    x = L.Dropout(0.2)(x)
    x = L.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=x)
    model.compile('adam', loss='sparse_categorical_crossentropy')
    
    return model


class F1Callback(callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X = X_val
        self.y = y_val.reshape(-1)
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            return
        pred = (
            model
            .predict(self.X, batch_size=64)
            .argmax(axis=-1)
            .reshape(-1)
        )
        
        score = f1_score(self.y, pred, average='macro')        
        print(f"val_f1_macro: {score:.4f}")


if __name__ == '__main__':

    seq_len = 5000
    groups = 1000  # 5000 * 1000 = 5_000_000
    n_classes, train_input, train_target, X_test, sub = load_train_test_data(seq_len)
    X_train, X_valid, y_train, y_valid = add_extra_data(train_input, train_target, seq_len, groups)

    model = build_model(n_classes, seq_len)
    model.summary()

    model.fit(
        X_train, y_train, 
        batch_size=16,
        epochs=30,
        callbacks=[
            callbacks.ReduceLROnPlateau(),
            F1Callback(X_valid, y_valid),
            callbacks.ModelCheckpoint('model.h5')
        ],
        validation_data=(X_valid, y_valid)
    )

    model.load_weights('model.h5')

    valid_pred = model.predict(X_valid, batch_size=64).argmax(axis=-1)
    f1_score(y_valid.reshape(-1), valid_pred.reshape(-1), average='macro')

    test_pred = model.predict(X_test, batch_size=64).argmax(axis=-1)
    sub.open_channels = test_pred.reshape(-1)
    sub.to_csv('submission.csv', index=False)
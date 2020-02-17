'''
https://tensorflow.google.cn/tutorials/keras/regression?hl=zh_cn
本 notebook 使用经典的 Auto MPG 数据集，构建了一个用来预测70年代末到80年代初汽车燃油效率的模型。
为了做到这一点，我们将为该模型提供许多那个时期的汽车描述。这个描述包含：气缸数，排量，马力以及重量。

mpg stands for Miles Per Gallon 
– a measure of how far a car can travel if you put just one gallon of petrol or diesel in its tank. 

'''

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


def load_auto_mpg_data():

    data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    dataset_path = keras.utils.get_file("auto-mpg.data", data_url)

    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight', 'Acceleration', 'Model Year', 'Origin']

    raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values = "?", 
                                comment='\t', sep=" ", skipinitialspace=True)

    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    origin = dataset.pop('Origin')

    dataset['USA'] = (origin == 1)*1.0
    dataset['Europe'] = (origin == 2)*1.0
    dataset['Japan'] = (origin == 3)*1.0

    train_dataset = dataset.sample(frac=0.8,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    print(train_dataset.tail())

    # sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    return dataset, (train_dataset, train_labels), (test_dataset, test_labels)


def normalize_dataset(train_dataset, test_dataset, dataset):

    train_stats = dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()

    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']
        
    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    print(normed_train_data.tail())

    return normed_train_data, normed_test_data


def build_model(input_shape):

    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[input_shape]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae', 'mse']
    )

    return model

def plot_history(history):

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(2*6, 6))
    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()

def evaulate_predictions(test_labels, test_predictions):
    
    plt.figure(figsize=(2*6, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    
    plt.subplot(1, 2, 2)
    error = test_predictions - test_labels
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")
    plt.show()

# 通过为每个完成的时期打印一个点来显示训练进度
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: 
            print('')
        print('.', end='')


if __name__ == '__main__':

    EPOCHS = 1000

    dataset, (train_dataset, train_labels), (test_dataset, test_labels) = load_auto_mpg_data()

    normed_train_data, normed_test_data = normalize_dataset(train_dataset, test_dataset, dataset)

    input_shape = len(normed_train_data.keys())

    model = build_model(input_shape)

    # patience 值用来检查改进 epochs 的数量, 能够容忍多少个epoch内都没有improvement。
    # monitor: 监控的数据接口，有’acc’,’val_acc’,’loss’,’val_loss’等等。
    # 正常情况下如果有验证集，就用’val_acc’或者’val_loss’
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                        validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

    plot_history(history)

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

    test_predictions = model.predict(normed_test_data).flatten()

    evaulate_predictions(test_labels, test_predictions)

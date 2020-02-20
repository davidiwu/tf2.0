'''
GRU对LSTM做了很多简化，同时却保持着和LSTM相同的效果。因此，GRU最近变得越来越流行。

GRU对LSTM做了两个大改动：
    将输入门、遗忘门、输出门变为两个门：更新门（Update Gate）z 和重置门（Reset Gate） r。
    将单元状态与输出合并为一个状态。只有一个输出h。
    GRU有3个全连接层：2个门+1个tanh层计算h`（当前输出的临时值）

GRUCell:
    Unlike GRU layers, which processes whole batches of input sequences, the GRU cell only processes a single timestep.
'''

import numpy as np
import tensorflow as tf

vocab_size = 100
seq_length = 10
embedding = 64
batch_size = 16
gru_units = 32

def print_layer_trainable_variables(layer):
    vars = layer.trainable_variables
    for var in vars:
        print(f'variable name is {var.name}, variable shape is {var.shape}')


inputs = np.random.randint(vocab_size, size=(batch_size, seq_length))

embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding)

gru_layer = tf.keras.layers.GRU(gru_units)
gru_layer_seq = tf.keras.layers.GRU(gru_units, return_sequences=True) # use this one, its more than enough.
gru_layer_all = tf.keras.layers.GRU(gru_units, return_sequences=True, return_state=True)
gru_layer_sta = tf.keras.layers.GRU(gru_units, return_state=True)

embedded = embedding_layer(inputs)
print(f'output of embedding layer shape is {embedded.shape}')  # 16, 10, 64

y1 = gru_layer(embedded)
print(f'output of gru layer shape is {y1.shape}')  # 16, 32
print_layer_trainable_variables(gru_layer)
'''
variable name is gru/kernel:0, variable shape is (64, 96)  
variable name is gru/recurrent_kernel:0, variable shape is (32, 96) 
variable name is gru/bias:0, variable shape is (2, 96)

total variables: 64 = embedding width, 32 = previous gru output state, 3 dense layers: 
(64 + 32) * 3 * 32  + 2 * 3 * 32 = 9408
'''

y2 = gru_layer_seq(embedded)
print(f'output of gru layer with sequence shape is {y2.shape}')  # 16, 10, 32
print_layer_trainable_variables(gru_layer_seq)

y3, final_state = gru_layer_all(embedded)
print(f'output of gru layer with sequence shape is {y3.shape}')  # 16, 10, 32
print(f'output of gru layer final_state shape is {final_state.shape}')  # 16, 32

y3_last = y3[:,-1,:]
is_equal_with_final_state = (y3_last.numpy() == final_state.numpy()).all() # True

print(f'is_equal_with_final_state: {is_equal_with_final_state}.')

y4, final_state = gru_layer_sta(embedded)
print(f'output of gru layer with state shape is {y4.shape}')  # 16, 32
print(f'output of gru layer final_state shape is {final_state.shape}')  # 16, 32

is_equal_with_final_state = (y4.numpy() == final_state.numpy()).all()  # True
print(f'is_equal_with_final_state: {is_equal_with_final_state}.')



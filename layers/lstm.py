'''
门实际上就是一层全连接层，它的输入是一个向量，输出是一个0到1之间的实数向量。
LSTM每个cell有2个输出：单元状态c，和单元输出h
LSTM有3个门：遗忘门，输入门，输出门, 他们都用sigmoid函数作为激活函数，输出（0，1）
LSTM有4个全连接层：3个门+1个tanh层计算c`（当前状态的临时值），共有4个W，4个b

LSTMCell:
    Unlike LSTM layers, which processes whole batches of input sequences, the LSTM cell only processes a single timestep.
    RNN(LSTMCell(10)) = LSTM(10)
'''

import numpy as np
import tensorflow as tf

vocab_size = 100
seq_length = 10
embedding = 64
batch_size = 16
lstm_units = 32

def print_layer_trainable_variables(layer):
    vars = layer.trainable_variables
    for var in vars:
        print(f'variable name is {var.name}, variable shape is {var.shape}')


inputs = np.random.randint(vocab_size, size=(batch_size, seq_length))

embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding)

lstm_layer = tf.keras.layers.LSTM(lstm_units)
lstm_layer_seq = tf.keras.layers.LSTM(lstm_units, return_sequences=True) # use this one, its more than enough.
lstm_layer_all = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
lstm_layer_sta = tf.keras.layers.LSTM(lstm_units, return_state=True)

embedded = embedding_layer(inputs)
print(f'output of embedding layer shape is {embedded.shape}')  # 16, 10, 64

y1 = lstm_layer(embedded)
print(f'output of lstm layer shape is {y1.shape}')  # 16, 32
print_layer_trainable_variables(lstm_layer)
'''
variable name is lstm/kernel:0, variable shape is (64, 128)  
variable name is lstm/recurrent_kernel:0, variable shape is (32, 128) 
variable name is lstm/bias:0, variable shape is (128,)

total variables: 64 = embedding width, 32 = previous lstm output state, 4 dense layers: 
(64 + 32) * 4 * 32  + 1 * 4 * 32 = 12416
'''

y2 = lstm_layer_seq(embedded)
print(f'output of lstm layer with sequence shape is {y2.shape}')  # 16, 10, 32
print_layer_trainable_variables(lstm_layer_seq)

y3, final_memory_state, final_carry_state = lstm_layer_all(embedded)
print(f'output of lstm layer with sequence shape is {y3.shape}')  # 16, 10, 32
print(f'output of lstm layer final_memory_state shape is {final_memory_state.shape}')  # 16, 32
print(f'output of lstm layer final_carry_state shape is {final_carry_state.shape}')  # 16, 32

y3_last = y3[:,-1,:]
is_equal_with_memory = (y3_last.numpy() == final_memory_state.numpy()).all() # True
is_equal_with_carry = (y3_last.numpy() == final_carry_state.numpy()).all()   # False

print(f'is_equal_with_memory: {is_equal_with_memory}, is_equal_with_carry: {is_equal_with_carry}')

y4, final_memory_state, final_carry_state= lstm_layer_sta(embedded)
print(f'output of lstm layer with state shape is {y4.shape}')  # 16, 32
print(f'output of lstm layer final_memory_state shape is {final_memory_state.shape}')  # 16, 32
print(f'output of lstm layer final_carry_state shape is {final_carry_state.shape}')  # 16, 32

is_equal_with_memory = (y4.numpy() == final_memory_state.numpy()).all()  # True
is_equal_with_carry = (y4.numpy() == final_carry_state.numpy()).all()    # False

print(f'is_equal_with_memory: {is_equal_with_memory}, is_equal_with_carry: {is_equal_with_carry}')



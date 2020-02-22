import numpy as np
import tensorflow as tf

'''
tensor permutation functions:

    tf.stack(inputs, axis=-1)  
    tf.reshape(x, (batch_size, -1, d_model))
    tf.transpose(x, perm=[0, 2, 1, 3])
    tf.squeeze(expand, axis=1)
    tf.split(ones, axis= -1, num_or_size_splits= 2)
    tf.unstack(ones, axis= -1)

    broadcasting

    dense layer output
'''

# stack
def demo_tf_stack_function():

    inputs = {key: tf.keras.layers.Input(shape=(5,), name='key_' + str(key)) for key in range(10)}
    inputs = list(inputs.values())

    x = tf.stack(inputs, axis=-1)  
    print(x.shape)  # (None, 5, 10)

    y = tf.stack(inputs)  
    print(y.shape)  # (10, None, 5)


# transpose from bert
def demo_tf_transpose_reshape():

    batch_size = 1
    d_model = 512
    num_heads = 8
    encoder_sequence = 128

    x = tf.random.uniform((1, encoder_sequence, d_model)) 

    x = tf.reshape(x, (batch_size, -1, num_heads, 64)) 
    print(x.shape)  # (1, 128, 8, 64)

    x = tf.transpose(x, perm=[0, 2, 1, 3]) 
    print(x.shape)  # (1, 8, 128, 64)

    x = tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, encoder_sequence, num_heads, depth)
    print(x.shape)  # (1, 128, 8, 64)

    x = tf.reshape(x, (batch_size, -1, d_model))  # (batch_size, encoder_sequence, d_model)
    print(x.shape)  # (1, 128, 512)


# expand and squeeze
def demo_tf_squeeze_function():

    ri = np.random.randint(8,size=(6,6))

    expand = ri[:, np.newaxis, np.newaxis, :]
    print(expand.shape)  # (6, 1, 1, 6)

    squeeze = tf.squeeze(expand, axis=1)
    print(squeeze.shape)  # (6, 1, 6)

    squeeze = tf.squeeze(squeeze, axis=1)
    print(squeeze.shape)  # (6, 6)


# split and unstack:
def demo_tf_split_unstack():

    ones = tf.ones([5, 5, 2])

    splits = tf.split(ones, axis= -1, num_or_size_splits= 2)  # 2 * [5, 5, 1]
    print(len(splits))  # 2

    squeezes = tf.squeeze(splits[0], axis= -1)  # [5, 5]
    print(squeezes.shape)  # (5, 5)

    unstack = tf.unstack(ones, axis= -1)   # 2 * [5, 5]
    print(unstack[0].shape)

    assert (squeezes.numpy() == unstack[0].numpy()).all()  # True

# broadcasting
def demo_tf_broadcasting():

    to_mask = tf.ones(shape=[2, 1, 16], dtype=tf.float32)
    to_ones = tf.ones(shape=[2, 16, 1], dtype=tf.float32) 

    mask = to_mask * to_ones
    print(mask.shape)  # (2, 16, 16)
    return mask

# Dense layer output
def demo_tf_dense_output(mask):

    dense = tf.keras.layers.Dense(32)
    output = dense(mask)
    print(output.shape)  # (2, 16, 32)


if __name__ == '__main__':

    demo_tf_stack_function()

    demo_tf_transpose_reshape()

    demo_tf_squeeze_function()

    demo_tf_split_unstack()

    mask = demo_tf_broadcasting()

    demo_tf_dense_output(mask)

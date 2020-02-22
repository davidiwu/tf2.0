'''
batch_norm就是通过一定的规范化手段，把每层神经网络任意神经元这个激活输入值的分布强行拉回到均值为0方差为1的标准正态分布.

BatchNorm的好处:
	不仅仅极大提升了训练速度，收敛过程大大加快；
	还能增加分类效果，一种解释是这是类似于Dropout的一种防止过拟合的正则化表达方式，所以不用Dropout也能达到相当的效果；
    另外调参过程也简单多了，对于初始化要求没那么高，而且可以使用大的学习率等

'''
import numpy as np
import tensorflow as tf

inputs = np.random.randint(100, size=(10, 5, 5))
inputs = tf.cast(inputs, tf.float32)

mean_before = tf.reduce_mean(inputs).numpy()
std_before = tf.math.reduce_std(inputs).numpy()
item_before = inputs[0][0][0].numpy()

layer = tf.keras.layers.BatchNormalization()
outputs = layer(inputs, training=True)

mean_after = tf.reduce_mean(outputs).numpy()
std_after = tf.math.reduce_std(outputs).numpy()
item_after = outputs[0][0][0].numpy()

print(f'inputs shape: {inputs.shape}, outputs shape: {outputs.shape}')
print(f'inputs data mean, before batch_norm value is {mean_before}, after value is {mean_after}')
print(f'inputs data std, before batch_norm value is {std_before}, after value is {std_after}')
print(f'inputs first item, before batch_norm value is {item_before}, after value is {item_after}')
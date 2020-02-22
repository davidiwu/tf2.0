import tensorflow as tf


print(tf.executing_eagerly())

class MyLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(MyLayer, self).__init__()
        self.my_var = tf.Variable(1.0)
        self.my_var_list = [tf.Variable(x) for x in range(10)]

print(dir(MyLayer))

my_layer = MyLayer()

my_layer_var_number = len(my_layer.variables)

print(my_layer_var_number)
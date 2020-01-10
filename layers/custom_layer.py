import tensorflow as tf

'''
* __init__ , where you can do all input-independent initialization 
* build, where you know the shapes of the input tensors and can do the rest of the initialization 
* call, where you do the forward computation
'''

class MyDenseLayer(tf.keras.layers.Layer):

    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    #  `build()`: Called once from `__call__`, when we know the shapes of inputs
    #   and `dtype`. Should have the calls to `add_weight()`, and then
    #   call the super's `build()` (which sets `self.built = True`, which is
    #   nice in case the user wants to call `build()` manually before the
    #   first `__call__`).
    def build(self, input_shape):
        self.kernel = self.add_weight('kernel', 
                            shape=[int(input_shape[-1]), self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)

def test_custom_layer():

    layer = MyDenseLayer(10)

    output = layer(tf.ones([10, 5]))  # call build method first, then call 'call method'

    print(output.shape)
    print([var.name for var in layer.trainable_variables])


class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        print(input_tensor.shape)
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)
        print(x.shape)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        print(x.shape)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)
        print(x.shape)

        x += input_tensor
        return tf.nn.relu(x)


def test_custom_model():
    block = ResnetIdentityBlock(3, [64, 64, 256])
    output = block(tf.ones([1, 128, 128, 256]))

    print(output.shape)
    print(block.layers)
    print(len(block.variables))
    block.summary()

if __name__ == '__main__':
    test_custom_model()
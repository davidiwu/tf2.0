import tensorflow as tf
from tensorflow.keras import layers


def residual_v2_block(inputs, filters, kernel_size=3, 
                        stride=1, conv_shortcut=False):
    """A residual version 2 block.

    # Arguments
        inputs: input image tensor. channel axis = 3
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
            otherwise identity shortcut.

    # Returns
        Output tensor for the residual block.
    """

    preact = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(inputs)
    preact = layers.Activation('relu')(preact)

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4*filters, 1, strides=stride)(preact)
    else:
        shortcut = layers.MaxPooling2D(1, strides=stride)(inputs) if stride > 1 else inputs

    x = layers.Conv2D(filters, 1, stride=1, use_bias=False)(preact)
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)
    x = layers.Activation('relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.Conv2D(filters, kernel_size, strides=stride, use_bias=False)(x)
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(4*filters, 1)(x)
    x = layers.Add()([shortcut, x])

    return x
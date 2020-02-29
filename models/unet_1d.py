import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def cbr(x, out_layer, kernel, stride, dilation):
    x = layers.Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def se_block(x_in, layer_n):
    x = layers.GlobalAveragePooling1D()(x_in)
    x = layers.Dense(layer_n//8, activation="relu")(x)
    x = layers.Dense(layer_n, activation="sigmoid")(x)
    x_out=layers.Multiply()([x_in, x])
    return x_out

def resblock(x_in, layer_n, kernel, dilation, use_se=True):
    x = cbr(x_in, layer_n, kernel, 1, dilation)
    x = cbr(x, layer_n, kernel, 1, dilation)
    if use_se:
        x = se_block(x, layer_n)
    x = layers.Add()([x_in, x])
    return x  

def Unet(input_shape=(None,1)):
    layer_n = 64
    kernel_size = 7
    depth = 2

    input_layer = layers.Input(input_shape)    
    input_layer_1 = layers.AveragePooling1D(5)(input_layer)
    input_layer_2 = layers.AveragePooling1D(25)(input_layer)
    # input_layer_3 = layers.AveragePooling1D(125)(input_layer)
    
    ########## Encoder
    x = cbr(input_layer, layer_n, kernel_size, 1, 1)
    for i in range(depth):
        x = resblock(x, layer_n, kernel_size, 1)
    out_0 = x

    x = cbr(x, layer_n*2, kernel_size, 5, 1)
    for i in range(depth):
        x = resblock(x, layer_n*2, kernel_size, 1)
    out_1 = x

    x = layers.Concatenate()([x, input_layer_1])    
    x = cbr(x, layer_n*3, kernel_size, 5, 1)
    for i in range(depth):
        x = resblock(x, layer_n*3, kernel_size, 1)
    out_2 = x

    x = layers.Concatenate()([x, input_layer_2])    
    x = cbr(x, layer_n*4, kernel_size, 5, 1)
    for i in range(depth):
        x = resblock(x, layer_n*4, kernel_size, 1)
    
    x = layers.UpSampling1D(5)(x)
    x = layers.Concatenate()([x, out_2])
    x = cbr(x, layer_n*3, kernel_size, 1, 1)

    x = layers.UpSampling1D(5)(x)
    x = layers.Concatenate()([x, out_1])
    x = cbr(x, layer_n*2, kernel_size, 1, 1)

    x = layers.UpSampling1D(5)(x)
    x = layers.Concatenate()([x, out_0])
    x = cbr(x, layer_n, kernel_size, 1, 1)    

    x = layers.Conv1D(11, kernel_size=kernel_size, strides=1, padding="same")(x)
    out = layers.Activation("softmax")(x)
    
    model = keras.Model(input_layer, out)
    
    return model


if __name__ == '__main__':

    model = Unet(input_shape=(4000,1))
    keras.utils.plot_model(model, to_file='unet_1d.png', show_shapes=True)
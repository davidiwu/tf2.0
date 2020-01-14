
# pre-trained models in Keras
Code and pre-trained weights are available for the following image classification models:

    Xception
    VGG16
    VGG19
    ResNet
    ResNet v2
    ResNeXt
    Inception v3
    Inception-ResNet v2
    MobileNet v1
    MobileNet v2
    DenseNet
    NASNet

# use pre-trained models
They can be imported from the module keras.applications:

    from keras.applications.xception import Xception
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg19 import VGG19
    from keras.applications.resnet import ResNet50
    from keras.applications.resnet import ResNet101
    from keras.applications.resnet import ResNet152
    from keras.applications.resnet_v2 import ResNet50V2
    from keras.applications.resnet_v2 import ResNet101V2
    from keras.applications.resnet_v2 import ResNet152V2
    from keras.applications.resnext import ResNeXt50
    from keras.applications.resnext import ResNeXt101
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.applications.mobilenet import MobileNet
    from keras.applications.mobilenet_v2 import MobileNetV2
    from keras.applications.densenet import DenseNet121
    from keras.applications.densenet import DenseNet169
    from keras.applications.densenet import DenseNet201
    from keras.applications.nasnet import NASNetLarge
    from keras.applications.nasnet import NASNetMobile

    model = VGG16(weights='imagenet', include_top=True)

# Documentation for individual models

|Model	|Size	|Top-1 Accuracy	|Top-5 Accuracy	|Parameters	|Depth|
|----|----|----|----|----|----|
|Xception	|88 MB	|0.790	|0.945	|22,910,480	|126|
|VGG16	|528 MB	|0.713	|0.901	|138,357,544	|23|
|VGG19	|549 MB	|0.713	|0.900	|143,667,240	|26|
|ResNet50	|98 MB	|0.749	|0.921	|25,636,712	|-|
|ResNet101	|171 MB	|0.764	|0.928	|44,707,176	|-|
|ResNet152	|232 MB	|0.766	|0.931	|60,419,944	|-|
|ResNet50V2	|98 MB	|0.760	|0.930	|25,613,800	|-|
|ResNet101V2	|171 MB	|0.772	|0.938	|44,675,560	|-|
|ResNet152V2	|232 MB	|0.780	|0.942	|60,380,648	|-|
|InceptionV3	|92 MB	|0.779	|0.937	|23,851,784	|159|
|InceptionResNetV2	|215 MB	|0.803	|0.953	|55,873,736	|572|
|MobileNet	|16 MB	|0.704	|0.895	|4,253,864	|88|
|MobileNetV2	|14 MB	|0.713	|0.901	|3,538,984	|88|
|DenseNet121	|33 MB	|0.750	|0.923	|8,062,504	|121|
|DenseNet169	|57 MB	|0.762	|0.932	|14,307,880	|169|
|DenseNet201	|80 MB	|0.773	|0.936	|20,242,984	|201|
|NASNetMobile	|23 MB	|0.744	|0.919	|5,326,716	|-|
|NASNetLarge	|343 MB	|0.825	|0.960	|88,949,818	|-|

The top-1 and top-5 accuracy refers to the model's performance on the ImageNet validation dataset.

Depth refers to the topological depth of the network. This includes activation layers, batch normalization layers etc.
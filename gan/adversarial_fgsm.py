'''

Fast Gradient Signed Method (FGSM) attack

Y = sign(x) returns an array Y the same size as x, where each element of Y is:

    1 if the corresponding element of x is greater than 0.
    0 if the corresponding element of x equals 0.
    -1 if the corresponding element of x is less than 0.
    x./abs(x) if x is complex.

'''

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

def get_pretrained_mobilenetv2_model():

    pretrained_model = tf.keras.applications.MobileNetV2(
        include_top = True,
        weights='imagenet'
    )
    pretrained_model.trainable = False

    # ImageNet labels
    decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

    return pretrained_model, decode_predictions


# Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = image/255
    image = tf.image.resize(image, (224, 224))
    image = image[None, ...]
    return image


# Helper function to extract labels from probability vector
def get_imagenet_label(probs, decode_predictions):
    docoded_predictions = decode_predictions(probs, top=1)
    return docoded_predictions[0][0]

def get_raw_labrador_image():

    image_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 
        'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_raw)

    image = preprocess(image)

    return image

def predict_raw_labrador_image(image, pretrained_model, decode_predictions):

    image_probs = pretrained_model.predict(image)

    plt.figure()
    plt.imshow(image[0])
    _, image_class, class_confidence = get_imagenet_label(image_probs, decode_predictions)
    plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
    plt.show()

    return image_probs


def create_adversarial_pattern(input_image, input_label, pretrained_model, loss_object):

    with tf.GradientTape() as tape:

        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad


def display_images(pretrained_model, image, description, decode_predictions):

    prediction = pretrained_model.predict(image)
    
    _, label, confidence = get_imagenet_label(prediction, decode_predictions)
    plt.figure()
    plt.imshow(image[0])
    plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                    label, confidence*100))
    plt.show()

def get_image_perturbations(image, pretrained_model, loss_object, total_types):

    # Get the input label of the image.
    labrador_retriever_index = 208
    label = tf.one_hot(labrador_retriever_index, total_types)

    perturbations = create_adversarial_pattern(image, label, pretrained_model, loss_object)
    plt.imshow(perturbations[0])
    plt.show()

    return perturbations

def prediction_after_adding_perturbations(image, perturbations, pretrained_model, decode_predictions):

    epsilons = [0, 0.01, 0.1, 0.15]
    descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                    for eps in epsilons]

    for i, eps in enumerate(epsilons):
        adv_x = image + eps*perturbations
        adv_x = tf.clip_by_value(adv_x, 0, 1)
        display_images(pretrained_model, adv_x, descriptions[i], decode_predictions)


if __name__ == '__main__':

    image = get_raw_labrador_image()

    pretrained_model, decode_predictions = get_pretrained_mobilenetv2_model()

    image_probs = predict_raw_labrador_image(image, pretrained_model, decode_predictions)
    total_types = image_probs.shape[-1]

    loss_object = tf.keras.losses.CategoricalCrossentropy()

    perturbations = get_image_perturbations(image, pretrained_model, loss_object, total_types)

    prediction_after_adding_perturbations(image, perturbations, pretrained_model, decode_predictions)
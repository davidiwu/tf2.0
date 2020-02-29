'''
Saving and Loading model:
    save weight
        model.save_weights('./weights/my_model') # Save weights to a TensorFlow Checkpoint file
        model.load_weights('./weights/my_model')

        model.save_weights('my_model.h5', save_format='h5') # Save weights to a HDF5 file
        model.load_weights('my_model.h5')

    save model configuration only
        json_string = model.to_json()
        fresh_model = tf.keras.models.model_from_json(json_string)

    save the whole model
        model.save('my_model.h5')  # after model compile and fit
        model = tf.keras.models.load_model('my_model.h5')
'''
from tensorflow.keras import layers, models  # type: ignore

def get_cnn(X_train, input_shape, num_labels):
    """Return a convolutional neural network suitable for analysing audio spectrograms."""

    # input_shape = X_train.shape[1:]
    print('Input shape:', input_shape)

    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=X_train)

    # This design was provided by tensorflow here: https://www.tensorflow.org/tutorials/audio/simple_audio
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Downsample the input.
        layers.Resizing(32, 32),
        # Normalize.
        # norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.summary()
    
    return model
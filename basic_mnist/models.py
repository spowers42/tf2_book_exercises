from tensorflow import keras


def get_single_layer_model(
    input_size: int, n_classes: int, activation="softmax"
) -> keras.models.Sequential:
    """create a simple single layer demo NN"""
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            n_classes,
            input_shape=(input_size,),
            activation=activation,
            name="dense_layer",
        )
    )
    return model


def get_multilayer_model(
    input_size: int,
    n_classes: int,
    n_hidden: int,
    activation_hidden="relu",
    output_activation="softmax",
) -> keras.models.Sequential:
    """Create a simle NN with a hidden layer"""
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            n_hidden,
            input_shape=(input_size,),
            name="input_layer",
            activation=activation_hidden,
        )
    )
    model.add(
        keras.layers.Dense(n_hidden, name="hidden_layer", activation=activation_hidden)
    )
    model.add(
        keras.layers.Dense(n_classes, name="output_layer", activation=output_activation)
    )
    return model

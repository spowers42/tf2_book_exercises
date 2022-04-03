from tensorflow import keras


def get_single_layer_model(
    input_size: int, n_classes: int, activation: str = "softmax"
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
    activation_hidden: str = "relu",
    output_activation: str = "softmax",
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


def get_larger_multilayer_model(
    input_size: int,
    n_classes: int,
    n_hidden: int,
    activation_hidden: str = "relu",
    activation_output: str = "softmax",
) -> keras.models.Sequential:
    """Creates a slightly larger multilayer NN"""
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
        keras.layers.Dense(n_hidden, name="hidden1", activation=activation_hidden)
    )
    model.add(
        keras.layers.Dense(
            int(n_hidden / 2), name="hidden2", activation=activation_hidden
        )
    )
    model.add(
        keras.layers.Dense(n_classes, name="output", activation=activation_output)
    )
    return model

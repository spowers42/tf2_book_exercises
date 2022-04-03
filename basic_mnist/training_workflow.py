from metaflow import FlowSpec, step, Parameter, conda_base

# uncomment when using conda to run remotely/on AWS
# @conda_base(libraries={'tensorflow':'2.8.0'})
class MnistTrainingFlow(FlowSpec):
    epochs = Parameter("epochs", help="training epochs", default=200)
    batch_size = Parameter("batch_size", help="training batch size", default=128)
    verbose = Parameter("verbose", help="verbose, 1 or 0", default=1)
    validation_split = Parameter(
        "split", help="training validation split percentage", default=0.2
    )
    n_hidden = Parameter("n_hidden", help="number of hidden layers to use", default=128)

    @step
    def start(self):
        """Start/setup"""
        self.reshaped = 28 * 28
        self.classes = 10
        self.next(self.load_data, self.compile_model)

    @step
    def load_data(self):
        """Load the MNIST dataset and split into training and test"""
        from tensorflow import keras

        mnist = keras.datasets.mnist
        (self.X_train, Y_train), (self.X_test, Y_test) = mnist.load_data()
        self.Y_train = keras.utils.to_categorical(Y_train, self.classes)
        self.Y_test = keras.utils.to_categorical(Y_test, self.classes)
        self.next(self.normalize)

    @step
    def normalize(self):
        """Normalize and reshape the data"""
        self.X_train = (
            self.X_train.reshape(60000, self.reshaped).astype("float32") / 255
        )
        self.X_test = self.X_test.reshape(10000, self.reshaped).astype("float32") / 255
        self.next(self.train_model)

    @step
    def compile_model(self):
        from models import get_larger_multilayer_model as model

        model = model(self.reshaped, self.classes, n_hidden=self.n_hidden)
        model.compile(
            optimizer="SGD", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        self.model = model
        """Compile the tensorflow model"""
        self.next(self.train_model)

    @step
    def train_model(self, inputs):
        """Training time"""
        self.merge_artifacts(inputs)
        self.model.fit(
            self.X_train,
            self.Y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_split=self.validation_split,
        )
        self.next(self.test_model)

    @step
    def test_model(self):
        """Perform validation testing on the trained model"""
        loss, accuracy = self.model.evaluate(self.X_test, self.Y_test)
        print(f"Loss ${loss}, accuracy ${accuracy}")
        self.next(self.end)

    @step
    def end(self):
        """End/cleanup"""
        pass


if __name__ == "__main__":
    MnistTrainingFlow()

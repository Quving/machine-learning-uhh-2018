from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.optimizers as Optimizers


class Nn:

    def __init__(self):
        self.epochs = 250
        self.learning_rate = 0.001
        self.batch_size = 20

        self.model = None
        self.input_dim = 0
        self.output_dim = 0

    def create_model(self, input_dim=11, output_dim=2):
        """
        Returns a compiled untrained model.
        :param input_dim:
        :param output_dim:
        :return:
        """
        self.output_dim = output_dim
        self.input_dim = input_dim

        model = Sequential()
        model.add(
            Dense(input_dim, input_dim=input_dim, kernel_initializer="uniform", activation='tanh', name='layer_0'))
        model.add(Dense(128, activation="tanh", use_bias=False, kernel_initializer="uniform", name='layer_1'))
        model.add(Dense(256, activation="tanh", use_bias=False, kernel_initializer="uniform", name='layer_2'))
        model.add(Dense(128, activation="tanh", use_bias=False, kernel_initializer="uniform", name='layer_3'))
        model.add(Dense(output_dim, activation="tanh", use_bias=False, kernel_initializer="uniform", name='layer_4'))

        opt = Optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06, decay=0.0)
        model.compile(optimizer=opt, loss='mse')

        model.summary()
        self.model = model

    def train(self, x, y):
        """
        Trains the model accordingly to to class variables.
        :return:
        """
        if self.model is None:
            raise ModelNotCreatedException("The model has not been created yet. call create_model() first.")

        if not isinstance(self.model, Sequential):
            raise ModelTypeMissmatchException("Expected object of Sequential. Given " + str(type(self.model)) + ".")


        self.model.fit(x=x, y=y, epochs=self.epochs, batch_size=self.batch_size)



class ModelTypeMissmatchException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class ModelNotCreatedException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)

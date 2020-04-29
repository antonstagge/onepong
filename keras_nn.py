from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Input, Flatten, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_squared_error
import numpy as np
import os


def special_loss(y_true, y_pred):
    # l = K.mean((y_true - y_pred) /
    #            K.cast(K.shape(y_true)[0], 'float32'), axis=1)
    l = mean_squared_error(y_true, y_pred)
    # print(t)
    # print(l)
    # K.print_tensor(t)
    return l


class KerasNetwork:
    """ A neural network"""

    def __init__(self,
                 n_in=None, n_hidden=None, n_out=None,
                 learning_rate=0.001,
                 saveName="Weights",
                 target=False,
                 load=False,
                 conv=False,
                 **kwargs):
        """ Constructor """
        self.learning_rate = learning_rate
        self.is_target_network = target
        self.saveName = saveName

        self.conv = conv
        self.epochs = 2

        # Initialise network
        if load:
            self.loadWeights()
            self.nin = self._model.layers[0].input_shape
            if self.conv:
                self.nin = self.nin[-3:]
            else:
                self.nin = self.nin[-1]

            self.nhidden = self._model.layers[0].output_shape[-1]
            self.nout = self._model.layers[-1].output_shape[-1]
        else:
            # Set up network size
            assert all([x != None for x in [n_in, n_out, n_hidden]])
            self.nin = n_in
            self.nout = n_out
            self.nhidden = n_hidden
            self.epsilon = 1.0

            self._model = Sequential()

            if self.conv:
                self._model.add(Input(shape=(self.nin[0], self.nin[1], 1,)))
                self._model.add(Conv2D(32, 6, strides=3))
                self._model.add(LeakyReLU())
                self._model.add(Conv2D(64, 3, strides=2))
                self._model.add(LeakyReLU())
                self._model.add(Conv2D(64, 2, strides=1))
                self._model.add(LeakyReLU())
                self._model.add(Flatten())
            else:
                self._model.add(Input(shape=(self.nin)))

            self._model.add(
                Dense(self.nhidden, activation='sigmoid'))
            self._model.add(Dense(self.nout, activation=None))

            #opt = RMSprop(learning_rate=self.learning_rate)
            #opt = SGD(learning_rate=self.learning_rate, momentum=self.momentum)
            opt = Adam(learning_rate=self.learning_rate)
            self._model.compile(optimizer=opt, loss='mse')

    def train(self, inputs, targets, batch_size):
        self._model.fit(inputs, targets, epochs=self.epochs,
                        verbose=0, batch_size=batch_size)
        # for i in range(50):
        # self._model.train_on_batch(inputs, targets)

    def forward(self, inputs, add_bias=None):
        """ Predict, add_bias is a NOP """
        return self._model(inputs).numpy()

    def predict(self, input):
        """ Just as forward but with only one vector  """
        inputs = np.array([input])
        out = self._model.predict(inputs)
        return out[0]

    def saveWeights(self):
        root_name = 'weights/keras/' + self.saveName
        if not os.path.isdir(root_name):
            os.mkdir(root_name)
        root_name += '/'
        if self.is_target_network:
            root_name += 't_'
        self._model.save(root_name + 'model.h5')
        np.save(root_name + 'epsilon' + '.npy', self.epsilon)

    def loadWeights(self):
        root_name = 'weights/keras/' + self.saveName + '/'
        if self.is_target_network:
            root_name += 't_'
        self._model = load_model(
            root_name + 'model.h5', custom_objects={'special_loss': special_loss})
        self.epsilon = np.load(root_name + 'epsilon' + '.npy')

    def get_weights(self):
        return self._model.get_weights()

    def set_weights(self, weights):
        self._model.set_weights(weights)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import os


class KerasNetwork:
    """ A neural network"""

    def __init__(self, n_in, n_hidden, n_out, eta=0.001, load=False, beta=1, momentum=0.95, saveName="Weights", target=False):
        """ Constructor """
        # Set up network size
        self.nin = n_in
        self.nout = n_out
        self.nhidden = n_hidden

        self.eta = eta
        self.beta = beta
        self.momentum = momentum

        # Initialise network
        self.saveName = saveName
        if load:
            self.loadWeights(target=target)
        else:
            self.epsilon = 1.0
            self._model = Sequential([
                Dense(self.nhidden, input_dim=self.nin, activation='sigmoid'),
                Dense(self.nout, activation='linear')
            ])
            self._model.compile(
                loss='categorical_crossentropy', optimizer=Adam(learning_rate=self.eta))

    def train(self, inputs, targets):
        self._model.fit(inputs, targets)

    def forward(self, inputs, add_bias=None):
        """ Predict, add_bias is a NOP """
        out = self._model.predict(inputs)
        return out

    def predict(self, input):
        """ Just as forward but with only one vector  """
        inputs = np.array([input])
        out = self._model.predict(inputs)
        return out[0]

    def saveWeights(self, target=False):
        root_name = 'weights/keras/' + self.saveName
        if not os.path.isdir(root_name):
            os.mkdir(root_name)
        root_name += '/'
        if target:
            root_name += 't_'
        self._model.save(root_name + 'model.h5')
        np.save(root_name + 'epsilon' + '.npy', self.epsilon)

    def loadWeights(self, target=False):
        root_name = 'weights/keras/' + self.saveName + '/'
        if target:
            root_name += 't_'
        self._model = load_model(root_name + 'model.h5')
        self.epsilon = np.load(root_name + 'epsilon' + '.npy')

    def get_weights(self):
        return self._model.get_weights()

    def set_weights(self, weights):
        self._model.set_weights(weights)

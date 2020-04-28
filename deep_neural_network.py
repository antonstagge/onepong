
# Initial Code by Stephen Marsland
# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes. -Stephen Marsland
#
# Improved and edited to be used by Deep Q-learning by Anton Stagge 2018

import numpy as np
import os


class Network:
    """ A neural network"""

    def __init__(self,
                 n_in=None, n_hidden=None, n_out=None,
                 learning_rate=0.001, momentum=0.95,
                 saveName="Weights",
                 target=False,
                 load=False):
        """ Constructor """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta = 0.99
        self.saveName = saveName
        self.is_target_network = target

        # Initialise network
        if load:
            self.loadWeights()
            self.nin = self.weights1.shape[0] - 1
            self.nhidden = self.weights1.shape[1]
            self.nout = self.weights2.shape[1]
        else:
            # new network
            # Set up network size
            assert all([x != None for x in [n_in, n_out, n_hidden]])
            self.nin = n_in
            self.nout = n_out
            self.nhidden = n_hidden
            self.weights1 = (np.random.rand(
                self.nin+1, self.nhidden) - 0.5)*2 / np.sqrt(self.nin)
            self.weights2 = (np.random.rand(
                self.nhidden+1, self.nout)-0.5)*2/np.sqrt(self.nhidden)
            self.epsilon = 1.0

    def train(self, inputs, targets, *args):
        """ Train the thing """
        self.ndata = np.shape(inputs)[0]
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs, -np.ones((self.ndata, 1))), axis=1)
        change = list(range(self.ndata))

        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))

        for n in range(50):
            self.outputs = self.forward(inputs)

            deltao = (targets-self.outputs)/self.ndata

            deltah = self.hidden*(1.0-self.hidden) * \
                (np.dot(deltao, np.transpose(self.weights2)))

            updatew1 = self.learning_rate * \
                (np.dot(np.transpose(inputs),
                        deltah[:, :-1])) + self.momentum*updatew1
            updatew2 = self.learning_rate*(np.dot(np.transpose(self.hidden), deltao)
                                           ) + self.momentum*updatew2

            self.weights1 += updatew1
            self.weights2 += updatew2

            # Randomise order of inputs
            np.random.shuffle(change)
            inputs = inputs[change, :]
            targets = targets[change, :]

    def forward(self, inputs, add_bias=False):
        """ Run the network forward"""
        self.ndata = np.shape(inputs)[0]

        if add_bias:
            inputs = np.concatenate(
                (inputs, -np.ones((self.ndata, 1))), axis=1)

        self.hidden = np.dot(inputs, self.weights1)
        self.hidden = self.sigmoid(self.hidden)
        self.hidden = np.concatenate(
            (self.hidden, -np.ones((np.shape(inputs)[0], 1))), axis=1)

        outputs = np.dot(self.hidden, self.weights2)
        self.outputs = self.linear(outputs)
        return outputs

    def predict(self, input):
        """ Just as forward but with only one vector  """
        inputs = np.array([input])
        out = self.forward(inputs, True)
        return out[0]

    def linear(self, outputs):
        return outputs

    def sigmoid(self, hidden):
        return 1.0/(1.0+np.exp(-self.beta*hidden))

    def saveWeights(self):
        root_name = 'weights/' + self.saveName
        if not os.path.isdir(root_name):
            os.mkdir(root_name)
        root_name += '/'
        if self.is_target_network:
            root_name += 't_'
        np.save(root_name + 'w' + str(1) + '.npy', self.weights1)
        np.save(root_name + 'w' + str(2) + '.npy', self.weights2)
        np.save(root_name + 'epsilon' + '.npy', self.epsilon)

    def loadWeights(self):
        root_name = 'weights/' + self.saveName + '/'
        if self.is_target_network:
            root_name += 't_'
        self.weights1 = np.load(root_name + 'w' + str(1) + '.npy')
        self.weights2 = np.load(root_name + 'w' + str(2) + '.npy')
        self.epsilon = np.load(root_name + 'epsilon' + '.npy')

    def get_weights(self):
        return (np.copy(self.weights1), np.copy(self.weights2))

    def set_weights(self, weights):
        self.weights1 = weights[0]
        self.weights2 = weights[1]

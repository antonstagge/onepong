
# Initial Code by Stephen Marsland
# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes. -Stephen Marsland
#
# Improved and edited to be used by Deep Q-learning by Anton Stagge 2018

from numpy import *

class network:
    """ A neural network"""

    def __init__(self,n_in,n_hidden,n_out, load = False, beta=1, momentum=0.95, saveName = "Weights"):
        """ Constructor """
        # Set up network size
        self.nin = n_in
        self.nout = n_out
        self.ndata = 1
        self.nhidden = n_hidden

        self.beta = beta
        self.momentum = momentum

        # Initialise network
        self.saveName = saveName
        if load:
            self.loadWeights()
        else:
            # new network
            self.weights1 = (random.rand(self.nin+1,self.nhidden)-0.5)*2/sqrt(self.nin)
            self.weights2 = (random.rand(self.nhidden+1,self.nout)-0.5)*2/sqrt(self.nhidden)
            self.epsilon = 1.0

    def train(self, inputs, targets, eta):
        """ Train the thing """
        self.ndata = shape(inputs)[0]
        # Add the inputs that match the bias node
        inputs = concatenate((inputs,-ones((self.ndata,1))),axis=1)
        change = list(range(self.ndata))

        updatew1 = zeros((shape(self.weights1)))
        updatew2 = zeros((shape(self.weights2)))

        for n in range(50):
            self.outputs = self.forward(inputs)

            deltao = (targets-self.outputs)/self.ndata

            deltah = self.hidden*(1.0-self.hidden)*(dot(deltao,transpose(self.weights2)))

            updatew1 = eta*(dot(transpose(inputs),deltah[:,:-1])) + self.momentum*updatew1
            updatew2 = eta*(dot(transpose(self.hidden),deltao)) + self.momentum*updatew2

            self.weights1 += updatew1
            self.weights2 += updatew2

            # Randomise order of inputs
            random.shuffle(change)
            inputs = inputs[change,:]
            targets = targets[change,:]


    def forward(self, inputs, add_bias=False):
        """ Run the network forward"""
        self.ndata = shape(inputs)[0]

        if add_bias:
            inputs = concatenate((inputs,-ones((self.ndata,1))),axis=1)

        self.hidden = dot(inputs, self.weights1)
        self.hidden = self.sigmoid(self.hidden)
        self.hidden = concatenate((self.hidden,-ones((shape(inputs)[0],1))),axis=1)

        outputs = dot(self.hidden, self.weights2)
        outputs = self.linear(outputs)
        return outputs

    def predict(self, input):
        """ Just as forward but with only one vector  """
        inputs = array([input])
        out = self.forward(inputs, True)
        return out[0]

    def relu(self, hidden):
        return maximum(0, hidden)

    def linear(self, outputs):
        return outputs

    def softmax(self, outputs):
        normalisers = sum(exp(outputs),axis=1)*ones((1,shape(outputs)[0]))
        return transpose(transpose(exp(outputs))/normalisers)

    def stable_softmax(self, outputs):
        try:
            temp = -0.5
            z = outputs - temp
            normalisers = sum(exp(z),axis=1)*ones((1,shape(z)[0]))
            soft_max = transpose(transpose(exp(z))/normalisers)
            return soft_max
        except FloatingPointError:
            z = outputs
            normalisers = sum(z,axis=1)*ones((1,shape(z)[0]))
            soft_max = transpose(transpose(z)/normalisers)
            return soft_max

    def sigmoid(self, hidden):
        return 1.0/(1.0+exp(-self.beta*hidden))

    def saveWeights(self):
        save((self.saveName + "_" + str(1) + ".npy"), self.weights1)
        save((self.saveName + "_" + str(2) + ".npy"), self.weights2)
        save((self.saveName + "_epsilon" + ".npy"), self.epsilon)

    def loadWeights(self):
        self.weights1 = load((self.saveName + "_" + str(1) + ".npy"))
        self.weights2 = load((self.saveName + "_" + str(2) + ".npy"))
        self.epsilon = load((self.saveName + "_epsilon" + ".npy"))

    def get_weights(self):
        return (copy(self.weights1), copy(self.weights2))
    def set_weights(self, weights):
        self.weights1 = weights[0]
        self.weights2 = weights[1]

# # Example how to use it!
# inputs = array(
#     [
#     [1, 1, 2],
#     [3, 2, 1]
#     ])
# targets = array(
#     [
#     [0, 3],
#     [-1, 0]
#     ])
#
# p = network(inputs, targets, 10, loadW = False)
# #print(p.softmax(inputs))
# print("relu")
# print(p.relu(temp))
# print(p.stable_softmax(inputs))
# p.train(inputs, targets, 0.01, 100)
# print(p.predict(inputs[0]))

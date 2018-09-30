
# Code from Chapter 3 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008
# Totally edited by Anton Stagge

from numpy import *

class mlp:
    """ A Multi-Layer Perceptron"""

    def __init__(self,inputs,targets,nhidden, load = False, beta=1,momentum=0.95, saveName = "Weights"):
        """ Constructor """
        # Set up network size
        self.nin = shape(inputs)[1]
        self.nout = shape(targets)[1]
        self.ndata = shape(inputs)[0]
        self.nhidden = nhidden

        self.beta = beta
        self.momentum = momentum

        # Initialise network
        self.saveName = saveName
        if load:
            self.loadWeights()
        else:
            self.weights1 = (random.rand(self.nin+1,self.nhidden)-0.5)*2/sqrt(self.nin)
            self.weights2 = (random.rand(self.nhidden+1,self.nout)-0.5)*2/sqrt(self.nhidden)
            self.epsilon = 1.0

        # if self.epsilon < 0.2:
        #     self.epsilon = 0.8


    def mlptrain(self, inputs, targets, eta, w1, w2):
        """ Train the thing """
        self.ndata = shape(inputs)[0]
        # Add the inputs that match the bias node
        inputs = concatenate((inputs,-ones((self.ndata,1))),axis=1)
        change = list(range(self.ndata))

        updatew1 = zeros((shape(w1)))
        updatew2 = zeros((shape(w2)))

        for n in range(100):
            self.outputs = self.mlpfwd(inputs, w1, w2)

            deltao = (targets-self.outputs)/self.ndata
            # if n == 1:
            #     print(" deltao")
            #     print(deltao)
            #     if targets2 is not None:
            #         print(" TARGETS 2")
            #         print(targets2/self.ndata)
            #deltao = targets/self.ndata
            #print(sum(deltao))

            deltah = self.hidden*(1.0-self.hidden)*(dot(deltao,transpose(w2)))

            updatew1 = eta*(dot(transpose(inputs),deltah[:,:-1])) + self.momentum*updatew1
            updatew2 = eta*(dot(transpose(self.hidden),deltao)) + self.momentum*updatew2

            self.weights1 += updatew1
            self.weights2 += updatew2

            # Randomise order of inputs
            random.shuffle(change)
            inputs = inputs[change,:]
            targets = targets[change,:]


    def mlpfwd(self, inputs, w1, w2):
        """ Run the network forward using weight w1 and w2"""
        self.ndata = shape(inputs)[0]

        self.hidden = dot(inputs, w1)
        self.hidden = self.relu(self.hidden)
        self.hidden = concatenate((self.hidden,-ones((shape(inputs)[0],1))),axis=1)

        outputs = dot(self.hidden, w2)
        outputs = self.linear(outputs)
        return outputs

    def predict(self, input):
        """ Just as mlpfwd but with only one vector and with append bias """
        inputs = array([input])
        inputs = concatenate((inputs,-ones((1,1))),axis=1)

        self.ndata = shape(inputs)[0]

        out = self.mlpfwd(inputs, self.weights1, self.weights2)
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
        save((str(1)+ "_" + self.saveName + ".npy"), self.weights1)
        save((str(2)+ "_" + self.saveName + ".npy"), self.weights2)
        save("epsilon_" + self.saveName + ".npy", self.epsilon)

    def loadWeights(self):
        self.weights1 = load((str(1) + "_" + self.saveName + ".npy"))
        self.weights2 = load((str(2) + "_" + self.saveName + ".npy"))
        self.epsilon = load("epsilon_" + self.saveName + ".npy")

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
# temp = [-128.69569334, -127.83431717 ,-127.56004046]
#
#
# test = amax(inputs, axis=1).shape
# print(test)
# print(inputs - test)
#
# p = mlp(inputs, targets, 10, loadW = False)
# #print(p.softmax(inputs))
# print("relu")
# print(p.relu(temp))
# print(p.stable_softmax(inputs))
# p.mlptrain(inputs, targets, 0.01, 100)
# print(p.predict(inputs[0]))
# p.saveWeights()
#print(log(-1))


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

    def __init__(self,inputs,targets,nhidden, loadW = False, beta=1,momentum=0.9, saveName = "Weights"):
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
        if loadW:
            self.loadWeights()
        else:
            self.weights1 = (random.rand(self.nin+1,self.nhidden)-0.5)*2/sqrt(self.nin)
            self.weights2 = (random.rand(self.nhidden+1,self.nout)-0.5)*2/sqrt(self.nhidden)



    def mlptrain(self, inputs, targets, eta, w1, w2):
        """ Train the thing """
        self.ndata = shape(inputs)[0]
        # Add the inputs that match the bias node
        inputs = concatenate((inputs,-ones((self.ndata,1))),axis=1)
        change = list(range(self.ndata))

        updatew1 = zeros((shape(self.weights1)))
        updatew2 = zeros((shape(self.weights2)))

        self.outputs = self.mlpfwd(inputs, w1, w2)

        deltao = targets #(targets-self.outputs)/self.ndata

        deltah = self.hidden*(1.0-self.hidden)*(dot(deltao,transpose(self.weights2)))

        updatew1 = eta*(dot(transpose(inputs),deltah[:,:-1])) + self.momentum*updatew1
        updatew2 = eta*(dot(transpose(self.hidden),deltao)) + self.momentum*updatew2

        self.weights1 += updatew1
        self.weights2 += updatew2


    def mlpfwd(self, inputs, w1, w2):
        """ Run the network forward using weight w1 and w2"""

        self.hidden = dot(inputs, w1)
        self.hidden = 1.0/(1.0+exp(-self.beta*self.hidden))
        self.hidden = concatenate((self.hidden,-ones((shape(inputs)[0],1))),axis=1)

        outputs = dot(self.hidden, w2)
        return outputs

    def predict(self, input):
        """ Just as mlpfwd but with only one vector and with append bias """
        inputs = array([input])
        inputs = concatenate((inputs,-ones((1,1))),axis=1)

        self.hidden = dot(inputs,self.weights1)
        self.hidden = 1.0/(1.0+exp(-self.beta*self.hidden))
        self.hidden = concatenate((self.hidden,-ones((shape(inputs)[0],1))),axis=1)

        outputs = dot(self.hidden,self.weights2)
        return self.softmax(outputs)[0]



    def softmax(self, outputs):
        normalisers = sum(exp(outputs),axis=1)*ones((1,shape(outputs)[0]))
        return transpose(transpose(exp(outputs))/normalisers)

    def saveWeights(self):
        save((str(1)+ "_" + self.saveName + ".npy"), self.weights1)
        save((str(2)+ "_" + self.saveName + ".npy"), self.weights2)

    def loadWeights(self):
        self.weights1 = load((str(1) + "_" + self.saveName + ".npy"))
        self.weights2 = load((str(2) + "_" + self.saveName + ".npy"))

# # Example how to use it!
# inputs = array(
#     [
#     [1, 2, 3],
#     [3, 2, 1]
#     ])
# targets = array(
#     [
#     [0, 1],
#     [1, 0]
#     ])
#
# p = mlp(inputs, targets, 10, loadW = False)
# p.mlptrain(inputs, targets, 0.01, 100)
# print(p.predict(inputs[0]))
# p.saveWeights()

import numpy
import mlp
from onepong import *
import tensorflow as tf


# s = State()
#
# s.run()
# s.run()
# state1 = numpy.copy(s._state)
# s.run()
# state2 = numpy.copy(s._state)
#
# data1 = (state2-state1).flatten()
# s.run()
# s.run()
# state1 = numpy.copy(s._state)
# s.run()
# state2 = numpy.copy(s._state)
# data2 = (state2-state1).flatten()
#
#
# goal1 = numpy.array([13,0,0])
# goal2 = numpy.array([0,0,3])
#
# inputs = []
# targets = []
# for i in range(0, 10):
#     inputs.append(data1)
#     inputs.append(data2)
#     targets.append(goal1)
#     targets.append(goal2)
#
# inputs = numpy.array(inputs)
# targets = numpy.array(targets)
#
# #print(inputs.shape)
# #print(targets.shape)
#
# neural_net = mlp.mlp(inputs, targets, 75, False, beta=1, saveName = "TEEEST")
#
# for i in range(0, 10):
#     neural_net.mlptrain(inputs, targets, 0.001, neural_net.weights1, neural_net.weights2)
#
# print(neural_net.predict(data1))
# print(neural_net.predict(data2))

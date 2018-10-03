import matplotlib.pyplot as plt
import deep_neural_network


SAVE_NAME = "FIXED_GAME"


neural_net = deep_neural_network.network([[]], [[]], 1, saveName = SAVE_NAME)
neural_net.loadWeights()

fig, axes = plt.subplots(2, 3)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = neural_net.weights1.min(), neural_net.weights1.max()
for coef, ax in zip(neural_net.weights1, axes.ravel()):
    ax.matshow(coef.reshape(2,5), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()

fig, axes = plt.subplots(3, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = neural_net.weights2.min(), neural_net.weights2.max()
for coef, ax in zip(neural_net.weights2, axes.ravel()):
    ax.matshow(coef.reshape(1,3), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()

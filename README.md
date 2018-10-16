# A Double Deep Q-learning Network without tensorflow
This is a Double Deep Q-learning Network with a implementation of a small game I
made called Onepong. Onepong is very much like pong except that it's only for
one player. Much like playing tennis against a wall.

## File summary
### deep_neural_network.py
This file contains the neural network itself. It is not dependent on any other
file, which means that you can use it for whatever you want.

### DQN.py
This file contains the login that is tied to Deep Q-learning and is not tied to
any files, except for deep_neural_network.py which is used for it's networks.
This means that technically DQN.py can be used to create a DQN that can play
any game, not just Onepong.

##### The Double Deep
The reason this is a **Double** Deep Q-learning Network is that there are 2 deep
networks at play. The networks are doing the exact same thing, meaning they are
trained on the same type of data. However, they are not trained on the exakt
same data. One of the networks is used to predict the action to take given an
observation, the other is used to give the value for the estimated future value
in the Q-update function:

Q(s, a<sub>t</sub>) += R + gamma * max<sub>a</sub>Q(s', a)

where gamma is the DISCOUND_FACTOR.

So basically, the live network is used to pick the action a<sub>t</sub> and
which action a maximizes Q(s', a), but then the value of the taget networks
Q(s', a) is used instead of the value for the live network.

This is done to avoid the problem with DQNs where the DQN tend to overestimated
the Q-values. If the Q-network keeps overestimating certain suboptimal actions,
meaning they are going to get higher Q-values, the network is going to have a
hard time learning the optimal actions. By removing the link between the Q-values
and the choice of the action, we can reduce the overestimation and help the
training procedure substantially.

### onepong.py
This file includes only the logic and drawing of the actual game. To simply
play the game all you have to do is:
```python
from onepong import *
pong = PlayPong(player = True, draw = True)
done = False
while not done:
    done = pong.play_one_pong()
```
This file also includes the functins `get_reward()` and `get_observation()`
which are used for the training procedure.

### draw_neural_net.py
This file adds an extra 500 px to the pygame screen that is drawing the game.
This is so that it can draw the neurons and the values to show which neurons
are active in the process of choosing an action.

### main.py
This file ties the onepong game together with the DQN, or simply lets you play
a game of pong on your own.

##### To play a game of pong
`> python3 main.py -p`

##### To train a new network to play pong
`> python3 main.py -i DEMO`

`> python3 main.py -t DEMO`

##### To watch the ai play
`> python3 main.py -a DEMO`

or to use the other network

`> python3 main.py -a -s DEMO`

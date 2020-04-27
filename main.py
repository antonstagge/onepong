# All code written by Anton Stagge 2018
# Feel free to use/edit it to what ever extent but please
# keep the original authors name.

import sys
import getopt
import numpy as np
from onepong import *
from snake import *
import deep_neural_network
import keras_nn
import draw_neural_net
import DQN
from collections import deque

#network = deep_neural_network.network
network = keras_nn.KerasNetwork

OUTER_ITER = 1000
NUMBER_OF_PLAYS = 50
NETWORK_SYNC_FREQ = 100
MAX_POINTS = 200

HIDDEN = 10

TR_SPEED = 0.001
DISCOUND_FACTOR = 0.95

EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

BATCH_SIZE = 32 * 2 * 2 * 2

games = {
    'onepong': {
        'initialize': PlayPong,
        'play_one_iteration': PlayPong.play_one_iteration,
        'get_observation': PlayPong.get_observation,
        'get_reward': PlayPong.get_reward
    },
    'snake': {
        'initialize': PlaySnake,
        'play_one_iteration': PlaySnake.play_one_iteration,
        'get_observation': PlaySnake.get_observation,
        'get_reward': PlaySnake.get_reward
    },
}

#GAME = 'onepong'
GAME = 'snake'

if GAME == 'onepong':
    N_IN = 5
    N_OUT = 3
elif GAME == 'snake':
    N_IN = 12
    N_OUT = 4


def usage():
    string = "There are multiple things you can do with onepong!\n"
    string += "you run it by typing:\n\n"
    string += "python3 main.py [-h] [-p] [-a [-s <trial_name>] [-t <trial_name>] [-i <trial_name>]\n\n"
    string += "-h --help : shows this help message.\n\n"
    string += "-p --play : play a game of onepong yourself.\n\n"
    string += "-a --ai   : watch as the neural network plays onepong. Combine with\n"
    string += "            -s or --swap to use the second neural network for decision making.\n\n"
    string += "-t --train: train the neural network, updates the weights in the trial_name files\n\n"
    string += "-i --init : creates the files needed under name trial_name, and initialize the network with random weights. \n\n"
    return string


def main():
    """
    This main function will either let you play the pong game yourself with -p,
    watch the ai play the game with flags -a (swap which net is used with -s),
    train the ai with flag -t
    or initialize the weights randomly for training with flag -i
    use -h or --help to get more information about the usage.
    """
    player = False
    train = False
    ai = False
    swap = False
    init = False
    SAVE_NAME = 'DEFAULT'

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(
            argv, "hpa:st:i:", ["help", "play", "ai=", "swap", "train=", "init="])
    except getopt.GetoptError:
        print(usage())
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage())
        elif opt in ('-p', '--play'):
            player = True
        elif opt in ('-a', '--ai'):
            ai = True
            SAVE_NAME = arg
        elif opt in ('-s', '--swap'):
            swap = True
        elif opt in ('-t', '--train'):
            train = True
            SAVE_NAME = arg
        elif opt in ('-i', '--init'):
            init = True
            SAVE_NAME = arg

    if player:
        return normal_play()
    elif ai:
        return ai_play(swap, SAVE_NAME)
    elif init:
        sure = input('Are you sure? (y/n):\n')
        if sure == 'y':
            neural_net = network(
                N_IN, HIDDEN, N_OUT, False, saveName=SAVE_NAME)
            neural_net.saveWeights()
            target_net = network(
                N_IN, HIDDEN, N_OUT, False, saveName=SAVE_NAME)
            target_net.saveWeights(target=True)
        return
    elif train:
        DQN.training_iteration(network,
                               SAVE_NAME,
                               OUTER_ITER, NUMBER_OF_PLAYS, MAX_POINTS, NETWORK_SYNC_FREQ,
                               N_IN, HIDDEN, N_OUT,
                               TR_SPEED, DISCOUND_FACTOR,
                               EPSILON_DECAY, EPSILON_MIN,
                               BATCH_SIZE,
                               initialize=games[GAME]['initialize'],
                               play_one_iteration=games[GAME]['play_one_iteration'],
                               get_observation=games[GAME]['get_observation'],
                               get_reward=games[GAME]['get_reward'])
        return


def normal_play():
    # player True and draw True
    game = games[GAME]['initialize'](player=True, draw=True)
    done = False
    while not done:
        done = game.play_one_iteration()
    print(" GAME OVER!!\nYou got %d points" % game.state.points)


def ai_play(swap_network, SAVE_NAME):
    if swap_network:
        print("Swapped")
    neural_net = network(
        N_IN, HIDDEN, N_OUT, True, saveName=(SAVE_NAME), target=swap_network)
    # player False and draw True
    game = games[GAME]['initialize'](player=False, draw=True)
    done = False
    grow = True
    while not done:
        obs = game.get_observation()
        print(obs)
        action = DQN.act(neural_net, obs, training=False)
        print("Action choosen:", action)
        draw_neural_net.draw(game.screen, grow, obs,
                             neural_net.hidden[0], neural_net.outputs[0])
        grow = False
        done = game.play_one_iteration(action)
    print(" GAME OVER!!\nAI scored %d points" % game.state.points)
    while True:
        pass


if __name__ == "__main__":
    main()

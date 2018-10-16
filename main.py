#!/usr/bin/python3
# Code written by Anton Stagge 2018

import sys
import getopt
import numpy as np
from onepong import *
import deep_neural_network
import draw_neural_net
import DQN
from collections import deque

OUTER_ITER = 10000
NUMBER_OF_PLAYS = 20
NETWORK_SYNC_FREQ = 100
MAX_POINTS = 50

N_IN = 5
HIDDEN = 10
N_OUT = 3

TR_SPEED = 0.001
DISCOUND_FACTOR = 0.95

EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

BATCH_SIZE = 32

def usage():
    string =  "There are multiple things you can do with onepong!\n"
    string += "you run it by typing:\n\n"
    string += "python3 main.py [-h] [-p] [-a [-s <input_file.npy>] [-t <input_file.npy>] [-i <input_file.npy>]\n\n"
    string += "-h --help : shows this help message.\n\n"
    string += "-p --play : play a game of onepong yourself.\n\n"
    string += "-a --ai   : watch as the neural network plays onepong. Combine with\n"
    string += "            -s or --swap to use the second neural network for decision making.\n\n"
    string += "-t --train: train the neural network, updates the weights in input_file.npy\n\n"
    string += "-i --init : creates intput_file, and initialize the network with random weights. \n\n"
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
        opts, args = getopt.getopt(argv,"hpa:st:i:",["help", "play", "ai=", "swap", "train=", "init="])
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
            neural_net = deep_neural_network.network(N_IN, HIDDEN, N_OUT, False, saveName = SAVE_NAME)
            neural_net.saveWeights()
            target_net = deep_neural_network.network(N_IN, HIDDEN, N_OUT, False, saveName = (SAVE_NAME + "_target"))
            target_net.saveWeights()
        return
    elif train:
        DQN.training_iteration(SAVE_NAME,
            OUTER_ITER, NUMBER_OF_PLAYS, MAX_POINTS, NETWORK_SYNC_FREQ,
            N_IN, HIDDEN, N_OUT,
            TR_SPEED, DISCOUND_FACTOR, EPSILON_DECAY,
            BATCH_SIZE,
            initialize = PlayPong,
            play_one_iteration = PlayPong.play_one_pong,
            get_observation = PlayPong.get_observation,
            get_reward = PlayPong.get_reward)
        return

def normal_play():
    # player True and draw True
    pong = PlayPong(True, True)
    done = False
    while not done:
        done = pong.play_one_pong()
    print(" GAME OVER!!\nYou got %d points" % pong.state.points)

def ai_play(swap_network, SAVE_NAME):
    if swap_network:
        print("Swapped")
        neural_net = deep_neural_network.network(N_IN, HIDDEN, N_OUT, True, saveName = (SAVE_NAME + "_target"))
    else:
        neural_net = deep_neural_network.network(N_IN, HIDDEN, N_OUT, True, saveName = SAVE_NAME)
    # player False and draw True
    pong = PlayPong(False, True)
    done = False
    grow = True
    while not done:
        obs = pong.get_observation()
        action = DQN.act(neural_net, obs, training = False)
        draw_neural_net.draw(pong.screen, grow, obs, neural_net.hidden[0], neural_net.outputs[0])
        grow = False
        done = pong.play_one_pong(action)
    print(" GAME OVER!!\AI scored %d points" % pong.state.points)


if __name__ == "__main__":
    main()

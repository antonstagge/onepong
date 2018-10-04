#!/usr/bin/python3
# Code written by Anton Stagge 2018

import datetime
import pygame
import sys
import numpy as np
from onepong import *
import deep_neural_network
import DQN
from collections import deque
import random
import matplotlib.pyplot as plt

SAVE_NAME = "TESTING"



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

def main():
    """
    This main function will either let you play the pong game yourself with -play,
    watch the ai play the game with flags -ai (swap which net is used with -swap),
    train the ai with flag -train
    or innitialize the weights for training with flag -i
    """
    player = False
    train = False
    ai = False
    swap = False
    init = False

    draw = False

    if len(sys.argv) > 1:
        if "-play" in sys.argv:
            player = True
        elif "-ai" in sys.argv:
            ai = True
            if "-swap" in sys.argv:
                swap = True
        elif "-train" in sys.argv:
            train = True
        elif "-i" in sys.argv:
            init = True

    if player:
        return normal_play()
    elif ai:
        return ai_play(swap)
    elif init:
        sure = input('Are you sure? (y/n): ')
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
            get_observation = get_observation,
            get_reward = PlayPong.get_reward)
        return

def normal_play():
    # player True and draw True
    pong = PlayPong(True, True)
    done = False
    while not done:
        done = pong.play_one_pong()
    print(" GAME OVER!!\nYou got %d points" % pong.state.points)

def ai_play(swap_network):
    if swap_network:
        print("swapped")
        neural_net = deep_neural_network.network(N_IN, HIDDEN, N_OUT, True, saveName = (SAVE_NAME + "_target"))
    else:
        neural_net = deep_neural_network.network(N_IN, HIDDEN, N_OUT, True, saveName = SAVE_NAME)
    # player False and draw True
    pong = PlayPong(False, True)
    done = False
    while not done:
        obs = get_observation(pong)
        action = DQN.act(neural_net, obs, training = False)
        done = pong.play_one_pong(action)
    print(" GAME OVER!!\AI scored %d points" % pong.state.points)

def get_observation(pong):
    """ Return the vector representation of a pong state
        which is the balls position direction and the pads position.
    """
    state = pong.state
    obs = np.array([state._position[0], state._position[1],
        state._direction[0],
        state._direction[1],
        state._pad])
    return obs


if __name__ == "__main__":
    main()

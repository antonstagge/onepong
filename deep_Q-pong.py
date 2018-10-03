#!/usr/bin/python3
# Code written by Anton Stagge 2018

import datetime
import pygame
import sys
import numpy as np
from onepong import *
import deep_neural_network
from collections import deque
import random
import matplotlib.pyplot as plt

SAVE_NAME = "FIXED_GAME"

OUTER_ITER = 10000
NUMBER_OF_PLAYS = 20
TARGET_UPDATE_FREQ = 100

BETA = 0.99
HIDDEN = 10
TR_SPEED = 0.001
DISCOUND_FACTOR = 0.95

EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

BATCH_SIZE = 32

def main():
    """
    This main function will either let you play the pong game yourself (no flags),
    watch the ai play the game with flags -ai -d (swap which net is used with -swap),
    train the ai with flag -ai
    or reset the weight for training with flag -i

    The ai uses Deep Q-learning with double Q-nets.
    One for net is used to make prediction and decide on actions,
    the other is used to get the values for the Q-update.

    The training works iteratively
    First you gather a lot if data by simply playing the game,
    then you pick BATCH_SIZE of this data to use for training.
    """
    player = True
    draw = False
    swap = False
    if len(sys.argv) > 1:
        if "-ai" in sys.argv:
            player = False
        if "-d" in sys.argv:
            draw = True
            if "-swap" in sys.argv:
                swap = True
        if "-i" in sys.argv:
            sure = input('Are you sure? (y/n): ')
            if sure == 'y':
                set_up_input = np.array([np.zeros(5)])
                set_up_target = np.array([[0,0,0]])
                neural_net = deep_neural_network.network(set_up_input, set_up_target, HIDDEN, False, beta=BETA, saveName = SAVE_NAME)
                neural_net.saveWeights()
                target_net = deep_neural_network.network(set_up_input, set_up_target, HIDDEN, False, beta=BETA, saveName = (SAVE_NAME + "_target"))
                target_net.saveWeights()
            return

    if player:
        return normal_play()

    np.seterr(divide='raise', over='raise', under='warn', invalid='raise')
    np.set_printoptions(threshold=np.nan)

    total_amount_training_data = 0 # only for printing

    for train in range(0, OUTER_ITER):
        # Initialising networks
        set_up_input = np.array([np.zeros(5)])
        set_up_target = np.array([[0,0,0]])
        neural_net = deep_neural_network.network(set_up_input, set_up_target, HIDDEN, True, beta=BETA, saveName = SAVE_NAME)
        target_net = deep_neural_network.network(set_up_input, set_up_target, HIDDEN, True, beta=BETA, saveName = (SAVE_NAME + "_target"))

        # Sometime sync target_network
        if train % TARGET_UPDATE_FREQ == 0 and not draw:
            target_net.set_weights(neural_net.get_weights())

        # randomly swap the target and live networks
        if (np.random.uniform() < 0.5 and not draw) or swap:
            temp = neural_net
            neural_net = target_net
            target_net = temp

        if draw:
            print("live net is now %s" % neural_net.saveName)

        # GATHER TRAINING DATA
        memory = deque(maxlen = 2000)
        max_points = 0
        for t in range(0, NUMBER_OF_PLAYS):
            # Initialize game
            pong = PlayPong(False, draw)

            done = False
            last_points = 0
            while not done:
                obs = get_observation(pong.state)
                action = act(neural_net, obs, draw)
                done = pong.play_one_pong(action)
                current_points = pong.state.points
                if current_points > max_points:
                    max_points = current_points
                reward = 0
                if current_points > last_points:
                    reward = 1
                elif done:
                    reward = -1
                last_points = current_points
                memory.append((obs, action, reward, get_observation(pong.state), done))

        if max_points > 50:
            print("Reached more than 50 points training is complete!")
            exit()

        # REPLAY
        devalue_rewards(memory)
        #normalize_rewards(memory)

        # use a small batch of training data to train on
        batch = random.sample(memory, min(len(memory),BATCH_SIZE))

        states = np.array([each[0] for each in batch])
        next_states = np.array([each[3] for each in batch])


        # set targets to predicted outputs so that we only affect the action
        # that we took.
        targets = neural_net.forward(states, add_bias=True)
        # Get next state q values from target network
        next_states_q_values = target_net.forward(next_states, add_bias=True)
        # Get next state q values from live network
        next_states_q_values_live_network = neural_net.forward(next_states, add_bias=True)

        for i in range(len(batch)):
            (_, action, reward, _, is_terminal) = batch[i]
            if is_terminal:
                targets[i, action] = reward
            else:
                # get max action based on live network
                selected_action = np.argmax(next_states_q_values_live_network[i])
                # use target network value
                targets[i, action] = reward + DISCOUND_FACTOR * next_states_q_values[i, selected_action]

        # Actually train the live network
        neural_net.train(states, targets, TR_SPEED)

        # update epsilons
        if neural_net.epsilon > EPSILON_MIN:
            neural_net.epsilon *= EPSILON_DECAY
            target_net.epsilon = neural_net.epsilon

        if not draw:
            # only save weights when not drawing
            neural_net.saveWeights()
            target_net.saveWeights()

            total_amount_training_data += len(batch)
            print(" --------------------------------------------------")
            print("One training iteration done and saved!   Number %d" % (train+1))
            print("Epsilon now sits at: %.5f" % neural_net.epsilon)
            print("Max points reached was: %d" % max_points)
            print(" --------------------------------------------------")
            print("")
    if not draw:
        print("")
        print("%d training iterations done!" % (train+1))
        print("Total amount of training data: %d" % total_amount_training_data)

def act(ann, obs, draw):
    """
    Returns a random action when epsilon if high.
    Otherwise return the action that will maximize the predicted reward.
    """
    if np.random.rand() <= ann.epsilon and not draw:
        return np.random.randint(0, 3)
    pred = ann.predict(obs)
    return np.argmax(pred)

def get_observation(state):
    """ Return the vector representation of a pong state
        which is the balls position direction and the pads position.
    """
    obs = np.array([state._position[0], state._position[1],
        state._direction[0],
        state._direction[1],
        state._pad])
    return obs

def get_movement_from_action(action):
    """ Convert action into pad movement """
    if action == 0:
        return Movement.PAD_L
    if action == 1:
        return Movement.PAD_STILL
    if action == 2:
        return Movement.PAD_R

def normal_play():
    pong = PlayPong(True, True)
    done = False
    while not done:
        done = pong.play_one_pong()
    print(" GAME OVER!!\nYou got %d points" % pong.state.points)

def devalue_rewards(batch):
    val = 0
    for i in range(0, len(batch)):
        idx = len(batch)-1-i
        if (not batch[idx][2] == 0):
            val = batch[idx][2]*DISCOUND_FACTOR
        else:
            batch[idx] = (batch[idx][0],batch[idx][1], val, batch[idx][3], batch[idx][4])
        val = val*DISCOUND_FACTOR
        if abs(val) < 0.001:
            val = 0.0

def normalize_rewards(batch):
    rew = [reward for (_, _, reward, _, _) in batch]
    mean = np.mean(rew)
    std = np.std(rew)
    for idx in range(0, len(batch)):
        batch[idx] = (batch[idx][0],batch[idx][1], (batch[idx][2]-mean)/std, batch[idx][3],  batch[idx][4])

if __name__ == "__main__":
    main()

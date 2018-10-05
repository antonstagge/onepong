#!/usr/bin/python3
# Code written by Anton Stagge 2018

import datetime
import pygame
import sys
import numpy as np
import deep_neural_network
from collections import deque
import random
import matplotlib.pyplot as plt
import graphics

BETA = 0.99
MAX_GAME_ITER = 9999999

EPSILON_MIN = 0.01

def training_iteration(
    SAVE_NAME, OUTER_ITER, NUMBER_OF_PLAYS, MAX_POINTS, NETWORK_SYNC_FREQ,
    N_IN, N_HIDDEN, N_OUT,
    TR_SPEED, DISCOUND_FACTOR, EPSILON_DECAY,
    BATCH_SIZE,
    initialize,
    play_one_iteration,
    get_observation,
    get_reward):
    """
    The ai uses Deep Q-learning with double Q-nets.
    One for net is used to make prediction and decide on actions,
    the other is used to get the values for the Q-update.

    The training works iteratively
    First you gather a lot if data by simply playing the game,
    then you pick BATCH_SIZE of this data to use for training.

    SAVE_NAME: the file name of the saved weights
    OUTER_ITER: the amount of time to perform the outer training loop
    NUMBER_OF_PLAYS: the amount of time to play the game in each outer training loop
    MAX_POINTS: when the AI reaches MAX_POINTS it consider itself a master
    NETWORK_SYNC_FREQ: how often to sync target and live network

    N_IN: how many input nodes
    N_HIDDEN: how many hidden nodes
    N_OUT: how many output nodes

    TR_SPEED: how much to take into account updates for weights (use 0.001)
    DISCOUND_FACTOR: how much to discout rewards and next q values (use 0.95)
    EPSILON_DECAY: how much to lower epsilon by (use 0.995)

    BATCH_SIZE: the size of the randomly sampled batch actually used for training.
    initialize: constructor to init and return the game
    play_one_iteration: function to move game forward using parameter action
    get_observation: function to return a input vecor of len N_IN
    get_reward: function to return the reward for a state.
    """
    np.seterr(divide='raise', over='raise', under='warn', invalid='raise')
    np.set_printoptions(threshold=np.nan)

    total_amount_training_data = 0 # used for printing

    for train in range(0, OUTER_ITER):
        # Initialising networks
        neural_net = deep_neural_network.network(N_IN, N_HIDDEN, N_OUT, True, beta=BETA, saveName = SAVE_NAME)
        target_net = deep_neural_network.network(N_IN, N_HIDDEN, N_OUT, True, beta=BETA, saveName = (SAVE_NAME + "_target"))

        # Sometime sync target_network
        if train % NETWORK_SYNC_FREQ == 0:
            target_net.set_weights(neural_net.get_weights())

        # randomly swap the target and live networks
        if np.random.uniform() < 0.5:
            temp = neural_net
            neural_net = target_net
            target_net = temp

        # GATHER TRAINING DATA
        memory = deque(maxlen = 2000)
        max_reached = 0
        for t in range(0, NUMBER_OF_PLAYS):
            # Initialize game
            game = initialize()

            done = False
            game_iters = 0
            accumulated_reward = 0
            while (not done and game_iters < MAX_GAME_ITER):
                obs = get_observation(game)
                action = act(neural_net, obs)
                done = play_one_iteration(game, action)
                reward = get_reward(game)
                memory.append((obs, action, reward, get_observation(game), done))
                accumulated_reward += reward
                game_iters += 1

            if accumulated_reward > max_reached:
                max_reached = accumulated_reward

            # Stop training when AI reaches MAX_POINTS in a game
            if max_reached > MAX_POINTS:
                print("Reached more than 50 points training is complete!")
                exit()

        # REPLAY
        devalue_rewards(memory, DISCOUND_FACTOR)

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

        # Save weights
        neural_net.saveWeights()
        target_net.saveWeights()

        total_amount_training_data += len(batch)
        print(" --------------------------------------------------")
        print("One training iteration done and saved!   Number %d" % (train+1))
        print("Epsilon now sits at: %.5f" % neural_net.epsilon)
        print("Max points reached was: %d" % max_reached)
        print(" --------------------------------------------------")
        print("")

    print("")
    print("%d training iterations done!" % (train+1))
    print("Total amount of training data: %d" % total_amount_training_data)

def act(ann, obs, training = True):
    """
    Returns a random action when epsilon is high and training.
    Otherwise return the action that will maximize the predicted reward.
    """
    if np.random.rand() <= ann.epsilon and training:
        return np.random.randint(0, 3)
    pred = ann.predict(obs)
    return np.argmax(pred)

def devalue_rewards(batch, DISCOUND_FACTOR):
    """
    Most often no rewards are given for moves, so we propagate the rewards
    upwards by the DISCOUND_FACTOR (d)
    ex. [0, 0, 0, 0, 0, R]
     => [R*d^5, R*d^4, R*d^3, R*d^2, R*d, R]
    """
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

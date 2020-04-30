# All code written by Anton Stagge 2018
# Feel free to use/edit it to what ever extent but please
# keep the original authors name.

import datetime
import pygame
import sys
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from timeit import default_timer as timer

MAX_GAME_ITER = 10000


def training_iteration(
        neural_net, target_net,
        OUTER_ITER, NUMBER_OF_PLAYS,
        MAX_POINTS, NETWORK_SYNC_FREQ,
        DISCOUND_FACTOR,
        EPSILON_DECAY, EPSILON_MIN,
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

    OUTER_ITER: the amount of time to perform the outer training loop
    NUMBER_OF_PLAYS: the amount of time to play the game in each outer training loop
    MAX_POINTS: when the AI reaches MAX_POINTS it consider itself a master
    NETWORK_SYNC_FREQ: how often to sync target and live network

    DISCOUND_FACTOR: how much to discout rewards and next q values (use 0.95)
    EPSILON_DECAY: how much to lower epsilon by (use 0.995)

    BATCH_SIZE: the size of the randomly sampled batch actually used for training.
    initialize: constructor to init and return the game
    play_one_iteration: function to move game forward using parameter action
    get_observation: function to return a input vecor of len N_IN
    get_reward: function to return the reward for a state.
    """
    np.seterr(divide='raise', over='raise', under='warn', invalid='raise')
    np.set_printoptions(threshold=None)

    total_amount_training_data = 0  # used for printing

    memories = [deque(maxlen=10000)
                for i in range(NUMBER_OF_PLAYS)]

    for epoch in range(OUTER_ITER):
        start = timer()
        # Sometime sync target_network
        if epoch % NETWORK_SYNC_FREQ == 0:
            target_net.set_weights(neural_net.get_weights())

        # randomly swap the target and live networks
        if np.random.uniform() < 0.5:
            temp = neural_net
            neural_net = target_net
            target_net = temp
            # maybe change is_target

        # GATHER TRAINING DATA

        games = [initialize() for i in range(NUMBER_OF_PLAYS)]

        iters = 0  # synced
        dones = [False] * NUMBER_OF_PLAYS
        acc_rews = [1] * NUMBER_OF_PLAYS  # 1 to count points not reward

        while not all(dones) and iters < MAX_GAME_ITER:
            s_t = timer()
            observations = np.zeros(
                np.append([NUMBER_OF_PLAYS], neural_net.nin))

            for i in range(NUMBER_OF_PLAYS):
                if not dones[i]:
                    observations[i] = games[i].get_observation()

            if np.random.rand() <= neural_net.epsilon:
                actions = np.random.randint(
                    0, high=neural_net.nout, size=NUMBER_OF_PLAYS)
            else:
                actions = neural_net.forward(observations, add_bias=True)
                actions = np.argmax(actions, axis=1)

            dones_prel = [games[i].play_one_iteration(action=actions[i])
                          if not dones[i] else True
                          for i in range(NUMBER_OF_PLAYS)]

            rewards = [games[i].get_reward()
                       if not dones[i] else 0
                       for i in range(NUMBER_OF_PLAYS)]

            acc_rews = [(acc_rews[i] + rewards[i])
                        if not dones[i] else acc_rews[i]
                        for i in range(NUMBER_OF_PLAYS)]

            for i in range(NUMBER_OF_PLAYS):
                if not dones[i]:
                    memories[i].append(
                        (observations[i], actions[i], rewards[i],
                         games[i].get_observation(), dones_prel[i])
                    )
            iters += 1

            if any([len(m) < BATCH_SIZE for m in memories]):
                dones = dones_prel
                continue

            states_full = np.empty(np.append([0], neural_net.nin))
            targets_full = np.empty((0, neural_net.nout))
            # train on each game seperately
            for i in range(NUMBER_OF_PLAYS):

                if dones[i]:
                    continue

                memory = memories[i]
                # REPLAY
                #replay_rewards = devalue_rewards(memory, DISCOUND_FACTOR*0.8)
                replay_rewards = [m[2] for m in memory]
                # use a small batch of training data to train on
                choice = np.random.choice(len(memory), BATCH_SIZE)
                #batch = random.sample(memory, min(len(memory), BATCH_SIZE))
                batch = [memory[c] for c in choice]
                batch_rewards = [replay_rewards[c] for c in choice]

                states = np.array([each[0] for each in batch])
                next_states = np.array([each[3] for each in batch])

                # set targets to predicted outputs so that we only affect the action
                # that we took.
                # Get next state q values from live network in the same step
                comb = np.append(states, next_states, axis=0)
                comb_targets = neural_net.forward(comb, add_bias=True)
                targets = comb_targets[:states.shape[0]]
                next_states_q_values_live_network = comb_targets[states.shape[0]:]
                # Get next state q values from target network
                next_states_q_values = target_net.forward(
                    next_states, add_bias=True)

                for i in range(len(batch)):
                    (_, action, _, _, is_terminal) = batch[i]
                    reward = batch_rewards[i]
                    if is_terminal:
                        targets[i, action] = reward
                    else:
                        # get max action based on live network
                        selected_action = np.argmax(
                            next_states_q_values_live_network[i])
                        # use target network value
                        targets[i, action] = reward + DISCOUND_FACTOR * \
                            next_states_q_values[i, selected_action]

                states_full = np.append(states_full, states, axis=0)
                targets_full = np.append(targets_full, targets, axis=0)

            # Actually train the live network
            neural_net.train(states_full, targets_full, BATCH_SIZE)
            sys.stdout.write(
                'Game iteration count %d - time: %.3f\r' % (iters, timer() - s_t))
            dones = dones_prel

        if any([len(m) < BATCH_SIZE for m in memories]):
            continue

        # update epsilons
        if neural_net.epsilon > EPSILON_MIN:
            neural_net.epsilon *= EPSILON_DECAY
            target_net.epsilon = neural_net.epsilon

        # Save weights
        neural_net.saveWeights()
        target_net.saveWeights()

        total_amount_training_data += len(batch)
        print(" --------------------------------------------------")
        print("One training iteration done and saved!   Number %d" % (epoch+1))
        print("Epsilon now sits at: %.5f" % neural_net.epsilon)
        print("Max points reached was: %d" % np.max(acc_rews))
        print("Mean points: %.3f" % np.mean(acc_rews))
        print("Time: ", timer() - start)
        print(" --------------------------------------------------")
        print("")

    print("")
    print("%d training iterations done!" % (epoch+1))
    print("Total amount of training data: %d" % total_amount_training_data)


def act(ann, obs, training=True):
    """
    Returns a random action when epsilon is high and training.
    Otherwise return the action that will maximize the predicted reward.
    """
    if np.random.rand() <= ann.epsilon and training:
        return np.random.randint(0, ann.nout)
    pred = ann.predict(obs)
    return np.argmax(pred)


def devalue_rewards(batch, DISCOUND_FACTOR):
    """
    Most often no rewards are given for moves, so we propagate the rewards
    upwards by the DISCOUND_FACTOR (d)
    ex. [0, 0, 0, 0, 0, R]
     => [R*d^5, R*d^4, R*d^3, R*d^2, R*d, R]
    """
    replay_rewards = []
    val = 0
    for i in range(0, len(batch)):
        idx = len(batch)-1-i
        if (not batch[idx][2] == 0):
            val = batch[idx][2]*DISCOUND_FACTOR
            replay_rewards.insert(0, batch[idx][2])
        else:
            replay_rewards.insert(0, val)
        val = val*DISCOUND_FACTOR
        if abs(val) < 0.001:
            val = 0.0
    return replay_rewards


def normalize_rewards(batch):
    rew = [reward for (_, _, reward, _, _) in batch]
    mean = np.mean(rew)
    std = np.std(rew)
    for idx in range(0, len(batch)):
        batch[idx] = (batch[idx][0], batch[idx][1], (batch[idx]
                                                     [2]-mean)/std, batch[idx][3],  batch[idx][4])

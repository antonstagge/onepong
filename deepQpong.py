#!/usr/bin/python3
import datetime
import pygame
import sys
import numpy as np
from onepong import *
import mlp
from collections import deque
import random

SAVE_NAME = "DOUBLE_DQN"

OUTER_ITER = 1000
NUMBER_OF_PLAYS = 20
TARGET_UPDATE_FREQ = 100

BETA = 1
HIDDEN = 75
TR_SPEED = 0.001
DISCOUND_FACTOR = 0.75

EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

BATCH_SIZE = 32

def main():
    player = True
    draw = True
    if len(sys.argv) > 1:
        if "-ai" in sys.argv:
            player = False
            draw = False
        if "-d" in sys.argv:
            draw = True
        if "-i" in sys.argv:
            sure = input('Are you sure? (y/n): ')
            if sure == 'y':
                set_up_input = np.array([np.zeros(5)])
                set_up_target = np.array([[0,0,0]])
                neural_net = mlp.mlp(set_up_input, set_up_target, HIDDEN, False, beta=BETA, saveName = SAVE_NAME)
                neural_net.saveWeights()
                target_net = mlp.mlp(set_up_input, set_up_target, HIDDEN, False, beta=BETA, saveName = (SAVE_NAME + "_target"))
                target_net.saveWeights()
                return

    if player:
        return normal_play()

    np.seterr(divide='raise', over='raise', under='warn', invalid='raise')
    np.set_printoptions(threshold=np.nan)
    total_amount = 0

    epsilon = 1.0

    for train in range(0, OUTER_ITER):

        set_up_input = np.array([np.zeros(5)])
        set_up_target = np.array([[0,0,0]])

        neural_net = mlp.mlp(set_up_input, set_up_target, HIDDEN, True, beta=BETA, saveName = SAVE_NAME)
        target_net = mlp.mlp(set_up_input, set_up_target, HIDDEN, True, beta=BETA, saveName = (SAVE_NAME + "_target"))

        # Sometime update target_network
        if train % TARGET_UPDATE_FREQ == 0:
            target_net.set_weights(neural_net.get_weights())

        # randomly swap the target and active networks
        if np.random.uniform() < 0.5:
            temp = neural_net
            neural_net = target_net
            target_net = temp

        # GATHER DATA
        memory = deque(maxlen = 2000)
        max_points = 0
        for t in range(0, NUMBER_OF_PLAYS):
            # Initialize game
            pong = PlayPong(player, draw)

            done = False
            last_points = 0
            while not done:
                obs = get_observation(pong.state)
                action = act(neural_net, obs, epsilon)
                done = pong.play_one_pong(get_movement_from_action(action))
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


        # REPLAY
        batch = random.sample(memory, min(len(memory),BATCH_SIZE))

        states = np.array([each[0] for each in batch])
        next_states = np.array([each[3] for each in batch])

        targets = neural_net.mlpfwd(states, add_bias=True)
        next_states_q_values = target_net.mlpfwd(next_states, add_bias=True)
        next_states_q_values_live_network = neural_net.mlpfwd(next_states, add_bias=True)

        for i in range(len(batch)):
            (_, action, reward, _, is_terminal) = batch[i]
            if is_terminal:
                targets[i, action] = reward
            else:
                selected_action = np.argmax(next_states_q_values_live_network[i]) # get max action based on live network
                targets[i, action] = reward + DISCOUND_FACTOR * next_states_q_values[i, selected_action] # use target network value

        neural_net.mlptrain(states, targets, TR_SPEED)

        if neural_net.epsilon > EPSILON_MIN:
            neural_net.epsilon *= EPSILON_DECAY
            target_net.epsilon = neural_net.epsilon

        if not draw:
            neural_net.saveWeights()
            target_net.saveWeights()

        total_amount += len(batch)
        print(" --------------------------------------------------")
        print("One training iteration done and saved!   Number %d" % (train+1))
        print("Epsilon now sits at: %.5f" % neural_net.epsilon)
        print("Max points reached was: %d" % max_points)
        print(" --------------------------------------------------")
        print("")

    print("")
    print("%d training iterations done!" % (train+1))
    print("Total amount of training data: %d" % total_amount)


# Return an action
def act(ann, obs, epsilon):
    if np.random.rand() <= ann.epsilon:
        return np.random.randint(0, 3)
    pred = ann.predict(obs)
    return np.argmax(pred)

def get_observation(state):
    obs = np.array([state._position[0], state._position[1], state._direction[0], state._direction[1], state._pad])
    return obs

def get_movement_from_action(action):
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


if __name__ == "__main__":
    main()

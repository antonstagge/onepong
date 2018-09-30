#!/usr/bin/python3
import datetime
import pygame
import sys
import numpy as np
from onepong import *
import mlp


OUTER_ITER = 5
NUMBER_OF_PLAYS = 20
LOAD = True
SAVE_NAME = "sigmoid"

BETA = 1
HIDDEN = 200
TR_SPEED = 0.001

DISCOUND_FACTOR = 0.95

EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

def main():
    player = True
    draw = True
    if len(sys.argv) > 1:
        if "-ai" in sys.argv:
            player = False
            draw = False
        if "-d" in sys.argv:
            draw = True

    if player:
        return normal_play()

    np.seterr(divide='raise', over='raise', under='raise', invalid='raise')
    total_amount = 0

    for train in range(0, OUTER_ITER):

        set_up_input = np.array([np.zeros(ROWS*COLUMNS)])
        set_up_target = np.array([[0,0,0]])

        neural_net = mlp.mlp(set_up_input, set_up_target, HIDDEN, LOAD, beta=BETA, saveName = SAVE_NAME)

        observations = deque(maxlen=2000)
        actions = deque(maxlen=2000)
        predictions = deque(maxlen=2000)
        rewards = deque(maxlen=2000)

        for t in range(0, NUMBER_OF_PLAYS):
            # Initialize game
            pong = PlayPong(player, draw)

            done = False
            last_frame = np.copy(pong.state._state)
            scn_last_frame = None
            last_points = 0
            while not done:
                if last_frame is not None and scn_last_frame is not None:
                    obs = (last_frame-scn_last_frame).flatten()
                    observations.append(obs)
                    pred = neural_net.predict(obs)
                    #print(pred)
                    predictions.append(pred)
                    action = get_action_from_prediction(pred)
                    actions.append(action)
                    done = pong.play_one_pong(get_movement_from_action(action))

                    current_points = pong.state.points
                    if current_points > last_points:
                        rewards.append(1)
                    elif done:
                        rewards.append(-1)
                    else:
                        rewards.append(0)

                    scn_last_frame = last_frame
                    last_frame = np.copy(pong.state._state)
                    last_points = current_points
                else:
                    done = pong.play_one_pong()
                    scn_last_frame = last_frame
                    last_frame = np.copy(pong.state._state)
                    last_points = pong.state.points


        #devalue_rewards(rewards)
        #normalize_reward(rewards)
        #targets = calcTargets(observations, predictions, actions, rewards)


        # print(np.array(predictions))
        # print(np.array(actions))
        # print(np.array(rewards))
        # print(targets)


        # Save current weight for training
        w1 = np.copy(neural_net.weights1)
        w2 = np.copy(neural_net.weights2)

        neural_net.mlptrain(observations, targets, TR_SPEED, w1, w2)

        neural_net.saveWeights()
        current_amount = len(observations)
        total_amount += current_amount
        print("One training iteration done and saved! %d" % train)
        print("It had batch has size %d" % current_amount)
        print("Completed at: %s" % str(datetime.datetime.now()))

    print("%d training iterations done!" % train)
    print("Total amount of training data: %d" % total_amount)


def devalue_rewards(rewards):
    val = 0
    for i in range(0, len(rewards)):
        idx = len(rewards)-1-i
        if (not rewards[idx] == 0):
            val = rewards[idx]*DISCOUND_FACTOR
        else:
            rewards[idx] = val
        val = val*DISCOUND_FACTOR
        if abs(val) < 0.001:
            val = 0.0

def normalize_reward(rewards):
    mean = np.mean(rewards)
    std = np.std(rewards)
    for i in range(0, len(rewards)):
        rewards[i] = (rewards[i]-mean)/std

def calcTargets(observations, predictions, actions, rewards):
    targets = []
    for i in range(0, len(observations)):
        for j in range(0, 3):
            if predictions[i][j] <= 0.0:
                predictions[i][j] == 0.0000000001
        log = np.log(predictions[i])
        inner = actions[i]*log
        one_move = -rewards[i]*(inner)
        targets.append(one_move)

    return np.array(targets)


def get_action_from_prediction(pred):
    max = -999
    idx = -1
    for i in range(0, 3):
        if pred[i] > max:
            max = pred[i]
            idx = i

    temp = [0, 0, 0]
    temp[idx] = 1
    return temp

def get_movement_from_action(action):
    if action[0]:
        return Movement.PAD_L
    if action[1]:
        return Movement.PAD_STILL
    if action[2]:
        return Movement.PAD_R


def normal_play():
    pong = PlayPong(True, True)
    done = False
    while not done:
        done = pong.play_one_pong()
    print(" GAME OVER!!\nYou got %d points" % pong.state.points)
if __name__ == "__main__":
    main()
    # observation - move - prediction - reward

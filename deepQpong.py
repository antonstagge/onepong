#!/usr/bin/python3
import datetime
import pygame
import sys
import numpy as np
from onepong import *
import mlp
from collections import deque
import random

SAVE_NAME = "LESS_HIDDEN"

OUTER_ITER = 20
NUMBER_OF_PLAYS = 20

BETA = 1
HIDDEN = 75
TR_SPEED = 0.001
DISCOUND_FACTOR = 0.95

EPSILON_MIN = 0.01
EPSILON_DECAY = 0.998

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
            set_up_input = np.array([np.zeros(ROWS*COLUMNS)])
            set_up_target = np.array([[0,0,0]])
            neural_net = mlp.mlp(set_up_input, set_up_target, HIDDEN, False, beta=BETA, saveName = SAVE_NAME)
            neural_net.saveWeights()
            return

    if player:
        return normal_play()

    np.seterr(divide='raise', over='raise', under='raise', invalid='raise')
    np.set_printoptions(threshold=np.nan)
    total_amount = 0

    epsilon = 1.0

    for train in range(0, OUTER_ITER):

        set_up_input = np.array([np.zeros(ROWS*COLUMNS)])
        set_up_target = np.array([[0,0,0]])

        neural_net = mlp.mlp(set_up_input, set_up_target, HIDDEN, True, beta=BETA, saveName = SAVE_NAME)

        # GATHER DATA

        memory = deque(maxlen = 2000)

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
                    action = act(neural_net, obs, epsilon)
                    done = pong.play_one_pong(get_movement_from_action(action))
                    current_points = pong.state.points
                    reward = 0
                    if current_points > last_points:
                        reward = 1
                    elif done:
                        reward = -1

                    scn_last_frame = last_frame
                    last_frame = np.copy(pong.state._state)
                    last_points = current_points

                    memory.append((obs, action, reward, (last_frame-scn_last_frame).flatten()))

                else:
                    done = pong.play_one_pong()
                    scn_last_frame = last_frame
                    last_frame = np.copy(pong.state._state)
                    last_points = pong.state.points


        # REPLAY
        observations = []
        targets = []
        # targets2 = []

        devalue_rewards(memory)
        normalize_rewards(memory)

        batch = random.sample(memory, BATCH_SIZE)

        for (obs, action, reward, next_obs) in batch:
            observations.append(obs)
            #print(np.reshape(obs, (ROWS, COLUMNS)))
            target = reward
            target_f = neural_net.predict(obs)
            # print("what is predicted")
            # print(target_f)
            if not target == -1:
                target = (reward + DISCOUND_FACTOR * np.amax(neural_net.predict(next_obs)))
            target_f[action] = target
            targets.append(target_f)
            # print("what was supposed to be predicted")
            # print(target_f)
            #print(target_f)

            #print(target_f)
            # a = [0,0,0]
            # p = neural_net.predict(obs)
            # print(p)
            # a[action] = 1
            # t = calcTargets([obs],[p],[a],[reward])
            # targets2.append(t[0])

        targets = np.array(targets)

        w1 = np.copy(neural_net.weights1)
        w2 = np.copy(neural_net.weights2)

        neural_net.mlptrain(observations, targets, TR_SPEED, w1, w2)#neural_net.weights1, neural_net.weights2)

        if neural_net.epsilon > EPSILON_MIN:
            neural_net.epsilon *= EPSILON_DECAY

        neural_net.saveWeights()

        total_amount += BATCH_SIZE
        print(" --------------------------------------------------")
        print("One training iteration done and saved!   Number %d" % (train+1))
        print("Epsilon now sits at: %.5f" % neural_net.epsilon)
        print(" --------------------------------------------------")
        print("")

    print("")
    print("%d training iterations done!" % (train+1))
    print("Total amount of training data: %d" % total_amount)


# Return an action
def act(ann, obs, epsilon):
    # if np.random.rand() <= ann.epsilon:
    #     return np.random.randint(0, 3)
    pred = ann.predict(obs)
    return np.argmax(pred)


def devalue_rewards(batch):
    val = 0
    for i in range(0, len(batch)):
        idx = len(batch)-1-i
        if (not batch[idx][2] == 0):
            val = batch[idx][2]*DISCOUND_FACTOR
        else:
            batch[idx] = (batch[idx][0],batch[idx][1], val, batch[idx][3])
        val = val*DISCOUND_FACTOR
        if abs(val) < 0.001:
            val = 0.0

def normalize_rewards(batch):
    rew = [reward for (_, _, reward, _) in batch]
    mean = np.mean(rew)
    std = np.std(rew)
    for idx in range(0, len(batch)):
        batch[idx] = (batch[idx][0],batch[idx][1], (batch[idx][2]-mean)/std, batch[idx][3])

def calcTargets(observations, predictions, actions, rewards):
    targets = []
    for i in range(0, len(observations)):
        log = np.log(predictions[i])
        #print(log)
        inner = actions[i]*log
        #print(inner)
        one_move = -rewards[i]*(inner)
        #print(one_move)
        targets.append(one_move)

    return np.array(targets)


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
    # observation - move - prediction - reward

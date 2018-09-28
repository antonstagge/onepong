#!/usr/bin/python3
import pygame
import sys
import numpy as np
from onepong import *
import mlp



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


    for train in range(0, 3000):

        set_up_input = np.array([np.zeros(ROWS*COLUMNS)])
        set_up_target = np.array([[1,0,0]])

        neural_net = mlp.mlp(set_up_input, set_up_target, 200, False, beta=0.02)

        observations = []
        actions = []
        predictions = []
        rewards = []

        for t in range(0, 30):
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


        devalue_rewards(rewards)
        normalize_reward(rewards)

        targets = []
        for i in range(0, len(observations)):
            log = np.log(predictions[i])
            inner = actions[i]*log
            one_move = -rewards[i]*(inner)
            targets.append(one_move)

        targets = np.array(targets)
        neural_net.mlptrain(observations, targets, 0.002, 10)

        neural_net.saveWeights()
        print("One training iteration done and saved! %d" % train)

    print("%d training iterations done!" % train)


def devalue_rewards(rewards):
    val = 0
    for i in range(0, len(rewards)):
        idx = len(rewards)-1-i
        if (not rewards[idx] == 0):
            val = rewards[idx]/2
        else:
            rewards[idx] = val
        val = val/2
        if val < 0.00001:
            val = 0.0

def normalize_reward(rewards):
    mean = np.mean(rewards)
    std = np.std(rewards)
    for i in range(0, len(rewards)):
        rewards[i] = (rewards[i]-mean)/std


def get_action_from_prediction(pred):
    max = -999.0
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

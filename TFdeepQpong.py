#!/usr/bin/python3
import datetime
import pygame
import sys
import numpy as np
from onepong import *
import mlp
from collections import deque
import random
import tensorflow as tf
import DQNetwork as dqn

SAVE_NAME = "LESS_HIDDEN"

OUTER_ITER = 1
NUMBER_OF_PLAYS = 30
RESTORE = True

BETA = 1
HIDDEN = 75
TR_SPEED = 0.001
DISCOUND_FACTOR = 0.95

EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
EPSILON = 1.0

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
            return

    if player:
        return normal_play()

    np.seterr(divide='raise', over='raise', under='raise', invalid='raise')
    np.set_printoptions(threshold=np.nan)
    total_amount = 0

    epsilon = 1.0


    for train in range(0, OUTER_ITER):
        with tf.Session() as sess:
            set_up_input = np.array([np.zeros(ROWS*COLUMNS)])
            set_up_target = np.array([[0,0,0]])

            # Instantiate the DQNetwork
            DQNetwork = dqn.DQNetwork(ROWS*COLUMNS, 3, TR_SPEED, BATCH_SIZE, epsilon)
            saver = tf.train.Saver()
            if RESTORE:
                print("restoring")
                saver.restore(sess, '/tmp/model.ckpt')
            else:
                sess.run(tf.global_variables_initializer())

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
                        action = predict_action(DQNetwork, sess, obs)
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

                        memory.append((obs, action, reward, (last_frame-scn_last_frame).flatten(), done))

                    else:
                        done = pong.play_one_pong()
                        scn_last_frame = last_frame
                        last_frame = np.copy(pong.state._state)
                        last_points = pong.state.points


            # REPLAY
            devalue_rewards(memory)
            normalize_rewards(memory)

            batch = random.sample(memory, BATCH_SIZE)
            states_mb = np.array([each[0] for each in batch])
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            next_states_mb = np.array([each[3] for each in batch])
            done_md = np.array([each[4] for each in batch])

            temp = {DQNetwork.inputs_: states_mb,
                       DQNetwork.actions_: actions_mb,
                       DQNetwork.rewards_: rewards_mb}
            loss = sess.run(DQNetwork.train_op,
                                    feed_dict=temp)
            if DQNetwork.epsilon > EPSILON_MIN:
                DQNetwork.epsilon *= EPSILON_DECAY
            if epsilon > EPSILON_MIN:
                epsilon *= EPSILON_DECAY


            saver.save(sess, '/tmp/model.ckpt')
            total_amount += BATCH_SIZE
            print(" --------------------------------------------------")
            print("One training iteration done and saved!   Number %d" % (train+1))
            print("Epsilon now sits at: %.5f" % DQNetwork.epsilon)
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


def devalue_rewards(batch):
    val = 0
    for i in range(0, len(batch)):
        idx = len(batch)-1-i
        if (not batch[idx][2] == 0):
            val = batch[idx][2]*DISCOUND_FACTOR
        else:
            batch[idx] = (batch[idx][0],batch[idx][1], val, batch[idx][3],batch[idx][4])
        val = val*DISCOUND_FACTOR
        if abs(val) < 0.001:
            val = 0.0

def normalize_rewards(batch):
    rew = [reward for (_, _, reward, _, _) in batch]
    mean = np.mean(rew)
    std = np.std(rew)
    for idx in range(0, len(batch)):
        batch[idx] = (batch[idx][0],batch[idx][1], (batch[idx][2]-mean)/std, batch[idx][3],batch[idx][4])

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


def predict_action(DQNetwork, sess, obs):
    if np.random.rand() <= DQNetwork.epsilon:
        return np.random.randint(0, 3)

    temp = []
    for i in range(0, BATCH_SIZE):
        temp.append(obs)
    pred = sess.run(DQNetwork.sample_op, feed_dict={DQNetwork.inputs_: temp})
    #print(pred[0])
    return pred[0]

if __name__ == "__main__":
    main()
    # observation - move - prediction - reward

import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices

from collections import deque # Ordered collection with ends

import random


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, BATCH_SIZE, eps, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = eps

        # We create the placeholders
        # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
        # [None, 84, 84, 4]

        self.inputs_ = tf.placeholder(shape=(BATCH_SIZE, self.state_size), dtype=tf.float32)
        self.actions_ = tf.placeholder(shape=[BATCH_SIZE], dtype=tf.int32)
        self.rewards_ = tf.placeholder(shape=[BATCH_SIZE], dtype=tf.float32) # +1 -1 with discounts
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self.Y = tf.layers.dense(self.inputs_, 75, use_bias=False, activation=tf.nn.relu, name="Y")
            self.Ylogits = tf.layers.dense(self.Y, self.action_size, use_bias=False, name="Ylogits")

            self.cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.Ylogits,
                    labels=self.actions_
                )

            self.loss = tf.reduce_sum(self.rewards_ * self.cross_entropies)

            self.logits_for_sampling = tf.reshape(self.Ylogits, shape=(BATCH_SIZE, 3))

            # Sample the action to be played during rollout.
            self.sample_op = tf.squeeze(tf.multinomial(logits=self.logits_for_sampling, num_samples=1))

            #self.cross_entropies = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(self.actions_, 3), logits=self.Ylogits)

            #self.loss = tf.reduce_sum(self.rewards_ * self.cross_entropies)
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            #self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.99)
            self.train_op = self.optimizer.minimize(self.loss)

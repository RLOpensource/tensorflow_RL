import tensorflow as tf
import collections
import random
import numpy as np

class QRDQN:
    def __init__(self, sess, output_size, mainNet, targetNet, batch_size, max_length=1000000):
        self.memory = collections.deque(maxlen=max_length)
        self.lr = 0.00005
        self.output_size = output_size
        self.sess = sess
        self.batch_size = batch_size
        self.gamma = 0.99
        self.mainNet = mainNet
        self.targetNet = targetNet
        self.num_support = self.mainNet.num_support

        self.main_network = self.mainNet.net
        self.main_action_support = self.mainNet.net
        self.main_params = self.mainNet.get_trainable_variables()

        self.target_network = self.targetNet.net
        self.target_action_support = self.targetNet.net
        self.target_params = self.targetNet.get_trainable_variables()

        self.assign_ops = []
        for v_old, v in zip(self.target_params, self.main_params):
            self.assign_ops.append(tf.assign(v_old, v))

        self.action = tf.placeholder(tf.float32, [None, self.output_size])
        self.Y = tf.placeholder(tf.float32, [None, self.num_support])

        self.theta_s_a = self.main_network
        expand_dim_action = tf.expand_dims(self.action, -1)
        theta_s_a = tf.reduce_sum(self.main_network * expand_dim_action, axis=1)

        theta_loss_tile = tf.tile(tf.expand_dims(theta_s_a, axis=2), [1, 1, self.num_support])
        logit_valid_tile = tf.tile(tf.expand_dims(self.Y, axis=1), [1, self.num_support, 1])

        Huber_loss = tf.losses.huber_loss(logit_valid_tile, theta_loss_tile, reduction=tf.losses.Reduction.NONE)
        tau = tf.reshape(tf.range(1e-10, 1, 1 / self.num_support), [1, self.num_support])
        inv_tau = 1.0 - tau

        tau = tf.tile(tf.expand_dims(tau, axis=1), [1, self.num_support, 1])
        inv_tau = tf.tile(tf.expand_dims(inv_tau, axis=1), [1, self.num_support, 1])

        error_loss = logit_valid_tile - theta_loss_tile
        Loss = tf.where(tf.less(error_loss, 0.0), inv_tau * Huber_loss, tau * Huber_loss)
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(Loss, axis=2), axis=1))

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-2/self.batch_size).minimize(self.loss)


    def train_model(self):
        minibatch = random.sample(self.memory, self.batch_size)
        state_stack = [mini[0] for mini in minibatch]
        next_state_stack = [mini[1] for mini in minibatch]
        action_stack = [mini[2] for mini in minibatch]
        reward_stack = [mini[3] for mini in minibatch]
        done_stack = [mini[4] for mini in minibatch]
        done_stack = [int(i) for i in done_stack]
        onehotaction = np.zeros([self.batch_size, self.output_size])
        for i, j in zip(onehotaction, action_stack):
            i[j] = 1
        action_stack = np.stack(onehotaction)

        Q_next_state = self.sess.run(self.target_network, feed_dict={self.targetNet.input: next_state_stack})
        next_action = np.argmax(np.mean(Q_next_state, axis=2), axis=1)
        Q_next_state_next_action = [Q_next_state[i, action, :] for i, action in enumerate(next_action)]
        Q_next_state_next_action = np.sort(Q_next_state_next_action)
        T_theta = [np.ones(self.num_support) * reward if done else reward + self.gamma * Q for reward, Q, done in
                    zip(reward_stack, Q_next_state_next_action, done_stack)]
        _, l = self.sess.run([self.train_op, self.loss],
                                feed_dict={self.mainNet.input: state_stack, self.action: action_stack, self.Y: T_theta})
        return l

    def get_action(self, state):
        Q = self.sess.run(self.main_network, feed_dict={self.mainNet.input: state})
        Q_s_a = np.mean(Q, axis=2)
        action = np.argmax(Q_s_a, axis=1)
        return action

    def update_target(self):
        self.sess.run(self.assign_ops)

    def append(self, state, next_state, action_one_hot, reward, done):
        self.memory.append([state, next_state, action_one_hot, reward, done])

import gym
import tensorflow as tf
import numpy as np
import random
import collections

class MLP_Distributional_RL:
    def __init__(self, sess, model, learning_rate, state_size, action_size, num_support, max_length=1000000):
        self.memory = collections.deque(maxlen=max_length)
        self.learning_rate = learning_rate
        self.state_size = state_size
        self.action_size = action_size
        self.model = model
        self.sess = sess
        self.batch_size = 32
        self.gamma = 0.99
        self.quantile_embedding_dim = 128

        self.num_support = num_support
        self.V_max = 5
        self.V_min = -5
        self.dz = float(self.V_max - self.V_min) / (self.num_support - 1)
        self.z = [self.V_min + i * self.dz for i in range(self.num_support)]
        
        self.state = tf.placeholder(tf.float32, [None, self.state_size])
        self.action = tf.placeholder(tf.float32, [None, self.action_size])
        self.dqn_Y = tf.placeholder(tf.float32, [None, 1])
        self.Y = tf.placeholder(tf.float32, [None, self.num_support])
        self.M = tf.placeholder(tf.float32, [None, self.num_support])
        self.tau = tf.placeholder(tf.float32, [None, self.num_support])

        self.main_network, self.main_action_support, self.main_params = self._build_network('main')
        self.target_network, self.target_action_support, self.target_params = self._build_network('target')

        if self.model == 'IQN':
            expand_dim_action = tf.expand_dims(self.action, -1)
            main_support = tf.reduce_sum(self.main_network * expand_dim_action, axis=1)

            theta_loss_tile = tf.tile(tf.expand_dims(main_support, axis=2), [1, 1, self.num_support])
            logit_valid_tile = tf.tile(tf.expand_dims(self.Y, axis=1), [1, self.num_support, 1])
            Huber_loss = tf.losses.huber_loss(logit_valid_tile, theta_loss_tile, reduction=tf.losses.Reduction.NONE)
            tau = self.tau
            inv_tau = 1 - tau
            tau = tf.tile(tf.expand_dims(tau, axis=1), [1, self.num_support, 1])
            inv_tau = tf.tile(tf.expand_dims(inv_tau, axis=1), [1, self.num_support, 1])
            error_loss = logit_valid_tile - theta_loss_tile

            Loss = tf.where(tf.less(error_loss, 0.0), inv_tau * Huber_loss, tau * Huber_loss)
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(Loss, axis=2), axis=1))

            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        elif self.model == 'DQN':
            self.Q_s_a = self.main_network * self.action
            self.Q_s_a = tf.expand_dims(tf.reduce_sum(self.Q_s_a, axis=1), -1)
            self.loss = tf.losses.mean_squared_error(self.dqn_Y, self.Q_s_a)
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        elif self.model == 'QRDQN':
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

            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.assign_ops = []
        for v_old, v in zip(self.target_params, self.main_params):
            self.assign_ops.append(tf.assign(v_old, v))

    def update_target(self):
        self.sess.run(self.assign_ops)

    def append(self, state, next_state, action_one_hot, reward, done):
        self.memory.append([state, next_state, action_one_hot, reward, done])

    def train(self):
        minibatch = random.sample(self.memory, self.batch_size)
        state_stack = [mini[0] for mini in minibatch]
        next_state_stack = [mini[1] for mini in minibatch]
        action_stack = [mini[2] for mini in minibatch]
        reward_stack = [mini[3] for mini in minibatch]
        done_stack = [mini[4] for mini in minibatch]
        done_stack = [int(i) for i in done_stack]

        if self.model == 'IQN':
            t = np.random.rand(self.batch_size, self.num_support)
            Q_next_state = self.sess.run(self.target_network, feed_dict={self.state: next_state_stack, self.tau: t})
            next_action = np.argmax(np.mean(Q_next_state, axis=2), axis=1)
            Q_next_state_next_action = [Q_next_state[i, action, :] for i, action in enumerate(next_action)]
            T_theta = [reward + (1-done)*self.gamma*Q for reward, Q, done in zip(reward_stack, Q_next_state_next_action, done_stack)]
            return self.sess.run([self.train_op, self.loss], feed_dict={self.state: state_stack, self.action: action_stack, self.tau:t, self.Y: T_theta})


        elif self.model == 'DQN':
            Q_next_state = self.sess.run(self.target_network, feed_dict={self.state: next_state_stack})
            next_action = np.argmax(Q_next_state, axis=1)
            Q_next_state_next_action = [s[a] for s, a in zip(Q_next_state, next_action)]
            T_theta = [[reward + (1-done)*self.gamma * Q] for reward, Q, done in zip(reward_stack, Q_next_state_next_action, done_stack)]
            return self.sess.run([self.train_op, self.loss],
                                 feed_dict={self.state: state_stack, self.action: action_stack, self.dqn_Y: T_theta})

        elif self.model == 'QRDQN':
            Q_next_state = self.sess.run(self.target_network, feed_dict={self.state: next_state_stack})
            next_action = np.argmax(np.mean(Q_next_state, axis=2), axis=1)
            Q_next_state_next_action = [Q_next_state[i, action, :] for i, action in enumerate(next_action)]
            Q_next_state_next_action = np.sort(Q_next_state_next_action)
            T_theta = [np.ones(self.num_support) * reward if done else reward + self.gamma * Q for reward, Q, done in
                       zip(reward_stack, Q_next_state_next_action, done_stack)]
            return self.sess.run([self.train_op, self.loss],
                                 feed_dict={self.state: state_stack, self.action: action_stack, self.Y: T_theta})



    def _build_network(self, name):
        with tf.variable_scope(name):
            if self.model == 'DQN':
                layer_1 = tf.layers.dense(inputs=self.state, units=64, activation=tf.nn.relu, trainable=True)
                layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.nn.relu, trainable=True)
                layer_3 = tf.layers.dense(inputs=layer_2, units=64, activation=tf.nn.relu,
                                          trainable=True)
                net = tf.layers.dense(inputs=layer_3, units=self.action_size, activation=None)
                net_action = net

            elif self.model == 'QRDQN':
                layer_1 = tf.layers.dense(inputs=self.state, units=64, activation=tf.nn.relu, trainable=True)
                layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.nn.relu, trainable=True)
                layer_3 = tf.layers.dense(inputs=layer_2, units=64, activation=tf.nn.relu,
                                          trainable=True)
                layer_4 = tf.layers.dense(inputs=layer_3, units=self.action_size * self.num_support, activation=None,
                                          trainable=True)
                net = tf.reshape(layer_4, [-1, self.action_size, self.num_support])
                net_action = net

            elif self.model == 'IQN':
                state_tile = tf.tile(self.state, [1, self.num_support])
                state_reshape = tf.reshape(state_tile, [-1, self.state_size])
                state_net = tf.layers.dense(inputs=state_reshape, units=self.quantile_embedding_dim, activation=tf.nn.selu)

                tau = tf.reshape(self.tau, [-1, 1])
                pi_mtx = tf.constant(np.expand_dims(np.pi * np.arange(0, 64), axis=0), dtype=tf.float32)
                cos_tau = tf.cos(tf.matmul(tau, pi_mtx))
                phi = tf.layers.dense(inputs=cos_tau, units=self.quantile_embedding_dim, activation=tf.nn.relu)

                net = tf.multiply(state_net, phi)
                net = tf.layers.dense(inputs=net, units=512, activation=tf.nn.relu)
                net = tf.layers.dense(inputs=net, units=128, activation=tf.nn.relu)
                net = tf.layers.dense(inputs=net, units=self.action_size, activation=None)

                net_action = tf.transpose(tf.split(net, 1, axis=0), perm=[0, 2, 1])

                net = tf.transpose(tf.split(net, self.batch_size, axis=0), perm=[0, 2, 1])

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        return net, net_action, params


    def choose_action(self, state):
        if self.model == 'DQN':
            result = self.sess.run(self.main_network, feed_dict={self.state: [state]})[0]
            action = np.argmax(result)

        elif self.model == 'QRDQN':
            Q = self.sess.run(self.main_network, feed_dict={self.state: [state]})
            Q_s_a = np.mean(Q[0], axis=1)
            action = np.argmax(Q_s_a)

        elif self.model == 'IQN':
            t = np.random.rand(1, self.num_support)
            Q_s_a = self.sess.run(self.main_action_support, feed_dict={self.state: [state], self.tau: t})
            Q_s_a = Q_s_a[0]
            Q_a = np.sum(Q_s_a, axis=1)
            action = np.argmax(Q_a)
        return action
       
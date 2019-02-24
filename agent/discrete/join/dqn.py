import tensorflow as tf
import collections
import random
import numpy as np

class DQN:
    def __init__(self, sess, output_size, mainNet, targetNet, max_length=1000000):
        self.memory = collections.deque(maxlen=max_length)
        self.lr = 0.00025
        self.output_size = output_size
        self.mainNet = mainNet
        self.targetNet = targetNet
        self.batch_size = 8
        self.gamma = 0.99
        self.memory = collections.deque(maxlen=max_length)
        self.sess = sess

        self.main_network = self.mainNet.Q
        self.target_network = self.targetNet.Q
        self.target_params = self.targetNet.get_trainable_variables()
        self.main_params = self.mainNet.get_trainable_variables()

        self.assign_ops = []
        for v_old, v in zip(self.target_params, self.main_params):
            self.assign_ops.append(tf.assign(v_old, v))

        self.action = tf.placeholder(tf.float32, [None, self.output_size])
        self.Y = tf.placeholder(tf.float32, [None, 1])

        self.Q_s_a = self.main_network * self.action
        self.Q_s_a = tf.expand_dims(tf.reduce_sum(self.Q_s_a, axis=1), -1)
        self.loss = tf.losses.mean_squared_error(self.Y, self.Q_s_a)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self):
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
        next_action = np.argmax(Q_next_state, axis=1)
        Q_next_state_next_action = [s[a] for s, a in zip(Q_next_state, next_action)]
        T_theta = [[reward + (1-done)*self.gamma * Q] for reward, Q, done in zip(reward_stack, Q_next_state_next_action, done_stack)]
        
        return self.sess.run([self.train_op, self.loss],
                    feed_dict={self.mainNet.input: state_stack, self.action: action_stack, self.Y: T_theta})


    def update_target(self):
        self.sess.run(self.assign_ops)

    def append(self, state, next_state, action_one_hot, reward, done):
        self.memory.append([state, next_state, action_one_hot, reward, done])

    def get_action(self, state):
        Q = self.sess.run(self.main_network, feed_dict={self.mainNet.input: state})
        action = np.argmax(Q, axis=1)
        return action
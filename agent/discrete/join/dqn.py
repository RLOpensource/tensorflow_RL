import tensorflow as tf
import collections
import random
import numpy as np

class DQN:
    def __init__(self, sess, output_size, mainNet, targetNet, max_length=1000000):
        self.memory = collections.deque(maxlen=max_length)
        self.mainNet = mainNet
        self.targetNet = targetNet
        self.output_size = output_size
        self.batch_size = 8
        self.sess = sess
        self.gamma = 0.99
        self.lr = 0.00025

        self.target = tf.placeholder(tf.float32, [None])
        self.action = tf.placeholder(tf.int32, [None])

        self.target_vars = self.targetNet.get_trainable_variables()
        self.main_vars = self.mainNet.get_trainable_variables()

        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(self.main_vars, self.target_vars)]

        self.action_one_hot = tf.one_hot(self.action, self.output_size)
        self.q_val = self.mainNet.Q
        self.q_val_action = tf.reduce_sum(self.q_val * self.action_one_hot, axis=1)

        self.loss = tf.reduce_mean((self.target - self.q_val_action) ** 2)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-2).minimize(self.loss)

    def update_target(self):
        self.sess.run(self.update_oldpi_op)

    def append(self, state, next_state, action, reward, done):
        self.memory.append([state, next_state, action, reward, done])

    def get_action(self, state):
        q_val = self.sess.run(self.q_val, feed_dict={self.mainNet.input: state})
        maxq = np.argmax(q_val, axis=1)
        return maxq

    def train_model(self):
        minibatch = random.sample(self.memory, self.batch_size)
        state = [mini[0] for mini in minibatch]
        next_state = [mini[1] for mini in minibatch]
        action = [mini[2] for mini in minibatch]
        reward = [mini[3] for mini in minibatch]
        done = [mini[4] for mini in minibatch]

        nextQ = self.sess.run(self.targetNet.Q, feed_dict={self.targetNet.input: next_state})
        max_nextQ = np.max(nextQ, axis=1)
        targets = [r + self.gamma * (1-d) * mQ for r, d, mQ in zip(reward, done, max_nextQ)]
        _, l = self.sess.run([self.train_op, self.loss], feed_dict={self.mainNet.input: state,
                                                               self.target: targets,
                                                               self.action: action})

        return l
import tensorflow as tf
import numpy as np

class MLPIQN:
    def __init__(self, name, state_size, output_size, num_support, batch_size):
        self.state_size = state_size
        self.output_size = output_size
        self.num_support = num_support
        self.quantile_embedding_dim = 128
        self.batch_size = batch_size
        with tf.variable_scope(name):
            self.input = tf.placeholder(tf.float32, shape=[None, self.state_size])
            self.tau = tf.placeholder(tf.float32, shape=[None, self.num_support])

            state_tile = tf.tile(self.input, [1, self.num_support])
            state_reshape = tf.reshape(state_tile, [-1, self.state_size])
            state_net = tf.layers.dense(inputs=state_reshape, units=self.quantile_embedding_dim, activation=tf.nn.selu)

            tau = tf.reshape(self.tau, [-1, 1])
            pi_mtx = tf.constant(np.expand_dims(np.pi * np.arange(0, 64), axis=0), dtype=tf.float32)
            cos_tau = tf.cos(tf.matmul(tau, pi_mtx))
            phi = tf.layers.dense(inputs=cos_tau, units=self.quantile_embedding_dim, activation=tf.nn.relu)

            net = tf.multiply(state_net, phi)
            net = tf.layers.dense(inputs=net, units=512, activation=tf.nn.relu)
            net = tf.layers.dense(inputs=net, units=128, activation=tf.nn.relu)
            net = tf.layers.dense(inputs=net, units=self.output_size, activation=None)

            self.net_action = tf.transpose(tf.split(net, 1, axis=0), perm=[0, 2, 1])

            self.net = tf.transpose(tf.split(net, self.batch_size, axis=0), perm=[0, 2, 1])

            self.scope = tf.get_variable_scope().name
        
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class MLPQRDQN:
    def __init__(self, name, state_size, output_size, num_support):
        self.state_size = state_size
        self.output_size = output_size
        self.num_support = num_support
        with tf.variable_scope(name):
            self.input = tf.placeholder(tf.float32, shape=[None, self.state_size])
            self.l1 = tf.layers.dense(inputs=self.input, units=256, activation=tf.nn.relu)
            self.l2 = tf.layers.dense(inputs=self.l1, units=256, activation=tf.nn.relu)
            self.l3 = tf.layers.dense(inputs=self.l2, units=self.output_size * self.num_support, activation=None)
            self.net = tf.reshape(self.l3, [-1, self.output_size, self.num_support])

            self.scope = tf.get_variable_scope().name
        
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class MLPDQN:
    def __init__(self, name, state_size, output_size):
        self.state_size = state_size
        self.output_size = output_size
        with tf.variable_scope(name):
            self.input = tf.placeholder(tf.float32, shape=[None, self.state_size])
            self.l1 = tf.layers.dense(inputs=self.input, units=256, activation=tf.nn.relu)
            self.l2 = tf.layers.dense(inputs=self.l1, units=256, activation=tf.nn.relu)
            self.Q = tf.layers.dense(inputs=self.l2, units=self.output_size, activation=None)

            self.scope = tf.get_variable_scope().name

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
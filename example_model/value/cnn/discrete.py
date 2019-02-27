import tensorflow as tf
import numpy as np

class CNNIQN:
    def __init__(self, name, window_size, obs_stack, output_size, num_support, batch_size):
        self.window_size = window_size
        self.obs_stack = obs_stack
        self.output_size = output_size
        self.quantile_embedding_dim = 256
        self.batch_size = batch_size
        self.num_support = num_support
        with tf.variable_scope(name):
            self.input = tf.placeholder(tf.float32, shape=[None, self.window_size, self.window_size, self.obs_stack])
            self.tau = tf.placeholder(tf.float32, shape=[None, self.num_support])

            state_expand_dims = tf.expand_dims(self.input, 1)
            state_tile = tf.tile(state_expand_dims, [1, self.num_support, 1, 1, 1])
            state_reshape = tf.reshape(state_tile, [-1, self.window_size, self.window_size, self.obs_stack])
            self.conv1 = tf.layers.conv2d(inputs=state_reshape, filters=32, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu)
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu)
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=128, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=tf.nn.relu)
            self.reshape = tf.reshape(self.conv3, [-1, 7 * 7 * 128])
            reshape = tf.layers.dense(inputs=self.reshape, units=1024, activation=tf.nn.relu)
            reshape = tf.layers.dense(inputs=reshape, units=512, activation=tf.nn.relu)
            self.state_net = tf.layers.dense(inputs=reshape, units=self.quantile_embedding_dim, activation=tf.nn.relu)

            tau = tf.reshape(self.tau, [-1, 1])
            pi_mtx = tf.constant(np.expand_dims(np.pi * np.arange(0, 64), axis=0), dtype=tf.float32)
            cos_tau = tf.cos(tf.matmul(tau, pi_mtx))
            phi = tf.layers.dense(inputs=cos_tau, units=self.quantile_embedding_dim, activation=tf.nn.relu)

            net = tf.multiply(self.state_net, phi)
            net = tf.layers.dense(inputs=net, units=128, activation=tf.nn.relu)
            net = tf.layers.dense(inputs=net, units=self.output_size, activation=None)

            self.net_action = tf.transpose(tf.split(net, 1, axis=0), perm=[0, 2, 1])

            self.net = tf.transpose(tf.split(net, self.batch_size, axis=0), perm=[0, 2, 1])

            self.scope = tf.get_variable_scope().name

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)



class CNNQRDQN:
    def __init__(self, name, window_size, obs_stack, output_size, num_support):
        self.window_size = window_size
        self.obs_stack = obs_stack
        self.output_size = output_size
        self.num_support = num_support
        with tf.variable_scope(name):
            self.input = tf.placeholder(tf.float32, shape=[None, self.window_size, self.window_size, self.obs_stack])
            self.conv1 = tf.layers.conv2d(inputs=self.input, filters=32, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu)
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu)
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=tf.nn.relu)
            self.reshape = tf.reshape(self.conv3, [-1, 7 * 7 * 64])
            self.dense_1 = tf.layers.dense(inputs=self.reshape, units=64 * 7 * 7, activation=tf.nn.relu)
            self.dense_2 = tf.layers.dense(inputs=self.dense_1, units=self.output_size * self.num_support, activation=None)
            self.net = tf.reshape(self.dense_2, [-1, self.output_size, self.num_support])

            self.scope = tf.get_variable_scope().name
        
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class CNNDQN:
    def __init__(self, name, window_size, obs_stack, output_size):
        self.window_size = window_size
        self.obs_stack = obs_stack
        self.output_size = output_size
        with tf.variable_scope(name):
            self.input = tf.placeholder(tf.float32, shape=[None, self.window_size, self.window_size, self.obs_stack])
            self.conv1 = tf.layers.conv2d(inputs=self.input, filters=32, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu)
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu)
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=tf.nn.relu)
            self.reshape = tf.reshape(self.conv3, [-1, 7 * 7 * 64])
            self.dense_3 = tf.layers.dense(inputs=self.reshape, units=512, activation=tf.nn.relu)
            self.Q = tf.layers.dense(inputs=self.dense_3, units=self.output_size, activation=None)

            self.scope = tf.get_variable_scope().name

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
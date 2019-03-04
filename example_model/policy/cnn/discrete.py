import tensorflow as tf
import numpy as np

class CNNLSTMActor:
    def __init__(self, name, window_size, obs_stack, output_size, lstm_units, lstm_layers):
        self.window_size = window_size
        self.output_size = output_size
        self.obs_stack = obs_stack
        self.reuse = []
        for i in range(self.obs_stack):
            if i == 0:
                self.reuse.append(False)
            else:
                self.reuse.append(True)

        self.lstm_list = [lstm_units for i in range(lstm_layers)]

        with tf.variable_scope(name):
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.window_size, self.window_size, self.obs_stack])
            self.expand_input = tf.expand_dims(self.input, axis=3)
            self.split = [self.expand_input[:, :, :, :, i] for i in range(self.obs_stack)]
            self.conv1 = [tf.layers.conv2d(inputs=self.split[i], filters=8, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu, name='conv1', reuse=self.reuse[i]) for i in range(self.obs_stack)]
            self.conv2 = [tf.layers.conv2d(inputs=self.conv1[i], filters=16, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu, name='conv2', reuse=self.reuse[i]) for i in range(self.obs_stack)]
            self.conv3 = [tf.layers.conv2d(inputs=self.conv2[i], filters=16, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=tf.nn.relu, name='conv3', reuse=self.reuse[i]) for i in range(self.obs_stack)]
            self.reshape = [tf.reshape(self.conv3[i], [-1, 7 * 7 * 16]) for i in range(self.obs_stack)]
            self.concat = tf.stack(self.reshape, axis=1)
            
            enc_cell = [tf.nn.rnn_cell.GRUCell(size) for size in self.lstm_list]
            enc_cell = tf.nn.rnn_cell.MultiRNNCell(enc_cell)
            self.outputs_enc, enc_states = tf.nn.dynamic_rnn(cell=enc_cell, inputs=self.concat, dtype=tf.float32)
            
            self.last_layer = self.outputs_enc[:, -1]
            self.actor = tf.layers.dense(inputs=self.last_layer, units=self.output_size, activation=tf.nn.softmax)

            self.scope = tf.get_variable_scope().name
    
    def get_action_prob(self, obs):
        return self.sess.run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class CNNLSTMCritic:
    def __init__(self, name, window_size, obs_stack, output_size, lstm_units, lstm_layers):
        self.window_size = window_size
        self.output_size = output_size
        self.obs_stack = obs_stack
        self.reuse = []
        for i in range(self.obs_stack):
            if i == 0:
                self.reuse.append(False)
            else:
                self.reuse.append(True)

        self.lstm_list = [lstm_units for i in range(lstm_layers)]

        with tf.variable_scope(name):
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.window_size, self.window_size, self.obs_stack])
            self.expand_input = tf.expand_dims(self.input, axis=3)
            self.split = [self.expand_input[:, :, :, :, i] for i in range(self.obs_stack)]
            self.conv1 = [tf.layers.conv2d(inputs=self.split[i], filters=8, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu, name='conv1', reuse=self.reuse[i]) for i in range(self.obs_stack)]
            self.conv2 = [tf.layers.conv2d(inputs=self.conv1[i], filters=16, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu, name='conv2', reuse=self.reuse[i]) for i in range(self.obs_stack)]
            self.conv3 = [tf.layers.conv2d(inputs=self.conv2[i], filters=16, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=tf.nn.relu, name='conv3', reuse=self.reuse[i]) for i in range(self.obs_stack)]
            self.reshape = [tf.reshape(self.conv3[i], [-1, 7 * 7 * 16]) for i in range(self.obs_stack)]
            self.concat = tf.stack(self.reshape, axis=1)
            
            enc_cell = [tf.nn.rnn_cell.GRUCell(size) for size in self.lstm_list]
            enc_cell = tf.nn.rnn_cell.MultiRNNCell(enc_cell)
            self.outputs_enc, enc_states = tf.nn.dynamic_rnn(cell=enc_cell, inputs=self.concat, dtype=tf.float32)
            
            self.last_layer = self.outputs_enc[:, -1]
            self.critic = tf.layers.dense(inputs=self.last_layer, units=1, activation=None)


            self.scope = tf.get_variable_scope().name
    
    def get_action_prob(self, obs):
        return self.sess.run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class CNNActor:
    def __init__(self, name, window_size, obs_stack, output_size):
        self.window_size = window_size
        self.output_size = output_size
        self.obs_stack = obs_stack

        with tf.variable_scope(name):
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, window_size, window_size, obs_stack])
            self.conv1 = tf.layers.conv2d(inputs=self.input, filters=32, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu)
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu)
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=tf.nn.relu)

            self.reshape = tf.reshape(self.conv3, [-1, 7 * 7 * 64])
            self.dense_3 = tf.layers.dense(inputs=self.reshape, units=512, activation=tf.nn.relu)
            
            self.actor = tf.layers.dense(inputs=self.dense_3, units=self.output_size, activation=tf.nn.softmax)
            
            self.scope = tf.get_variable_scope().name

    
    def get_action_prob(self, obs):
        return self.sess.run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class CNNCritic:
    def __init__(self, name, window_size, obs_stack):
        self.window_size = window_size
        self.obs_stack = obs_stack

        with tf.variable_scope(name):
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, window_size, window_size, obs_stack])
            self.conv1 = tf.layers.conv2d(inputs=self.input, filters=32, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu)
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu)
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=tf.nn.relu)

            self.reshape = tf.reshape(self.conv3, [-1, 7 * 7 * 64])
            self.dense_3 = tf.layers.dense(inputs=self.reshape, units=512, activation=tf.nn.relu)
            
            self.critic = tf.layers.dense(inputs=self.dense_3, units=1, activation=None)

            self.scope = tf.get_variable_scope().name

    
    def get_action_prob(self, obs):
        return self.sess.run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
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


class CNNActorLSTM:
    def __init__(self, name, window_size, action_size, lstm_units, lstm_layers):
        self.window_size = window_size
        self.action_size = action_size
        self.lstm_units = lstm_units
        self.lstm_layers = lstm_layers
        self.scope = name

        with tf.variable_scope(name):
            self.batch_size = tf.placeholder(tf.int32, shape=[])
            self.input = tf.placeholder(tf.float32, shape=[None, self.window_size, self.window_size, 1])
            self.conv1 = tf.layers.conv2d(inputs=self.input, filters=32, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu)
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu)
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=tf.nn.relu)
            self.flatten = tf.reshape(self.conv3, [-1, 7 * 7 * 64])
            self.l1 = tf.layers.dense(inputs=self.flatten, units=self.lstm_units, activation=tf.nn.relu)
            
            lstm = tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_units)
            lstm = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm] * self.lstm_layers)

            self.init_state = lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            self.lstm_in = tf.expand_dims(self.l1, axis=1)

            self.outputs, self.final_state = tf.nn.dynamic_rnn(cell=lstm, inputs=self.lstm_in, initial_state=self.init_state)
            last_layer = tf.reshape(self.outputs, [-1, self.lstm_units])
            self.actor = tf.layers.dense(inputs=last_layer, units=self.action_size, activation=tf.nn.softmax)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class CNNCriticLSTM:
    def __init__(self, name, window_size, lstm_units, lstm_layers):
        self.window_size = window_size
        self.lstm_units = lstm_units
        self.lstm_layers = lstm_layers
        self.scope = name

        with tf.variable_scope(name):
            self.batch_size = tf.placeholder(tf.int32, shape=[])
            self.input = tf.placeholder(tf.float32, shape=[None, self.window_size, self.window_size, 1])
            self.conv1 = tf.layers.conv2d(inputs=self.input, filters=32, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu)
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu)
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=tf.nn.relu)
            self.flatten = tf.reshape(self.conv3, [-1, 7 * 7 * 64])
            self.l1 = tf.layers.dense(inputs=self.flatten, units=self.lstm_units, activation=tf.nn.relu)
            
            lstm = tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_units)
            lstm = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm] * self.lstm_layers)

            self.init_state = lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            self.lstm_in = tf.expand_dims(self.l1, axis=1)

            self.outputs, self.final_state = tf.nn.dynamic_rnn(cell=lstm, inputs=self.lstm_in, initial_state=self.init_state)
            last_layer = tf.reshape(self.outputs, [-1, self.lstm_units])
            self.critic = tf.layers.dense(inputs=last_layer, units=1, activation=None)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class MLPDDPGContinuousActor:
    def __init__(self,name,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.scope = name
        with tf.variable_scope(name):
            self.state = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size])
            self.l1 = tf.layers.dense(self.state,128,tf.nn.leaky_relu,trainable=True)
            self.l2 = tf.layers.dense(self.l1, 128, tf.nn.leaky_relu, trainable=True)
            self.l3 = tf.layers.dense(self.l2, 128, tf.nn.leaky_relu, trainable=True)
            self.actor = tf.layers.dense(self.l3, self.action_size, tf.tanh, trainable=True)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class MLPDDPGContinuousCritic:
    def __init__(self,name,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.scope = name
        with tf.variable_scope(name):
            self.state = tf.placeholder(dtype=tf.float32,shape=[None,self.state_size])
            self.action = tf.placeholder(dtype=tf.float32,shape=[None,self.action_size])
            self.cat_state_action = tf.concat([self.state,self.action],1)
            self.l1 = tf.layers.dense(self.cat_state_action,128,tf.nn.leaky_relu,trainable=True)
            self.l2 = tf.layers.dense(self.l1, 128,tf.nn.leaky_relu,trainable=True)
            self.l3 = tf.layers.dense(self.l2, 128, tf.nn.leaky_relu, trainable=True)
            self.critic = tf.layers.dense(self.l3,1,None,trainable=True)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class MLPContinuousActor:
    def __init__(self, name, state_size, output_size):
        self.state_size = state_size
        self.output_size = output_size

        with tf.variable_scope(name):
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size])
            self.l1 = tf.layers.dense(self.input, 128, tf.nn.tanh, trainable=True)
            self.l2 = tf.layers.dense(self.l1, 128, tf.nn.tanh, trainable=True)
            self.l3 = tf.layers.dense(self.l2, 128, tf.nn.tanh, trainable=True)
            self.mu = tf.layers.dense(self.l3, self.output_size, tf.nn.tanh, trainable=True)
            self.sigma = tf.ones_like(self.mu)
            #self.sigma = tf.layers.dense(self.l3, self.output_size, tf.nn.softplus, trainable=True)

            self.actor = tf.distributions.Normal(loc=self.mu, scale=self.sigma)
        
            self.scope = tf.get_variable_scope().name

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class MLPContinuousCritic:
    def __init__(self, name, state_size):
        self.state_size = state_size

        with tf.variable_scope(name):
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size])
            self.l1 = tf.layers.dense(self.input, 128, tf.nn.relu, trainable=True)
            self.l2 = tf.layers.dense(self.l1, 128, tf.nn.relu, trainable=True)
            self.l3 = tf.layers.dense(self.l2, 128, tf.nn.relu, trainable=True)
            self.critic = tf.layers.dense(self.l3, 1)

            self.scope = tf.get_variable_scope().name

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class MLPActor:
    def __init__(self, name, state_size, output_size):
        self.state_size = state_size
        self.output_size = output_size

        with tf.variable_scope(name):
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size])
            self.dense_1 = tf.layers.dense(inputs=self.input, units=256, activation=tf.nn.relu)
            self.dense_2 = tf.layers.dense(inputs=self.dense_1, units=256, activation=tf.nn.relu)
            self.dense_3 = tf.layers.dense(inputs=self.dense_2, units=256, activation=tf.nn.relu)
            self.actor = tf.layers.dense(inputs=self.dense_3, units=self.output_size, activation=tf.nn.softmax)

            self.scope = tf.get_variable_scope().name

    def get_action_prob(self, obs):
        return self.sess.run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class MLPCritic:
    def __init__(self, name, state_size):
        self.state_size = state_size

        with tf.variable_scope(name):
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size])
            self.dense_1 = tf.layers.dense(inputs=self.input, units=256, activation=tf.nn.relu)
            self.dense_2 = tf.layers.dense(inputs=self.dense_1, units=256, activation=tf.nn.relu)
            self.dense_3 = tf.layers.dense(inputs=self.dense_2, units=256, activation=tf.nn.relu)
            self.critic = tf.layers.dense(inputs=self.dense_3, units=1, activation=None)

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
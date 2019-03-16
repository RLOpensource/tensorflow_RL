import tensorflow as tf
import numpy as np
from agent.utils import gaussian_likelihood

class MLPContinuousActor:
    def __init__(self, name, state_size, output_size):
        self.state_size = state_size
        self.output_size = output_size

        with tf.variable_scope(name):
            self.input = tf.placeholder(tf.float32, shape=[None, self.state_size])
            self.action = tf.placeholder(tf.float32, shape=[None, self.output_size])

            self.l1 = tf.layers.dense(inputs=self.input, units=128, activation=tf.nn.relu)
            self.l2 = tf.layers.dense(inputs=self.l1,    units=128, activation=tf.nn.relu)
            self.l3 = tf.layers.dense(inputs=self.l2,    units=128, activation=tf.nn.relu)

            self.mu = tf.layers.dense(inputs=self.l3,    units=self.output_size, activation=None)
            self.log_std = tf.get_variable(name='log_std', initializer= -0.5 * np.ones(self.output_size, dtype=np.float32))
            self.std = tf.exp(self.log_std)
            self.pi = self.mu + tf.random_normal(tf.shape(self.mu)) * self.std
            self.logp = gaussian_likelihood(self.action, self.mu, self.log_std)
            self.logp_pi = gaussian_likelihood(self.pi, self.mu, self.log_std)
    
            self.scope = tf.get_variable_scope().name

    def get_action_prob(self, obs):
        return self.sess.run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class MLPContinuousCritic:
    def __init__(self, name, state_size):
        self.state_size = state_size
        
        with tf.variable_scope(name):
            self.input = tf.placeholder(tf.float32, shape=[None, self.state_size])

            self.l1 = tf.layers.dense(inputs=self.input, units=128, activation=tf.nn.relu)
            self.l2 = tf.layers.dense(inputs=self.l1,    units=128, activation=tf.nn.relu)
            self.l3 = tf.layers.dense(inputs=self.l2,    units=128, activation=tf.nn.relu)
            
            self.value = tf.layers.dense(inputs=self.l3, units=1,   activation=None)
            self.v = tf.squeeze(self.value, axis=1)

            self.scope = tf.get_variable_scope().name

    def get_action_prob(self, obs):
        return self.sess.run(self.act_probs, feed_dict={self.obs: obs})

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

class MLPTD3ContinousCritic:
    def __init__(self,name,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.scope = name
        with tf.variable_scope(name):
            self.state = tf.placeholder(dtype=tf.float32,shape=[None,self.state_size])
            self.action = tf.placeholder(dtype=tf.float32,shape=[None,self.action_size])
            self.cat_state_action = tf.concat([self.state,self.action],1)
            net1_l1 = tf.layers.dense(self.cat_state_action,128,tf.nn.leaky_relu,trainable=True)
            net1_l2 = tf.layers.dense(net1_l1, 128, tf.nn.leaky_relu, trainable=True)
            net1_l3 = tf.layers.dense(net1_l2, 128, tf.nn.leaky_relu, trainable=True)
            self.critic1 = tf.layers.dense(net1_l3,1,None,trainable=True)
            net2_l1 = tf.layers.dense(self.cat_state_action,128,tf.nn.leaky_relu,trainable=True)
            net2_l2 = tf.layers.dense(net2_l1, 128, tf.nn.leaky_relu, trainable=True)
            net2_l3 = tf.layers.dense(net2_l2, 128, tf.nn.leaky_relu, trainable=True)
            self.critic2 = tf.layers.dense(net2_l3,1,None,trainable=True)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
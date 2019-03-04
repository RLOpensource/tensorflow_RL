import tensorflow as tf
import numpy as np

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

class MLPContinuousActor:
    def __init__(self, name, state_size, output_size):
        self.state_size = state_size
        self.output_size = output_size

        with tf.variable_scope(name):
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size])
            self.l1 = tf.layers.dense(self.input, 256, tf.nn.relu, trainable=True)
            self.l2 = tf.layers.dense(self.l1, 256, tf.nn.relu, trainable=True)
            self.l3 = tf.layers.dense(self.l2, 256, tf.nn.relu, trainable=True)
            self.mu = tf.layers.dense(self.l3, self.output_size, tf.nn.tanh, trainable=True) * 0.5
            self.sigma = tf.ones_like(self.mu) * 0.3
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
            self.l1 = tf.layers.dense(self.input, 256, tf.nn.relu, trainable=True)
            self.l2 = tf.layers.dense(self.l1, 256, tf.nn.relu, trainable=True)
            self.l3 = tf.layers.dense(self.l2, 256, tf.nn.relu, trainable=True)
            self.critic = tf.layers.dense(self.l3, 1)

            self.scope = tf.get_variable_scope().name

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
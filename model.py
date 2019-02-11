import tensorflow as tf

class CNNActorLSTM:
    def __init__(self, name, window_size, obs_stack, action_size, lstm_shape):
        self.window_size = window_size
        self.action_size = action_size
        self.lstm_shape = lstm_shape
        self.obs_stack = obs_stack
        self.reuse = []
        self.scope = name
        for i in range(self.obs_stack):
            if i == 0:
                self.reuse.append(False)
            else:
                self.reuse.append(True)

        with tf.variable_scope(name):
            self.input = tf.placeholder(tf.float32, shape=[None, self.window_size, self.window_size, self.obs_stack])
            self.input_list = [self.input[:, :, :, i] for i in range(obs_stack)]
            self.input_channel = [tf.expand_dims(self.input_list[i], axis=3) for i in range(obs_stack)]
            self.conv1 = [tf.layers.conv2d(inputs=self.input_channel[i], filters=4, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu, name='conv1', reuse=self.reuse[i]) for i in range(obs_stack)]
            self.conv2 = [tf.layers.conv2d(inputs=self.conv1[i], filters=16, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu, name='conv2', reuse=self.reuse[i]) for i in range(obs_stack)]
            self.conv3 = [tf.layers.conv2d(inputs=self.conv2[i], filters=16, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=tf.nn.relu, name='conv3', reuse=self.reuse[i]) for i in range(obs_stack)]
            self.flatten = [tf.reshape(self.conv3[i], [-1, 7 * 7 * 16]) for i in range(obs_stack)]
            self.l1 = [tf.layers.dense(inputs=self.flatten[i], units=512, activation=tf.nn.relu) for i in range(self.obs_stack)]
            self.expand = [tf.expand_dims(self.l1[i], axis=2) for i in range(self.obs_stack)]
            self.concat = tf.concat(self.expand, axis=2)
            self.transpose = tf.transpose(self.concat, perm=[0, 2, 1])

            enc_cell = [tf.nn.rnn_cell.GRUCell(size) for size in self.lstm_shape]
            enc_cell = tf.nn.rnn_cell.MultiRNNCell(enc_cell)
            self.cell_out, self.enc_states = tf.nn.dynamic_rnn(cell=enc_cell, inputs=self.transpose, dtype=tf.float32)
            last_layer = self.cell_out[:, -1]
            self.actor = tf.layers.dense(inputs=last_layer, units=self.action_size, activation=tf.nn.softmax)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class CNNCriticLSTM:
    def __init__(self, name, window_size, obs_stack, lstm_shape):
        self.window_size = window_size
        self.lstm_shape = lstm_shape
        self.obs_stack = obs_stack
        self.reuse = []
        self.scope = name
        for i in range(self.obs_stack):
            if i == 0:
                self.reuse.append(False)
            else:
                self.reuse.append(True)

        with tf.variable_scope(name):
            self.input = tf.placeholder(tf.float32, shape=[None, self.window_size, self.window_size, self.obs_stack])
            
            self.input_list = [self.input[:, :, :, i] for i in range(obs_stack)]
            self.input_channel = [tf.expand_dims(self.input_list[i], axis=3) for i in range(obs_stack)]
            self.conv1 = [tf.layers.conv2d(inputs=self.input_channel[i], filters=4, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu, name='conv1', reuse=self.reuse[i]) for i in range(obs_stack)]
            self.conv2 = [tf.layers.conv2d(inputs=self.conv1[i], filters=16, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu, name='conv2', reuse=self.reuse[i]) for i in range(obs_stack)]
            self.conv3 = [tf.layers.conv2d(inputs=self.conv2[i], filters=16, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=tf.nn.relu, name='conv3', reuse=self.reuse[i]) for i in range(obs_stack)]
            self.flatten = [tf.reshape(self.conv3[i], [-1, 7 * 7 * 16]) for i in range(obs_stack)]
            self.l1 = [tf.layers.dense(inputs=self.flatten[i], units=512, activation=tf.nn.relu) for i in range(self.obs_stack)]
            self.expand = [tf.expand_dims(self.l1[i], axis=2) for i in range(self.obs_stack)]
            self.concat = tf.concat(self.expand, axis=2)
            self.transpose = tf.transpose(self.concat, perm=[0, 2, 1])

            enc_cell = [tf.nn.rnn_cell.GRUCell(size) for size in self.lstm_shape]
            enc_cell = tf.nn.rnn_cell.MultiRNNCell(enc_cell)
            self.cell_out, self.enc_states = tf.nn.dynamic_rnn(cell=enc_cell, inputs=self.transpose, dtype=tf.float32)
            last_layer = self.cell_out[:, -1]
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
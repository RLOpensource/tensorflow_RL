import numpy as np
import tensorflow as tf
import copy

class VPG:
    def __init__(self, sess, output_size, num_worker, num_step, actor, critic):
        self.sess = sess
        self.output_size = output_size

        self.actor = actor
        self.critic = critic

        self.gamma = 0.99
        self.lamda = 0.9
        self.lr = 0.00025
        self.batch_size = 32
        self.cof_entropy = 0.5
        
        self.actor_pi_trainable = self.actor.get_trainable_variables()
        self.critic_pi_trainable = self.critic.get_trainable_variables()

        self.actions = tf.placeholder(dtype=tf.int32,shape=[None])
        self.targets = tf.placeholder(dtype=tf.float32,shape=[None])
        self.adv = tf.placeholder(dtype=tf.float32,shape=[None])
        self.value = tf.squeeze(self.critic.critic)

        act_probs = self.actor.actor

        act_probs = tf.reduce_sum(tf.multiply(act_probs,tf.one_hot(indices=self.actions,depth=self.output_size)),axis=1)
        cross_entropy = tf.log(tf.clip_by_value(act_probs,1e-5, 1.0))*self.adv
        actor_loss = -tf.reduce_mean(cross_entropy)

        critic_loss = tf.losses.mean_squared_error(self.value,self.targets)

        entropy = - self.actor.actor * tf.log(self.actor.actor)
        entropy = tf.reduce_mean(entropy)


        total_actor_loss = actor_loss - self.cof_entropy * entropy
        actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.actor_train_op = actor_optimizer.minimize(total_actor_loss, var_list=self.actor_pi_trainable)

        critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.critic_train_op = critic_optimizer.minimize(critic_loss, var_list=self.critic_pi_trainable)


    def train_model(self, state, action, targets, advs): # targets에 rtgs 넣어주기
        sample_range = np.arange(len(state))
        np.random.shuffle(sample_range)
        if len(state) < self.batch_size:
            train_size = len(state)
        else:
            train_size = self.batch_size
        state_batch = [state[sample_range[i]] for i in range(train_size)]
        action_batch = [action[sample_range[i]] for i in range(train_size)]
        targets_batch = [targets[sample_range[i]] for i in range(train_size)]
        advs_batch = [advs[sample_range[i]] for i in range(train_size)]
        self.sess.run(self.critic_train_op, feed_dict={self.critic.input: state_batch,
                                                self.targets: targets_batch})
        
        self.sess.run(self.actor_train_op, feed_dict={self.actor.input: state_batch,
                                                self.actions: action_batch,
                                                self.adv: advs_batch})


    def get_action(self, state):
        action = self.sess.run(self.actor.actor, feed_dict={self.actor.input: state})
        action = [np.random.choice(self.output_size, p=i) for i in action]
        return np.stack(action)

    def get_value(self, state, next_state):
        value = self.sess.run(self.value, feed_dict={self.critic.input: state})
        next_value = self.sess.run(self.value, feed_dict={self.critic.input: next_state})
        return value, next_value

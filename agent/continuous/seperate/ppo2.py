import tensorflow as tf
import numpy as np

class PPO2:
    def __init__(self, sess, state_size, output_size, actor, critic):
        self.sess = sess
        self.state_size = state_size
        self.output_size = output_size
        self.gamma = 0.99
        self.lamda = 0.95
        self.epoch = 10
        self.actor = actor
        self.critic = critic
        self.lr = 0.00025
        self.ppo_eps = 0.2

        self.pi, self.logp, self.logp_pi = self.actor.pi, self.actor.logp, self.actor.logp_pi
        self.v = self.critic.v

        self.pi_params = self.actor.get_trainable_variables()
        self.v_params = self.critic.get_trainable_variables()

        self.adv_ph = tf.placeholder(tf.float32, shape=[None])
        self.ret_ph = tf.placeholder(tf.float32, shape=[None])
        self.logp_old_ph = tf.placeholder(tf.float32, shape=[None])
        self.old_value = tf.placeholder(tf.float32, shape=[None])

        self.all_phs = [self.actor.input, self.critic.input, self.actor.action, self.adv_ph,
                        self.ret_ph, self.logp_old_ph, self.old_value]
        self.get_action_ops = [self.pi, self.v, self.logp_pi]

        self.ratio = tf.exp(self.logp - self.logp_old_ph)
        self.min_adv = tf.where(self.adv_ph > 0, (1.0 + self.ppo_eps)*self.adv_ph, (1.0 - self.ppo_eps)*self.adv_ph)
        self.pi_loss = -tf.reduce_mean(tf.minimum(self.ratio * self.adv_ph, self.min_adv))
        
        self.clipped_value_loss = self.old_value + tf.clip_by_value(self.v - self.old_value, -self.ppo_eps, self.ppo_eps)
        self.v_loss1 = (self.ret_ph - self.clipped_value_loss) ** 2
        self.v_loss2 = (self.ret_ph - self.v) ** 2

        self.v_loss = 0.5 * tf.reduce_mean(tf.maximum(self.v_loss1, self.v_loss2))

        self.train_pi = tf.train.AdamOptimizer(self.lr).minimize(self.pi_loss)
        self.train_v = tf.train.AdamOptimizer(self.lr).minimize(self.v_loss)

        self.approx_kl = tf.reduce_mean(self.logp_old_ph - self.logp)
        self.approx_ent = tf.reduce_mean(-self.logp)

    def update(self, state, action, target, adv, logp_old, value):
        zip_ph = [state, state, action, adv, target, logp_old, value]
        inputs = {k:v for k,v in zip(self.all_phs, zip_ph)}
        
        value_loss, kl, ent = 0, 0, 0
        for i in range(self.epoch):
            _, _, v_loss, approxkl, approxent = self.sess.run([self.train_pi, self.train_v, self.v_loss, self.approx_kl, self.approx_ent], feed_dict=inputs)
            value_loss += v_loss
            kl += approxkl
            ent += approxent
        return value_loss, kl, ent

    def get_action(self, state):
        a, v, logp_t = self.sess.run(self.get_action_ops, feed_dict={
                                                            self.actor.input: state,
                                                            self.critic.input: state})
        return a, v, logp_t
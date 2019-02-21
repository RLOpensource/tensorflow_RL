import tensorflow as tf
import numpy as np

class PPO:
    def __init__(self, sess, state_size, output_size, num_worker, num_step, old_actor, actor, critic):
        self.sess = sess
        self.output_size = output_size
        self.state_size = state_size
        self.critic_lr = 0.0002
        self.actor_lr = 0.0001
        self.epoch = 3
        self.batch_size = 32
        self.gamma = 0.99
        self.lamda = 0.95
        self.ppo_eps = 0.2

        self.actor = actor
        self.old_actor = old_actor
        self.critic = critic
        self.value = self.critic.critic[:, 0]

        self.pi = self.actor.actor
        self.old_pi = self.old_actor.actor
        self.pi_params = self.actor.get_trainable_variables()
        self.old_pi_params = self.old_actor.get_trainable_variables()
        self.critic_params = self.critic.get_trainable_variables()

        self.sample_op = tf.squeeze(self.pi.sample(1), axis=0)
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(self.pi_params, self.old_pi_params)]

        self.action = tf.placeholder(tf.float32, [None, self.output_size])
        self.adv = tf.placeholder(tf.float32, [None])
        self.targets = tf.placeholder(tf.float32, [None])
        adv = tf.expand_dims(self.adv, 1)

        ratio = self.pi.prob(self.action) / (self.old_pi.prob(self.action) + 1e-10)
        surr = ratio * adv
        actor_loss = -tf.reduce_sum(tf.minimum(surr, tf.clip_by_value(ratio, 1-self.ppo_eps, 1+self.ppo_eps)*adv))

        critic_loss = tf.losses.mean_squared_error(self.value,self.targets)

        self.atrain_op = tf.train.AdamOptimizer(self.actor_lr).minimize(actor_loss)
        self.ctrain_op = tf.train.AdamOptimizer(self.critic_lr).minimize(critic_loss)

    def train_model(self, state, action, target, adv):
        self.sess.run(self.update_oldpi_op)
        sample_range = np.arange(len(state))
        for _ in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(state)/self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]
                state_batch = [state[i] for i in sample_idx]
                action_batch = [action[i] for i in sample_idx]
                advs_batch = [adv[i] for i in sample_idx]
                targets_batch = [target[i] for i in sample_idx]
                feed_dict={self.actor.input: state_batch,
                    self.critic.input: state_batch,
                    self.old_actor.input: state_batch,
                    self.targets: targets_batch,
                    self.adv: advs_batch,
                    self.action: action_batch}
                self.sess.run([self.atrain_op, self.ctrain_op], feed_dict=feed_dict)

    def get_action(self, state):
        action = self.sess.run(self.sample_op, feed_dict={self.actor.input: state})
        return np.clip(action, -1, 1)

    def get_value(self, state, next_state):
        value = self.sess.run(self.value, feed_dict={self.critic.input: state})
        next_value = self.sess.run(self.value, feed_dict={self.critic.input: next_state})
        return value, next_value


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

    def train_model(self):
        batch = random.sample(self.memory,self.batch_size)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        next_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        target_action_input = np.clip(
            self.sess.run(self.target_actor.actor,
                          feed_dict={self.target_actor.state:next_states})
            + np.clip(np.random.normal(0,0.1,[self.batch_size,self.action_size]),-0.1,0.1),-1,1)
        target_q_value = self.sess.run(self.target_q,
                                       feed_dict={self.target_critic.state:next_states,
                                                  self.target_critic.action:target_action_input})
        targets = np.asarray(
            [r + self.gamma * (1 - d) * tv for r, tv, d in
             zip(rewards, target_q_value, dones)])
        self.sess.run(self.ctrain_op,feed_dict=
        {
            self.critic.state:states,
            self.critic.action:actions,
            self.target_value:np.squeeze(targets)
        })
        action_for_train = self.sess.run(self.actor.actor,feed_dict={self.actor.state:states})
        self.sess.run(self.atrain_op,feed_dict=
        {
            self.actor.state:states,
            self.critic.state:states,
            self.critic.action:action_for_train
        })
        self.sess.run(self.update_target_soft)
        
    def get_action(self, state):
        action = self.sess.run(self.sample_op, feed_dict={self.actor.input: state})
        return np.clip(action, -1, 1)

    def get_value(self, state, next_state):
        value = self.sess.run(self.value, feed_dict={self.critic.input: state})
        next_value = self.sess.run(self.value, feed_dict={self.critic.input: next_state})
        return value, next_value


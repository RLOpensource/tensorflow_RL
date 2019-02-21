import numpy as np
import tensorflow as tf
import copy

class PPOLSTM:
    def __init__(self, sess, output_size, num_worker, num_step, old_actor, actor, critic):
        self.sess = sess
        self.output_size = output_size
        self.old_actor = old_actor
        self.actor = actor
        self.critic = critic
        self.num_worker = num_worker
        self.value = tf.squeeze(self.critic.critic)
        self.gamma = 0.99
        self.lamda = 0.95
        self.lr = 0.00025
        self.grad_clip_max = 1.0
        self.grad_clip_min = -1.0
        self.ppo_eps = 0.2
        self.epoch = 8
        self.batch_size = 16
        self.cof_entropy = 0.001

        self.old_actor_pi_trainable = self.old_actor.get_trainable_variables()
        self.actor_pi_trainable = self.actor.get_trainable_variables()
        self.critic_pi_trainable = self.critic.get_trainable_variables()

        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(self.actor_pi_trainable, self.old_actor_pi_trainable)]

        self.actions = tf.placeholder(dtype=tf.int32, shape=[None])
        self.targets = tf.placeholder(dtype=tf.float32, shape=[None])
        self.adv = tf.placeholder(dtype=tf.float32, shape=[None])
        self.value = tf.squeeze(self.critic.critic)

        act_probs = self.actor.actor
        act_probs_old = self.old_actor.actor

        act_probs = tf.reduce_sum(tf.multiply(act_probs,tf.one_hot(indices=self.actions,depth=self.output_size)),axis=1)
        act_probs_old = tf.reduce_sum(tf.multiply(act_probs_old, tf.one_hot(indices=self.actions, depth=self.output_size)), axis=1)

        act_probs = tf.clip_by_value(act_probs, 1e-10, 1.0)
        act_probs_old = tf.clip_by_value(act_probs_old, 1e-10, 1.0)

        ratios = tf.exp(tf.log(act_probs) - tf.log(act_probs_old))
        clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.ppo_eps, clip_value_max=1 + self.ppo_eps)
        actor_loss_minimum = tf.minimum(tf.multiply(self.adv, clipped_ratios), tf.multiply(self.adv, ratios))
        actor_loss = -tf.reduce_mean(actor_loss_minimum)

        critic_loss = tf.losses.mean_squared_error(self.value,self.targets)

        entropy = - self.actor.actor * tf.log(self.actor.actor)
        entropy = tf.reduce_mean(entropy)
        
        actor_loss = actor_loss - self.cof_entropy * entropy
        actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        actor_gvs = actor_optimizer.compute_gradients(actor_loss, var_list=self.actor_pi_trainable)
        actor_capped_gvs = [(tf.clip_by_value(grad, self.grad_clip_min, self.grad_clip_max), var) for grad, var in actor_gvs]
        self.actor_train_op = actor_optimizer.apply_gradients(actor_capped_gvs)

        critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        critic_gvs = critic_optimizer.compute_gradients(critic_loss, var_list=self.critic_pi_trainable)
        critic_capped_gvs = [(tf.clip_by_value(grad, self.grad_clip_min, self.grad_clip_max), var) for grad, var in critic_gvs]
        self.critic_train_op = critic_optimizer.apply_gradients(critic_capped_gvs)
       

    def get_init(self):
        actor_init, critic_init = self.sess.run([self.actor.init_state, self.critic.init_state],
                                                feed_dict={self.actor.batch_size: self.num_worker,
                                                           self.critic.batch_size: self.num_worker})
        return actor_init, critic_init


    def get_action(self, state, actor_init, critic_init):
        action, actor_final, value, value_final = self.sess.run([self.actor.actor, self.actor.final_state,
                                                                self.value, self.critic.final_state],
                                                                feed_dict={self.actor.input: state, self.critic.input: state,
                                                                        self.actor.init_state: actor_init, self.critic.init_state: critic_init})
        action = [np.random.choice(self.output_size, p=i) for i in action]
        return action, value, actor_final, value_final

class PPO:
    def __init__(self, sess, output_size, num_worker, num_step, actor, critic):
        self.sess = sess
        self.output_size = output_size
        
        self.actor = actor
        self.critic = critic

        self.gamma = 0.99
        self.lamda = 0.9
        self.lr = 0.00025
        self.batch_size = 32
        self.grad_clip_max = 1.0
        self.grad_clip_min = -1.0
        self.ppo_eps = 0.2
        self.epoch = 3
        self.cof_entropy = 0.001

        self.actor_pi_trainable = self.actor.get_trainable_variables()
        self.critic_pi_trainable = self.critic.get_trainable_variables()

        self.actions = tf.placeholder(dtype=tf.int32, shape=[None])
        self.targets = tf.placeholder(dtype=tf.float32, shape=[None])
        self.adv = tf.placeholder(dtype=tf.float32, shape=[None])
        self.old_policy = tf.placeholder(dtype=tf.float32, shape=[None, self.output_size])
        self.value = tf.squeeze(self.critic.critic)

        act_probs = self.actor.actor
        act_probs_old = self.old_policy

        act_probs = tf.reduce_sum(tf.multiply(act_probs,tf.one_hot(indices=self.actions,depth=self.output_size)),axis=1)
        act_probs_old = tf.reduce_sum(tf.multiply(act_probs_old, tf.one_hot(indices=self.actions, depth=self.output_size)), axis=1)

        act_probs = tf.clip_by_value(act_probs, 1e-10, 1.0)
        act_probs_old = tf.clip_by_value(act_probs_old, 1e-10, 1.0)

        ratios = tf.exp(tf.log(act_probs) - tf.log(act_probs_old))
        clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.ppo_eps, clip_value_max=1 + self.ppo_eps)
        actor_loss_minimum = tf.minimum(tf.multiply(self.adv, clipped_ratios), tf.multiply(self.adv, ratios))
        actor_loss = -tf.reduce_mean(actor_loss_minimum)

        critic_loss = tf.losses.mean_squared_error(self.value,self.targets)

        entropy = - self.actor.actor * tf.log(self.actor.actor)
        entropy = tf.reduce_mean(entropy)
        
        actor_loss = actor_loss - self.cof_entropy * entropy
        actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        actor_gvs = actor_optimizer.compute_gradients(actor_loss, var_list=self.actor_pi_trainable)
        actor_capped_gvs = [(tf.clip_by_value(grad, self.grad_clip_min, self.grad_clip_max), var) for grad, var in actor_gvs]
        self.actor_train_op = actor_optimizer.apply_gradients(actor_capped_gvs)

        critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        critic_gvs = critic_optimizer.compute_gradients(critic_loss, var_list=self.critic_pi_trainable)
        critic_capped_gvs = [(tf.clip_by_value(grad, self.grad_clip_min, self.grad_clip_max), var) for grad, var in critic_gvs]
        self.critic_train_op = critic_optimizer.apply_gradients(critic_capped_gvs)


    def train_model(self, state, action, targets, advs):
        old_policy = self.sess.run(self.actor.actor, feed_dict={self.actor.input: state})
        sample_range = np.arange(len(state))
        for ep in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(state) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]
                state_batch = [state[i] for i in sample_idx]
                action_batch = [action[i] for i in sample_idx]
                advs_batch = [advs[i] for i in sample_idx]
                targets_batch = [targets[i] for i in sample_idx]
                old_policy_batch = [old_policy[i] for i in sample_idx]
                self.sess.run(self.critic_train_op, feed_dict={self.critic.input: state_batch,
                                                               self.targets: targets_batch})
                self.sess.run(self.actor_train_op, feed_dict={self.actor.input: state_batch,
                                                            self.actions: action_batch,
                                                            self.adv: advs_batch,
                                                            self.old_policy: old_policy_batch})

    def get_action(self, state):
        action = self.sess.run(self.actor.actor, feed_dict={self.actor.input: state})
        action = [np.random.choice(self.output_size, p=i) for i in action]
        return np.stack(action)

    def get_value(self, state, next_state):
        value, next_value = [], []
        for i in range(len(state)):
            v = self.sess.run(self.value, feed_dict={self.critic.input: [state[i]]})
            nv = self.sess.run(self.value, feed_dict={self.critic.input: [next_state[i]]})
            value.append(v)
            next_value.append(nv)
        return value, next_value
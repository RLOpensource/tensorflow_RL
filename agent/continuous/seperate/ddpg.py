import numpy as np
import tensorflow as tf
import random
from collections import deque
from agent.utils import OU_noise

class DDPG:
    def __init__(self, sess, state_size, action_size, num_worker, num_step, target_actor, target_critic, actor, critic):
        self.sess = sess
        self.action_size = action_size
        self.state_size = state_size
        self.critic_lr = 0.0002
        self.actor_lr = 0.0001
        self.batch_size = 32
        self.gamma = 0.9
        self.target_update_rate = 1e-3

        self.actor = actor
        self.target_actor = target_actor
        self.critic = critic
        self.target_critic = target_critic

        self.memory = deque(maxlen=20000)
        self.noise_generator = OU_noise(action_size,num_worker)

        self.value = self.critic.critic
        self.pi = self.actor.actor

        self.pi_params = self.actor.get_trainable_variables()
        self.target_pi_params = self.target_actor.get_trainable_variables()
        self.critic_params = self.critic.get_trainable_variables()
        self.target_critic_params = self.target_critic.get_trainable_variables()

        self.update_target_soft = []
        for idx in range(len(self.pi_params)):
            self.update_target_soft.append(self.target_pi_params[idx].assign(self.target_update_rate*self.target_pi_params[idx].value() + (1 - self.target_update_rate)*self.pi_params[idx].value()))
        for idx in range(len(self.critic_params)):
            self.update_target_soft.append(self.target_critic_params[idx].assign(self.target_update_rate*self.target_critic_params[idx].value() + (1 - self.target_update_rate)*self.critic_params[idx].value()))

        self.target_value = tf.placeholder(tf.float32,shape=[None])
        self.action = tf.placeholder(tf.float32,shape=[None,self.action_size])

        critic_loss = tf.losses.huber_loss(self.target_value,tf.squeeze(self.value))

        action_grad = tf.clip_by_value(tf.gradients(self.value,self.critic.action),-0.1,0.1)
        pi_grad = tf.gradients(xs=self.pi_params,ys=self.pi,grad_ys=action_grad)
        for idx,grads in enumerate(pi_grad):
            pi_grad[idx] = -grads/self.batch_size

        with tf.control_dependencies(self.pi_params):
            self.atrain_op = tf.train.AdamOptimizer(self.actor_lr).apply_gradients(zip(pi_grad,self.pi_params))

        with tf.control_dependencies(self.critic_params):
            self.ctrain_op = tf.train.AdamOptimizer(self.actor_lr).minimize(critic_loss)

    def train_model(self):
        batch = random.sample(self.memory,self.batch_size)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        next_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        target_action_input = self.sess.run(self.target_actor.actor,feed_dict={self.target_actor.state:next_states})
        target_q_value = self.sess.run(self.target_critic.critic,feed_dict={self.target_critic.state:next_states,
                                                                            self.target_critic.action:target_action_input})
        targets = np.asarray([r + self.gamma * (1-d) * tv for r,tv,d in zip(rewards,target_q_value,dones)])
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

    def get_action(self, state, epsilon):
        action = self.sess.run(self.actor.actor,feed_dict={self.actor.state:state})
        return np.clip(action + epsilon*self.noise_generator.noise(), -1, 1)

    def get_sample(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

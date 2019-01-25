from breakout_environment import Environment
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from agent.discrete.seperate.a2c import A2C
from agent.discrete.seperate.ppo import PPO
from agent.utils import get_gaes
from tensorboardX import SummaryWriter
from model import *
import gym
import cv2

writer = SummaryWriter()
sess = tf.Session()
num_worker = 8
num_step = 64
window_size, output_size, obs_stack = 84, 3, 4
actor = CNNActor('actor', window_size, obs_stack, output_size)
critic = CNNCritic('critic', window_size, obs_stack)
agent = A2C(sess, output_size, num_worker, num_step, actor, critic)
#agent = PPO(sess, output_size, num_worker, num_step, actor, critic)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#saver.restore(sess, 'breakout/model')

env = gym.make('BreakoutDeterministic-v4')

normalize = True
train_size = 16
update_step = 0
episode = 0

total_state, total_next_state, total_reward, total_action, total_done = [], [], [], [], []

while True:
    episode += 1
    state = np.zeros([84, 84, 4])
    
    _ = env.reset()

    image = env.env.ale.getScreenGrayscale().squeeze().astype('float32')
    image = cv2.resize(image, (84, 84))
    image *= (1.0/255.0)

    state[:, :, 3] = image

    done = False

    while not done:
        #env.render()

        action = agent.get_action([state])
        action = action[0]
        _, reward, _, _ = env.step(action+1)

        image = env.env.ale.getScreenGrayscale().squeeze().astype('float32')
        image = cv2.resize(image, (84, 84))
        image *= (1.0/255.0)

        next_state = np.zeros([84, 84, 4])
        next_state[:, :, :3] = state[:, :, 1:]
        next_state[:, :, 3] = image

        lives = env.env.ale.lives()

        if lives == 5:
            done = False
        elif lives == 4:
            done = True

        if done:
            reward = -1
        
        total_state.append(state)
        total_next_state.append(next_state)
        total_reward.append(reward)
        total_done.append(done)
        total_action.append(action)
        
        state = next_state

    if episode % train_size == 0:
        update_step += 1
        total_state = np.stack(total_state)
        total_next_state = np.stack(total_next_state)
        total_action = np.stack(total_action)
        total_done = np.stack(total_done)
        total_reward = np.stack(total_reward)

        value, next_value = agent.get_value(total_state, total_next_state)
        adv, target = get_gaes(total_reward, total_done, value, next_value, agent.gamma, agent.lamda, normalize)

        agent.train_model(total_state, total_action, np.hstack(target), np.hstack(adv))
        writer.add_scalar('data/reward_per_episode', sum(total_reward) / train_size, update_step)
        print(sum(total_reward) / train_size, update_step)

        total_state, total_next_state, total_reward, total_action, total_done = [], [], [], [], []
        saver.save(sess, 'breakout/model')
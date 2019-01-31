import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
from lunarLander_environment import Environment
from multiprocessing import Process, Pipe
from agent.discrete.seperate.a2c import A2C
from agent.discrete.seperate.ppo import PPO
from agent.discrete.seperate.vpg import VPG
from agent.utils import get_gaes, get_rtgs
from model import *
import gym

num_worker = 8
num_step = 256
visualize = False
global_update = 0
sample_idx = 0
step = 0
score = 0
episode = 0

writer = SummaryWriter()
sess = tf.Session()
state_size, output_size = 8, 4
actor = MLPActor('actor', state_size, output_size)
critic = MLPCritic('critic', state_size)
agent = VPG(sess, output_size, num_worker, num_step, actor, critic)
#agent = PPO(sess, output_size, num_worker, num_step, actor, critic)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

normalize = True
env = gym.make('LunarLander-v2')
train_size = 16
update_step = 0
episode = 0
total_state, total_next_state, total_reward, total_action, total_done = [], [], [], [], []

while True:
    episode += 1
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action([state])
        action = action[0]
        next_state, reward, done, _ = env.step(action)

        total_state.append(state)
        total_next_state.append(next_state)
        total_reward.append(reward)
        total_done.append(done)
        total_action.append(action)

        score += reward

        state = next_state

    if episode % train_size == 0:
        update_step += 1
        total_state = np.stack(total_state)
        total_next_state = np.stack(total_next_state)
        total_action = np.stack(total_action)
        total_done = np.stack(total_done)
        total_reward = np.stack(total_reward)

        value, next_value = agent.get_value(total_state, total_next_state)
        adv, target = get_rtgs(total_reward, total_done, value, agent.gamma) # rtgs

        agent.train_model(total_state, total_action, np.hstack(target), np.hstack(adv))
        writer.add_scalar('data/reward_per_episode', sum(total_reward) / train_size, update_step)
        print(sum(total_reward) / train_size, update_step)

        total_state, total_next_state, total_reward, total_action, total_done = [], [], [], [], []
        saver.save(sess, 'lunarlander/model')
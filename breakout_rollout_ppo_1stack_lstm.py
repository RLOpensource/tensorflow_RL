import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from policy.discrete.seperate.ppo import PPOLSTM
from policy.utils import get_gaes, split_episode
from tensorboardX import SummaryWriter
from model import CNNActorLSTM, CNNCriticLSTM
import cv2
import gym

#writer = SummaryWriter()
sess = tf.Session()
num_worker = 1
window_size, output_size, lstm_units, lstm_layers = 84, 3, 512, 1
old_actor = CNNActorLSTM('old_actor', window_size, output_size, lstm_units, lstm_layers)
actor = CNNActorLSTM('actor', window_size, output_size, lstm_units, lstm_layers)
critic = CNNCriticLSTM('critic', window_size, lstm_units, lstm_layers)
agent = PPOLSTM(sess, output_size, num_worker, None, old_actor, actor, critic)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#saver.restore(sess, 'breakout_ppo_1env1lstm/model')

learning = True
normalize = True
global_update = 0
sample_idx = 0
score = 0
episode = 0

train_size = 16

env = gym.make('BreakoutDeterministic-v4')

total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []
total_value, total_next_value = [], []
episode = 0
while True:

    episode += 1
    _ = env.reset()
    env.step(1)
    state = cv2.resize(env.env.ale.getScreenGrayscale().squeeze().astype('float32'), (84, 84))
    state *= (1.0/255.0)
    state = np.reshape(state, [window_size, window_size, 1])
    actor_init, critic_init = agent.get_init()
    done = False

    while not done:
        #env.render()
        action, value, actor_init, value_init = agent.get_action([state], actor_init, critic_init)
        action = action[0]
        _, reward, done, info = env.step(action+1)

        next_state = cv2.resize(env.env.ale.getScreenGrayscale().squeeze().astype('float32'), (84, 84))
        next_state *= (1.0/255.0)
        next_state = np.reshape(next_state, [window_size, window_size, 1])

        _, next_value, _, _ = agent.get_action([next_state], actor_init, critic_init)

        if info['ale.lives'] != 5:
            done = True
            reward = -1

        total_next_value.append(next_value)
        total_value.append(value)
        total_state.append(state)
        total_next_state.append(next_state)
        total_done.append(done)
        total_reward.append(reward)
        total_action.append(action)

        state = next_state

    if episode % train_size == 0:
        total_state = np.stack(total_state)
        total_next_state = np.stack(total_next_state)
        total_value = np.stack(total_value)
        total_next_value = np.stack(total_next_value)
        total_action = np.stack(total_action)
        total_done = np.stack(total_done)
        total_reward = np.stack(total_reward)

        adv, target = get_gaes(total_reward, total_done, total_value, total_next_value,
                                agent.gamma, agent.lamda, True)

        total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []
        total_value, total_next_value = [], []


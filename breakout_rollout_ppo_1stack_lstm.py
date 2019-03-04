import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from breakout_environment_custom import Environment
from multiprocessing import Pipe
from agent.discrete.seperate.ppo import PPO
from agent.utils import get_gaes
from example_model.policy.cnn.discrete import CNNLSTMActor
from example_model.policy.cnn.discrete import CNNLSTMCritic
from tensorboardX import SummaryWriter
import time

writer = SummaryWriter()
works = []
parent_conns = []
child_conns = []
visualize = True
normalize = True
num_worker, num_step = 4, 256
sample_idx = 0
window_size, output_size, obs_stack = 84, 3, 3
lstm_units, lstm_layers = 256, 1

sess = tf.Session()
actor = CNNLSTMActor('actor', window_size, obs_stack, output_size, lstm_units, lstm_layers)
critic = CNNLSTMCritic('critic', window_size, obs_stack, output_size, lstm_units, lstm_layers)
agent = PPO(sess, output_size, num_worker, num_step, actor, critic)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#saver.restore(sess, 'breakout_ppo_lstm/model')

for idx in range(num_worker):
    parent_conn, child_conn = Pipe()
    work = Environment(visualize if sample_idx == idx else False, idx, child_conn, obs_stack)
    work.start()
    works.append(work)
    parent_conns.append(parent_conn)
    child_conns.append(child_conn)

states = np.zeros([num_worker, 84, 84, obs_stack])
score = 0
global_step = 0
episode = 0

while True:
    total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []
    global_step += 1

    for _ in range(num_step):
        actions = agent.get_action(states)

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)
        
        next_states, rewards, dones, real_dones = [], [], [], []
        
        for parent_conn in parent_conns:
            s, r, d, rd = parent_conn.recv()
            next_states.append(s)
            rewards.append(r)
            dones.append(d)
            real_dones.append(rd)

        next_states = np.stack(next_states)
        rewards = np.hstack(rewards)
        dones = np.hstack(dones)
        real_dones = np.hstack(real_dones)

        score += rewards[sample_idx]

        total_state.append(states)
        total_next_state.append(next_states)
        total_done.append(dones)
        total_reward.append(rewards)
        total_action.append(actions)

        states = next_states

        if real_dones[sample_idx]:
            episode += 1
            writer.add_scalar('data/reward_per_episode', score, episode)
            print(episode, score)
            score = 0

    total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, window_size, window_size, obs_stack])
    total_next_state = np.stack(total_next_state).transpose([1, 0, 2, 3, 4]).reshape([-1, window_size, window_size, obs_stack])
    total_action = np.stack(total_action).transpose([1, 0]).reshape([-1])
    total_done = np.stack(total_done).transpose([1, 0]).reshape([-1])
    total_reward = np.stack(total_reward).transpose([1, 0]).reshape([-1])

    total_target, total_adv = [], []
    for idx in range(num_worker):
        value, next_value = agent.get_value(total_state[idx * num_step:(idx + 1) * num_step],
                                                total_next_state[idx * num_step:(idx + 1) * num_step])
        adv, target = get_gaes(total_reward[idx * num_step:(idx + 1) * num_step],
                                    total_done[idx * num_step:(idx + 1) * num_step],
                                    value, next_value, agent.gamma, agent.lamda, normalize)
        total_target.append(target)
        total_adv.append(adv)

    agent.train_model(total_state, total_action, np.hstack(total_target), np.hstack(total_adv))

    writer.add_scalar('data/reward_per_rollout', sum(total_reward)/(num_worker), global_step)
    saver.save(sess, 'breakout_ppo_lstm/model')
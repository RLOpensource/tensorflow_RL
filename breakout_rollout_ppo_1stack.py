from breakout_environment import Environment
from multiprocessing import Pipe
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from agent.discrete.seperate.ppo import PPO
from example_model.policy.cnn.discrete import CNNActor
from example_model.policy.cnn.discrete import CNNCritic
from agent.utils import get_gaes, get_rtgs
from tensorboardX import SummaryWriter

writer = SummaryWriter()
sess = tf.Session()
num_worker = 4
num_step = 256
window_size, output_size, obs_stack = 84, 3, 1
actor = CNNActor('actor', window_size, obs_stack, output_size)
critic = CNNCritic('critic', window_size, obs_stack)
agent = PPO(sess, output_size, num_worker, num_step, actor, critic)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#saver.restore(sess, 'breakout_ppo/model')
learning = True

normalize = True
global_update = 0
sample_idx = 0
score = 0
episode = 0

works = []
parent_conns = []
child_conns = []
visualize = False
output_malloc = False

for idx in range(num_worker):
    parent_conn, child_conn = Pipe()
    work = Environment(visualize if sample_idx == idx else False, idx, child_conn)
    work.start()
    works.append(work)
    parent_conns.append(parent_conn)
    child_conns.append(child_conn)


states = np.zeros([num_worker, window_size, window_size, obs_stack])
while True:
    total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []
    global_update += 1
    next_states, rewards, dones, real_dones = [], [], [], []

    for _ in range(num_step):
        actions = agent.get_action(states)

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        next_states, rewards, dones, real_dones = [], [], [], []

        for parent_conn in parent_conns:
            s, r, d, rd = parent_conn.recv()
            s = np.stack(s)[:, :, -1].reshape([window_size, window_size, obs_stack])
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
            if episode < 1455:
                print(episode, score)
                writer.add_scalar('data/reward_per_episode', score, episode)
            score = 0

    if learning:
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

        writer.add_scalar('data/reward_per_rollout', sum(total_reward)/(num_worker), global_update)
        saver.save(sess, 'breakout_ppo/model')
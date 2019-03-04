import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
from bipedal_walker_environment import Environment
from multiprocessing import Process, Pipe
from agent.continuous.seperate.ppo import PPO
from example_model.policy.mlp.continuous import MLPContinuousActor
from example_model.policy.mlp.continuous import MLPContinuousCritic
from agent.utils import get_gaes, get_rtgs

num_worker = 16
num_step = 256
visualize = False
global_update = 0
sample_idx = 0
step = 0
score = 0
episode = 0
normalize = True

state_size, output_size = 24, 4
writer = SummaryWriter()
sess = tf.Session()
sess = tf.Session()
old_actor = MLPContinuousActor('old_pi', state_size, output_size)
actor = MLPContinuousActor('pi', state_size, output_size)
critic = MLPContinuousCritic('critic', state_size)
agent = PPO(sess, state_size, output_size, 4, 128, old_actor, actor, critic)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#saver.restore(sess, 'bipedal_rollout_ppo/model')

agent.lamda = 0.95
agent.epoch = 3

works = []
parent_conns = []
child_conns = []
for idx in range(num_worker):
    parent_conn, child_conn = Pipe()
    work = Environment(idx, child_conn, visualize)
    work.start()
    works.append(work)
    parent_conns.append(parent_conn)
    child_conns.append(child_conn)

states = np.zeros([num_worker, state_size])

while True:
    total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []
    global_update += 1

    for _ in range(num_step):
        step += 1
        actions = agent.get_action(states)
        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        next_states, rewards, dones = [], [], []
        for parent_conn in parent_conns:
            s, r, d, _ = parent_conn.recv()
            next_states.append(s)
            rewards.append(r)
            dones.append(d)

        next_states = np.vstack(next_states)
        rewards = np.hstack(rewards)
        dones = np.hstack(dones)

        total_state.append(states)
        total_next_state.append(next_states)
        total_reward.append(rewards)
        total_done.append(dones)
        total_action.append(actions)

        score += rewards[sample_idx]
        
        if dones[sample_idx]:
            episode += 1
            writer.add_scalar('data/reward_per_episode', score, episode)
            print(episode, score)
            score = 0

        states = next_states

    total_state = np.stack(total_state).transpose([1, 0, 2]).reshape([-1, state_size])
    total_next_state = np.stack(total_next_state).transpose([1, 0, 2]).reshape([-1, state_size])
    total_reward = np.stack(total_reward).transpose().reshape([-1])
    total_done = np.stack(total_done).transpose().reshape([-1])
    total_action = np.stack(total_action).transpose([1, 0, 2]).reshape([-1, output_size])
    
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
    saver.save(sess, 'bipedal_rollout_ppo/model')
from breakout_environment import Environment
from example_model.value.cnn.discrete import CNNDQN
from agent.discrete.join.dqn import DQN
from multiprocessing import Pipe    
import tensorflow as tf
import numpy as np
from tensorboardX import SummaryWriter

model_path = 'breakoutdqn/model'
writer = SummaryWriter()
sess = tf.Session()
num_worker = 8
num_step = 256
window_size, output_size, obs_stack = 84, 3, 4
num_support, batch_size = 8, 32
mainNet = CNNDQN('main', window_size, obs_stack, output_size)
targetNet = CNNDQN('target', window_size, obs_stack, output_size)
agent = DQN(sess, output_size, mainNet, targetNet, max_length=20000)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
agent.update_target()

sample_idx = 0

works = []
parent_conns = []
child_conns = []
visualize = False

for idx in range(num_worker):
    parent_conn, child_conn = Pipe()
    work = Environment(visualize if sample_idx == idx else False, idx, child_conn)
    work.start()
    works.append(work)
    parent_conns.append(parent_conn)
    child_conns.append(child_conn)

step = 0
states = np.zeros([num_worker, window_size, window_size, obs_stack])
episode = 0
train_step = 0
l = 0
score = 0
train_start_step = 10000
e = 1.0
step_per_episode = 0

while True:
    for _ in range(num_step):
        step += 1
        step_per_episode += 1
        x = np.random.rand()
        if x < e:
            actions = np.random.randint(0, output_size, num_worker)
        else:
            actions = [agent.get_action([states[i]])[0]for i in range(num_worker)]

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
        
        if step > 1:
            for s, ns, r, a, d in zip(states, next_states, rewards, actions, dones):
                agent.append(s, ns, a, r, d)

        states = next_states
        score += rewards[sample_idx]

        if len(agent.memory) > train_start_step:
            train_step += 1
            _, loss = agent.train()
            l += loss
            if train_step % 5 == 0:
                agent.update_target()

        if real_dones[sample_idx]:
            if len(agent.memory) > train_start_step:
                episode += 1
            e = 1.0 - (episode / 30)
            writer.add_scalar('data/reward', score, episode)
            writer.add_scalar('data/loss', l / step_per_episode, episode)
            writer.add_scalar('data/epsilon', e, episode)
            print('episode:', episode, 'reward:', score, 'expectation loss:', l / step_per_episode, 'epsilon:', e, 'memory_size:', len(agent.memory))
            saver.save(sess, model_path)
            score = 0
            step_per_episode = 0
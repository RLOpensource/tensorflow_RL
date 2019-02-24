import tensorflow as tf
from tensorboardX import SummaryWriter
from agent.discrete.join.dqn import DQN
from model import MLPDQNNetwork, MLPDuelDQNNetwork
from lunarLander_environment import Environment
from multiprocessing import Pipe
import numpy as np

sess = tf.Session()
state_size, output_size = 8, 4
double = False
memory_size = 150
mainNet = MLPDuelDQNNetwork('main', state_size, output_size)
targetNet = MLPDuelDQNNetwork('target', state_size, output_size)
agent = DQN(sess, output_size, mainNet, targetNet, double, memory_size)
sess.run(tf.global_variables_initializer())
agent.update_target()

num_worker = 4
num_step = 128
visualize = False

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
sample_idx = 0
e = 0
episode = 0
score = 0
print_step = 0
l_total = 0

while True:
    episode += 1
    e = 1. / ((episode / 10) + 1)
    for _ in range(num_step):
        if np.random.rand() < e:
            actions = np.random.randint(0, output_size, size=num_worker)
        else:
            actions = agent.get_action(states)
        
        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        next_states, rewards, dones = [], [], []
        for parent_conn in parent_conns:
            s, r, d, _ = parent_conn.recv()
            next_states.append(s)
            rewards.append(r)
            dones.append(d)

        next_states = np.stack(next_states)
        rewards = np.stack(rewards)
        dones = np.stack(dones)

        for s, a, ns, r, d in zip(states, actions, next_states, rewards, dones):
            agent.append(s, ns, a, r, d)
        
        states = next_states
        score += rewards[sample_idx]

        if dones[sample_idx]:
            print_step += 1
            print(print_step, score)
            if len(agent.memory) > 1000:
                for i in range(5):
                    _, l = agent.train_model()
                    l_total += l
            if print_step % 5 == 0:
                agent.update_target()
            print('episode:', print_step, '| reward:', score, '| loss:', l_total)
            l_total = 0
            score = 0
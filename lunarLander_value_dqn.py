from example_model.value.mlp.discrete import MLPDQN
from agent.discrete.join.dqn import DQN
import tensorflow as tf
import numpy as np
from lunarLander_environment import Environment
from tensorboardX import SummaryWriter
from multiprocessing import Pipe

writer = SummaryWriter()

num_worker, num_step = 4, 256
sess = tf.Session()
state_size, output_size = 8, 4
mainNet = MLPDQN('name', state_size, output_size)
targetNet = MLPDQN('target', state_size, output_size)
agent = DQN(sess, output_size, mainNet, targetNet, max_length=100000)
sess.run(tf.global_variables_initializer())
agent.batch_size = 8

sample_idx = 0
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
episode = 0
train_start = 1000
update_target_start = 1
step = 0
score= 0
e = 1.0
episode_loss = 0
episode_step = 0

while True:
    for _ in range(num_step):
        step += 1
        episode_step += 1
        if np.random.rand() < e:
            actions = np.random.randint(0, output_size, num_worker)
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
        
        score += rewards[sample_idx]

        for s, ns, a, r, d in zip(states, next_states, actions, rewards, dones):
            agent.append(s, ns, a, r, d)

        states = next_states
        if len(agent.memory) > 1000:
            for i in range(num_worker):
                loss = agent.train_model()
                episode_loss += loss
            if step % update_target_start == 0:
                agent.update_target()
            
        if dones[sample_idx]:
            episode += 1
            e = 1. / ((episode + 10) + 1)
            if episode < 250:
                writer.add_scalar('data/loss', episode_loss / episode_step, episode)
                writer.add_scalar('data/reward', score, episode)
            print(episode, score)
            episode_loss = 0
            score = 0
            episode_step = 0
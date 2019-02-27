from example_model.value.mlp.discrete import MLPDQN
from agent.discrete.join.dqn import DQN
import tensorflow as tf
import numpy as np
import gym
from tensorboardX import SummaryWriter

writer = SummaryWriter()
model_path = 'dqn/model'
sess = tf.Session()
state_size, output_size, num_support = 8, 4, 8
mainNet = MLPDQN('main', state_size, output_size)
targetNet = MLPDQN('target', state_size, output_size)
agent = DQN(sess, output_size, mainNet, targetNet)
sess.run(tf.global_variables_initializer())
agent.update_target()
saver = tf.train.Saver()

env = gym.make('LunarLander-v2')
episode = 0
while True:
    episode += 1
    e = 1. / ((episode / 10) + 1)
    done = False
    state = env.reset()
    global_step = 0
    l = 0
    score = 0
    while not done:
        global_step += 1
        if np.random.rand() < e:
            action = np.random.randint(0, output_size, 1)
        else:
            action = agent.get_action([state])
        action = action[0]

        next_state, reward, done, _ = env.step(action)
        score += reward

        if len(agent.memory) > 1000:
            _, loss = agent.train()
            l += loss
            if global_step % 5 == 0:
                agent.update_target()
        agent.append(state, next_state, action, reward, done)
        state = next_state
        if done:
            if episode < 467:
                writer.add_scalar('data/reward', score, episode)
                writer.add_scalar('data/loss', l, episode)
                writer.add_scalar('data/epsilon', e, episode)
            print('episode:', episode, 'reward:', score, 'expectation loss:', l, 'epsilon:', e)
            saver.save(sess, model_path)
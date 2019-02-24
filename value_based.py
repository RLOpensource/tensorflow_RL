import gym
from agent.discrete.join.distributional_rl import MLP_Distributional_RL
import tensorflow as tf
import numpy as np
import random
from tensorboardX import SummaryWriter

writer = SummaryWriter()

state_size, output_size, num_support = 8, 4, 8
sess = tf.Session()
env = gym.make('LunarLander-v2')
learning_rate = 0.00025
model = 'IQN'
model_path = model + '/model'
agent = MLP_Distributional_RL(sess, model, learning_rate, state_size, output_size, num_support)
sess.run(tf.global_variables_initializer())
agent.update_target()
saver = tf.train.Saver()

episode = 0
training = True
if not training: 
    saver.restore(sess, model_path)

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
        if training:
            if np.random.rand() < e:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(state)
        else:
            env.render()
            action = agent.choose_action(state)
        
        next_state, reward, done, _ = env.step(action)

        if len(agent.memory) > 1000:
            _, loss = agent.train()
            l += loss
            if global_step % 5 == 0:
                agent.update_target()
        score += reward
        action_one_hot = np.zeros(output_size)
        action_one_hot[action] = 1
        agent.append(state, next_state, action_one_hot, reward, done)
        state = next_state
        if done:
            if episode < 300:
                writer.add_scalar('data/reward', score, episode)
                writer.add_scalar('data/loss', l, episode)
                writer.add_scalar('data/epsilon', e, episode)
            print('episode:', episode, 'reward:', score, 'expectation loss:', l, 'epsilon:', e)
            saver.save(sess, model_path)
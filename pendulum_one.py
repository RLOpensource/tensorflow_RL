import gym
from agent.continuous.seperate.ppo import PPO
from agent import utils
import tensorflow as tf
import numpy as np
from tensorboardX import SummaryWriter
from model import MLPContinuousActor
from model import MLPContinuousCritic

writer = SummaryWriter()

state_size = 3
output_size = 1
clip = 2
env = gym.make('Pendulum-v0')
sess = tf.Session()
old_actor = MLPContinuousActor('old_pi', state_size, output_size, clip)
actor = MLPContinuousActor('pi', state_size, output_size, clip)
critic = MLPContinuousCritic('critic', state_size)
agent = PPO(sess, state_size, output_size, 4, 128, old_actor, actor, critic)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, 'pendulum/model')
agent.epoch = 4
agent.lamda = 0

ep_len = 200
ep = 0
train_size = 1
total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []
score = 0
update_step = 0

while True:
    ep += 1
    state = env.reset()
    done = False
    for t in range(ep_len):
        env.render()
        action = agent.get_action([state], clip)
        action = action[0]
        print(action)
        next_state, reward, done, _ = env.step(action)

        score += reward

        total_state.append(state)
        total_next_state.append(next_state)
        total_done.append(done)
        total_reward.append((reward+8)/8)
        total_action.append(action)

        state = next_state

    if ep  % train_size == 0:
        update_step += 1
        total_state = np.stack(total_state)
        total_next_state = np.stack(total_next_state)
        total_reward = np.stack(total_reward)
        total_done = np.stack(total_done)
        total_action = np.stack(total_action)

        value, next_value = agent.get_value(total_state, total_next_state)
        adv, target = utils.get_gaes(total_reward, total_done, value, next_value, agent.gamma, agent.lamda, False)

        agent.train_model(total_state, total_action, target, adv)
        print(update_step, score/train_size)
        if update_step < 1000000000000:
            writer.add_scalar('data/reward', score/train_size, update_step)
            saver.save(sess, 'pendulum/model')
        total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []
        score = 0
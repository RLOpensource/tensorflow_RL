import gym
from agent.continuous.seperate.ppo import PPO
from agent.continuous.seperate.ddpg import DDPG
from agent import utils
import tensorflow as tf
import numpy as np
from tensorboardX import SummaryWriter
from model import MLPContinuousActor
from model import MLPContinuousCritic
from model import MLPDDPGContinuousActor
from model import MLPDDPGContinuousCritic

writer = SummaryWriter()

state_size = 3
output_size = 1
env = gym.make('Pendulum-v0')
sess = tf.Session()
#old_actor = MLPContinuousActor('old_pi', state_size, output_size)
#actor = MLPContinuousActor('pi', state_size, output_size)
#critic = MLPContinuousCritic('critic', state_size)
#agent = PPO(sess, state_size, output_size, 4, 128, old_actor, actor, critic)
target_actor = MLPDDPGContinuousActor('target_actor',state_size,output_size)
target_critic = MLPDDPGContinuousCritic('target_critic',state_size,output_size)
actor = MLPDDPGContinuousActor('actor',state_size,output_size)
critic = MLPDDPGContinuousCritic('critic',state_size,output_size)
agent = DDPG(sess,state_size,output_size,1,1,target_actor,target_critic,actor,critic)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#saver.restore(sess,'pendulum/model')

ep_len = 300
ep = 0
clip = 1.0
epsilon = 1.0
while True:
    ep += 1
    state = env.reset()
    done = False
    total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []
    score = 0
    for t in range(ep_len):
        env.render()
        action = agent.get_action([state], clip,epsilon)

        action = action[0]
        next_state, reward, done, _ = env.step(2*action)

        score += reward
        reward -= 0.1*np.abs(action)

        agent.get_sample(state,action,reward,next_state,done)
        #total_state.append(state)
        #total_next_state.append(next_state)
        #total_done.append(done)
        #total_reward.append((reward+8)/8)
        #total_action.append(action)

        state = next_state
        if len(agent.memory) >= 1000:
            agent.train_model()

    agent.noise_generator.reset()
    if len(agent.memory) >= 1000:
        epsilon *= 0.995
    #total_state = np.stack(total_state)
    #total_next_state = np.stack(total_next_state)
    #total_reward = np.stack(total_reward)
    #total_done = np.stack(total_done)
    #total_action = np.stack(total_action)

    #value, next_value = agent.get_value(total_state, total_next_state)
    #adv, target = utils.get_gaes(total_reward, total_done, value, next_value, agent.gamma, agent.lamda, False)

    #agent.train_model(total_state, total_action, target, adv)
    print(ep, score, epsilon)
    writer.add_scalar('data/reward', score, ep)
    saver.save(sess, 'pendulum/model')
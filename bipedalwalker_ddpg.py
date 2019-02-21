import gym
from policy.continuous.seperate.ppo import PPO
from policy.continuous.seperate.ddpg import DDPG
from policy import utils
import tensorflow as tf
import numpy as np
from tensorboardX import SummaryWriter
from model import MLPContinuousActor
from model import MLPContinuousCritic
from model import MLPDDPGContinuousActor
from model import MLPDDPGContinuousCritic

writer = SummaryWriter()

state_size = 24
output_size = 4
env = gym.make('BipedalWalker-v2')
sess = tf.Session()
target_actor = MLPDDPGContinuousActor('target_actor',state_size,output_size)
target_critic = MLPDDPGContinuousCritic('target_critic',state_size,output_size)
actor = MLPDDPGContinuousActor('actor',state_size,output_size)
critic = MLPDDPGContinuousCritic('critic',state_size,output_size)
agent = DDPG(sess,state_size,output_size,1,1,target_actor,target_critic,actor,critic)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#saver.restore(sess,'bipedalwalker_ddpg/model')

ep = 0
clip = 1.0
epsilon = 1.0

while True:
    ep += 1
    state = env.reset()
    done = False
    total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []
    score = 0
    while not done:
        #env.render()
        action = agent.get_action([state], epsilon)

        action = action[0]
        next_state, reward, done, _ = env.step(clip * action)

        score += reward

        agent.get_sample(state,action,reward,next_state,done)

        state = next_state
        if len(agent.memory) >= 1000:
            agent.train_model()

    agent.noise_generator.reset()
    if len(agent.memory) >= 1000:
        epsilon *= 0.995
    if ep < 2165:
        print(ep, score, epsilon)
        writer.add_scalar('data/reward', score, ep)
        saver.save(sess, 'bipedalwalker_ddpg/model')
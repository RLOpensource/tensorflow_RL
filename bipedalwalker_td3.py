import gym
from agent.continuous.seperate.ppo import PPO
from agent.continuous.seperate.td3 import TD3
from agent import utils
from example_model.policy.mlp.continuous import MLPDDPGContinuousActor
from example_model.policy.mlp.continuous import MLPTD3ContinousCritic
import tensorflow as tf
import numpy as np
from tensorboardX import SummaryWriter

writer = SummaryWriter()

state_size = 24
output_size = 4
env = gym.make('BipedalWalker-v2')
sess = tf.Session()
target_actor = MLPDDPGContinuousActor('target_actor',state_size,output_size)
target_critic = MLPTD3ContinousCritic('target_critic',state_size,output_size)
actor = MLPDDPGContinuousActor('actor',state_size,output_size)
critic = MLPTD3ContinousCritic('critic',state_size,output_size)
agent = TD3(sess,state_size,output_size,1,1,target_actor,target_critic,actor,critic)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#saver.restore(sess,'bipedalwalker_td3/model')

ep = 0
clip = 1.0
epsilon = 1.0
scores = []
update_step = 0
while True:
    ep += 1
    state = env.reset()
    done = False
    score = 0
    while not done:

        #if ep % 10 == 0:
        #    env.render()
        #env.render()
        action = agent.get_action([state], epsilon)

        action = action[0]
        next_state, reward, done, _ = env.step(clip * action)

        score += reward

        agent.get_sample(state,action,reward,next_state,done)

        state = next_state
        if len(agent.memory) >= 10000:
            agent.train_model()

    
    scores.append(score)
    agent.noise_generator.reset()
    if len(agent.memory) >= 10000 and epsilon >= 0.01:
        epsilon *= 0.999
        epsilon -= 0.000001
    if ep % 10 == 0:
        update_step += 1
        print(update_step, np.mean(scores), epsilon)
        writer.add_scalar('data/reward_per_episode', np.mean(scores), update_step)
        saver.save(sess, 'bipedalwalker_td3/model')
        scores = []
    
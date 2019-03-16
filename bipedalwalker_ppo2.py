
import gym
import tensorflow as tf
from tensorboardX import SummaryWriter
from agent.continuous.seperate.ppo2 import PPO2
from agent.utils import get_gaes
from example_model.policy.mlp.continuous import MLPContinuousActor
from example_model.policy.mlp.continuous import MLPContinuousCritic

env = gym.make('BipedalWalker-v2')
sess = tf.Session()
state_size = 24
output_size = 4
actor = MLPContinuousActor('actor', state_size, output_size)
critic = MLPContinuousCritic('critic', state_size)
agent = PPO2(sess, state_size, output_size, actor, critic)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#saver.restore(sess, 'model/model')

writer = SummaryWriter()

values, states, actions, dones, logp_ts, rewards = [], [], [], [], [], []
o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

rollout = 0
score = 0
ep = 0
n_step = 512
record_score_size = 10
record_score = 0

while True:
    score_rollout = 0
    rollout += 1
    for t in range(n_step):
        #if ep % 10 == 0:
        #    env.render()
        a, v_t, logp_t = agent.get_action([o])
        a, v_t, logp_t = a[0], v_t[0], logp_t[0]
        n_o, r, d, _ = env.step(a)

        score += r
        score_rollout += r
        record_score += r

        values.append(v_t)
        states.append(o)
        actions.append(a)
        dones.append(d)
        rewards.append(r)
        logp_ts.append(logp_t)

        o = n_o

        if d:
            ep += 1
            if ep % record_score_size == 0:
                if int(ep / record_score_size) < 600:
                    writer.add_scalar('data/reward', record_score / record_score_size, int(ep / record_score_size))
                    record_score = 0
            writer.add_scalar('data/reward_per_episode', score, ep)
            print(score, ep)
            score = 0
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    a, v_t, logp_t = agent.get_action([o])
    values.append(v_t[0])
    next_value = values[1:]
    value = values[:-1]
    adv, target = get_gaes(rewards, dones, value, next_value, agent.gamma, agent.lamda, True)
    value_loss, kl, ent = agent.update(states, actions, target, adv, logp_ts, value)

    writer.add_scalar('data/value_loss_per_rollout', value_loss, rollout)
    writer.add_scalar('data/kl_per_rollout', kl, rollout)
    writer.add_scalar('data/ent_per_rollout', ent, rollout)
    writer.add_scalar('data/reward_per_rollout', score_rollout, rollout)

    values, states, actions, dones, logp_ts, rewards = [], [], [], [], [], []
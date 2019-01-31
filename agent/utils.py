import numpy as np
import copy

def get_gaes(rewards, dones, values, next_values, gamma, lamda, normalize):
    deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
    deltas = np.stack(deltas)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(deltas) - 1)):
        gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

    target = gaes + values
    if normalize:
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-30)
    return gaes, target

def get_rtgs(rewards, dones, values, gamma): # get_rtgs for VPG
    deltas = rewards
    deltas = np.stack(deltas)
    rtgs = copy.deepcopy(deltas)
    
    for t in reversed(range(len(deltas) - 1)):
        rtgs[t] = rtgs[t] + (1 - dones[t]) * gamma * rtgs[t + 1]

    advs = rtgs - values
    return advs, rtgs

class OU_noise:
    def __init__(self,action_size,worker_size,mu=0,theta=0.05,sigma=0.2):
        self.action_size = action_size
        self.worker_size = worker_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_size)*self.mu

    def noise(self):
        self.state = (1 - self.theta)*self.state + self.sigma*np.random.randn(self.worker_size,self.action_size)
        return self.state
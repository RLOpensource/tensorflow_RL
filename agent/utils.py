import numpy as np
import copy
import collections

class Memory:
    def __init__(self, max_length, prioritized):
        self.max_length = max_length
        self.prioritized = prioritized
        self.state_memory = collections.deque(maxlen=self.max_length)
        self.action_memory = collections.deque(maxlen=self.max_length)
        self.next_state_memory = collections.deque(maxlen=self.max_length)
        self.reward_memory = collections.deque(maxlen=self.max_length)
        self.done_memory = collections.deque(maxlen=self.max_length)

    def append(self, state, action, next_state, reward, done):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.next_state_memory.append(next_state)
        self.reward_memory.append(reward)
        self.done_memory.append(done)

    def sample(self, sample_number):
        if not self.prioritized:
            length = len(self.state_memory)
            sample_idx = np.random.randint(sample_number, size=length)
            s = [self.state_memory[i] for i in sample_idx]
            n_s = [self.next_state_memory[i] for i in sample_idx]
            r = [self.reward_memory[i] for i in sample_idx]
            a = [self.action_memory[i] for i in sample_idx]
            d = [self.done_memory[i] for i in sample_idx]
            
            return s, a, n_s, r, d


def split_episode(data, done):
    total_data_list = []
    data_list = []
    for i, j in zip(data, done):
        if not j:
            data_list.append(i)
        else:
            data_list.append(i)
            total_data_list.append(data_list)
            data_list = []
    return total_data_list

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

def get_rtgs_like_get_gaes(rewards, dones, values, gamma): # get_rtgs for VPG
    deltas = rewards
    deltas = np.stack(deltas)
    rtgs = copy.deepcopy(deltas)
    
    for t in reversed(range(len(deltas) - 1)):
        rtgs[t] = rtgs[t] + (1 - dones[t]) * gamma * rtgs[t + 1]
    
    advs = rtgs - values        
    return advs, rtgs

def get_rtgs_discount_rollout(rewards, dones, values, gamma):
    total_step = len(rewards)
    targets = copy.deepcopy(rewards)

    for i in reversed(range(total_step - 1)):
        targets[i] = rewards[i] + gamma * targets[i+1] * (1 - dones[i])

    adv = targets - values

    return adv, targets
    
def get_rtgs(rewards, dones, values): # get_rtgs for VPG
    total_step = len(rewards)
    rtgs = np.zeros_like(rewards)
    done_prev = 0
    
    for i in range(total_step):
        if dones[i] == True:
            for j in range(i, done_prev-1, -1):
                rtgs[j] = rewards[j] + (rtgs[j+1] if j+1 <= i else 0)
            done_prev = i+1

    advs = rtgs - values
    return advs, rtgs

def get_rtgs_discount(rewards, dones, values, gamma): # get_rtgs for VPG
    total_step = len(rewards)
    rtgs = np.zeros_like(rewards)
    done_prev = 0

    for i in range(total_step):
        if dones[i] == True:
            for j in range(i, done_prev-1, -1):
                rtgs[j] = rewards[j] + (gamma*rtgs[j+1] if j+1 <= i else 0)
            done_prev = i+1

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
import gym
import cv2
from multiprocessing import Process, Pipe
import numpy as np
import matplotlib.pyplot as plt

class Environment(Process):
    def __init__(self, is_render, env_idx, child_conn):
        super(Environment, self).__init__()
        self.is_render = is_render
        self.env_idx = env_idx
        self.child_conn = child_conn
        self.steps = 0
        self.episode = 0
        self.score = 0
        self.history = np.zeros([84, 84, 4])
        self.env = gym.make('BreakoutDeterministic-v4')
        self.reset()
        self.lives = self.env.env.ale.lives()

    def run(self):
        super(Environment, self).run()
        while True:
            action = self.child_conn.recv()
            if self.is_render == True:
                self.env.render()

            _, reward, done, info = self.env.step(action + 1)

            if self.lives > info['ale.lives'] and info['ale.lives'] > 0:
                force_done = True
                self.lives = info['ale.lives']
            else:
                force_done = done

            if force_done:
                reward = -1

            self.score += reward
            self.history[:, :, :3] = self.history[:, :, 1:]
            self.history[:, :, 3] = self.pre_proc(
                self.env.env.ale.getScreenGrayscale().squeeze().astype('float32'))

            if done:
                self.history = self.reset()

            self.child_conn.send([self.history[:, :, :], reward, force_done, done])

    def reset(self):
        self.episode += 1
        self.env.reset()
        self.lives = self.env.env.ale.lives()
        self.get_init_state(self.env.env.ale.getScreenGrayscale().squeeze().astype('float32'))
        return self.history[:, :, :]

    def pre_proc(self, x):
        x = cv2.resize(x, (84, 84))
        x *= (1.0 / 255.0)
        return x


    def get_init_state(self, s):
        for i in range(4):
            self.history[:, :, i] = self.pre_proc(s)

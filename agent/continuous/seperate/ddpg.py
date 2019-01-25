import numpy as np
import tensorflow as tf
import copy

class DDPG:
    def __init__(self, sess, output_size, num_worker, num_step, old_actor, actor, critic):
        self.sess = sess
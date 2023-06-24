import torch
import numpy as np
from collections import deque
import gym
from gym import spaces


class FrameStack:
    def __init__(self, observation_space, count, flat = True):
        """
        Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        self.count = count
        self.flat = flat
        self.frames = deque([], maxlen=count)
        observation_space = observation_space
        self.orig_shape = shp = observation_space.shape
        #TODO: remove consts -1 and 1
        if flat:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(shp[:-1] + (shp[-1] * count,)), dtype=observation_space.dtype)
        else:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(count, shp[0]), dtype=observation_space.dtype)
        print('self.observation_space: ', self.observation_space)

    def init_obs(self, obs):
        for _ in range(self.count):
            self.frames.append(torch.zeros_like(obs))
        return self._get_ob()

    def get_obs(self, ob, done):
        self.frames.append(ob)
        for i in range(self.count-1):
            self.frames[i] = (1.0 - done.unsqueeze(1)) * self.frames[i]
        return self._get_ob()


    def _get_ob(self):
        assert len(self.frames) == self.count
        if self.flat:
            return torch.cat(list(self.frames), axis=-1)
        else:
            return torch.stack(list(self.frames), axis=1)

import h5py

class SaveH5VecEnv:
    def __init__(self, vec_env):
        self.vec_env = vec_env

    def step(self, actions):
        pass

    def reset(self):
        return self.vec_env.reset()
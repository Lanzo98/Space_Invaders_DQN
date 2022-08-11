from collections import deque

import gym
import numpy as np

from baselines_wrappers import VecEnvWrapper
from baselines_wrappers.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, ScaledFloatFrame,ClipRewardEnv, WarpFrame

import time


def make_atari(env_id, min_y='', max_y='', min_x='', max_x='', crop=False, scale_values=True, clip_rewards=True, render_mode=None):
    env = gym.make(env_id,render_mode=render_mode)
    env = NoopResetEnv(env, noop_max=30)

    env = EpisodicLifeEnv(env)
    
    env = WarpFrame(env, min_y=min_y, max_y=max_y, min_x=min_x, max_x=max_x, crop=crop)

    if scale_values:
        env = ScaledFloatFrame(env)

    if clip_rewards:
        env = ClipRewardEnv(env)

    env = TransposeImageObs(env, op=[2, 0, 1])  # Convert to torch order (C, H, W)

    return env

class TransposeImageObs(gym.ObservationWrapper):
    def __init__(self, env, op):
        super().__init__(env)

        self.op = op

        observation_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [
                observation_shape[self.op[0]],
                observation_shape[self.op[1]],
                observation_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, obs):
        return obs.transpose(self.op[0], self.op[1], self.op[2])


class BatchedFrameStack(VecEnvWrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        super().__init__(env)
        self.k = k
        self.batch_stacks = [deque([], maxlen=k) for _ in range(env.num_envs)]
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[0] * k, shp[1], shp[2]), dtype=env.observation_space.dtype)
        self.env = env

    def reset(self):
        observations = self.env.reset()
        for _ in range(self.k):
            for i, observation in enumerate(observations):
                self.batch_stacks[i].append(observation.copy())
        return self._get_ob()

    def step_wait(self):
        observations, rewards, dones, infos = self.env.step_wait()
        for i, observation_frame in enumerate(observations):
            self.batch_stacks[i].append(observation_frame)

        ret_ob = self._get_ob()
        return ret_ob, rewards, dones, infos

    def _get_ob(self):
        return [LazyFrames(list(batch_stack), axis=0) for batch_stack in self.batch_stacks]

class LazyFrames(object):
    def __init__(self, frames, axis=0):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None
        self.axis = axis

    def __len__(self):
        return len(self.get_frames())

    def get_frames(self):
        """Get Numpy representation without dumping the frames."""
        return np.concatenate(self._frames, axis=self.axis)

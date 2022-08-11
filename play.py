import torch
import itertools
import time
import numpy as np
from baselines_wrappers import DummyVecEnv
from dqn import Network
from space_wrappers import make_atari, BatchedFrameStack, LazyFrames
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

make_env = lambda: make_atari('ALE/SpaceInvaders-v5', min_y=20, max_y=-15, min_x=4, max_x=-15, crop=True, render_mode='human', scale_values=True, clip_rewards=False)

vec_env = DummyVecEnv([make_env])

env = BatchedFrameStack(vec_env, k=4)

net = Network(env, device)
net = net.to(device)

#PATH = './torch_models/space_2.5e-5_cropV2_rew_17.49.pt'
#PATH = './torch_models/space_2.5e-5_cropV2.pt'
PATH = './torch_models/space_2.5e-5_cropV2_rew_17.39.pt'

net.load(PATH)

obs = env.reset()

episode_reward = 0.0
rewards = []
episode_count = 0
for t in itertools.count():
    act_obs = np.stack([o.get_frames() for o in obs])
    action = net.compute_actions(act_obs, 0.0)

    obs, rew, done, info = env.step(action)
    episode_reward += rew
    #time.sleep(0.02)

    if done[0] and info[0]['lives'] == 0:
        print('Reward:', episode_reward)
        rewards.append(episode_reward)
        episode_reward = 0.0
        episode_count += 1
    
    if episode_count == 30:
        break

print('Avg reward', np.mean(np.array(rewards)))
print('Standard deviation:', np.std(np.array(rewards)))
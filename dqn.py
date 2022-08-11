import os
import time
import math
import random
import gym
import torch
import itertools
import numpy as np
from torch import nn
from collections import deque
from matplotlib import pyplot as plt

from space_wrappers import make_atari, BatchedFrameStack, LazyFrames
from baselines_wrappers import DummyVecEnv, Monitor, SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter

NUM_ENVS = 4
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 300000
#MIN_REPLAY_SIZE = 50000
MIN_REPLAY_SIZE = 32
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 300000
TARGET_UPDATE_FREQ = 10000 // NUM_ENVS
LR = 2.5e-5
#SAVE_PATH = './torch_models/space_2.5e-5_cropV2_rew_17.49.pt'
SAVE_PATH = './delete/delete.pt'
SAVE_FREQ = 10000
#LOG_DIR = './logs/space_2.5e-5_cropV2'
LOG_DIR = './logs/delete'
LOG_FREQ = 1000

def make_cnn(observation_space):
    n_input_channels = observation_space.shape[0]   #torch order (C, H, W)

    cnn = nn.Sequential(   
        nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten())

    with torch.no_grad():
        n_flatten = cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
    out = nn.Sequential(
        cnn,
        nn.Linear(n_flatten, 512),
        nn.ReLU())
    
    return out

class Network(nn.Module):
    def __init__(self, env, device, double=False):
        super().__init__()
        
        self.num_actions = env.action_space.n
        self.device = device
        self.double = double

        cnn = make_cnn(env.observation_space)

        self.net = nn.Sequential(cnn, nn.Linear(512, self.num_actions))

    def forward(self, x):
        return self.net(x)

    def compute_actions(self, observations, epsilon):
        observations_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        q_values = self(observations_t)
        max_q_indices = torch.argmax(q_values, dim=1)
        actions = max_q_indices.detach().tolist()
        
        for i in range(len(actions)):
        	random_sample = random.random()
        	if random_sample <= epsilon:
        		actions[i] = random.randint(0, self.num_actions -1)

        return actions
        

    def compute_loss(self, transitions, target_net):
        observations = [t[0] for t in transitions]
        observations = np.stack([o.get_frames() for o in observations])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_observations = [t[4] for t in transitions]        
        new_observations = np.stack([o.get_frames() for o in new_observations])


        observations_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        new_observations_t = torch.as_tensor(new_observations, dtype=torch.float32, device=self.device)

        #targets
        with torch.no_grad():
            if self.double:
                targets_online_q_values = self(new_observations_t)
                targets_online_best_q_indices = targets_online_q_values.argmax(dim=1, keepdim=True)

                targets_target_q_values = target_net(new_observations_t)
                targets_selected_q_values = torch.gather(input=targets_target_q_values,dim=1, index=targets_online_best_q_indices)

                targets = rewards_t + GAMMA * (1 - dones_t) * targets_selected_q_values
            else:
                target_q_values = target_net(new_observations_t)
                max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

                targets = rewards_t + GAMMA * (1 - dones_t) * max_target_q_values
        #loss
        q_values = self(observations_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        return loss

    def save(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)

    def load(self, load_path):
        params = torch.load(load_path)
        self.load_state_dict(params)
        self.eval()


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    #[9:-15,4:-15]   #cropV1
    #[20:-15,4:-15]  #cropV2

    make_env = lambda: Monitor(make_atari('ALE/SpaceInvaders-v5', min_y=20, max_y=-15, min_x=4, max_x=-15, crop=True), allow_early_resets=True)

    #vec_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])     #used for testing
    vec_env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])

    env = BatchedFrameStack(vec_env, k=4)

    replay_buffer = deque(maxlen=BUFFER_SIZE)
    infos_buffer = deque(maxlen=100)

    summary_writer = SummaryWriter(LOG_DIR)

    episode_count = 0

    online_net = Network(env, device)
    target_net = Network(env, device)

    online_net = online_net.to(device)
    target_net = target_net.to(device)

    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

    print('Initialize replay buffer')
    observations = env.reset()

    for i in range(MIN_REPLAY_SIZE):
        actions = [env.action_space.sample() for _ in range(NUM_ENVS)]
        new_observations, rews, dones, _ = env.step(actions)

        #if i == 30:
            #plt.imshow(new_observations[0].get_frames()[0],cmap='gray')
            #plt.show()
            #matplotlib.image.imsave('original.png', new_observations[0].get_frames()[0])
            #time.sleep(100)
        
        for observation, action, rew, done, new_observation in zip(observations, actions, rews, dones, new_observations):
            transition = (observation, action, rew, done, new_observation)
            replay_buffer.append(transition)
            
        observations = new_observations

    print('Main Training')   
    observations = env.reset()

    for step in itertools.count():
        epsilon = np.interp(step * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

        random_sample = random.random()

        act_observations = np.stack([o.get_frames() for o in observations])
        actions = online_net.compute_actions(act_observations, epsilon)

        new_observations, rews, dones, infos = env.step(actions)
        
        for observation, action, rew, done, new_observation, info in zip(observations, actions, rews, dones, new_observations, infos):
            transition = (observation, action, rew, done, new_observation)
            replay_buffer.append(transition)
            if done:
                infos_buffer.append(info['episode'])
                episode_count += 1
            
        observations = new_observations

        transitions = random.sample(replay_buffer, BATCH_SIZE)
        loss = online_net.compute_loss(transitions, target_net)

        # gradient descent step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update target network
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

        # log
        if step % LOG_FREQ == 0:
            mean_reward = np.mean([e['r'] for e in infos_buffer])
            mean_len = np.mean([e['l'] for e in infos_buffer])

            if math.isnan(mean_reward):
                mean_reward = 0

            if math.isnan(mean_len):
                mean_len = 0
            
            print()
            print('Step:', step)
            print('Avg Reward:', mean_reward)   
            print('Avg Episode Length:', mean_len)                     
            print('Episodes', episode_count)
            print('Saving Summary...')
            print()
            
            summary_writer.add_scalar('AvgReward', mean_reward, global_step = step)
            summary_writer.add_scalar('AvgEpLen', mean_len, global_step = step)
            summary_writer.add_scalar('Episodes', episode_count, global_step = step)
            
        # save model
        if step % SAVE_FREQ == 0 and step != 0:
            print('Saving model...')
            online_net.save(SAVE_PATH)









import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np




def make_env(env_name):
    env = gym.make(env_name)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env) 
    env = BufferWraper(env, 4)
    return ScaledFloatFrame(env)

env = make_env("ALE/Breakout-v5")

plt.ion()

# We would like to use MPS for MAC or CPU
#device = torch.device("mps")
device = torch.device("cpu")


# This is a repdocution of the the Deep Q-Network (DQN) algorithm from the paper
# Playing Atari with Deep Reinforcement Learning https://arxiv.org/abs/1312.5602
# Based on the PyTorch tutorial https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# Expereince or state action described by e_t = (s_t, a_t, r_t, s_t+1) 
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

# Section 4
# Replay Memory D = e1, ..., eN of transitions (si, ai, ri, si+1)
# Memory pool used to for the experience replay idea e ~ D
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
# Model Architecture as described in section 4.1
# Input: 84x84 image
# First Layer: 16 8x8 filters with stride 4
# Second Layer: 32 4x4 filters with stride 2
# Third Layer: 256 fully connected
# Output Layer: fully connected linear layer with a single output for each valid action
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer2 = nn.Conv2d(8, 8, 16, stride=4, device=device)
        self.layer3 = nn.Conv2d(4, 4, 32, stride=2, device=device)
        self.layer4 = nn.Linear(4, 256, device=device)
        self.layer5 = nn.Linear(256, n_actions, device=device)

    # Forwrard pass RELUs added two layer 2 and 3
    def forward(self, x):
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return self.layer5(x)
    


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(84, 84, 1), 
                                                dtype=np.uint8)
        
    def observation(self, obs):
        return ProcessFrame84.process(obs)
    
    @staticmethod
    def process(frame):
        img = np.reshape(frame, [210, 160,  3]).astype(np.float32)
        return img

    

class BufferWraper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWraper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.oservation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                               old_space.high.repeat(n_steps, axis=0), 
                                               dtype=dtype)
        
    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())
    
    def observeration(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer
    
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,            
                                shape=(old_shape[-1], 
                                old_shape[0], old_shape[1]),
                                dtype=np.float32)
        
        def observation(self, observation):
            return np.moveaxis(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
     def observation(self, obs):
         return np.array(obs).astype(np.float32) / 255.0
     


# Hyper Parameters
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(state, n_actions).to(device)
target_net = DQN(state, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def preprocess(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((110, 84)),
        transforms.CenterCrop((84, 84)),
    ])
    tensor = transform(image)
    return tensor

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated



def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


num_episodes = 600

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    print(state.shape)
    state = torch.tensor(state, dtype=torch.float32, device=device)
    torch.flip(state, [0])
    state = preprocess(state)

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            next_state = preprocess(observation)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)

        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break


print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()


# Create simulation for the agent, and test it
env = gym.make("ALE/Breakout-v5", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
    observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    action = select_action(observation) 
    observation, reward, terminated, truncated, info = env.step(action.item())

    if terminated or truncated:
        observation, info = env.reset()
      
env.close()

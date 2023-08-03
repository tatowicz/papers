import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib


# See if the environment is working and check basic
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    print("IPython envionment detected")


# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print("Using device: ", device)

# if MPS for accleration on MAC is available
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print("Example Tensor:", x)
else:
    print ("MPS device not found.")

# Open up gym environment and test basic functionality
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(100):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()


env.close()
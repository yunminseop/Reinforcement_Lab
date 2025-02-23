"""pip install gym
in your virtual environment"""

import gymnasium as gym
import numpy as np
my_bool = np.bool_(True) 


env = gym.make("FrozenLake-v1", is_slippery=False)
print(env.observation_space)
print(env.action_space)

n_trial = 20

env.reset()
episode=[]

for i in range(n_trial):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    episode.append([action, reward, obs])
    env.render()
    if done:
        break

print(episode)
env.close()
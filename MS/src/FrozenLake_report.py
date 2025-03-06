import gymnasium as gym
from gymnasium.envs.registration import register
import sys, tty, termios
import numpy as np
import matplotlib.pyplot as plt
import random


def rargmax(vector):
    m = np.argmax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)


register(
    id="FrozenLake-v3",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name" : "4x4", "is_slippery" : False}
)

env = gym.make("FrozenLake-v1", render_mode="human")

Q = np.zeros([env.observation_space.n, env.action_space.n])

print(Q[0], Q[1])
num_episodes = 2000

rList = []
for i in range(num_episodes):
    state, _ = env.reset()
    state = int(state)
    rAll = 0
    done = False

    print(f"state: {state}")
    while not done:
        action = rargmax(Q[state, :]) #rargmax = random or argmax

        new_state, reward, terminated, truncated, info = env.step(action)

        Q[state, action] = reward + np.max(Q[new_state, :])

        state = new_state

    rList.append(rAll)

    print(f"Success rate: {str(sum(rList)/num_episodes)}")
    print("Final Q-Table Values")
    print("Left Down Right Up")
    print(Q)
    plt.bar(range(len(rList)), rList, color="blue")
    plt.show()
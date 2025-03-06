import gymnasium as gym
from gymnasium.envs.registration import register
import sys, tty, termios
import numpy as np
import matplotlib.pyplot as plt
import random

class _Getch:
    def __call__(Self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

inkey = _Getch()

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    '\x2b[A' : UP,
    '\x2b[B' : DOWN,
    '\x2b[C' : RIGHT,
    '\x2b[D' : LEFT}

register(
    id="FrozenLake-v3",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name" : "4x4", "is_slippery" : False}
)

env = gym.make("FrozenLake-v1", render_mode="human")
env.reset()
env.render()

while True:
    key = inkey()
    if key not in arrow_keys.keys():
        print("Game Aborted!")
        break

    action = arrow_keys[key]
    state, reward, terminated, truncated, info = env.step(action)

    env.render()
    print("State:", state, "Action:", action, "Reward:", reward, "Info:", info)

    if terminated or truncated:
        print("Finished with reward", reward)
        break
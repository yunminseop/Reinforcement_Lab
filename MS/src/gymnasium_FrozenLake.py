import gymnasium
from gymnasium.envs.registration import register
import sys, tty, termios

register(
    id="FrozenLake-v3",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name" : "4x4", "is_slippery" : False}
)

env = gymnasium.make("FrozenLake-v3")
env.render()

while True:
    key = inkey()
    if key not in arrow_keys.keys():
        print("Game Aborted!")
        break

    action = arrow_keys[keys]
    state, reward, terminated, truncated, info = env.step(action)

    env.render()
    print("State:", state, "Action:", action, "Reward:", reward, "Info:", info)

    if terminated or truncated:
        print("Finished with reward", reward)
        break
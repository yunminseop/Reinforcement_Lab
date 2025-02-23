import numpy as np
import gym
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

model = load_model("./MS/data/cartpole_by_DQN.h5", custom_objects={"mse": MeanSquaredError()})

env = gym.make("CartPole-v0")
long_reward = 0

s, _ = env.reset()
while True:
    q = model.predict(np.reshape(s, [1, 4]))
    a = np.argmax(q[0])
    s1, r, done, truncate, _ = env.step(a)
    done = done or truncate

    s = s1
    long_reward += r

    env.render()
    time.sleep(0.02)

    if done:
        print("episode score:", long_reward)
        break

env.close()
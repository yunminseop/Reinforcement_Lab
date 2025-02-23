"""it needs venv for neural network"""

import numpy as np
import random
import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

alpha = 0.9 # learning rate
gamma = 0.99
epsilon = 0.9
eps_decay = 0.999
batch_size = 64
n_episode = 100

def deep_network():
    mlp=Sequential()
    mlp.add(Dense(32, input_dim=env.observation_space.shape[0], activation="relu"))
    mlp.add(Dense(32, activation="relu"))
    mlp.add(Dense(env.action_space.n, activation="linear"))
    mlp.compile(loss="mse", optimizer="Adam")
    return mlp

def model_learning():
    mini_batch = np.asarray(random.sample(D, batch_size))
    state = np.asarray([mini_batch[i, 0] for i in range(batch_size)])
    action = mini_batch[:, 1]
    reward = mini_batch[:, 2]
    state1 = np.asarray([mini_batch[i, 3] for i in range(batch_size)])
    done = mini_batch[:, 4]

    target = model.predict(state)
    target1 = model.predict(state1)

    for i in range(batch_size):
        if done[i]:
            target[i][action[i]] = reward[i]

        else:
            target[i][action[i]] += alpha*((reward[i] + gamma*np.amax(target1[i])) - target[i][action[i]])

    model.fit(state, target, batch_size=batch_size, epochs=1, verbose=0)

env = gym.make("CartPole-v0")

model = deep_network()
D = deque(maxlen=2000)
scores = []
max_steps = env.spec.max_episode_steps

for i in range(n_episode):
    s, _ = env.reset()
    long_reward = 0

    while True:
        r = np.random.random()
        epsilon = max(0.01, epsilon*eps_decay)
        if (r < epsilon):
            a = np.random.randint(0, env.action_space.n)
        else:
            q = model.predict(np.reshape(s, [1, 4]))
            a = np.argmax(q[0])
        
        s1, r, done, truncated, _ = env.step(a)
        done = done or truncated
        # done = np.bool_(done) 


        if done and long_reward < max_steps - 1:
            r = -100

        D.append((s, a, r, s1, done))

        if len(D) > batch_size*3:
            model_learning()

        s = s1
        long_reward += r

        if done:
            long_reward = long_reward if long_reward == max_steps else long_reward + 100
            print(f"Score in episode {i}: {long_reward}")
            scores.append(long_reward)
            break

    if i>10 and np.mean(scores[-5:]) > (0.95*max_steps):
        break


model.save("./cartpole_by_DQN.h5")
env.close()

import matplotlib.pyplot as plt

plt.plot(range(1, len(scores) + 1), scores)
plt.title("DQN scores for CartPole-v0")
plt.ylabel("Score")
plt.xlabel("Episode")
plt.grid()
plt.show()